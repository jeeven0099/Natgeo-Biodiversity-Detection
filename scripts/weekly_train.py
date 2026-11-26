"""Weekly refit of the review app's learners, with a swap rule that protects what is live.

    python -X utf8 scripts/weekly_train.py                      # the server's weekly run (natgeo-train.timer)
    python -X utf8 scripts/weekly_train.py --no-swap            # fit and report, leave the live files alone
    python -X utf8 scripts/weekly_train.py --rebaseline         # once: replace live fits trained on the test set

Steps, one learner at a time (memory: the app keeps running beside this):
  1. lock      data/review/weekly/.lock (pid + time); a lock older than 24 h, or whose process is gone
               (a run that was killed), is taken as stale.
  2. snapshot  annotation_log.csv, box_shown.csv, species_shown.csv, audit_windows.csv (when present)
               into data/review/weekly/<stamp>/logs/. The app appends to them while this runs, so each
               copy is cut after its last complete line and rows whose field count is not the header's
               are dropped. Every learner then reads the same snapshot.
  3. fit       review_taxon.py, review_species.py (--cache data/review/embeddings), box_refine.py
               (--cache data/review/refine_cache) and review_calibrate.py (a report only: the app does not
               read it), each with --out data/review/weekly/<stamp>/run/<learner> and --uploads-dir, so
               uploads count as well as the survey recordings (review_sources.py). Output goes to
               run/<learner>.log; seconds, return code and peak memory go into the report. The learners
               leave out the frozen test recordings (config/frozen_test.csv; review_labels.py).
  4. score     when data/review/frozen_test/ holds a snapshot (frozen_test.py freeze): frozen_test.py score
               --fit live=calibration --fit new=run, into <stamp>/frozen_test/ (log <stamp>/frozen_test.log),
               its scores.txt appended to the report. Report only: it never decides a swap, and a failed
               score is exit 1 but swaps still go ahead.
  5. swap      taxon, species and refiner only. Never a fit whose spec's held_out marker is not
               config/frozen_test.csv's (it may have trained on the frozen recordings); a live fit
               whose marker is not that list's is kept, with the reason, until a --rebaseline run.
               --rebaseline (once, after the frozen test set is set up): every new fit is swapped in
               when its own adopted units have their interval low end above 0, without the live checks
               below; a unit the new fit does not adopt falls back to zero-shot in the app.
               Otherwise only when the new fit
                 - adopts every group (taxon, refiner) or the learner (species) that the live fit adopts,
                 - has at least the live support there: taxon pos and recordings per adopted group,
                   species n_rows and n_recordings, refiner n_boxes and n_recordings per adopted group,
                 - and (taxon, species) has its adopted interval low end above 0.
               Same counts and same decisions as live = nothing new to learn: no swap. A missing live file
               = swap. Anything else keeps the live files and lists the reasons in the report: a
               regression is for the user to look at, never applied on its own.
               A swap copies the live files to data/review/calibration_history/<stamp>/<learner>/ and
               moves the new ones in (tmp + os.replace, the json last).
  6. report    data/review/weekly/<stamp>/report.json + report.txt; data/review/weekly/restart-app is
               written when something was swapped (the service restarts callfinder on it, since the app
               reads the learners only at start).
Older runs past --keep (default 12) are removed from weekly/ and calibration_history/.
Exit 0 when every learner ran (swapped or not), 1 when one failed, 2 when another run holds the lock.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import frozen_test as FT                                 # stdlib only at import (scripts/ is sys.path[0])

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
REVIEW_DIR = ROOT / "data" / "review"
UPLOADS_DIR = ROOT / "data" / "uploads"
LOGS = ("annotation_log.csv", "box_shown.csv", "species_shown.csv", "audit_windows.csv")
STALE_S = 24 * 3600
KEEP = 12
RESTART_FLAG = "restart-app"

# learner -> (script, files it writes that the swap moves in; the json goes last)
LEARNERS = {
    "taxon": ("review_taxon.py", ("taxon.npz", "taxon_eval.csv", "taxon_segments.csv", "taxon.json")),
    "species": ("review_species.py", ("species_protos.npy", "species_eval.csv", "species_tau.csv", "species.json")),
    "refiner": ("box_refine.py", ("refine_settings.csv", "refine_boxes.csv", "refine_grid.csv", "refiner.json")),
    "calibrate": ("review_calibrate.py", ()),            # report only: not swapped
}
SPEC = {"taxon": "taxon.json", "species": "species.json", "refiner": "refiner.json"}


# ---------------------------------------------------------------- lock

def _alive(pid: int) -> bool:
    """Is pid running? POSIX: signal 0. Windows: psutil when present (os.kill there would end the
    process), else assume it is, so only the 24 h age frees the lock."""
    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    try:
        import psutil
    except ImportError:
        return True
    return psutil.pid_exists(pid)


class Lock:
    def __init__(self, path: Path):
        self.path, self.held = path, False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                age = time.time() - self.path.stat().st_mtime
                pid = self._pid()
                if age < STALE_S and (pid is None or _alive(pid)):
                    return False
                why = f"{age / 3600:.1f} h old" if age >= STALE_S else f"its process {pid} is gone"
                print(f"lock {self.path} is stale ({why}): removed")
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w") as f:
                f.write(f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}\n")
            self.held = True
            return True
        return False

    def _pid(self) -> int | None:
        try:
            return int(self.path.read_text().split()[0])
        except (OSError, ValueError, IndexError):
            return None

    def release(self) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False


# ---------------------------------------------------------------- snapshot

def snapshot_csv(src: Path, dst: Path) -> dict:
    """Copy src as it stands, cut after its last complete line; drop rows whose field count is not the
    header's (a row the app was writing at that moment). Returns counts for the report."""
    raw = src.read_bytes()
    cut = raw.rfind(b"\n") + 1
    rows = list(csv.reader(io.StringIO(raw[:cut].decode("utf-8"), newline="")))
    if not rows:
        dst.write_bytes(b"")
        return {"rows": 0, "dropped": 0, "partial_bytes": len(raw) - cut}
    head, body = rows[0], rows[1:]
    good = [r for r in body if len(r) == len(head)]
    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(head)
    w.writerows(good)
    dst.write_text(buf.getvalue(), encoding="utf-8", newline="")
    return {"rows": len(good), "dropped": len(body) - len(good), "partial_bytes": len(raw) - cut}


def snapshot(review: Path, logs: Path) -> dict:
    logs.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in LOGS:
        src = review / name
        out[name] = snapshot_csv(src, logs / name) if src.exists() else None
    return out


# ---------------------------------------------------------------- run one learner

def run_measured(cmd: list[str], log: Path, cwd: Path) -> dict:
    """Run cmd with stdout+stderr to log; seconds, return code and peak resident memory (MB) of the child."""
    t0 = time.time()
    with log.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                             env={**os.environ, "PYTHONUNBUFFERED": "1"})   # the log fills as it runs
        peak = None
        if hasattr(os, "wait4"):
            _, status, ru = os.wait4(p.pid, 0)
            p.returncode = os.waitstatus_to_exitcode(status)
            peak = ru.ru_maxrss / 1024                       # Linux: KB
        else:
            try:                                         # Windows: a venv python.exe is a launcher that starts
                import psutil                            # the real interpreter as its child, so take the tree
                proc, seen = psutil.Process(p.pid), {}
                while p.poll() is None:
                    try:
                        for q in [proc] + proc.children(recursive=True):
                            try:
                                seen[q.pid] = max(seen.get(q.pid, 0), q.memory_info().peak_wset)
                            except psutil.Error:
                                pass
                    except psutil.Error:
                        pass
                    peak = max(seen.values()) / 2 ** 20 if seen else None
                    time.sleep(0.5)
            except ImportError:
                pass
            p.wait()
    return {"seconds": round(time.time() - t0, 1), "returncode": p.returncode,
            "peak_mb": None if peak is None else round(peak, 1)}


def learner_cmd(name: str, py: str, logs: Path, out: Path, review: Path, uploads: Path) -> list[str]:
    script = LEARNERS[name][0]
    cmd = [py, "-X", "utf8", str(SCRIPTS / script), "--log", str(logs / "annotation_log.csv"),
           "--out", str(out), "--uploads-dir", str(uploads)]
    if name in ("species", "refiner"):
        cmd += ["--audit", str(logs / "audit_windows.csv"), "--shown", str(logs / "box_shown.csv")]
    if name == "calibrate":
        cmd += ["--shown", str(logs / "box_shown.csv")]
    if name == "species":
        cmd += ["--species-shown", str(logs / "species_shown.csv"), "--cache", str(review / "embeddings")]
    if name == "refiner":
        cmd += ["--cache", str(review / "refine_cache")]
    return cmd


# ---------------------------------------------------------------- swap rule

def _units(name: str, spec: dict) -> dict:
    """unit -> (adopted, support counts compared for the swap, every count for 'nothing new', ci_lo)."""
    if name == "species":
        m = spec.get("metrics", {})
        sup = {k: m.get(k) for k in ("n_rows", "n_recordings")}
        allc = {k: m.get(k) for k in ("n_rows", "n_boxes", "n_recordings", "n_eval")}
        return {"species": (spec.get("decision") == "adopt", sup, allc, m.get("ci_lo"))}
    out = {}
    for g, e in spec.get("groups", {}).items():
        if name == "taxon":
            sup = {k: e.get(k) for k in ("pos", "recordings")}
            allc = {k: e.get(k) for k in ("pos", "negative", "unlabelled", "recordings", "negative_recordings")}
            ci = e.get("ci_lo")
        else:
            sup = {k: e.get(k) for k in ("n_boxes", "n_recordings")}
            allc, ci = dict(sup), None
        out[g] = (e.get("decision") == "adopt", sup, allc, ci)
    return out


def swap_decision(name: str, live: dict | None, new: dict, held_sha: str | None = None,
                  rebaseline: bool = False) -> tuple[str, list[str]]:
    """('swap' | 'same' | 'keep', reasons). held_sha: frozen_test.marker()'s sha256 of
    config/frozen_test.csv (None: no list); a fit is comparable only when its spec's held_out marker
    is that one. rebaseline: the live fit was trained on other recordings (the frozen ones), so the
    live adoption and support checks are skipped once; the new fit's own checks still hold."""
    if FT.marker_sha(new) != held_sha:
        return "keep", [f"the new fit's held_out marker {FT.marker_sha(new)} is not config/frozen_test.csv's "
                        f"{held_sha}: it may have trained on the frozen test recordings"]
    if live is None:
        return "swap", ["no live file"]
    L, N = _units(name, live), _units(name, new)
    reasons = []
    if rebaseline:
        reasons = [f"{u}: adopted with interval low end {ci}, not above 0" for u, (a, _, _, ci) in N.items()
                   if a and name != "refiner" and not (ci is not None and ci > 0)]
        if reasons:
            return "keep", reasons
        lost = [u for u, (a, *_) in L.items() if a and not N.get(u, (False,))[0]]
        return "swap", ["rebaseline: live replaced by the fit without the frozen test recordings"] + (
            [f"no longer adopted (zero-shot from now on): {', '.join(lost)}"] if lost else [])
    if FT.marker_sha(live) != held_sha:
        return "keep", ["the live fit trained on the frozen test recordings (its held_out marker is not "
                        "config/frozen_test.csv's): run weekly_train.py --rebaseline once"]
    for u, (adopted, sup, _, _) in L.items():
        if not adopted:
            continue
        if u not in N or not N[u][0]:
            reasons.append(f"{u}: live adopts, the new fit does not")
            continue
        for k, v in sup.items():
            nv = N[u][1].get(k)
            if v is not None and (nv is None or nv < v):
                reasons.append(f"{u}: {k} {nv} < live {v}")
    for u, (adopted, _, _, ci) in N.items():
        if adopted and name != "refiner" and not (ci is not None and ci > 0):
            reasons.append(f"{u}: adopted with interval low end {ci}, not above 0")
    if reasons:
        return "keep", reasons
    if {u: (a, c) for u, (a, _, c, _) in L.items()} == {u: (a, c) for u, (a, _, c, _) in N.items()}:
        return "same", ["same counts and decisions as live: nothing new to learn"]
    gained = [u for u, (a, *_) in N.items() if a and not L.get(u, (False,))[0]]
    return "swap", ["support kept or grown on every adopted unit"] + ([f"newly adopted: {', '.join(gained)}"]
                                                                     if gained else [])


def adopted(name: str, spec: dict | None) -> list[str]:
    return [] if spec is None else [u for u, (a, *_) in _units(name, spec).items() if a]


def swap_in(name: str, new_dir: Path, cal: Path, hist: Path) -> list[str]:
    files = LEARNERS[name][1]
    missing = [f for f in files if not (new_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"{name}: the new fit lacks {missing}")
    old = [f for f in files if (cal / f).exists()]
    if old:
        (hist / name).mkdir(parents=True, exist_ok=True)
        for f in old:
            shutil.copy2(cal / f, hist / name / f)
    cal.mkdir(parents=True, exist_ok=True)
    for f in files:                                       # the json last: a half swap leaves the old spec
        tmp = cal / (f + ".weekly-tmp")
        shutil.copy2(new_dir / f, tmp)
        os.replace(tmp, cal / f)
    return list(files)


def load_json(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def prune(folder: Path, keep: int) -> list[str]:
    if not folder.exists():
        return []
    stamps = sorted(d for d in folder.iterdir() if d.is_dir() and d.name[:8].isdigit())
    gone = stamps[:-keep] if keep > 0 else []
    for d in gone:
        shutil.rmtree(d)
    return [d.name for d in gone]


# ---------------------------------------------------------------- main

def report_text(rep: dict) -> str:
    out = [f"weekly training {rep['stamp']} (UTC), swap {'on' if rep['swap'] else 'off'}"
           + (", REBASELINE" if rep.get("rebaseline") else ""), ""]
    for n, s in (rep["snapshot"] or {}).items():
        out.append(f"  {n}: " + ("absent" if s is None else
                                 f"{s['rows']} rows, {s['dropped']} dropped, {s['partial_bytes']} partial bytes"))
    out.append("")
    for n, r in rep["learners"].items():
        out.append(f"{n}: rc {r['returncode']}, {r['seconds']} s, peak {r['peak_mb']} MB -> {r.get('result', '')}")
        if "adopted_live" in r:
            out.append(f"    adopted live: {', '.join(r['adopted_live']) or 'none'}; "
                       f"new: {', '.join(r['adopted_new']) or 'none'}")
        for why in r.get("reasons", []) + r.get("left_out", []):
            out.append(f"    {why}")
    out += ["", f"swapped: {', '.join(rep['swapped']) or 'nothing'}; restart flag: {rep['restart_flag']}",
            f"pruned: {rep['pruned']}", ""]
    ft = rep.get("frozen_test")
    if ft is not None:
        out.append(f"frozen test score (report only; live = before this run's swap): rc {ft['returncode']}, "
                   f"{ft['seconds']} s, peak {ft['peak_mb']} MB")
        txt = Path(ft["out"]) / "scores.txt"
        out += [txt.read_text(encoding="utf-8") if txt.exists() else "  no scores.txt: see frozen_test.log", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review-dir", type=Path, default=REVIEW_DIR, help="logs, calibration/ and caches")
    ap.add_argument("--uploads-dir", type=Path, default=UPLOADS_DIR, help="audio added in the app")
    ap.add_argument("--learners", nargs="+", choices=list(LEARNERS), default=list(LEARNERS))
    ap.add_argument("--no-swap", action="store_true", help="fit and report only")
    ap.add_argument("--rebaseline", action="store_true",
                    help="once, after the frozen test set is set up: swap in every valid new fit without "
                         "comparing it to the live one (which trained on the frozen recordings)")
    ap.add_argument("--keep", type=int, default=KEEP, help=f"weekly runs and history kept (default {KEEP})")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    review = args.review_dir.resolve()
    weekly, cal, hist_root = review / "weekly", review / "calibration", review / "calibration_history"
    lock = Lock(weekly / ".lock")
    if not lock.acquire():
        print(f"another weekly run holds {lock.path}; nothing done")
        return 2
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        base = weekly / stamp
        logs, run = base / "logs", base / "run"
        run.mkdir(parents=True, exist_ok=True)
        rep = {"stamp": stamp, "swap": not args.no_swap, "rebaseline": args.rebaseline, "review_dir": str(review),
               "uploads_dir": str(args.uploads_dir.resolve()), "snapshot": snapshot(review, logs),
               "learners": {}, "swapped": [], "restart_flag": False, "pruned": {}}
        print(f"snapshot -> {logs}")
        failed = False
        held_sha = (FT.marker() or {}).get("sha256")
        for name in args.learners:                        # 3. fit
            out = run / name
            cmd = learner_cmd(name, args.python, logs, out, review, args.uploads_dir.resolve())
            print(f"{name}: running ({run / (name + '.log')})", flush=True)
            r = run_measured(cmd, run / f"{name}.log", ROOT)
            rep["learners"][name] = r
            r["left_out"] = [ln.strip() for ln in (run / f"{name}.log").read_text(encoding="utf-8").splitlines()
                             if ln.startswith(("left out", "held out"))]   # e.g. survey boxes with no audio here
            print(f"  rc {r['returncode']}, {r['seconds']} s, peak {r['peak_mb']} MB", flush=True)
        fits = [n for n in args.learners if n in SPEC and rep["learners"][n]["returncode"] == 0]
        if (review / FT.SNAP / "manifest.json").exists() and fits:   # 4. frozen test score (report only)
            cmd = [args.python, "-X", "utf8", str(SCRIPTS / "frozen_test.py"), "score", "--review-dir", str(review),
                   "--uploads-dir", str(args.uploads_dir.resolve()), "--fit", f"live={cal}", "--fit", f"new={run}",
                   "--learners", *fits, "--out", str(base / "frozen_test")]
            print(f"frozen test: scoring ({base / 'frozen_test.log'})", flush=True)
            ft = run_measured(cmd, base / "frozen_test.log", ROOT)
            ft["out"] = str(base / "frozen_test")
            rep["frozen_test"] = ft
            print(f"  rc {ft['returncode']}, {ft['seconds']} s, peak {ft['peak_mb']} MB", flush=True)
            failed |= ft["returncode"] != 0               # a failed score is reported; it never blocks a swap
        for name in args.learners:                        # 5. swap
            out, r = run / name, rep["learners"][name]
            if r["returncode"] != 0:
                failed = True
                r["result"] = "failed: live files kept"
                continue
            if name not in SPEC:
                r["result"] = "report only"
                continue
            live, new = load_json(cal / SPEC[name]), load_json(out / SPEC[name])
            if new is None:
                failed = True
                r["result"] = f"failed: no {SPEC[name]} written; live files kept"
                continue
            decision, reasons = swap_decision(name, live, new, held_sha, args.rebaseline)
            r.update(adopted_live=adopted(name, live), adopted_new=adopted(name, new), reasons=reasons)
            if decision == "swap" and not args.no_swap:
                r["files"] = swap_in(name, out, cal, hist_root / stamp)
                rep["swapped"].append(name)
                r["result"] = "swapped"
            elif decision == "swap":
                r["result"] = "would swap (--no-swap)"
            else:
                r["result"] = {"same": "no swap: nothing new", "keep": "kept live: see reasons"}[decision]
            print(f"{name}: {r['result']}", flush=True)
        if rep["swapped"]:
            (weekly / RESTART_FLAG).write_text(stamp + "\n", encoding="utf-8")
            rep["restart_flag"] = True
        rep["pruned"] = {"weekly": prune(weekly, args.keep), "calibration_history": prune(hist_root, args.keep)}
        (base / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        txt = report_text(rep)
        (base / "report.txt").write_text(txt, encoding="utf-8")
        print("\n" + txt)
        return 1 if failed else 0
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
