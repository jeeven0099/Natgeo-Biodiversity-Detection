"""Frozen held-out test set: a fixed snapshot of the test recordings' reviews, scored, never trained on.

    python -X utf8 scripts/frozen_test.py freeze              # once, on the server (its log is the authoritative one)
    python -X utf8 scripts/frozen_test.py freeze --refresh    # a new snapshot; the old one moves to frozen_test_history/
    python -X utf8 scripts/frozen_test.py score               # the live learners (data/review/calibration)
    python -X utf8 scripts/frozen_test.py score --fit live=data/review/calibration --fit new=data/review/weekly/<stamp>/run

The recordings: config/frozen_test.csv, the 34 recordings of the test split in data/splits.csv (user
decision 2026-11-25). Everything else follows from that list:
  training   review_labels.labels_from_logs leaves every row of these recordings out, whenever it was
             logged (all four learners; review_labels.py --include-test keeps them), and each learner
             writes the list's marker (held_out: list, n_recordings, sha256) into its spec.
             weekly_train.py swaps in only a fit whose marker is the list's.
  reviewing  the reviewer page does not list these recordings (review_app.py /api/meta).
  freeze     copies the rows of these recordings in annotation_log.csv and audit_windows.csv, and the
             box_shown.csv / species_shown.csv rows of those log rows (by log_id), as they stand, to
             data/review/frozen_test/, with manifest.json (frozen_at, the list and its sha256, per file
             the rows kept and their sha256, the last logged_at). A copy is cut after its last
             complete line, and rows whose field count is not the header's are dropped (as
             weekly_train.py snapshots). Rows logged later stay in the logs, used by neither training
             nor this test until a --refresh. Refuses when a snapshot exists.
  score      rebuilds the labels from the snapshot (review_labels.labels_from_logs, the same rules,
             the hold-out off) and scores each fit (--fit LABEL=DIR; the learner files in DIR or
             DIR/<learner>/) as the app would use it, against the zero-shot baseline:
               taxon    per group, AUROC positive vs rest over the test recordings' 2.5 s segments
                        (review_taxon.group_data: positives, the "no" segments, the unlabelled rest
                        of recordings holding a positive). The app's score is the head probability
                        when the group is adopted, else the zero-shot score. AUROC P vs N is also shown.
               species  MRR@5, hit@1 and hit@5 of the named names in the list the app shows (learned
                        names >= tau when adopted, then the logged zero-shot top 5), on rows where
                        the reviewer saw no learned name (review_species' evaluation rows).
               refiner  mean IoU with the accepted box of what the app shows (refined when the
                        group's spec is adopted and confident, else the grid box) vs the grid box,
                        on boxes shown plain (box_refine.CONTROL_SHOWN).
             Each comes with a 95% bootstrap interval over the test recordings (BOOT draws, SEED):
             app - zero-shot per fit, and later fits - the first fit on the same rows (paired).
             A fit whose marker is not the snapshot's is flagged: it was trained on these recordings.
Report only: nothing here adopts or swaps. Adoption stays on each learner's cross-validation over
the training recordings.
score writes to --out (default data/review/frozen_test_scores/<stamp>/): scores.json, scores.txt
and per-row CSVs (taxon_rows.csv, species_rows.csv, refiner_rows.csv). Box embeddings and refined
boxes missing from the caches (data/review/embeddings, refine_cache) are computed from the audio
and added to the caches, as the learners do.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIST_FILE = ROOT / "config" / "frozen_test.csv"
REVIEW_DIR = ROOT / "data" / "review"
UPLOADS_DIR = ROOT / "data" / "uploads"
SNAP, HISTORY, SCORES = "frozen_test", "frozen_test_history", "frozen_test_scores"   # under the review dir
LOG, AUDIT, SHOWN, SPECIES_SHOWN = "annotation_log.csv", "audit_windows.csv", "box_shown.csv", "species_shown.csv"
LEARNER_FILE = {"taxon": "taxon.json", "species": "species.json", "refiner": "refiner.json"}
BOOT, SEED = 2000, 0


# ---- the list -------------------------------------------------------------------------------
def recordings(path: Path = LIST_FILE) -> list[str]:
    """The frozen test recordings, sorted ([] without the file)."""
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8", newline="") as f:
        return sorted({r["recording"].strip() for r in csv.DictReader(f) if (r.get("recording") or "").strip()})


def list_sha(recs) -> str:
    return hashlib.sha256("\n".join(sorted(set(recs))).encode("utf-8")).hexdigest()


def marker(recs=None) -> dict | None:
    """The held_out entry of a learner's spec: which recordings its labels left out (None: none)."""
    recs = recordings() if recs is None else sorted(set(recs))
    if not recs:
        return None
    return dict(list=LIST_FILE.relative_to(ROOT).as_posix(), n_recordings=len(recs), sha256=list_sha(recs))


def marker_sha(spec: dict | None) -> str | None:
    return ((spec or {}).get("held_out") or {}).get("sha256")


# ---- freeze ---------------------------------------------------------------------------------
def read_rows(path: Path) -> tuple[list[str], list[dict], dict | None]:
    """(header, rows, counts) of a log as it stands: cut after its last complete line, rows whose
    field count is not the header's dropped. ([], [], None) without the file."""
    if not path.exists():
        return [], [], None
    raw = path.read_bytes()
    cut = raw.rfind(b"\n") + 1
    rows = list(csv.reader(io.StringIO(raw[:cut].decode("utf-8"), newline="")))
    if not rows:
        return [], [], dict(rows=0, dropped=0, partial_bytes=len(raw) - cut)
    head, body = rows[0], rows[1:]
    good = [r for r in body if len(r) == len(head)]
    return head, [dict(zip(head, r, strict=True)) for r in good], dict(rows=len(good), dropped=len(body) - len(good),
                                                                      partial_bytes=len(raw) - cut)


def write_rows(path: Path, head: list[str], rows: list[dict]) -> str:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=head)
    w.writeheader()
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def freeze(review: Path, recs: list[str], refresh: bool = False) -> dict:
    snap = review / SNAP
    if not recs:
        raise SystemExit(f"{LIST_FILE} lists no recordings: nothing to freeze")
    if (snap / "manifest.json").exists():
        old = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
        if not refresh:
            raise SystemExit(f"{snap} exists (frozen {old.get('frozen_at')}): --refresh replaces it "
                             f"(the old snapshot moves to {HISTORY}/)")
    elif snap.exists() and any(snap.iterdir()):
        raise SystemExit(f"{snap} holds files but no manifest.json: look at it and remove it by hand")
    head, rows, cnt = read_rows(review / LOG)
    if not rows:
        raise SystemExit(f"no rows in {review / LOG}: nothing to freeze")
    keep = set(recs)
    tmp = review / (SNAP + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    log = [r for r in rows if r["recording"] in keep]
    ids = {r["log_id"] for r in log if r.get("log_id")}
    files = {LOG: dict(rows=len(log), source_rows=cnt["rows"], source_dropped=cnt["dropped"],
                       sha256=write_rows(tmp / LOG, head, log))}
    for name, by in ((SHOWN, "log_id"), (SPECIES_SHOWN, "log_id"), (AUDIT, "recording")):
        h, rr, c = read_rows(review / name)
        if c is None:
            files[name] = None
            continue
        sel = [r for r in rr if (r.get(by) in ids if by == "log_id" else r.get(by) in keep)]
        files[name] = dict(rows=len(sel), source_rows=c["rows"], source_dropped=c["dropped"],
                           sha256=write_rows(tmp / name, h, sel) if h else None)
    with_rows = sorted({r["recording"] for r in log})
    man = dict(frozen_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), **marker(recs),
               recordings=sorted(recs), recordings_with_rows=with_rows,
               last_logged_at=max((r.get("logged_at", "") for r in log), default=""), files=files)
    dest = None
    if (snap / "manifest.json").exists():
        dest = review / HISTORY / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        man["previous"] = str(dest)
    (tmp / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(snap), str(dest))
    elif snap.exists():
        snap.rmdir()                                       # empty (checked above)
    os.replace(tmp, snap)
    return man


# ---- score ----------------------------------------------------------------------------------
def boot_ci(recs, stat, n_boot: int = BOOT, seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval of stat(row indices) over bootstrap draws of whole recordings."""
    import numpy as np
    recs = np.asarray(recs)
    keys = sorted(set(recs.tolist()))
    if not keys:
        return float("nan"), float("nan")
    idx = {k: np.flatnonzero(recs == k) for k in keys}
    rng = np.random.default_rng(seed)
    out = np.array([stat(np.concatenate([idx[keys[i]] for i in rng.integers(0, len(keys), len(keys))]))
                    for _ in range(n_boot)], float)
    if not np.isfinite(out).any():
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def find(fit_dir: Path, learner: str) -> Path | None:
    for p in (fit_dir / LEARNER_FILE[learner], fit_dir / learner / LEARNER_FILE[learner]):
        if p.exists():
            return p
    return None


def score_taxon(lab: dict, src, fits: dict, n_boot: int, seed: int) -> tuple[dict, object]:
    import numpy as np
    import pandas as pd
    import review_labels as L
    import review_taxon as RT
    taxon = pd.DataFrame(lab["taxon"], columns=L.TAXON_COLS)
    lab01 = taxon["label"].astype(int)
    taxon = taxon[(lab01 == 1) | ((lab01 == 0) & (taxon["reason"] == "decision"))]   # as review_taxon.main
    heads = {n: RT.TaxonHead.load(p, p.with_name("taxon.npz")) for n, p in fits.items()}
    win = src.windows() if len(taxon) else None
    out, parts = {}, []
    for g in RT.GROUPS:
        if not (taxon.loc[taxon["group"] == g, "label"].astype(int) == 1).any():
            continue
        lane0, by_lane = RT.default_lane(g), {}

        def data(lane, g=g, by_lane=by_lane):
            if lane not in by_lane:
                by_lane[lane] = RT.group_data(taxon, win, g, lane, negatives=True)
            return by_lane[lane]
        base = data(lane0)
        y, recs, neg = base.y, base.rec, base.neg
        z = base.rows["zero_shot"].to_numpy(float)
        e = dict(lane=lane0, n_rows=len(y), n_recordings=len(set(recs)), n_pos=int(y.sum()),
                 n_pos_recordings=len(set(recs[y == 1])), n_negative=int(neg.sum()), fits={})
        if not 0 < y.sum() < len(y):
            out[g] = {**e, "note": "one class only: no AUROC"}
            continue
        pn = (y == 1) | neg
        e.update(auc_zero=RT.auroc(y, z), auc_pn_zero=RT.auroc(y[pn], z[pn]) if neg.any() else float("nan"))
        scores = {}
        for n, h in heads.items():
            f = dict(adopted=g in h.adopted())
            s = z
            if f["adopted"]:
                d = data(h.lane(g))
                if not d.rows[["recording", "start_s"]].equals(base.rows[["recording", "start_s"]]):
                    e["fits"][n] = {**f, "note": f"lane {h.lane(g)} gives other segments than {lane0}: not scored"}
                    continue
                s = h.prob(g, d.X)
            scores[n] = s
            f.update(auc=RT.auroc(y, s), auc_pn=RT.auroc(y[pn], s[pn]) if neg.any() else float("nan"))
            f["gain_lo"], f["gain_hi"] = ((0.0, 0.0) if s is z else
                                          boot_ci(recs, lambda k, s=s: RT.auroc(y[k], s[k]) - RT.auroc(y[k], z[k]),
                                                  n_boot, seed))
            e["fits"][n] = f
        e["paired"] = {}
        names = list(scores)
        for b in names[1:]:
            sa, sb = scores[names[0]], scores[b]
            ci = (0.0, 0.0) if np.array_equal(sa, sb) else boot_ci(
                recs, lambda k, sa=sa, sb=sb: RT.auroc(y[k], sb[k]) - RT.auroc(y[k], sa[k]), n_boot, seed)
            e["paired"][f"{b} - {names[0]}"] = dict(diff=RT.auroc(y, sb) - RT.auroc(y, sa), ci_lo=ci[0], ci_hi=ci[1])
        out[g] = e
        parts.append(base.rows.assign(group=g, **{f"app_{n}": np.round(s, 5) for n, s in scores.items()}))
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out, rows


def score_species(lab: dict, log_rows: list[dict], src, review: Path, fits: dict, n_boot: int, seed: int,
                  device: str) -> tuple[dict, object]:
    import numpy as np
    import pandas as pd
    import box_embed as BE
    import review_labels as L
    import review_species as RS
    species = pd.DataFrame(lab["species"], columns=L.SPECIES_COLS)
    names, memo = L.Names(), {}

    def zero_key(n: str) -> str:
        if n not in memo:
            x = names.resolve(n)
            memo[n] = RS.name_key(n, x["canonical"], x["rank"])
        return memo[n]
    rows, boxes = RS.build_rows(species, log_rows, L.R.lane_group, zero_key)
    if not len(rows):
        return dict(note="no named rows"), pd.DataFrame()
    cache, audio = BE.EmbeddingCache(review / "embeddings"), src.audio()
    rows, boxes, n_out = RS.embeddable(rows, boxes, audio, cache)
    if n_out:
        print(f"left out: {n_out[0]} boxes ({n_out[1]} named rows) with no audio here and no cached embedding")
    X = (BE.embed_boxes(boxes, BE.BoxEmbedder(BE.PerchSpatial(device, wavs=audio), starts=src.window_starts()),
                        cache, progress=True) if boxes else np.zeros((0, BE.DIM), np.float32))
    D = RS.Data.make(rows, X)
    ev = np.flatnonzero(D.rows["eval"].to_numpy(bool))
    if not len(ev):
        return dict(note="no evaluation rows (every row was shown a learned name)"), pd.DataFrame()
    R = D.rows.iloc[ev].reset_index(drop=True)
    recs = R["recording"].to_numpy()
    ranks = {"zero": np.array([RS.rank_of(D.rows.at[i, "key"], RS.merged([], D.rows.at[i, "zero"]), D.others[i])
                               for i in ev])}
    learners = {n: RS.SpeciesLearner.load(p, p.with_name("species_protos.npy")) for n, p in fits.items()}
    for n, lr in learners.items():
        tau = lr.tau if lr.adopted else float("inf")
        ml, k = int(lr.spec.get("max_learned", RS.MAX_LEARNED)), int(lr.spec.get("top_k", RS.TOP_K))
        ranks[n] = np.array([RS.rank_of(D.rows.at[i, "key"], RS.merged(
            [x for x, _ in RS.learned_for(D.X[D.rows.at[i, "box"]], D.rows.at[i, "group"], lr.meta, lr.P, tau, ml)],
            D.rows.at[i, "zero"], k), D.others[i]) for i in ev])
    rr = {n: np.where(r > 0, 1.0 / np.maximum(r, 1), 0.0) for n, r in ranks.items()}

    def metrics(m):
        return {n: dict(mrr=float(rr[n][m].mean()), hit1=float((ranks[n][m] == 1).mean()),
                        hit5=float((ranks[n][m] > 0).mean())) for n in ranks}
    out = dict(n_rows=len(R), n_recordings=len(set(recs)), n_boxes=int(R["box"].nunique()),
               n_rows_all=len(D.rows), left_out_boxes=n_out[0] if n_out else 0, all=metrics(np.ones(len(R), bool)),
               groups={g: dict(n_rows=int(m.sum()), n_recordings=len(set(recs[m])), **metrics(m))
                       for g in sorted(set(R["group"])) for m in [R["group"].to_numpy() == g]},
               fits={n: dict(adopted=bool(learners[n].adopted)) for n in learners}, paired={})
    for n in learners:
        a, z = rr[n], rr["zero"]
        out["fits"][n]["gain_lo"], out["fits"][n]["gain_hi"] = (
            (0.0, 0.0) if np.array_equal(a, z) else boot_ci(recs, lambda k, a=a: a[k].mean() - z[k].mean(), n_boot, seed))
    names_ = list(learners)
    for b in names_[1:]:
        a0, b0 = rr[names_[0]], rr[b]
        ci = (0.0, 0.0) if np.array_equal(a0, b0) else boot_ci(recs, lambda k, a0=a0, b0=b0: b0[k].mean() - a0[k].mean(),
                                                               n_boot, seed)
        out["paired"][f"{b} - {names_[0]}"] = dict(diff=float(b0.mean() - a0.mean()), ci_lo=ci[0], ci_hi=ci[1])
    keep = ["recording", "box_id", "log_id", "group", "name", "key"]
    return out, R[keep].assign(**{f"rank_{n}": r for n, r in ranks.items()})


def score_refiner(lab: dict, src, review: Path, fits: dict, n_boot: int, seed: int) -> tuple[dict, object]:
    import numpy as np
    import pandas as pd
    import box_refine as B
    import review_labels as L
    box = B.pipeline_yes(pd.DataFrame(lab["box"], columns=L.BOX_COLS))
    audio, cache = src.audio(), B.BoxCache(review / "refine_cache")
    here = np.array([t["recording"] in audio or cache.get(t["recording"], [t[f"tight_{e}"] for e in B.EDGES]) is not None
                     for t in box.to_dict("records")], bool).reshape(-1)
    if (~here).any():
        print(f"left out: {int((~here).sum())} boxes with no audio here and not in {review / 'refine_cache'}")
    box = box[here].sort_values(["recording", "box_begin_s"]).reset_index(drop=True)
    T = B.build_table(box, lambda b, coarse: B.make_patch(audio[b.recording], coarse), cache)
    cache.save()
    ctrl = np.flatnonzero(T.info["control"].to_numpy(bool)) if len(T.info) else np.zeros(0, int)
    if not len(ctrl):
        return dict(note="no boxes shown plain"), pd.DataFrame()
    info = T.info.iloc[ctrl].reset_index(drop=True)
    recs, grp = info["recording"].to_numpy(), info["group"].to_numpy()
    iou = {"grid": T.grid_iou[ctrl]}
    shown = {}
    refs = {n: B.Refiner.load(p) for n, p in fits.items()}
    for n, ref in refs.items():
        a, sh = T.grid_iou[ctrl].copy(), np.zeros(len(ctrl), bool)
        for j, i in enumerate(ctrl):
            s = ref.group_spec(info.at[j, "group"])
            k = B.GRID_INDEX[tuple(float(x) for x in s["setting"])]
            nb = [B.GRID_INDEX[tuple(float(x) for x in v)] for v in s["neighbours"]]
            cut = s.get("conf_min")
            if (s.get("decision") == "adopt" and cut is not None and not np.isnan(T.B[i, k]).any()
                    and B.box_conf(T, i, dict(k=k, nb=nb)) >= cut):
                a[j], sh[j] = T.I[i, k], True
        iou[n], shown[n] = a, sh

    def block(m):
        r = recs[m]
        e = dict(n_boxes=int(m.sum()), n_recordings=len(set(r)), iou_grid=float(iou["grid"][m].mean()), fits={},
                 paired={})
        for n in refs:
            a, gd = iou[n][m], iou["grid"][m]
            ci = (0.0, 0.0) if np.array_equal(a, gd) else boot_ci(r, lambda k, a=a, gd=gd: a[k].mean() - gd[k].mean(),
                                                                  n_boot, seed)
            e["fits"][n] = dict(iou=float(a.mean()), refined=float(shown[n][m].mean()), gain_lo=ci[0], gain_hi=ci[1])
        names_ = list(refs)
        for b in names_[1:]:
            a0, b0 = iou[names_[0]][m], iou[b][m]
            ci = (0.0, 0.0) if np.array_equal(a0, b0) else boot_ci(
                r, lambda k, a0=a0, b0=b0: b0[k].mean() - a0[k].mean(), n_boot, seed)
            e["paired"][f"{b} - {names_[0]}"] = dict(diff=float(b0.mean() - a0.mean()), ci_lo=ci[0], ci_hi=ci[1])
        return e
    out = dict(all=block(np.ones(len(ctrl), bool)), groups={g: block(grp == g) for g in sorted(set(grp))},
               adopted={n: ref.adopted() for n, ref in refs.items()}, left_out_boxes=int((~here).sum()))
    rows = info[["box_id", "recording", "group", "shown_box"]].assign(
        **{f"iou_{n}": np.round(v, 4) for n, v in iou.items()}, **{f"refined_{n}": v for n, v in shown.items()})
    return out, rows


def fmt_ci(lo, hi) -> str:
    return f"[{lo:+.3f}, {hi:+.3f}]"


def report_text(rep: dict) -> str:
    m = rep["frozen"]
    out = [f"frozen test set: {m['n_recordings']} recordings ({len(m['recordings_with_rows'])} with reviews), "
           f"frozen {m['frozen_at']}, last row logged {m['last_logged_at']}"]
    if rep.get("list_changed"):
        out.append("  WARNING: config/frozen_test.csv is not the list frozen: freeze --refresh")
    for n, f in rep["fits"].items():
        for lrn, w in f["learners"].items():
            out.append(f"  fit {n} {lrn}: {w}")
    t = rep.get("taxon")
    if t is not None:
        out += ["", "taxon: AUROC positive vs rest (P vs N where 'no' segments exist), app vs zero-shot"]
        for g, e in t.items():
            head = (f"  {g:<14} {e['n_pos']} positive / {e['n_negative']} negative / {e['n_rows']} segments, "
                    f"{e['n_recordings']} recordings ({e['n_pos_recordings']} with positives)")
            if "note" in e:
                out.append(f"{head}: {e['note']}")
                continue
            pn = e["n_negative"] > 0
            out.append(f"{head}; zero-shot {e['auc_zero']:.3f}" + (f" (P vs N {e['auc_pn_zero']:.3f})" if pn else ""))
            for n, f in e["fits"].items():
                if "note" in f:
                    out.append(f"      {n:<6} {f['note']}")
                    continue
                out.append(f"      {n:<6} {'head' if f['adopted'] else 'zero-shot (not adopted)':<24} {f['auc']:.3f}"
                           + (f" (P vs N {f['auc_pn']:.3f})" if pn else "")
                           + f"; gain {fmt_ci(f['gain_lo'], f['gain_hi'])}"
                           + (" (1 recording: no interval)" if e["n_recordings"] < 2 else ""))
            for k, p in e.get("paired", {}).items():
                out.append(f"      {k}: {p['diff']:+.3f} {fmt_ci(p['ci_lo'], p['ci_hi'])}")
    s = rep.get("species")
    if s is not None:
        out += ["", "species: MRR@5 / hit@1 / hit@5 on rows shown no learned name"]
        if "note" in s:
            out.append(f"  {s['note']}")
        else:
            out.append(f"  {s['n_rows']} rows ({s['n_rows_all']} named), {s['n_boxes']} boxes, {s['n_recordings']} "
                       f"recordings" + (f"; left out {s['left_out_boxes']} boxes" if s["left_out_boxes"] else ""))
            for label, blk in [("all", s["all"]), *[(g, v) for g, v in s["groups"].items()]]:
                extra = "" if label == "all" else f" ({blk['n_rows']} rows, {blk['n_recordings']} recordings)"
                out.append(f"  {label}{extra}: " + "; ".join(
                    f"{n} {blk[n]['mrr']:.3f} / {blk[n]['hit1']:.3f} / {blk[n]['hit5']:.3f}"
                    for n in blk if n not in ("n_rows", "n_recordings")))
            for n, f in s["fits"].items():
                out.append(f"  {n}: {'adopted' if f['adopted'] else 'not adopted'}; MRR gain "
                           f"{fmt_ci(f['gain_lo'], f['gain_hi'])}")
            for k, p in s["paired"].items():
                out.append(f"  {k}: MRR {p['diff']:+.3f} {fmt_ci(p['ci_lo'], p['ci_hi'])}")
    r = rep.get("refiner")
    if r is not None:
        out += ["", "refiner: IoU with the accepted box, boxes shown plain, app vs grid box"]
        if "note" in r:
            out.append(f"  {r['note']}")
        else:
            for n, a in r["adopted"].items():
                out.append(f"  {n} adopts: {', '.join(a) or 'none'}")
            for label, e in [("all", r["all"]), *r["groups"].items()]:
                out.append(f"  {label:<14} {e['n_boxes']} boxes, {e['n_recordings']} recordings; grid {e['iou_grid']:.3f}; "
                           + "; ".join(f"{n} {f['iou']:.3f} (refined {f['refined']:.0%}) "
                                       f"{fmt_ci(f['gain_lo'], f['gain_hi'])}" for n, f in e["fits"].items()))
                for k, p in e["paired"].items():
                    out.append(f"      {k}: {p['diff']:+.3f} {fmt_ci(p['ci_lo'], p['ci_hi'])}")
    return "\n".join(out) + "\n"


def score(review: Path, uploads: Path, fit_args: list[str], out_dir: Path, learners: list[str], n_boot: int,
          seed: int, device: str) -> dict:
    snap = review / SNAP
    if not (snap / "manifest.json").exists():
        raise SystemExit(f"no frozen test set in {snap}: run freeze first")
    man = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    import review_labels as L
    fit_dirs = {}
    for a in fit_args:
        n, sep, d = a.partition("=")
        if not sep or not n or not d:
            raise SystemExit(f"--fit wants LABEL=DIR, got {a!r}")
        fit_dirs[n] = Path(d)
    rep = dict(frozen={k: man[k] for k in ("frozen_at", "n_recordings", "sha256", "recordings", "recordings_with_rows",
                                           "last_logged_at")},
               list_changed=list_sha(recordings()) != man["sha256"], fits={})
    paths = {lrn: {} for lrn in learners}
    for n, d in fit_dirs.items():
        rep["fits"][n] = dict(dir=str(d), learners={})
        for lrn in learners:
            p = find(d, lrn)
            if p is None:
                rep["fits"][n]["learners"][lrn] = f"no {LEARNER_FILE[lrn]}: not scored"
                continue
            spec = json.loads(p.read_text(encoding="utf-8"))
            ok = marker_sha(spec) == man["sha256"]
            rep["fits"][n]["learners"][lrn] = (f"{p} (held out: the frozen recordings)" if ok else
                                               f"{p} (TRAINED ON THE FROZEN RECORDINGS: not a held-out score)")
            paths[lrn][n] = p
    src = L.SRC.Sources(uploads)
    print(src.describe())
    log_rows = L.read_csv_rows(snap / LOG)
    lab = L.labels_from_logs(log_rows, L.read_csv_rows(snap / AUDIT), L.read_csv_rows(snap / SHOWN),
                             L.read_csv_rows(snap / SPECIES_SHOWN), src, holdout=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "taxon" in learners:
        rep["taxon"], rows = score_taxon(lab, src, paths["taxon"], n_boot, seed)
        rows.to_csv(out_dir / "taxon_rows.csv", index=False)
    if "species" in learners:
        rep["species"], rows = score_species(lab, log_rows, src, review, paths["species"], n_boot, seed, device)
        rows.to_csv(out_dir / "species_rows.csv", index=False)
    if "refiner" in learners:
        rep["refiner"], rows = score_refiner(lab, src, review, paths["refiner"], n_boot, seed)
        rows.to_csv(out_dir / "refiner_rows.csv", index=False)
    (out_dir / "scores.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    txt = report_text(rep)
    (out_dir / "scores.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fz = sub.add_parser("freeze", help="copy the test recordings' rows of the logs to data/review/frozen_test/")
    fz.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    fz.add_argument("--refresh", action="store_true", help="replace an existing snapshot (moved to frozen_test_history/)")
    sc = sub.add_parser("score", help="score fits on the frozen snapshot (report only)")
    sc.add_argument("--review-dir", type=Path, default=REVIEW_DIR, help="the snapshot and the caches")
    sc.add_argument("--uploads-dir", type=Path, default=UPLOADS_DIR)
    sc.add_argument("--fit", action="append", default=None, metavar="LABEL=DIR",
                    help="learner files in DIR or DIR/<learner>/ (default: live=<review-dir>/calibration)")
    sc.add_argument("--learners", nargs="+", choices=list(LEARNER_FILE), default=list(LEARNER_FILE))
    sc.add_argument("--out", type=Path, default=None, help=f"default <review-dir>/{SCORES}/<stamp>")
    sc.add_argument("--boot", type=int, default=BOOT)
    sc.add_argument("--seed", type=int, default=SEED)
    sc.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = ap.parse_args()
    review = args.review_dir.resolve()
    if args.cmd == "freeze":
        man = freeze(review, recordings(), args.refresh)
        f = man["files"]
        print(f"frozen {man['n_recordings']} recordings ({len(man['recordings_with_rows'])} with reviews) at "
              f"{man['frozen_at']} into {review / SNAP}; last row logged {man['last_logged_at']}")
        for name, v in f.items():
            print(f"  {name}: " + ("absent" if v is None else f"{v['rows']} of {v['source_rows']} rows"))
        if man.get("previous"):
            print(f"  the previous snapshot moved to {man['previous']}")
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    score(review, args.uploads_dir.resolve(), args.fit or [f"live={review / 'calibration'}"],
          args.out or review / SCORES / stamp, args.learners, args.boot, args.seed, args.device)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
