"""Log the Raven boxes as reviews: model boxes that match a Raven box are fitted to it, and Raven boxes
the model missed become new boxes. Raven's annotators are people, so these are human labels.

    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\raven_autolabel.py            # dry run: counts + plan
    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\raven_autolabel.py --write    # backs up, then appends

Matching is raven_match.py (time IoU >= 0.3 on the merged span of a Raven box's model boxes, same group,
each model box to one Raven box). For every Raven box that takes part (Unknown is skipped):
  one / merge  each of its model boxes gets a "yes" row whose accepted box is the Raven box (time and
               frequency); several model boxes inside one Raven box all become that one box.
               box_shown.csv: "raven-auto" (one) or "raven-auto merge" (merge). The box refiner and the
               threshold calibration fit boxes on "raven-auto" only; review_labels counts a merged
               Raven box once for taxon and species.
  new          the Raven box becomes a box of its own (model "Raven", the group's default lane), "yes";
               shown_box "raven". Below the cut its model boxes stay as they are.
Model boxes no Raven box of their group matches get nothing: they stay for the reviewer, and are
never negatives. Names come from the label (raven_match.label_names): group-only labels (Bird, Insect,
Frog, sounds, any label ending in "?") give no species; Cicada is logged as Cicadidae, Collared
Scops Owl as Otus lettia (Perch's Collared Scops-Owl); typos merged.
Named rows get the model's top 5 over the Raven box's span (Perch / EarthChirp, as the app computes
it), so a learner's comparison with the zero-shot ranking is fair.

Every row: reviewer "raven-auto", written by review_app.Store.append, so the columns and the box ids
are the app's. A box that already has any log row is left alone (a person's answer is never
overwritten, and a rerun adds nothing twice); a person's later answer on a box overrides its row as
usual (the newest row wins). --write first copies annotation_log.csv, box_shown.csv and
species_shown.csv to <review-dir>/backup/<time>/. The plan (one row per row to write or skipped) goes
to <review-dir>/raven_plan.csv on every run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raven_match as RM  # noqa: E402
import review_labels as L  # noqa: E402

R = L.R
pv = R.pv
PLAN_COLS = ["recording", "kind", "action", "why", "group", "label", "names", "lane", "box_id", "n_models", "iou",
             "raven_begin_s", "raven_end_s", "raven_lo_khz", "raven_hi_khz", "box_begin_s", "box_end_s"]
SHOWN = {"one": RM.SHOWN_ONE, "merge": RM.SHOWN_MERGE, "new": RM.SHOWN_NEW}


def set_review_dir(d: Path) -> None:
    """Points review_app's files at d (as its --review-dir does)."""
    d = d.resolve()
    R.REVIEW_DIR, R.LOG_FILE, R.CACHE_FILE = d, d / "annotation_log.csv", d / "prediction_cache.jsonl"
    R.AUDIT_FILE, R.SHOWN_FILE = d / "audit_windows.csv", d / "box_shown.csv"
    R.SPECIES_SHOWN_FILE = d / "species_shown.csv"


def plan(S, recs: list[str]) -> list[dict]:
    """One task per row the Raven boxes give: the queue item it is logged on, the Raven box, the names."""
    tasks = []
    for rec in recs:
        p = S.raven_plan(rec)
        if not p:
            continue
        items = {d["id"]: d for d in S.items(rec, pv.KAPPA)}
        for e in p:
            r = e["raven"]
            names = RM.label_names(r["label"])
            base = dict(recording=rec, kind=e["kind"], why=e["why"], group=r["group"], label=r["label"],
                        names=names, n_models=len(e["models"]), iou=round(e["iou"], 4), raven=r)
            if e["kind"] == "new":
                rid = R.item_id(rec, "Raven", "", r["begin_s"], r["end_s"], r["lo_khz"], r["hi_khz"])
                d = items[rid]
                lanes = S.raven_lanes(r["group"])
                tasks.append({**base, "item": d, "lane": lanes[0]})
            else:
                for m in e["models"]:
                    d = items[m["id"]]
                    assert d["source"] == "model", d["id"]
                    tasks.append({**base, "item": d, "lane": d["lane"]})
    return tasks


def mark_existing(tasks: list[dict], log_rows: list[dict]) -> None:
    """action: write, or skip when the box has a log row (a person's, or this script's from a run before),
    or when the Raven box is a point (no time or frequency extent: the log cannot hold it; it stays in
    the app's Raven source for a person to box)."""
    who: dict[str, set[str]] = {}
    for x in log_rows:
        who.setdefault(R.row_id(x), set()).add("auto" if x["reviewer"] == RM.REVIEWER else "human")
    for t in tasks:
        w, r = who.get(t["item"]["id"], set()), t["raven"]
        flat = not (r["begin_s"] < r["end_s"] and r["lo_khz"] < r["hi_khz"])
        t["action"] = ("skip: a person's row" if "human" in w else "skip: logged before" if w
                       else "skip: Raven box has no extent" if flat else "write")


def top5(S, t: dict, cache: dict) -> tuple[list[dict], str]:
    """The model's top 5 for the task's lane over the Raven box's span, once per Raven box and lane."""
    r = t["raven"]
    key = (t["recording"], r["key"], t["lane"])
    if key not in cache:
        d = {**t["item"], "lane": t["lane"], "predictor": R.model_of(t["lane"])}
        res = S.predict(d, r["begin_s"], r["end_s"])
        cache[key] = (res["top"], f"{res['span'][0]:.2f}-{res['span'][1]:.2f}")
    return cache[key]


def body_of(t: dict, top: list[dict], span: str) -> dict:
    r = t["raven"]
    note = f"Raven: {r['label']}; " + {"one": f"time IoU {t['iou']:.2f}",
                                       "merge": f"{t['n_models']} model boxes merged, time IoU {t['iou']:.2f}",
                                       "new": f"new box ({t['why']})"}[t["kind"]]
    return {"id": t["item"]["id"], "kappa": pv.KAPPA, "decision": "yes", "reviewer": RM.REVIEWER,
            "begin_s": r["begin_s"], "end_s": r["end_s"], "lo_khz": r["lo_khz"], "hi_khz": r["hi_khz"],
            "lane": t["lane"], "species": [{"name": n, "source": RM.NAME_SOURCE} for n in t["names"]],
            "top": top, "prediction_span": span, "comment": note}


def plan_frame(tasks: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([dict(recording=t["recording"], kind=t["kind"], action=t["action"], why=t["why"],
                              group=t["group"], label=t["label"], names="; ".join(t["names"]), lane=t["lane"],
                              box_id=t["item"]["id"], n_models=t["n_models"], iou=t["iou"],
                              raven_begin_s=t["raven"]["begin_s"], raven_end_s=t["raven"]["end_s"],
                              raven_lo_khz=t["raven"]["lo_khz"], raven_hi_khz=t["raven"]["hi_khz"],
                              box_begin_s=t["item"]["begin_s"], box_end_s=t["item"]["end_s"]) for t in tasks],
                        columns=PLAN_COLS)


def report(S, recs: list[str], tasks: list[dict], names: L.Names) -> None:
    pf = plan_frame(tasks)
    rv = pd.concat([pd.DataFrame(S.raven_boxes(r)).assign(recording=r) for r in recs], ignore_index=True)
    skipped = rv[[RM.label_names(x) is None or g == "Unknown" for x, g in zip(rv["label"], rv["group"], strict=True)]]
    print(f"{len(recs)} recordings, {len(rv)} distinct Raven boxes ({len(skipped)} skipped: Unknown)")
    per = pf.drop_duplicates(["recording", "raven_begin_s", "raven_end_s", "raven_lo_khz", "raven_hi_khz"])
    print("\nRaven boxes by outcome (new: why)")
    t = per.assign(out=[k if k != "new" else f"new: {w}" for k, w in zip(per["kind"], per["why"].str.replace(
        r"IoU [0-9.]+ < .*", "below the IoU cut", regex=True), strict=True)])
    print(pd.crosstab(t["group"], t["out"], margins=True).to_string())
    fit = pf[pf["kind"] != "new"]
    allm = pd.concat([pv.perch_boxes(S.seg_by_rec[r], S.thresholds).assign(recording=r) for r in recs],
                     ignore_index=True)
    print("\nmodel boxes (model source, all lanes): fitted to a Raven box / left for review")
    n_all = allm.groupby("group").size()
    n_fit = fit.groupby("lane").size().reindex(n_all.index, fill_value=0)
    print(pd.DataFrame({"model boxes": n_all, "fitted": n_fit, "left": n_all - n_fit}).to_string())
    print("\nrows by action")
    print(pd.crosstab(pf["kind"], pf["action"], margins=True).to_string())
    print("\nnames logged (label -> name: resolved rank, group; Raven boxes)")
    for (lab, nm, g), n in per.groupby(["label", "names", "group"]).size().items():
        if not nm:
            continue
        x = names.resolve(nm)
        flag = "" if x["rank"] != "unresolved" and x["group"] == g else "   <-- check"
        print(f"  {lab!r:34} -> {nm!r:32} {x['rank']:<12} {x['canonical']:<28} {x['group'] or '-':<14} {n:>5}{flag}")
    gonly = Counter(lab for lab, nm in zip(per["label"], per["names"], strict=True) if not nm)
    print("\ngroup only (no species): " + ", ".join(f"{k} {v}" for k, v in sorted(gonly.items())))
    dup = rv.groupby(["recording", "begin_s", "end_s", "lo_khz", "hi_khz"])["label"].nunique()
    print(f"\nRaven boxes listed twice with different labels: {int((dup > 1).sum())}")
    hum = [x for x in S.log_rows() if x["reviewer"] != RM.REVIEWER]
    hid = {R.row_id(x): x["decision"] for x in L.latest_boxes(hum)}
    clash = [t for t in tasks if t["item"]["id"] in hid and hid[t["item"]["id"]] == "no"]
    print(f"boxes a person answered 'no' that a Raven box matches (left as the person answered): {len(clash)}")
    named = {(t["recording"], t["raven"]["key"], t["lane"]) for t in tasks if t["action"] == "write" and t["names"]}
    print(f"top-5 runs needed: {len(named)} (Raven boxes with a name)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="append the rows (default: dry run)")
    ap.add_argument("--review-dir", type=Path, default=R.REVIEW_DIR)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="for Perch's top 5")
    ap.add_argument("--recordings", nargs="*", help="only these (default: every recording with Raven boxes)")
    args = ap.parse_args()
    set_review_dir(args.review_dir)
    S = R.Store(args.device)                 # no refiner or learners: the rows name what was shown
    recs = [r for r in S.recs if (not args.recordings or r in args.recordings)
            and (S.raven["recording"] == r).any()]
    tasks = plan(S, recs)
    mark_existing(tasks, S.log_rows())
    report(S, recs, tasks, L.Names())
    pf = plan_frame(tasks)
    pf.to_csv(R.REVIEW_DIR / "raven_plan.csv", index=False)
    print(f"plan: {R.rel(R.REVIEW_DIR / 'raven_plan.csv')}")
    todo = [t for t in tasks if t["action"] == "write"]
    if not args.write:
        print(f"dry run: {len(todo)} rows would be written; rerun with --write")
        return 0
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = R.REVIEW_DIR / "backup" / stamp
    bak.mkdir(parents=True, exist_ok=True)
    for f in (R.LOG_FILE, R.SHOWN_FILE, R.SPECIES_SHOWN_FILE):
        if f.exists():
            shutil.copy2(f, bak / f.name)
    print(f"backed up to {R.rel(bak)}")
    cache, n, bad = {}, 0, []
    for i, t in enumerate(todo):
        top, span = top5(S, t, cache) if t["names"] else ([], "")
        try:
            S.append(body_of(t, top, span), shown_as=SHOWN[t["kind"]])
            n += 1
        except ValueError as e:                  # a Raven box with no extent
            bad.append((t["item"]["id"], str(e)))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1} / {len(todo)}", flush=True)
    print(f"wrote {n} rows to {R.rel(R.LOG_FILE)} and {R.rel(R.SHOWN_FILE)}; {len(bad)} not written")
    for b in bad:
        print("  not written:", *b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
