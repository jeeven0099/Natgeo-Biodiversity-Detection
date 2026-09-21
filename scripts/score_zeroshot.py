"""Score Perch 2.0's zero-shot output against our labels. Needs only numpy/sklearn.

Reads the cache written by extract_perch.py and reports, per target class:
  AP / ROC-AUC on the test fold, and precision/recall/F1 at a threshold chosen
  on the val fold. Perch's own model card warns that species logits are
  uncalibrated, so a threshold fitted on our own data is not optional.

Window scores are aggregated to clip level with a max. Our labels say the sound
occurs somewhere in the clip, not in every window of it - which is also why the
36% of clips longer than 10 s are flagged weak_label in the manifest.

Two caveats, both of which belong in the write-up:

1. The corpus is single-label (one folder per clip), so negatives are "clips
   filed under some other label". A clip filed as Bird may well also contain
   cicadas, so some counted false positives are really annotation gaps.
   Precision reported here is therefore a lower bound.

2. Clip-level max-aggregation is biased toward long clips: the max of n noisy
   window scores grows with n. Class and duration are strongly correlated here
   (Aircraft runs ~79 s, Collared Scops Owl ~1.1 s), so a clip-level score can
   partly be measuring clip length. A synthetic control confirmed this - a
   deliberately uninformative target still reached AUC 0.72 at clip level.
   Hence both levels are reported, alongside auc_dur: the AUC obtained from
   window count alone. If a target's auc_clip is not comfortably above its
   auc_dur, the result is a duration artefact, not detection skill.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def load_positive_specs(path: Path) -> dict[str, tuple[str, set[str]]]:
    """fine -> (field, {values}) describing which clips count as positive."""
    out = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        field, _, values = row["positives"].partition(":")
        out[row["fine"]] = (field, {v.strip() for v in values.split(",") if v.strip()})
    return out


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1, scanning only at observed score values."""
    if y.sum() == 0:
        return float("nan"), float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    fn = y.sum() - tp
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float),
                   where=(2 * tp + fp + fn) > 0)
    i = int(np.argmax(f1))
    return float(s[order][i]), float(f1[i])


def prf(y: np.ndarray, s: np.ndarray, thr: float) -> tuple[float, float, float]:
    pred = s >= thr
    tp = float((pred & (y == 1)).sum())
    fp = float((pred & (y == 0)).sum())
    fn = float((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--perchdir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = args.root
    pdir = args.perchdir or root / "data" / "perch"

    target_names = (pdir / "target_names.txt").read_text(encoding="utf-8").split()
    logits = np.load(pdir / "target_logits.npy")
    windows = list(csv.DictReader((pdir / "windows.csv").open(encoding="utf-8")))
    if len(windows) != logits.shape[0]:
        raise SystemExit(f"windows.csv has {len(windows)} rows but logits has {logits.shape[0]}")

    splits = {r["session_id"]: r["split"]
              for r in csv.DictReader((root / "data" / "splits.csv").open(encoding="utf-8"))}
    specs = load_positive_specs(root / "config" / "perch_targets.csv")

    # Window -> clip, taking the max score per target.
    fine_w = np.array([w["fine"] for w in windows])
    coarse_w = np.array([w["coarse"] for w in windows])
    fold_w = np.array([splits.get(w["session_id"], "unassigned") for w in windows])

    clip_rows: dict[str, dict] = {}
    clip_scores: dict[str, np.ndarray] = {}
    clip_nwin: dict[str, int] = defaultdict(int)
    for i, w in enumerate(windows):
        cid = w["clip_id"]
        row = logits[i]
        clip_nwin[cid] += 1
        if cid in clip_scores:
            np.maximum(clip_scores[cid], row, out=clip_scores[cid])
        else:
            clip_scores[cid] = row.astype(np.float64).copy()
            clip_rows[cid] = w

    clips = sorted(clip_scores)
    S = np.stack([clip_scores[c] for c in clips])
    fold = np.array([splits.get(clip_rows[c]["session_id"], "unassigned") for c in clips])
    fine = np.array([clip_rows[c]["fine"] for c in clips])
    coarse = np.array([clip_rows[c]["coarse"] for c in clips])
    n_win = np.array([clip_nwin[c] for c in clips], dtype=float)
    print(f"{len(windows)} windows -> {len(clips)} clips "
          f"(train={int((fold=='train').sum())} val={int((fold=='val').sum())} test={int((fold=='test').sum())})")

    results = []
    for j, name in enumerate(target_names):
        field, values = specs.get(name, ("fine", {name}))
        src = fine if field == "fine" else coarse
        y_all = np.isin(src, list(values)).astype(int)
        s_all = S[:, j]

        m_val, m_test = fold == "val", fold == "test"
        y_val, s_val = y_all[m_val], s_all[m_val]
        y_test, s_test = y_all[m_test], s_all[m_test]

        # Window level: no aggregation, so no length bias. Positives are noisy
        # instead, since every window of a weakly-labelled clip inherits its tag.
        yw = np.isin(fine_w if field == "fine" else coarse_w, list(values)).astype(int)
        mw = fold_w == "test"
        yw_t, sw_t = yw[mw], logits[mw, j]

        n_pos_test = int(y_test.sum())
        row = {"target": name, "n_pos_val": int(y_val.sum()), "n_pos_test": n_pos_test,
               "n_test": int(m_test.sum())}
        if n_pos_test == 0 or n_pos_test == m_test.sum():
            row.update({k: float("nan") for k in
                        ("ap", "auc", "ap_win", "auc_win", "auc_dur", "thr", "p", "r", "f1")})
            row["note"] = "no positives in test fold"
        else:
            # Control: how well does sheer window count separate this class?
            row["auc_dur"] = roc_auc_score(y_test, n_win[m_test])
            row["ap_win"] = (average_precision_score(yw_t, sw_t)
                             if 0 < yw_t.sum() < len(yw_t) else float("nan"))
            row["auc_win"] = (roc_auc_score(yw_t, sw_t)
                              if 0 < yw_t.sum() < len(yw_t) else float("nan"))
            thr, _ = best_f1_threshold(y_val, s_val) if y_val.sum() else (float(np.median(s_all)), 0.0)
            if not np.isfinite(thr):
                thr = float(np.median(s_all))
            p, r, f = prf(y_test, s_test, thr)
            row.update({
                "ap": average_precision_score(y_test, s_test),
                "auc": roc_auc_score(y_test, s_test),
                "thr": thr, "p": p, "r": r, "f1": f,
            })
            notes = []
            if y_val.sum() < 5:
                notes.append("few val positives - threshold unreliable")
            if row["auc"] - row["auc_dur"] < 0.05:
                notes.append("no better than clip length - likely duration artefact")
            row["note"] = "; ".join(notes)
        results.append(row)

    results.sort(key=lambda d: (-(d["ap"] if np.isfinite(d.get("ap", np.nan)) else -1)))
    out = args.out or root / "data" / "zeroshot_report.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["target", "n_pos_val", "n_pos_test", "n_test",
                                           "ap", "auc", "ap_win", "auc_win", "auc_dur",
                                           "thr", "p", "r", "f1", "note"])
        w.writeheader()
        w.writerows(results)

    def fmt(v):
        return "  -  " if not np.isfinite(v) else f"{v:5.3f}"

    print(f"\n{'target':28s} {'nPos':>5s} {'AP':>6s} {'AUC':>6s} | {'APwin':>6s} {'AUCwin':>6s} "
          f"| {'AUCdur':>6s} | {'P':>6s} {'R':>6s} {'F1':>6s}")
    for d in results:
        print(f"{d['target']:28s} {d['n_pos_test']:5d} {fmt(d.get('ap', np.nan))} "
              f"{fmt(d.get('auc', np.nan))} | {fmt(d.get('ap_win', np.nan))} "
              f"{fmt(d.get('auc_win', np.nan))} | {fmt(d.get('auc_dur', np.nan))} | "
              f"{fmt(d.get('p', np.nan))} {fmt(d.get('r', np.nan))} {fmt(d.get('f1', np.nan))}"
              + (f"   [{d['note']}]" if d.get("note") else ""))
    print("\nauc_dur = AUC from window count alone. A target whose auc is not clearly "
          "above\nits auc_dur is tracking clip length, not the animal.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
