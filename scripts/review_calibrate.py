"""Learner 1: per-lane thresholds and box-tightening settings, fitted to the review labels.

CPU only. Reads the review logs (through review_labels.py), segment_scores.csv (+ the
EarthChirp lanes) and cell_maps.npy, with the scans of the audio added in the app behind them
(review_sources.py, --uploads-dir). Writes recommendations to data/review/calibration/;
nothing is applied: the viewer and the app keep perch_viewer.THRESHOLDS / KAPPA / ROW_SUPPORT
until someone edits them.

A fitted value is marked "adopt" only when it beats the current one on held-out recordings:
leave one recording out (fit on the others, score the held-out one), pool, and bootstrap the
difference over recordings; adopt when the 95% interval is above 0. Too few labels -> "keep".

(a) Thresholds, per lane, from positives only (positive vs unlabelled; user decision 2026-11-17).
    Units: every 2.5 s segment of each recording holding at least one positive of the lane's
    group (EC-Insecta and EC-margin use the Insecta positives from every lane). A segment touching
    a positive interval is positive; every other segment is unlabelled, not negative ("no"
    answers and audit gaps are not used).
    - recall = share of positive segments flagged (score >= threshold).
    - flagged = share of all segments flagged.
    - gain = recall - flagged; the threshold maximising it (ties -> the one nearest the current).
    If the positives are a random sample of the calls (Elkan & Noto 2008), gain = (1 - share of
    call time) x (true-positive rate - false-positive rate), so its best threshold is the one true
    negatives would give. Positives that are not such a sample (long stitched boxes with silent
    gaps inside) pull the threshold down.
(b) Box settings (kappa, row support), per lane and for all lanes together, on the yes boxes
    the pipeline suggested (model and below-threshold boxes): each is re-tightened from its
    model box at every setting on the grid and scored by its time x frequency IoU (kHz linear)
    with the box the reviewer accepted. Unedited boxes count (they agree with what was shown).
    Boxes the reviewer started from a refined or earlier box (shown_box "refined" / "previous")
    are left out: the accepted box is anchored to that start, not to the tightened box. So are
    model boxes merged into one Raven box (shown_box "raven-auto merge", raven_autolabel.py): their
    accepted box spans the other model boxes too. One model box fitted to a Raven box counts.
    Accepted boxes reaching outside the model box (time, or above Perch's 16 kHz) are counted:
    no setting can reach them.

    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\review_calibrate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_labels as L  # noqa: E402

R, pv = L.R, L.pv
OUT_DIR = R.REVIEW_DIR / "calibration"
EPS_S = 0.05                     # overlaps shorter than this are edge contact, not overlap
KAPPAS = tuple(round(float(k), 2) for k in np.arange(0.0, 0.95, 0.1))   # Python floats, as the app passes
ROW_SUPPORTS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.75)
MAX_CANDIDATES = 400             # threshold candidates per fit (quantiles beyond this)
MIN = dict(pos=30, boxes=20, recordings=3)   # support before a fit is tried
BOOT, SEED = 2000, 0

UNIT_COLS = ["lane", "group", "recording", "split", "label", "score", "begin_s", "end_s"]
THR_COLS = ["lane", "group", "current", "fitted", "decision", "reason", "reliability", "n_recordings",
            "n_pos_segments", "n_segments", "gain_current_all", "gain_fitted_all",
            "loro_gain_current", "loro_gain_fitted", "loro_gain_diff", "ci_lo", "ci_hi",
            "loro_recall_current", "loro_recall_fitted", "loro_flagged_current", "loro_flagged_fitted"]
CURVE_COLS = ["lane", "threshold", "recall", "flagged", "gain", "is_current"]
BOX_SET_COLS = ["lane", "current_kappa", "current_row_support", "fitted_kappa", "fitted_row_support",
                "decision", "reason", "n_boxes", "n_recordings", "n_edited", "share_beyond",
                "iou_current_all", "iou_fitted_all", "loro_iou_current", "loro_iou_fitted", "loro_iou_diff",
                "ci_lo", "ci_hi"]
IOU_COLS = ["box_id", "recording", "split", "lane", "kappa", "row_support", "begin_s", "end_s", "lo_khz",
            "hi_khz", "iou", "time_iou", "band_iou"]


# ---- intervals ---------------------------------------------------------------------------
def overlap(s: np.ndarray, e: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.clip(np.minimum(e, b) - np.maximum(s, a), 0, None)


# ---- (a) thresholds ----------------------------------------------------------------------
def threshold_units(taxon: pd.DataFrame, seg: pd.DataFrame, lanes: list[str]) -> pd.DataFrame:
    """Per lane: one row per 2.5 s segment of each recording holding a positive of the lane's group;
    label 1 if the segment touches a positive interval (> EPS_S), else 0 (unlabelled)."""
    out = []
    by_rec = {r: d.sort_values("start_s") for r, d in seg.groupby("recording", sort=False)}
    pos = taxon[taxon["label"].astype(int) == 1]
    for lane in lanes:
        g = R.lane_group(lane)
        col = f"prob_{lane}"
        for rec, d in pos[pos["group"] == g].groupby("recording", sort=True):
            if rec not in by_rec or col not in by_rec[rec]:
                continue
            sd = by_rec[rec]
            s = sd["start_s"].to_numpy(float)
            e = s + pv.SEG_S
            sc = sd[col].to_numpy(float)
            lab = np.zeros(len(s), bool)
            for a, b in zip(d["begin_s"].astype(float), d["end_s"].astype(float), strict=True):
                lab |= overlap(s, e, a, b) > EPS_S
            ok = np.isfinite(sc)
            out.append(pd.DataFrame(dict(lane=lane, group=g, recording=rec, split=d["split"].iloc[0],
                                         label=lab[ok].astype(int), score=sc[ok], begin_s=s[ok], end_s=e[ok])))
    u = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=UNIT_COLS)
    return u[UNIT_COLS].astype({"label": int, "score": float, "begin_s": float, "end_s": float})


def counts(u: pd.DataFrame, t) -> np.ndarray:
    """(positive segments flagged, positive segments, segments flagged, segments) at threshold t;
    t may be an array (one row of counts per threshold). Flagged = score >= t."""
    sc = np.sort(u["score"].to_numpy(float))
    ps = np.sort(u.loc[u["label"].to_numpy() == 1, "score"].to_numpy(float))
    t = np.asarray(t, float)
    return np.stack([len(ps) - np.searchsorted(ps, t, side="left"), np.full(t.shape, len(ps)),
                     len(sc) - np.searchsorted(sc, t, side="left"), np.full(t.shape, len(sc))], -1).astype(float)


def recall_flagged(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recall (share of positive segments flagged) and flagged share of all segments; NaN without any."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(c[..., 1] > 0, c[..., 0] / np.maximum(c[..., 1], 1), np.nan)
        f = np.where(c[..., 3] > 0, c[..., 2] / np.maximum(c[..., 3], 1), np.nan)
    return r, f


def gain(c: np.ndarray) -> np.ndarray:
    """recall - flagged share, from counts()."""
    r, f = recall_flagged(c)
    return r - f


def support(u: pd.DataFrame) -> dict[str, int]:
    return dict(pos=int((u["label"] == 1).sum()), segments=len(u), recordings=u["recording"].nunique())


def too_few(sup: dict[str, int], mins: dict[str, int], keys=("pos", "recordings")) -> str:
    low = [f"{k} {sup[k]} < {mins[k]}" for k in keys if sup[k] < mins[k]]
    return "too few labels: " + ", ".join(low) if low else ""


def candidates(u: pd.DataFrame, current: float) -> np.ndarray:
    """Midpoints between neighbouring scores (quantiles beyond MAX_CANDIDATES), 4 significant
    digits, + the current threshold. From scores only, no labels."""
    v = np.unique(u["score"].to_numpy(float))
    if len(v) > 1:
        v = (v[1:] + v[:-1]) / 2
    if len(v) > MAX_CANDIDATES:
        v = np.quantile(v, np.linspace(0, 1, MAX_CANDIDATES))
    v = np.array([float(f"{x:.4g}") for x in v])
    return np.unique(np.r_[v, current])


def best(g: np.ndarray, cand: np.ndarray, current: float) -> int:
    """Index of the highest gain; ties -> the candidate nearest the current threshold."""
    top = np.flatnonzero(g >= np.nanmax(g) - 1e-12)
    return int(top[np.argmin(np.abs(cand[top] - current))])


def fit_threshold(u: pd.DataFrame, current: float, mins: dict[str, int]) -> tuple[float, str]:
    """(threshold, '' or why it was not fitted)."""
    why = too_few(support(u), mins)
    if why:
        return current, why
    c = candidates(u, current)
    return float(c[best(gain(counts(u, c)), c, current)]), ""


def boot_diff(new: np.ndarray, cur: np.ndarray, stat, n_boot: int, seed: int) -> tuple[float, float]:
    """95% interval of stat(sum new) - stat(sum cur) over recordings drawn with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(new), (n_boot, len(new)))
    d = stat(new[idx].sum(1)) - stat(cur[idx].sum(1))
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def calibrate_thresholds(units: pd.DataFrame, current: dict[str, float], mins: dict[str, int],
                         n_boot: int = BOOT, seed: int = SEED) -> pd.DataFrame:
    """Leave one recording out on a fixed candidate grid (from all scores, no labels): per-recording
    counts at every candidate, so a fit on the other recordings is the sum minus one row."""
    rows = []
    for lane, cur in current.items():
        u = units[units["lane"] == lane]
        sup = support(u)
        t, why = fit_threshold(u, cur, mins)
        row = dict(lane=lane, group=R.lane_group(lane), current=cur, fitted=t, reliability=pv.RELIABILITY.get(lane, ""),
                   n_recordings=sup["recordings"], n_pos_segments=sup["pos"], n_segments=sup["segments"],
                   gain_current_all=float(gain(counts(u, cur))) if len(u) else np.nan,
                   gain_fitted_all=float(gain(counts(u, t))) if len(u) else np.nan)
        if why:
            rows.append({**row, "decision": "keep", "reason": why})
            continue
        cand = candidates(u, cur)
        ic = int(np.flatnonzero(cand == cur)[0])
        recs = sorted(u["recording"].unique())
        K = np.stack([counts(u[u["recording"] == r], cand) for r in recs])     # (recordings, candidates, 4)
        total = K.sum(0)
        new, old = [], []
        for i in range(len(recs)):
            tr = total - K[i]
            low = too_few(dict(pos=int(tr[0, 1]), recordings=len(recs) - 1), mins)
            k = ic if low else best(gain(tr), cand, cur)
            new.append(K[i, k])
            old.append(K[i, ic])
        new, old = np.array(new), np.array(old)
        rn, fn = recall_flagged(new.sum(0))
        rc, fc = recall_flagged(old.sum(0))
        lo, hi = boot_diff(new, old, gain, n_boot, seed)
        adopt = lo > 0 and t != cur
        rows.append({**row, "decision": "adopt" if adopt else "keep",
                     "reason": "held-out gain higher, interval above 0" if adopt else
                     ("fitted = current" if t == cur else "held-out gain not shown (interval reaches 0)"),
                     "loro_gain_current": float(rc - fc), "loro_gain_fitted": float(rn - fn),
                     "loro_gain_diff": float((rn - fn) - (rc - fc)), "ci_lo": lo, "ci_hi": hi,
                     "loro_recall_current": float(rc), "loro_recall_fitted": float(rn),
                     "loro_flagged_current": float(fc), "loro_flagged_fitted": float(fn)})
    return pd.DataFrame(rows, columns=THR_COLS)


def threshold_curve(units: pd.DataFrame, current: dict[str, float]) -> pd.DataFrame:
    out = []
    for lane, cur in current.items():
        u = units[units["lane"] == lane]
        if not len(u):
            continue
        c = candidates(u, cur)
        r, f = recall_flagged(counts(u, c))
        out.append(pd.DataFrame(dict(lane=lane, threshold=c, recall=r, flagged=f, gain=r - f, is_current=c == cur)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=CURVE_COLS)


# ---- (b) box settings --------------------------------------------------------------------
def tight_box(sub: np.ndarray, s0: int, kappa: float, row_support: float) -> tuple[float, float, float, float]:
    """As CellMaps.tighten_boxes: (begin_s, end_s, lo_khz, hi_khz). kappa as a Python float: against
    float32 cells, np.float64(0.7) and 0.7 cut a borderline cell differently (NumPy 2 promotion)."""
    t0, t1, r0, r1, _ = pv.tighten(sub, float(kappa), float(row_support))
    return (s0 + t0) * pv.CELL_S, (s0 + t1) * pv.CELL_S, pv.ROW_EDGES_KHZ[r0], pv.ROW_EDGES_KHZ[r1]


def iou_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def iou_2d(a: tuple, b: tuple) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    union = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / union if union > 0 else 0.0


ANCHORED = ("refined", "previous",    # shown_box values whose accepted box is anchored elsewhere,
            "raven-auto merge")     # or is one Raven box over several model boxes (raven_autolabel.py)


def pipeline_boxes(box: pd.DataFrame) -> pd.DataFrame:
    keep = (box["proposal"] == "pipeline") & box["lane"].isin(pv.CELL_LANES)
    if "shown_box" in box:
        keep &= ~box["shown_box"].isin(ANCHORED)
    return box[keep].reset_index(drop=True)


def box_ious(box: pd.DataFrame, sub_of, kappas=KAPPAS, row_supports=ROW_SUPPORTS) -> pd.DataFrame:
    """IoU of every pipeline yes box re-tightened at every setting against the accepted box.
    sub_of(recording, lane, begin_s, end_s) -> (cells (slots, 4), first slot) or None."""
    out = []
    for b in pipeline_boxes(box).itertuples():
        got = sub_of(b.recording, b.lane, float(b.box_begin_s), float(b.box_end_s))
        if got is None:
            continue
        sub, s0 = got
        acc = tuple(float(getattr(b, f"accepted_{e}")) for e in L.EDGES)
        for k in kappas:
            for rs in row_supports:
                t = tight_box(sub, s0, k, rs)
                out.append((b.box_id, b.recording, b.split, b.lane, k, rs, *t, iou_2d(t, acc),
                            iou_1d(t[0], t[1], acc[0], acc[1]), iou_1d(t[2], t[3], acc[2], acc[3])))
    return pd.DataFrame(out, columns=IOU_COLS)


def beyond(box: pd.DataFrame) -> pd.Series:
    """Accepted box reaching outside the model box in time, or above Perch's band."""
    return ((box["accepted_begin_s"].astype(float) < box["box_begin_s"].astype(float) - EPS_S)
            | (box["accepted_end_s"].astype(float) > box["box_end_s"].astype(float) + EPS_S)
            | (box["accepted_hi_khz"].astype(float) > pv.PERCH_BAND[1] + 1e-3))


def fit_box(iou: pd.DataFrame, current: tuple[float, float], mins: dict[str, int]) -> tuple[tuple[float, float], str]:
    """Setting with the highest mean IoU (ties -> nearest the current one)."""
    n_box, n_rec = iou["box_id"].nunique(), iou["recording"].nunique()
    low = [f"{k} {v} < {mins[k]}" for k, v in (("boxes", n_box), ("recordings", n_rec)) if v < mins[k]]
    if low:
        return current, "too few labels: " + ", ".join(low)
    m = iou.groupby(["kappa", "row_support"])["iou"].mean()
    best = m[m >= m.max() - 1e-12].index
    dist = [abs(k - current[0]) + abs(rs - current[1]) for k, rs in best]
    return tuple(float(x) for x in best[int(np.argmin(dist))]), ""


def at(iou: pd.DataFrame, setting: tuple[float, float]) -> pd.Series:
    """Per-box IoU at one setting, indexed by box id."""
    k, rs = setting
    d = iou[np.isclose(iou["kappa"], k) & np.isclose(iou["row_support"], rs)]
    return d.set_index("box_id")["iou"]


def mean_ratio(x: np.ndarray) -> np.ndarray:
    """(IoU sum, box count) -> mean IoU."""
    return x[..., 0] / x[..., 1]


def calibrate_boxes(box: pd.DataFrame, iou: pd.DataFrame, current: tuple[float, float], mins: dict[str, int],
                    n_boot: int = BOOT, seed: int = SEED) -> pd.DataFrame:
    pb = pipeline_boxes(box)
    rows = []
    lanes = sorted(iou["lane"].unique(), key=pv.CELL_LANES.index) if len(iou) else []
    for lane in [*lanes, "all"]:
        d = iou if lane == "all" else iou[iou["lane"] == lane]
        bx = pb[pb["box_id"].isin(d["box_id"].unique())]
        setting, why = fit_box(d, current, mins)
        row = dict(lane=lane, current_kappa=current[0], current_row_support=current[1],
                   fitted_kappa=setting[0], fitted_row_support=setting[1], n_boxes=d["box_id"].nunique(),
                   n_recordings=d["recording"].nunique(), n_edited=int((bx["box_edited"].astype(str) == "1").sum()),
                   share_beyond=float(beyond(bx).mean()) if len(bx) else np.nan,
                   iou_current_all=float(at(d, current).mean()) if len(d) else np.nan,
                   iou_fitted_all=float(at(d, setting).mean()) if len(d) else np.nan)
        if why:
            rows.append({**row, "decision": "keep", "reason": why})
            continue
        recs = sorted(d["recording"].unique())
        new, old = [], []
        for r in recs:
            sr, _ = fit_box(d[d["recording"] != r], current, mins)
            held = d[d["recording"] == r]
            a, c = at(held, sr), at(held, current)
            new.append((a.sum(), len(a)))
            old.append((c.sum(), len(c)))
        new, old = np.array(new, float), np.array(old, float)
        lo, hi = boot_diff(new, old, mean_ratio, n_boot, seed)
        fn, fc = mean_ratio(new.sum(0)), mean_ratio(old.sum(0))
        adopt = lo > 0 and setting != current
        rows.append({**row, "decision": "adopt" if adopt else "keep",
                     "reason": "held-out IoU higher, interval above 0" if adopt else
                     ("fitted = current" if setting == current else "held-out gain not shown (interval reaches 0)"),
                     "loro_iou_current": float(fc), "loro_iou_fitted": float(fn), "loro_iou_diff": float(fn - fc),
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows, columns=BOX_SET_COLS)


def cell_sub_of(cells: pv.CellMaps):
    def sub_of(rec: str, lane: str, b0: float, b1: float):
        g = cells.grid(rec)
        s0, s1 = int(round(b0 / pv.CELL_S)), int(round(b1 / pv.CELL_S))
        if g is None or lane not in pv.CELL_LANES or s1 > len(g) or s1 <= s0:
            return None
        return np.asarray(g[s0:s1, :, pv.CELL_LANES.index(lane)]), s0
    return sub_of


# ---- main --------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=R.LOG_FILE)
    ap.add_argument("--shown", type=Path, default=R.SHOWN_FILE)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    for k, v in MIN.items():
        ap.add_argument(f"--min-{k}", type=int, default=v, help=f"support needed before fitting (default {v})")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--uploads-dir", type=Path, default=L.SRC.SA.UPLOADS_DIR, help="audio added in the app")
    args = ap.parse_args()
    mins = {k: getattr(args, f"min_{k}") for k in MIN}

    src = L.SRC.Sources(args.uploads_dir)
    print(src.describe())
    lab = L.labels_from_logs(L.read_csv_rows(args.log), [], L.read_csv_rows(args.shown), sources=src)   # positives only
    taxon = pd.DataFrame(lab["taxon"], columns=L.TAXON_COLS)
    box = pd.DataFrame(lab["box"], columns=L.BOX_COLS)
    seg = src.segments()
    current = {g: t for g, t in pv.THRESHOLDS.items() if f"prob_{g}" in seg}
    taxon = taxon[taxon["recording"].isin(set(seg["recording"]))]
    units = threshold_units(taxon, seg, list(current))
    thr = calibrate_thresholds(units, current, mins, args.boot, args.seed)
    curve = threshold_curve(units, current)

    cur_box = (pv.KAPPA, pv.ROW_SUPPORT)
    if cur_box[0] not in KAPPAS or cur_box[1] not in ROW_SUPPORTS:
        raise SystemExit(f"current box setting {cur_box} is not on the grid")
    cells = src.cell_maps() if len(pipeline_boxes(box)) else None
    if cells is not None:
        iou = box_ious(box.sort_values("recording"), cell_sub_of(cells))
    else:
        iou = pd.DataFrame(columns=IOU_COLS)
    bset = calibrate_boxes(box, iou, cur_box, mins, args.boot, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    thr.to_csv(args.out / "thresholds.csv", index=False)
    units.to_csv(args.out / "threshold_units.csv", index=False)
    curve.to_csv(args.out / "threshold_curve.csv", index=False)
    bset.to_csv(args.out / "box_settings.csv", index=False)
    iou.to_csv(args.out / "box_iou.csv", index=False)
    rec = {"thresholds": {r.lane: r.fitted for r in thr.itertuples() if r.decision == "adopt"},
           "box_settings": {r.lane: {"kappa": r.fitted_kappa, "row_support": r.fitted_row_support}
                            for r in bset.itertuples() if r.decision == "adopt"},
           "threshold_objective": "positive vs unlabelled: recall - flagged share",
           "minimum_support": mins, "log": str(args.log), "held_out": lab["held_out"]}
    (args.out / "calibration.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    print(f"{len(lab['taxon'])} taxon labels, {len(box)} yes boxes ({len(pipeline_boxes(box))} pipeline)")
    print("thresholds (gain = recall - flagged share)")
    for r in thr.itertuples():
        extra = (f"  held-out gain {r.loro_gain_current:.3f} -> {r.loro_gain_fitted:.3f} "
                 f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] (recall {r.loro_recall_current:.2f} -> {r.loro_recall_fitted:.2f}, "
                 f"flagged {r.loro_flagged_current:.2f} -> {r.loro_flagged_fitted:.2f})"
                 if np.isfinite(r.loro_gain_diff) else "")
        print(f"  {r.lane:<14} {r.current:<7g} -> {r.fitted:<8g} {r.decision:<6} {r.reason}{extra}")
    print("box settings (kappa, row support)")
    for r in bset.itertuples():
        extra = (f"  held-out IoU {r.loro_iou_current:.3f} -> {r.loro_iou_fitted:.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
                 if np.isfinite(r.loro_iou_diff) else "")
        print(f"  {r.lane:<14} ({r.current_kappa:g}, {r.current_row_support:g}) -> "
              f"({r.fitted_kappa:g}, {r.fitted_row_support:g}) {r.decision:<6} {r.reason}{extra}")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
