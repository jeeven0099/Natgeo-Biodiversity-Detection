"""Perch 2.0 zero-shot, per 2.5 s segment, against the Raven boxes of the long-form recordings.

Every annotated 20-min recording in dataset/ goes through Perch as 5 s windows that
start every --hop seconds (default 2.5 s, so neighbouring windows share half their
audio; the tail is zero-padded). Nothing is trained.

Scoring unit = one hop-long SEGMENT (2.5 s). Each segment lies inside two windows
(the one starting at the segment and the one starting a hop earlier; the first
segment has only one), and its score is the MAX of their scores: a call is credited
if either window that contains the segment hears it. Each stretch of audio is
therefore scored exactly once, at 2.5 s resolution. With --hop 5 a segment is a
window and this reduces to plain back-to-back 5 s windows.

Groups (from the annotation table; '?' labels count for their group):
    Aves, Amphibia, Insecta, Mammalia, Anthropogenic   - the five target groups
    Weather                                            - scored too, reported apart
    Unknown                                            - never scored; see "ignored"

Ground truth for (segment, group):
    1   a box of the group covers >= MIN_OVERLAP_S of the segment, or all of a shorter box
   -1   ignored: a box of the group only grazes the segment, or an Unknown box touches it
        and the group has no box of its own there - neither is a fair negative
    0   otherwise

Perch side. Perch 2.0 is an EfficientNet-B3 on a 128-bin log-mel (60 Hz-16 kHz,
32 kHz, 500 frames per 5 s), one flat head of 14,795 classes (14,597 species +
198 FSD50K sound events), trained with SOFTMAX cross-entropy (targets 1/k for k
species). So a sigmoid of a logit is not a probability, and the natural group score
is the summed softmax probability of the group's classes:
    Aves / Amphibia / Insecta / Mammalia: every species of that taxonomic class
        (config/perch_class_index.csv) plus the FSD50K classes of that animal group
    Anthropogenic / Weather: FSD50K classes listed below
Everything else (water, generic noise, reptiles, ...) belongs to no group. The max
logit over the same classes is saved too, as an alternative score.

Operating points (both reported):
    fixed  - label-free: group present if its summed probability >= --threshold (0.5,
             i.e. Perch puts most of its belief on that group)
    lodo   - leave-one-day-out: for each recording day, the F1-best threshold on the
             OTHER days is applied to this day. Every segment is scored by a threshold
             that never saw it. A group with no positives outside a day falls back to
             the fixed threshold for that day.
Threshold-free AP and ROC-AUC are reported alongside.

Recall view (the annotations are incomplete, so an unlabelled flag is not surely a false
positive and precision is only a lower bound). Thresholds here never look at labels:
    prob>=0.5      the fixed rule above
    top 5/10/20/50%  flag that fraction of ALL segments, by the group score (a quantile)
    segment recall = flagged positive segments / positive segments
    box recall     = Raven boxes of the group with >= 1 flagged segment they overlap / boxes
Recall rises with how much is flagged, so each is read against two nulls that keep the
amount and the temporal clustering of the flags and move only their position:
    null any       each recording takes the flags of a random recording, rotated
    null within    each recording's own flags, rotated  (= is the TIME right inside it?)
p = share of N_NULL null draws with recall >= observed. 95% CIs resample recordings.

Model: the local ONNX export (external/bacpipe_model_checkpoints/perch_v2/
perch_v2_no_dft.onnx) on CUDA with TF32 OFF. With onnxruntime's default TF32 the
logits differ from the official TF SavedModel by up to 3.09; with it off, 0.0003
(16 real windows, same as onnxruntime on CPU). --verify-tf repeats that check.
onnxruntime logs 'cudnn_frontend ... No execution plans support the graph' the first
time it sees batch sizes 1 or 2 (recording tails); it falls back to another kernel
and the logits still match the CPU to 0.0001. The noise is harmless.

    .venv-perch/Scripts/python -X utf8 scripts/zeroshot_windows.py            # extract + score
    .venv-perch/Scripts/python -X utf8 scripts/zeroshot_windows.py --score-only

Writes data/perch_windows/:
    window_scores.csv   Perch output per 5 s window: group probabilities, max logits, top-1
                        (the cache; --score-only rebuilds everything below from it)
    segment_scores.csv  one row per segment: truth, max-over-covering-windows scores
    metrics_fixed.csv / metrics_lodo.csv   per group P, R, F1, accuracy (+ 95% CIs)
    thresholds_lodo.csv the threshold used for each (group, held-out day)
    metrics_recall.csv  segment and box recall per group and label-free rule, with nulls
    label_map.csv       every raw annotation label -> group, with counts
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import glob
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_perch import add_nvidia_dll_dirs, normalize_windows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PERCH_SR, WINDOW_S = 32_000, 5.0
WINDOW_N = int(PERCH_SR * WINDOW_S)
MIN_OVERLAP_S = 0.5
TARGET_GROUPS = ("Aves", "Amphibia", "Insecta", "Mammalia", "Anthropogenic")
GROUPS = TARGET_GROUPS + ("Weather",)
BUDGETS = (0.05, 0.10, 0.20, 0.50)      # recall view: share of all segments flagged
N_NULL = 200                            # recall view: null draws per null

# ---- annotation labels -> group ------------------------------------------------------
# Matched after lower-casing, trimming and dropping a trailing '?'. Every label in the
# tables must land somewhere; an unseen label stops the run instead of being guessed.
LABEL_GROUPS = {
    "Aves": ["bird", "birds", "bid", "3 birds", "collared scops owl", "red-legged crake",
             "brown boobook", "brown boobokok", "greater racket-tailed drongo",
             "pin-striped tit-babbler", "pin-striped tit babbler", "pin-striped tit-babler",
             "orange-bellied flowerpecker", "red-crowned barbet", "banded woodpecker",
             "short-tailed babbler", "western hooded pitta", "common hill myna",
             "changeable hawk-eagle", "crow", "grey-headed fish eagle",
             "square-tailed drongo-cuckoo", "long-tailed parakeet", "common flameback"],
    "Amphibia": ["frog"],
    "Insecta": ["orthoptera", "cicada", "cicafa", "insect", "purana ubina",
                "chremistica pontianaka"],
    "Mammalia": ["raffles' banded langur"],
    "Anthropogenic": ["aircraft", "aircrfat", "gunfire", "helicopter", "siren", "human voice"],
    "Weather": ["thunder", "rain", "wind"],
    "Unknown": ["unknown", "unknwon"],
}
LABEL_TO_GROUP = {lab: g for g, labs in LABEL_GROUPS.items() for lab in labs}
# The grouping table this run must reproduce, box for box.
EXPECTED_BOXES = {"Aves": 2907, "Amphibia": 39, "Insecta": 1900, "Mammalia": 6,
                  "Anthropogenic": 342, "Weather": 14, "Unknown": 47}

# ---- Perch classes -> group ----------------------------------------------------------
SPECIES_CLASS = {"Aves": "Aves", "Amphibia": "Amphibia", "Insecta": "Insecta",
                 "Mammalia": "Mammalia"}
EVENT_CLASSES = {
    "Aves": ["Chirp_and_tweet", "Crow", "Fowl", "Gull_and_seagull", "Chicken_and_rooster"],
    "Amphibia": ["Frog"],
    "Insecta": ["Insect", "Cricket"],
    "Mammalia": ["Dog", "Bark", "Cat", "Meow", "Purr"],
    "Anthropogenic": [
        # vehicles and engines
        "Accelerating_and_revving_and_vroom", "Aircraft", "Fixed-wing_aircraft_and_airplane",
        "Boat_and_Water_vehicle", "Bus", "Car", "Car_passing_by", "Engine", "Engine_starting",
        "Idling", "Motor_vehicle_(road)", "Motorcycle", "Race_car_and_auto_racing",
        "Rail_transport", "Train", "Subway_and_metro_and_underground",
        "Traffic_noise_and_roadway_noise", "Truck", "Vehicle",
        "Vehicle_horn_and_car_horn_and_honking",
        # alarms, guns, explosions
        "Siren", "Alarm", "Gunshot_and_gunfire", "Explosion", "Fireworks", "Boom",
        # human voice
        "Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking",
        "Child_speech_and_kid_speaking", "Conversation", "Human_voice", "Shout", "Yell",
        "Screaming", "Laughter", "Singing", "Male_singing", "Female_singing", "Whispering",
        "Crowd", "Chatter", "Cheering",
        # machines, tools, music
        "Power_tool", "Drill", "Sawing", "Hammer", "Tools", "Mechanical_fan", "Music",
        "Musical_instrument", "Bell", "Church_bell", "Telephone", "Ringtone"],
    "Weather": ["Thunder", "Thunderstorm", "Rain", "Raindrop", "Wind"],
}


def norm_label(s: str) -> str:
    return " ".join(s.strip().rstrip("?").strip().lower().split())


def read_boxes(ann_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(ann_dir.glob("*.selections.txt")):
        t = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
        rec = f.name.split(".")[0]
        for _, r in t.iterrows():
            rows.append((rec, float(r["Begin Time (s)"]), float(r["End Time (s)"]),
                         r["Annotation"]))
    b = pd.DataFrame(rows, columns=["recording", "begin_s", "end_s", "label"])
    b["key"] = b["label"].map(norm_label)
    unmapped = sorted(set(b["key"]) - set(LABEL_TO_GROUP))
    if unmapped:
        raise SystemExit(f"annotation labels with no group: {unmapped}")
    b["group"] = b["key"].map(LABEL_TO_GROUP)
    got = b["group"].value_counts().to_dict()
    if got != EXPECTED_BOXES:
        raise SystemExit(f"group box counts {got} differ from the table {EXPECTED_BOXES}")
    return b


def class_groups(labels: list[str]) -> dict[str, np.ndarray]:
    idx = pd.read_csv(ROOT / "config" / "perch_class_index.csv", keep_default_na=False)
    if idx["label"].tolist() != labels:
        raise SystemExit("config/perch_class_index.csv order differs from the model's labels")
    pos = {n: i for i, n in enumerate(labels)}
    sp = (idx["kind"] == "species").to_numpy()
    out = {}
    for g in GROUPS:
        missing = [n for n in EVENT_CLASSES[g] if n not in pos]
        if missing:
            raise SystemExit(f"{g}: not Perch classes: {missing}")
        ids = [pos[n] for n in EVENT_CLASSES[g]]
        if g in SPECIES_CLASS:
            ids += np.flatnonzero(sp & (idx["class"] == SPECIES_CLASS[g]).to_numpy()).tolist()
        out[g] = np.unique(np.asarray(ids, dtype=int))
    seen = np.concatenate(list(out.values()))
    if len(seen) != len(np.unique(seen)):
        raise SystemExit("a Perch class is assigned to two groups")
    return out


def segment_truth(boxes: pd.DataFrame, rec: str, n_win: int, seg_s: float) -> np.ndarray:
    """(n_seg, len(GROUPS)) of 1 / 0 / -1 for segments [k*seg_s, (k+1)*seg_s), as in the docstring."""
    y = np.zeros((n_win, len(GROUPS)), dtype=np.int8)
    graze = np.zeros_like(y, dtype=bool)
    unknown = np.zeros(n_win, dtype=bool)
    starts = np.arange(n_win) * seg_s
    for b in boxes[boxes["recording"] == rec].itertuples():
        ov = np.minimum(starts + seg_s, b.end_s) - np.maximum(starts, b.begin_s)
        touch = ov > 0
        if b.group == "Unknown":
            unknown |= touch
            continue
        need = min(MIN_OVERLAP_S, b.end_s - b.begin_s) - 1e-9
        j = GROUPS.index(b.group)
        y[ov >= need, j] = 1
        graze[touch & (ov < need), j] = True
    y[(y == 0) & graze] = -1
    y[(y == 0) & unknown[:, None]] = -1
    return y


def load_windows(path: str, hop_s: float = WINDOW_S) -> np.ndarray:
    """(n, WINDOW_N): window k starts at k*hop_s; n = ceil(duration / hop_s), so every
    hop-long segment of the recording is the start of a window. Zero-padded past the end."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != PERCH_SR:
        g = int(np.gcd(sr, PERCH_SR))
        x = resample_poly(x, PERCH_SR // g, sr // g).astype(np.float32)
    hop_n = int(round(hop_s * PERCH_SR))
    n_win = max(1, int(np.ceil(len(x) / hop_n - 1e-9)))
    x = np.pad(x, (0, (n_win - 1) * hop_n + WINDOW_N - len(x)))
    return np.lib.stride_tricks.sliding_window_view(x, WINDOW_N)[::hop_n][:n_win].copy()


def open_session(model: str):
    import onnxruntime as ort
    add_nvidia_dll_dirs()
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    # TF32 off: with it on, logits drift up to 3.09 from the official SavedModel.
    sess = ort.InferenceSession(model, providers=[("CUDAExecutionProvider", {"use_tf32": 0}),
                                                  "CPUExecutionProvider"])
    print("onnxruntime providers:", ", ".join(sess.get_providers()), flush=True)
    return sess


def verify_tf(sess, wins: np.ndarray, tf_dir: Path) -> None:
    import tensorflow as tf
    fn = tf.saved_model.load(str(tf_dir)).signatures["serving_default"]
    x = normalize_windows(wins[:16].copy(), 0.25)
    want = fn(inputs=tf.constant(x))["label"].numpy()
    got = sess.run(["label"], {"inputs": x})[0]
    d = float(np.abs(got - want).max())
    print(f"verify-tf: logit max|diff| vs SavedModel = {d:.5f} on 16 windows", flush=True)
    if d > 0.01:
        raise SystemExit("ONNX output does not match the TF SavedModel")


def segments(win: pd.DataFrame, boxes: pd.DataFrame, hop_s: float) -> pd.DataFrame:
    """Hop-long segments: score = max over the windows containing the segment; truth from boxes."""
    r = int(round(WINDOW_S / hop_s))
    score_cols = [c for c in win.columns if c.startswith(("prob_", "maxlogit_")) and c != "prob_other"]
    out = []
    for rec, w in win.groupby("recording", sort=True):
        w = w.sort_values("start_s").reset_index(drop=True)
        k = np.rint(w["start_s"].to_numpy() / hop_s).astype(int)
        if not np.array_equal(k, np.arange(len(w))):
            raise SystemExit(f"{rec}: window starts are not every {hop_s} s - wrong --hop for this cache?")
        # segment k lies inside windows k-r+1 .. k; a rolling max over r windows is exactly that
        seg = w[score_cols].rolling(r, min_periods=1).max()
        seg.insert(0, "recording", rec)
        seg.insert(1, "day", w["day"].astype(str).to_numpy())
        seg.insert(2, "start_s", w["start_s"].to_numpy())
        seg.insert(3, "silent", w["silent"].astype(int).rolling(r, min_periods=1).min().astype(bool).to_numpy())
        y = segment_truth(boxes, rec, len(w), hop_s)
        for j, g in enumerate(GROUPS):
            seg.insert(4 + j, f"truth_{g}", y[:, j])
        out.append(seg)
    return pd.concat(out, ignore_index=True)


def extract(args, boxes: pd.DataFrame, labels: list[str], groups: dict) -> pd.DataFrame:
    wavs = {Path(p).stem: p for p in glob.glob(str(args.audio_dir / "S01634_*" / "*" / "*.wav"))}
    recs = sorted(boxes["recording"].unique())
    missing = [r for r in recs if r not in wavs]
    if missing:
        raise SystemExit(f"annotated recordings with no wav: {missing}")
    if args.limit:
        recs = recs[:args.limit]
    sess = open_session(str(args.model))
    if args.verify_tf:
        verify_tf(sess, load_windows(wavs[recs[0]]), args.tf_dir)
    member = np.full(len(labels), -1, dtype=int)
    for j, g in enumerate(GROUPS):
        member[groups[g]] = j
    out, t0, n_done = [], time.monotonic(), 0
    with cf.ThreadPoolExecutor(1) as pool:          # read the next file while the GPU runs
        nxt = pool.submit(load_windows, wavs[recs[0]], args.hop)
        for i, rec in enumerate(recs):
            wins = nxt.result()
            if i + 1 < len(recs):
                nxt = pool.submit(load_windows, wavs[recs[i + 1]], args.hop)
            logit = np.concatenate([
                sess.run(["label"], {"inputs": normalize_windows(wins[k:k + args.batch].copy(), 0.25)})[0]
                for k in range(0, len(wins), args.batch)]).astype(np.float64)
            p = np.exp(logit - logit.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            df = pd.DataFrame({"recording": rec, "day": rec.split("_")[1],
                               "start_s": np.arange(len(wins)) * args.hop})
            for j, g in enumerate(GROUPS):
                df[f"prob_{g}"] = p[:, groups[g]].sum(axis=1)
                df[f"maxlogit_{g}"] = logit[:, groups[g]].max(axis=1)
            df["prob_other"] = p[:, member < 0].sum(axis=1)
            top = p.argmax(axis=1)
            df["top1_class"] = [labels[k] for k in top]
            df["top1_group"] = [GROUPS[member[k]] if member[k] >= 0 else "other" for k in top]
            df["top1_prob"] = p[np.arange(len(p)), top]
            df["silent"] = np.abs(wins).max(axis=1) == 0
            out.append(df)
            n_done += len(wins)
            if (i + 1) % 10 == 0 or i + 1 == len(recs):
                el = time.monotonic() - t0
                print(f"  {i + 1}/{len(recs)} recordings, {n_done} windows, {n_done / el:.1f} win/s, "
                      f"~{(len(recs) - i - 1) * len(wins) / (n_done / el) / 60:.1f} min left", flush=True)
    return pd.concat(out, ignore_index=True)


# ---- scoring -------------------------------------------------------------------------
def counts(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    return (int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()),
            int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum()))


def prf(tp, fp, fn, tn) -> dict:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"precision": p, "recall": r, "f1": f, "accuracy": (tp + tn) / (tp + fp + fn + tn)}


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float | None:
    if not (y == 1).any():
        return None
    o = np.argsort(-s, kind="stable")
    ys, ss = y[o] == 1, s[o]
    tp, fp = np.cumsum(ys), np.cumsum(~ys)
    last = np.r_[ss[1:] != ss[:-1], True]           # cut only between distinct scores
    f1 = np.where(last, 2 * tp / (tp + fp + ys.sum()), -1)
    k = int(f1.argmax())
    return float(ss[k])                              # predict present if score >= this


def bootstrap(per_rec: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """95% CIs for P/R/F1/accuracy, resampling recordings. per_rec: (n_rec, 4) TP FP FN TN."""
    rng = np.random.default_rng(seed)
    c = per_rec[rng.integers(0, len(per_rec), (n, len(per_rec)))].sum(axis=1).astype(float)
    tp, fp, fn, tn = c.T
    with np.errstate(invalid="ignore", divide="ignore"):
        m = {"precision": tp / (tp + fp), "recall": tp / (tp + fn),
             "f1": 2 * tp / (2 * tp + fp + fn), "accuracy": (tp + tn) / c.sum(axis=1)}
    with warnings.catch_warnings():                 # a group with no positives has no CI
        warnings.simplefilter("ignore", RuntimeWarning)
        return {k: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for k, v in m.items()}


def score(w: pd.DataFrame, args, outdir: Path) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score
    days = sorted(w["day"].unique())
    pred_fixed, pred_lodo, thr_rows = {}, {}, []
    for g in GROUPS:
        y, s = w[f"truth_{g}"].to_numpy(), w[f"{args.score}_{g}"].to_numpy()
        pred_fixed[g] = w[f"prob_{g}"].to_numpy() >= args.threshold
        pl = np.zeros(len(w), dtype=bool)
        for d in days:
            here = (w["day"] == d).to_numpy()
            fit = ~here & (y >= 0)
            t = best_f1_threshold(y[fit], s[fit])
            # No positives to fit on: fall back to the fixed probability rule for this day.
            pl[here] = pred_fixed[g][here] if t is None else s[here] >= t
            thr_rows.append({"group": g, "held_out_day": d,
                             "threshold": f"prob>={args.threshold}" if t is None else t,
                             "fallback_fixed": t is None,
                             "positives_in_fit_days": int((y[fit] == 1).sum())})
        pred_lodo[g] = pl
    pd.DataFrame(thr_rows).to_csv(outdir / f"thresholds_lodo{args.suffix}.csv", index=False)

    rec_codes = pd.factorize(w["recording"])[0]
    for name, preds in (("fixed", pred_fixed), ("lodo", pred_lodo)):
        rows, tot = [], np.zeros(4, dtype=int)
        for g in GROUPS:
            y, s = w[f"truth_{g}"].to_numpy(), w[f"{args.score}_{g}"].to_numpy()
            k = y >= 0
            c = counts(y[k], preds[g][k])
            if g in TARGET_GROUPS:
                tot += c
            per_rec = np.zeros((rec_codes.max() + 1, 4), dtype=int)
            for j, v in enumerate((preds[g] & (y == 1), preds[g] & (y == 0),
                                   ~preds[g] & (y == 1), ~preds[g] & (y == 0))):
                np.add.at(per_rec[:, j], rec_codes[v], 1)
            ci = bootstrap(per_rec)
            m = prf(*c)
            row = {"group": g, "positives": int((y == 1).sum()), "negatives": int((y == 0).sum()),
                   "ignored": int((y < 0).sum()), "TP": c[0], "FP": c[1], "FN": c[2], "TN": c[3]}
            for key in ("precision", "recall", "f1", "accuracy"):
                row[key] = m[key]
                row[f"{key}_ci"] = f"{ci[key][0]:.3f}-{ci[key][1]:.3f}"
            row["accuracy_all_absent"] = (y[k] == 0).mean()
            both = (y[k] == 1).any() and (y[k] == 0).any()
            row["AP"] = average_precision_score(y[k], s[k]) if both else float("nan")
            row["AP_chance"] = (y[k] == 1).mean()
            row["ROC_AUC"] = roc_auc_score(y[k], s[k]) if both else float("nan")
            rows.append(row)
        df = pd.DataFrame(rows)
        tgt = df[df["group"].isin(TARGET_GROUPS)]
        micro = prf(*tot)
        df = pd.concat([df, pd.DataFrame([
            {"group": "micro (5 targets)", "TP": tot[0], "FP": tot[1], "FN": tot[2], "TN": tot[3], **micro},
            {"group": "macro (5 targets)", **{k: tgt[k].mean() for k in
                                              ("precision", "recall", "f1", "accuracy", "AP", "ROC_AUC")}}])],
            ignore_index=True)
        # Segment exact match over the 5 targets, on segments with no ignored target cell.
        Y = np.stack([w[f"truth_{g}"].to_numpy() for g in TARGET_GROUPS], 1)
        P = np.stack([preds[g] for g in TARGET_GROUPS], 1)
        ok = (Y >= 0).all(axis=1)
        exact = float(((Y[ok] == 1) == P[ok]).all(axis=1).mean())
        df.to_csv(outdir / f"metrics_{name}{args.suffix}.csv", index=False)
        title = (f"fixed threshold: summed probability >= {args.threshold}" if name == "fixed"
                 else f"leave-one-day-out F1-best threshold on {args.score} (never fit on the scored day)")
        print(f"\n=== {title} ===")
        print(f"{'group':18s}{'pos':>6s}{'neg':>7s}{'ign':>5s}  {'Prec':>6s} {'Rec':>6s} {'F1':>6s} "
              f"{'Acc':>6s} {'Acc0':>6s}  {'AP':>6s} {'AP0':>6s} {'AUC':>6s}   F1 95% CI")
        for r in df.itertuples():
            f = lambda v: "   -  " if pd.isna(v) else f"{v:6.3f}"  # noqa: E731
            pos = "" if pd.isna(r.positives) else f"{int(r.positives):6d}{int(r.negatives):7d}{int(r.ignored):5d}"
            print(f"{r.group:18s}{pos:>18s}  {f(r.precision)} {f(r.recall)} {f(r.f1)} {f(r.accuracy)} "
                  f"{f(r.accuracy_all_absent)}  {f(r.AP)} {f(r.AP_chance)} {f(r.ROC_AUC)}   "
                  f"{'' if pd.isna(r.f1_ci) else r.f1_ci}")
        print(f"segment exact match over the 5 targets: {exact:.3f} ({ok.sum()} segments with no ignored cell; "
              f"all-absent rule {float((Y[ok] == 0).all(axis=1).mean()):.3f})")
    print("\nAcc0 = accuracy of always saying 'absent'; AP0 = AP of a random score (= prevalence).")


# ---- recall view ---------------------------------------------------------------------
def recording_bounds(w: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row codes, first row and length of each recording; rows must be contiguous and in time order."""
    codes = pd.factorize(w["recording"])[0]
    n = np.bincount(codes)
    lo = np.r_[0, np.cumsum(n)[:-1]]
    if not np.array_equal(codes, np.repeat(np.arange(len(n)), n)):
        raise SystemExit("segment rows of a recording are not contiguous")
    if (np.diff(w["start_s"].to_numpy())[np.diff(codes) == 0] <= 0).any():
        raise SystemExit("segment rows are not in time order")
    return codes, lo, n


def box_segments(boxes: pd.DataFrame, w: pd.DataFrame, hop_s: float, lo: np.ndarray,
                 n: np.ndarray) -> pd.DataFrame:
    """Every non-Unknown box of a scored recording with the first and last segment row it overlaps."""
    code = {r: i for i, r in enumerate(pd.unique(w["recording"]))}
    b = boxes[(boxes["group"] != "Unknown") & boxes["recording"].isin(code)].copy()
    c = b["recording"].map(code).to_numpy()
    # clip: 8 Insecta boxes start before 0 s (Raven times of about -510 s)
    i0 = np.clip(np.floor(b["begin_s"].to_numpy() / hop_s).astype(int), 0, n[c] - 1)
    i1 = np.minimum(np.maximum(np.ceil(b["end_s"].to_numpy() / hop_s).astype(int) - 1, i0), n[c] - 1)
    b["code"], b["lo"], b["hi"] = c, lo[c] + i0, lo[c] + i1
    return b.reset_index(drop=True)


def null_index(lo: np.ndarray, n: np.ndarray, rng, across: bool) -> np.ndarray:
    """Row gather index for one null draw: recording r reads the flags of recording src[r]
    (itself unless `across`) rotated by a random offset. Amount and clustering are kept."""
    src = rng.permutation(len(n)) if across else np.arange(len(n))
    idx = np.empty(int(n.sum()), dtype=np.int64)
    for r, s in enumerate(src):
        o = rng.integers(n[s])
        idx[lo[r]:lo[r] + n[r]] = lo[s] + (np.arange(n[r]) + o) % n[s]
    return idx


def recalls(flag: np.ndarray, y: np.ndarray, blo: np.ndarray, bhi: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Segment recall, box recall and the per-box hit vector for one flag vector."""
    cs = np.r_[0, np.cumsum(flag)]
    hit = cs[bhi + 1] - cs[blo] > 0
    seg = (flag & y).sum() / y.sum() if y.any() else float("nan")
    return seg, (hit.mean() if len(hit) else float("nan")), hit


def recall_report(w: pd.DataFrame, boxes: pd.DataFrame, args, outdir: Path) -> pd.DataFrame:
    codes, lo, n = recording_bounds(w)
    bx = box_segments(boxes, w, args.hop, lo, n)
    rng = np.random.default_rng(0)
    nulls = {k: [null_index(lo, n, rng, k == "any") for _ in range(N_NULL)] for k in ("any", "within")}
    pick = np.random.default_rng(1).integers(0, len(n), (1000, len(n)))

    def ci(num, den):
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v = num[pick].sum(axis=1) / den[pick].sum(axis=1)
            return f"{np.nanpercentile(v, 2.5):.3f}-{np.nanpercentile(v, 97.5):.3f}"

    rows = []
    for g in GROUPS:
        y = w[f"truth_{g}"].to_numpy() == 1
        s = w[f"{args.score}_{g}"].to_numpy()
        b = bx[bx["group"] == g]
        blo, bhi, bc = b["lo"].to_numpy(), b["hi"].to_numpy(), b["code"].to_numpy()
        ops = [(f"prob>={args.threshold:g}", w[f"prob_{g}"].to_numpy() >= args.threshold, args.threshold)]
        for q in BUDGETS:
            t = float(np.quantile(s, 1 - q))
            ops.append((f"top {q:.0%}", s >= t, t))
        for rule, f, t in ops:
            seg, box, hit = recalls(f, y, blo, bhi)
            row = {"group": g, "rule": rule, "threshold": t, "flagged": f.mean(),
                   "pos_segments": int(y.sum()), "boxes": len(b),
                   "recordings": int(len(np.unique(codes[y]))),
                   "seg_recall": seg, "seg_recall_ci": ci(np.bincount(codes[f & y], minlength=len(n)),
                                                          np.bincount(codes[y], minlength=len(n))),
                   "box_recall": box, "box_recall_ci": ci(np.bincount(bc[hit], minlength=len(n)),
                                                          np.bincount(bc, minlength=len(n)))}
            for k, draws in nulls.items():
                nr = np.array([recalls(f[ix], y, blo, bhi)[:2] for ix in draws])
                for j, unit in enumerate(("seg", "box")):
                    obs = row[f"{unit}_recall"]
                    row[f"{unit}_null_{k}"] = nr[:, j].mean()
                    row[f"{unit}_p_{k}"] = ((nr[:, j] >= obs).sum() + 1) / (len(nr) + 1)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"metrics_recall{args.suffix}.csv", index=False)

    for unit, what in (("seg", "SEGMENT recall: share of positive segments flagged"),
                       ("box", "BOX recall: share of Raven boxes with >= 1 flagged segment")):
        print(f"\n=== {what} (thresholds label-free; score {args.score}) ===")
        print(f"{'group':14s}{'n':>6s} {'rule':10s}{'flag%':>6s}  {'recall':>6s}  {'95% CI':11s}"
              f"  {'null any':>8s} {'p':>6s}  {'null within':>11s} {'p':>6s}")
        for r in df.itertuples():
            cnt = r.pos_segments if unit == "seg" else r.boxes
            print(f"{r.group:14s}{cnt:6d} {r.rule:10s}{100 * r.flagged:6.1f}  "
                  f"{getattr(r, unit + '_recall'):6.3f}  {getattr(r, unit + '_recall_ci'):11s}"
                  f"  {getattr(r, unit + '_null_any'):8.3f} {getattr(r, unit + '_p_any'):6.3f}"
                  f"  {getattr(r, unit + '_null_within'):11.3f} {getattr(r, unit + '_p_within'):6.3f}")
    print(f"\nnull any = flags of a random recording, rotated; null within = the recording's own flags, "
          f"rotated ({N_NULL} draws each; smallest possible p = {1 / (N_NULL + 1):.3f}).")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audio-dir", type=Path, default=ROOT / "dataset")
    ap.add_argument("--ann-dir", type=Path, default=ROOT / "dataset" / "annotaions")
    ap.add_argument("--model", type=Path,
                    default=ROOT / "external" / "bacpipe_model_checkpoints" / "perch_v2" / "perch_v2_no_dft.onnx")
    ap.add_argument("--tf-dir", type=Path, default=ROOT / ".cache" / "kagglehub" / "models" / "google" /
                    "bird-vocalization-classifier" / "tensorFlow2" / "perch_v2_cpu" / "1")
    ap.add_argument("--outdir", type=Path, default=ROOT / "data" / "perch_windows")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="fixed rule: summed group probability >= this")
    ap.add_argument("--score", choices=("prob", "maxlogit"), default="prob",
                    help="group score for the lodo thresholds and AP/AUC (fixed rule always uses prob)")
    ap.add_argument("--hop", type=float, default=2.5,
                    help="window hop = segment length, s; must divide 5 (2.5 -> half-overlapping windows)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N recordings only (smoke test)")
    ap.add_argument("--score-only", action="store_true", help="reuse window_scores.csv")
    ap.add_argument("--verify-tf", action="store_true", help="check ONNX against the TF SavedModel first")
    args = ap.parse_args()
    if abs(WINDOW_S / args.hop - round(WINDOW_S / args.hop)) > 1e-9:
        raise SystemExit("--hop must divide 5 s (e.g. 5, 2.5, 1.25)")
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.suffix = "" if args.score == "prob" else f"_{args.score}"

    boxes = read_boxes(args.ann_dir)
    lab = (boxes.groupby(["label", "group"]).size().rename("boxes").reset_index()
           .sort_values(["group", "boxes"], ascending=[True, False]))
    lab.to_csv(args.outdir / "label_map.csv", index=False)
    print("boxes per group:", dict(Counter(boxes["group"])), flush=True)

    cache = args.outdir / "window_scores.csv"
    if args.score_only:
        w = pd.read_csv(cache, dtype={"day": str})
    else:
        labels = pd.read_csv(ROOT / "config" / "perch_class_index.csv", keep_default_na=False)["label"].tolist()
        groups = class_groups(labels)
        print("Perch classes per group:", {g: len(v) for g, v in groups.items()}, flush=True)
        w = extract(args, boxes, labels, groups)
        w.to_csv(cache, index=False)
        print(f"wrote {cache} ({len(w)} windows)", flush=True)
    print(f"{len(w)} Perch windows (5 s, hop {args.hop} s) from {w['recording'].nunique()} recordings "
          f"over {w['day'].nunique()} days", flush=True)
    seg = segments(w, boxes, args.hop)
    seg.to_csv(args.outdir / "segment_scores.csv", index=False)
    print(f"{len(seg)} scored segments of {args.hop} s; {int(seg['silent'].sum())} digitally silent")
    score(seg, args, args.outdir)
    recall_report(seg, boxes, args, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
