"""Hybrid detector on the long-form recordings: EarthChirp for Insecta, Perch zero-shot for the rest.

The Perch zero-shot run (zeroshot_windows.py) kept only group scores per window. EarthChirp needs
the 1536-d Perch embedding of each window, so this script runs Perch once more on exactly the
same windows (5 s, every 2.5 s, same ONNX model, TF32 off) and keeps the embeddings. It then
scores each window with EarthChirp's insect prototypes (insecta_compare.EarthChirpScore: max
cosine to the 2,012 species + 619 genus + 2 group prototypes, and the margin = that cosine minus
the cosine to the ESC-50 urban / geophony outgroup). Segment scores are the max over the two
windows covering each 2.5 s segment, as in zeroshot_windows.segments().

The hybrid itself is a choice of lanes in the viewer (scripts/perch_viewer.py): Perch for Aves,
Amphibia, Anthropogenic and Weather; EarthChirp in place of Perch's Insecta. Nothing is trained
and nothing here looks at labels except the metrics.

Stages:
  extract  Perch on the 64,368 long-form windows -> earthchirp_embeddings.npy (float16)
           + earthchirp_rebuild_check.csv                                       (~20 min GPU)
           Rebuild check: the group probabilities must match window_scores.csv (the cache the
           listening-confirmed thresholds were set on), else nothing is written.
  score    EarthChirp per window and segment, and Insecta metrics against the Raven boxes
                                                                                (about 1 min)

    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\earthchirp_windows.py                 # both stages
    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\earthchirp_windows.py --stage score
    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\earthchirp_windows.py --limit 2 --device cpu   # smoke

Reads data/perch_windows/{window_scores,segment_scores}.csv, dataset/, data/insecta_compare/
esc50_emb.npy + esc50_meta.csv (the outgroup) and external/earthchirp_lib. Writes
data/perch_windows/ (never rewrites the Perch files there):
  earthchirp_embeddings.npy        (n_windows, 1536) float16, rows in window_scores.csv order
  earthchirp_rebuild_check.csv     per recording: windows, max |prob diff| vs window_scores.csv
  earthchirp_windows.csv           per window: ec_cos, ec_og, ec_margin
  earthchirp_segments.csv          per 2.5 s segment: ec_cos, ec_margin (max of covering windows)
  hybrid_insecta_auc.csv           segment AUROC vs Raven Insecta, recording-bootstrap CIs
  hybrid_insecta_operating.csv     flag share, segment / box recall, off-label flag shares
--limit N runs the first N recordings into *_limit files (the viewer ignores those).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import insecta_compare as IC  # noqa: E402
import zeroshot_windows as z  # noqa: E402

ROOT = z.ROOT
HOP_S = 2.5
EMB_DIM = 1536
MAX_PROB_DIFF = 0.01          # rebuild check: group probability vs the cached run
# Starting thresholds for the viewer, both from the clip windows (insecta_compare.py): the cut
# that flags the same 68% of clip windows as Perch Insecta >= 0.032. Not yet set by listening.
EC_COS_THR, EC_MARGIN_THR = 0.402, 0.280
OFF_LABEL = ("Aves", "Amphibia", "Anthropogenic", "Weather")


def suffix(args) -> str:
    return "_limit" if args.limit else ""


def wav_paths(audio_dir: Path) -> dict[str, str]:
    return {Path(p).stem: p for p in glob.glob(str(audio_dir / "S01634_*" / "*" / "*.wav"))}


def cached_windows(args) -> pd.DataFrame:
    w = pd.read_csv(args.pdir / "window_scores.csv", dtype={"day": str})
    if args.limit:
        w = w[w["recording"].isin(pd.unique(w["recording"])[:args.limit])].reset_index(drop=True)
    return w


# ---- extract ------------------------------------------------------------------------------
def stage_extract(args) -> None:
    w = cached_windows(args)
    recs = list(pd.unique(w["recording"]))
    rows = {r: g for r, g in w.groupby("recording", sort=False)}
    wavs = wav_paths(args.audio_dir)
    missing = [r for r in recs if r not in wavs]
    if missing:
        raise SystemExit(f"recordings in window_scores.csv with no wav: {missing}")
    _, groups = IC.labels_and_groups()
    sess = IC.open_session(args.model, args.device)
    prob_cols = [f"prob_{g}" for g in z.GROUPS]

    final = args.pdir / f"earthchirp_embeddings{suffix(args)}.npy"
    tmp = final.with_suffix(".partial.npy")
    E = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(w), EMB_DIM))
    check, row, t0 = [], 0, time.monotonic()
    with cf.ThreadPoolExecutor(1) as pool:          # read the next file while the GPU runs
        nxt = pool.submit(z.load_windows, wavs[recs[0]], HOP_S)
        for i, rec in enumerate(recs):
            wins = nxt.result()
            if i + 1 < len(recs):
                nxt = pool.submit(z.load_windows, wavs[recs[i + 1]], HOP_S)
            want = rows[rec]
            if len(wins) != len(want) or not np.allclose(want["start_s"], np.arange(len(wins)) * HOP_S):
                raise SystemExit(f"{rec}: {len(wins)} rebuilt windows != {len(want)} in window_scores.csv")
            if want.index[0] != row:
                raise SystemExit(f"{rec}: window_scores.csv rows are not contiguous per recording")
            e, p, _ = IC.run_perch(sess, wins, groups, args.batch)
            d = np.abs(p[:, :len(z.GROUPS)] - want[prob_cols].to_numpy())
            check.append({"recording": rec, "windows": len(wins), "max_prob_diff": float(d.max()),
                          "worst_group": z.GROUPS[int(d.max(axis=0).argmax())]})
            E[row:row + len(e)] = e.astype(np.float16)
            row += len(e)
            if (i + 1) % 10 == 0 or i + 1 == len(recs):
                el = time.monotonic() - t0
                print(f"  {i + 1}/{len(recs)} recordings, {row}/{len(w)} windows, {row / el:.1f} win/s, "
                      f"~{(len(w) - row) / (row / el) / 60:.1f} min left; "
                      f"max prob diff so far {max(c['max_prob_diff'] for c in check):.2e}", flush=True)
    E.flush()
    del E
    chk = pd.DataFrame(check)
    chk.to_csv(args.pdir / f"earthchirp_rebuild_check{suffix(args)}.csv", index=False)
    worst = chk["max_prob_diff"].max()
    print(f"rebuild check: max |group prob diff| vs window_scores.csv = {worst:.2e} "
          f"(limit {MAX_PROB_DIFF}) over {len(chk)} recordings", flush=True)
    if worst > MAX_PROB_DIFF:
        raise SystemExit(f"Perch output differs from the cached run; embeddings left in {tmp}")
    os.replace(tmp, final)
    print(f"wrote {final} ({row} windows)", flush=True)


# ---- score --------------------------------------------------------------------------------
def earthchirp(args) -> IC.EarthChirpScore:
    cdir = ROOT / "data" / "insecta_compare"
    return IC.EarthChirpScore(args.lib, np.load(cdir / "esc50_emb.npy"), pd.read_csv(cdir / "esc50_meta.csv"))


def segment_max(win: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per recording, segment k = max over windows k-1 and k (zeroshot_windows.segments, r = 2)."""
    r = int(round(z.WINDOW_S / HOP_S))
    out = []
    for rec, g in win.groupby("recording", sort=True):
        g = g.sort_values("start_s")
        s = g[cols].rolling(r, min_periods=1).max()
        s.insert(0, "recording", rec)
        s.insert(1, "start_s", g["start_s"].to_numpy())
        out.append(s)
    return pd.concat(out, ignore_index=True)


def ratio_ci(num: np.ndarray, den: np.ndarray, pick: np.ndarray) -> str:
    v = num[pick].sum(axis=1) / np.maximum(den[pick].sum(axis=1), 1)
    return f"{np.percentile(v, 2.5):.3f}-{np.percentile(v, 97.5):.3f}"


def stage_score(args) -> None:
    w = cached_windows(args)
    E = np.load(args.pdir / f"earthchirp_embeddings{suffix(args)}.npy", mmap_mode="r")
    if len(E) != len(w):
        raise SystemExit(f"embeddings have {len(E)} rows, window_scores.csv {len(w)}: rerun --stage extract")
    ec = earthchirp(args)
    cos, og = np.empty(len(w), np.float32), np.empty(len(w), np.float32)
    for k in range(0, len(w), 8192):
        X = np.asarray(E[k:k + 8192], dtype=np.float32)
        cos[k:k + 8192], og[k:k + 8192] = ec.cos(X), ec.og(X)
    win = w[["recording", "day", "start_s"]].assign(ec_cos=cos, ec_og=og, ec_margin=cos - og)
    win.to_csv(args.pdir / f"earthchirp_windows{suffix(args)}.csv", index=False)
    segs = segment_max(win, ["ec_cos", "ec_margin"])
    seg = pd.read_csv(args.pdir / "segment_scores.csv", dtype={"day": str})
    seg = seg[seg["recording"].isin(set(win["recording"]))].reset_index(drop=True)
    if not (seg["recording"].equals(segs["recording"]) and np.allclose(seg["start_s"], segs["start_s"])):
        raise SystemExit("EarthChirp segments do not line up with segment_scores.csv")
    segs.to_csv(args.pdir / f"earthchirp_segments{suffix(args)}.csv", index=False)
    print(f"wrote EarthChirp scores for {len(win)} windows / {len(segs)} segments "
          f"of {win['recording'].nunique()} recordings", flush=True)
    if args.limit:
        print("--limit: metrics skipped (they need every recording)")
        return
    seg["ec_cos"], seg["ec_margin"] = segs["ec_cos"].to_numpy(), segs["ec_margin"].to_numpy()

    # threshold-free: AUROC vs Raven Insecta, ignored segments left out, CIs resample recordings
    t = seg["truth_Insecta"].to_numpy()
    sc = {"perch_prob": seg["prob_Insecta"].to_numpy(), "ec_cos": seg["ec_cos"].to_numpy(),
          "ec_margin": seg["ec_margin"].to_numpy()}
    auc = IC.boot_auc(t == 1, sc, seg["recording"].to_numpy(), t >= 0)
    auc = pd.DataFrame([{"score": k, **v} for k, v in auc.items()])
    auc.to_csv(args.pdir / "hybrid_insecta_auc.csv", index=False)

    # operating points
    boxes = z.read_boxes(args.ann_dir)
    codes, lo, n = z.recording_bounds(seg)
    bx = z.box_segments(boxes, seg, HOP_S, lo, n)
    b = bx[bx["group"] == "Insecta"]
    blo, bhi, bc = b["lo"].to_numpy(), b["hi"].to_numpy(), b["code"].to_numpy()
    y = t == 1
    perch_flag = sc["perch_prob"] >= IC.PERCH_INSECTA_THR
    share = perch_flag.mean()
    q_cos = float(np.quantile(sc["ec_cos"], 1 - share))
    q_mar = float(np.quantile(sc["ec_margin"], 1 - share))
    rules = {f"Perch prob >= {IC.PERCH_INSECTA_THR}": perch_flag,
             f"EC cos > {EC_COS_THR} (clip-matched)": sc["ec_cos"] > EC_COS_THR,
             f"EC margin > {EC_MARGIN_THR} (clip-matched)": sc["ec_margin"] > EC_MARGIN_THR,
             f"EC cos > {q_cos:.3f} (Perch's long-form flag share)": sc["ec_cos"] > q_cos,
             f"EC margin > {q_mar:.3f} (Perch's long-form flag share)": sc["ec_margin"] > q_mar}
    pick = np.random.default_rng(1).integers(0, len(n), (1000, len(n)))
    rows = []
    for name, f in rules.items():
        seg_r, box_r, hit = z.recalls(f, y, blo, bhi)
        row = {"rule": name, "flagged_all": f.mean(), "pos_segments": int(y.sum()), "boxes": len(b),
               "seg_recall": seg_r,
               "seg_recall_ci": ratio_ci(np.bincount(codes[f & y], minlength=len(n)),
                                         np.bincount(codes[y], minlength=len(n)), pick),
               "box_recall": box_r,
               "box_recall_ci": ratio_ci(np.bincount(bc[hit], minlength=len(n)),
                                         np.bincount(bc, minlength=len(n)), pick)}
        for g in OFF_LABEL:   # other group's box, no Insecta box: an upper bound on false alarms
            sel = (seg[f"truth_{g}"] == 1).to_numpy() & (t == 0)
            row[f"flagged_{g}_only"] = f[sel].mean() if sel.any() else np.nan
            row[f"n_{g}_only"] = int(sel.sum())
        unl = (seg[[f"truth_{g}" for g in z.GROUPS]] == 0).all(axis=1).to_numpy()
        row["flagged_unlabelled"] = f[unl].mean()
        rows.append(row)
    op = pd.DataFrame(rows)
    op.to_csv(args.pdir / "hybrid_insecta_operating.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"\nInsecta segment AUROC vs Raven ({auc.n_pos.iloc[0]} positive, {auc.n_neg.iloc[0]} negative "
          f"segments, {auc.n_sessions.iloc[0]} recordings; 95% recording-bootstrap CI; d = minus Perch, paired)")
    for r in auc.itertuples():
        d = "" if r.score == "perch_prob" else f"   d {r.d_vs_perch:+.3f} [{r.d_lo:+.3f}, {r.d_hi:+.3f}]"
        print(f"  {r.score:11s} {r.auroc:.3f} [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}]   AP {r.ap:.3f}{d}")
    print("\nInsecta operating points (recall = share of Raven Insecta flagged; *_only = share flagged of "
          "segments with that group's box and no Insecta box; unlabelled = no Raven box at all)")
    show = ["rule", "flagged_all", "seg_recall", "seg_recall_ci", "box_recall", "box_recall_ci",
            *[f"flagged_{g}_only" for g in OFF_LABEL], "flagged_unlabelled"]
    print(op[show].round(3).to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("all", "extract", "score"), default="all")
    ap.add_argument("--model", type=Path,
                    default=ROOT / "external" / "bacpipe_model_checkpoints" / "perch_v2" / "perch_v2_no_dft.onnx")
    ap.add_argument("--lib", type=Path, default=ROOT / "external" / "earthchirp_lib")
    ap.add_argument("--audio-dir", type=Path, default=ROOT / "dataset")
    ap.add_argument("--ann-dir", type=Path, default=ROOT / "dataset" / "annotaions")
    ap.add_argument("--pdir", type=Path, default=ROOT / "data" / "perch_windows")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N recordings only, to *_limit files")
    args = ap.parse_args()
    if args.stage in ("all", "extract"):
        stage_extract(args)
    if args.stage in ("all", "score"):
        stage_score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
