"""A label-free minimum-size cutoff for speckle (NOT in the paper).

The paper keeps every connected segment, and most of them are background speckle:
noise cells just above the Otsu level that survive the 3x3 opening. To choose a size
cutoff without labels, a random sample of 60 s pieces is phase-randomised: the same
long-term power spectrum, but no calls and no temporal structure. The segmentation then
runs on each noise copy in two ways:

  * paper     the full pipeline, with the copy's own Otsu threshold;
  * same_thr  the real piece's Otsu level (dB above noise) applied to the copy's S'.

For each min-cells cutoff, (noise boxes per piece) / (real boxes per piece) estimates
the share of the surviving real boxes that noise alone would produce.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy.ndimage import mean as label_mean
from scipy.ndimage import sum as label_sum

from . import segmentation as seg

CUTS = [0, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1000]
SEED = 20261015
KINDS = ["real", "null_paper", "null_same_thr"]


def components(x: np.ndarray, sr: int, thr_db: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """n_cells and mean dB of every segment; the Otsu threshold unless thr_db is given."""
    _, _, s1 = seg.noise_reduced(x, sr)
    if thr_db is None:
        mask, thr_db = seg.graythresh_binarize(s1)
    else:
        mask = s1 > thr_db
    mask = seg.clean_mask(mask)
    lab, n = label(mask, structure=seg.CONN8)
    ids = np.arange(1, n + 1)
    return label_sum(mask, lab, ids).astype(int), label_mean(s1, lab, ids), thr_db


def phase_randomise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same magnitude spectrum as x, uniformly random phases (DC and Nyquist kept real)."""
    spec = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, len(spec))
    ph[0] = 0.0
    if len(x) % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(spec) * np.exp(1j * ph), n=len(x))


def run_piece(job: tuple) -> tuple[pd.DataFrame, dict]:
    k, name, path, start, duration, seed = job
    x, sr = seg.read_mono(path, start, start + duration)
    real_n, real_db, thr = components(x, sr)
    xn = phase_randomise(x, np.random.default_rng(seed + k))
    null_n, null_db, null_thr = components(xn, sr)
    same_n, same_db, _ = components(xn, sr, thr)
    rows = [pd.DataFrame({"piece": k, "file": name, "start_s": start, "kind": kind, "n_cells": n, "mean_db": db})
            for kind, n, db in zip(KINDS, [real_n, null_n, same_n], [real_db, null_db, same_db], strict=True)]
    return pd.concat(rows), {"piece": k, "file": name, "start_s": start, "otsu_real": thr, "otsu_null": null_thr}


def cutoff_table(comp: pd.DataFrame, n_pieces: int, cuts: list[int] = CUTS) -> pd.DataFrame:
    """Per cutoff: real and noise boxes kept per piece, and the noise share of the kept boxes."""
    n_real = (comp["kind"] == "real").sum()
    rows = []
    for c in cuts:
        kept = comp[comp["n_cells"] >= c].groupby("kind").size().reindex(KINDS, fill_value=0) / n_pieces
        rows.append({"min_cells": c, "real_per_piece": kept["real"],
                     "null_per_piece_paper": kept["null_paper"], "null_per_piece_same_thr": kept["null_same_thr"],
                     "noise_share_paper": kept["null_paper"] / max(kept["real"], 1e-9),
                     "noise_share_same_thr": kept["null_same_thr"] / max(kept["real"], 1e-9),
                     "real_kept_frac": kept["real"] * n_pieces / max(n_real, 1)})
    return pd.DataFrame(rows)


def run(box_dir: Path, pieces: int = 60, seed: int = SEED, workers: int = 1) -> pd.DataFrame:
    """Sample pieces of a segmentation run (use the run WITHOUT --min-cells), write box_dir/speckle_null/."""
    box_dir = Path(box_dir)
    paths = pd.read_csv(box_dir / "files.csv").set_index("file")["path"]
    pp = pd.read_csv(box_dir / "per_piece.csv")
    pp = pp.sample(n=min(pieces, len(pp)), random_state=seed).sort_values(["file", "start_s"])
    jobs = [(k, r.file, paths[r.file], r.start_s, r.duration_s, seed) for k, r in enumerate(pp.itertuples())]
    res = list(seg._map(run_piece, jobs, workers))
    comp = pd.concat([r[0] for r in res], ignore_index=True)
    thr = pd.DataFrame([r[1] for r in res])
    tab = cutoff_table(comp, len(jobs))
    out = box_dir / "speckle_null"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "cutoff_table.csv", index=False)
    thr.to_csv(out / "thresholds.csv", index=False)
    comp.to_csv(out / "components.csv.gz", index=False)
    return tab
