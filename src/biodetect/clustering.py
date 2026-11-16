"""Features + LAMDA over every box of a segmentation run (Guerrero et al. 2023, 2.3.3-2.3.4).

Output folder layout (`out` defaults to <box_dir>/lamda):

  out/features.csv.gz              27 features per box (shared by every scaling and order)
  out/                             minmax scaling (the paper)
  out/rank/                        rank scaling (NOT in the paper)
  .../shuffle<seed>/               the same scaling with the boxes presented in a random order

Each clustering folder holds boxes_clustered.csv.gz, prototypes.csv, representatives.csv
(the paper's representative element: the highest-GAD box of each cluster) and run_info.json.
run_info.json reports how well the clusters line up with three label-free descriptions of
each box (recording, frequency band, duration) as AMI/ARI. They are diagnostics, not scores:
high agreement with band or recording means the clusters follow those, not species.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari

from . import features as feat
from .lamda import N_ITER, NIC, lamda, scale

BAND_EDGES_HZ = [0, 1000, 3000, 7000, 16000, np.inf]
BAND_NAMES = ["<1 kHz", "1-3 kHz", "3-7 kHz", "7-16 kHz", ">16 kHz"]
DUR_EDGES_S = [0, 0.05, 0.15, 0.5, np.inf]
DUR_NAMES = ["<50 ms", "50-150 ms", "150-500 ms", ">500 ms"]
DIAGNOSTICS = ["file", "band", "duration"]


def cluster_dir(out: Path, scaling: str, shuffle_seed: int | None) -> Path:
    d = Path(out) if scaling == "minmax" else Path(out) / scaling
    return d if shuffle_seed is None else d / f"shuffle{shuffle_seed}"


def run(box_dir: Path, out: Path | None = None, scaling: str = "minmax", shuffle_seed: int | None = None,
        iterations: int = N_ITER, frame_cols: int = 0, skip_features: bool = False,
        files: list[str] | None = None, workers: int = 1, log=print) -> dict:
    """Describe and cluster the boxes of `box_dir`. Returns the run_info dictionary."""
    box_dir = Path(box_dir)
    out = Path(out) if out is not None else box_dir / "lamda"
    out.mkdir(parents=True, exist_ok=True)
    clus = cluster_dir(out, scaling, shuffle_seed)
    clus.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    feat_path = out / "features.csv.gz"
    if skip_features:
        if not feat_path.exists():
            raise FileNotFoundError(f"{feat_path} not found; run once without --skip-features")
        feats = pd.read_csv(feat_path)
    else:
        feats = feat.extract(box_dir, files, frame_cols, workers, log)
        feats.to_csv(feat_path, index=False)
    t_feat = time.time() - t_start

    raw = feats[feat.FEATURES].to_numpy(np.float64)
    bad = ~np.isfinite(raw).all(1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} boxes have non-finite features")
    data = scale(raw, scaling)
    order = np.arange(len(data))
    if shuffle_seed is not None:
        order = np.random.default_rng(shuffle_seed).permutation(len(data))
    t0 = time.time()
    idx_o, gad_o, cp, elements = lamda(data[order], iterations)
    t_lamda = time.time() - t0
    idx, gad = np.empty_like(idx_o), np.empty_like(gad_o)
    idx[order], gad[order] = idx_o, gad_o

    centre = (feats.low_hz + feats.high_hz) / 2
    b = feats[["file", "selection", "begin_s", "end_s", "low_hz", "high_hz", "n_cells", "mean_db"]].assign(
        segment=np.arange(len(feats)), cluster=idx, gad=gad,
        band=pd.cut(centre, BAND_EDGES_HZ, labels=BAND_NAMES, right=False).astype(str),
        duration=pd.cut(feats.end_s - feats.begin_s, DUR_EDGES_S, labels=DUR_NAMES, right=False).astype(str))
    b.to_csv(clus / "boxes_clustered.csv.gz", index=False)
    sizes = np.bincount(idx, minlength=len(cp))
    pd.DataFrame(cp, columns=feat.FEATURES).assign(
        cluster=np.arange(len(cp)), learn_count=elements, boxes=sizes).to_csv(clus / "prototypes.csv", index=False)
    b.loc[b.groupby("cluster").gad.idxmax()].to_csv(clus / "representatives.csv", index=False)

    lo, hi = raw.min(0), raw.max(0)
    info = {
        "paper": "Guerrero et al. 2023, Methods Ecol Evol 14:1500, sections 2.3.3-2.3.4",
        "box_dir": str(box_dir.resolve()),
        "boxes": int(len(b)), "files": int(b.file.nunique()),
        "features": feat.FEATURES, "n_ceps": feat.N_CEPS,
        "power": "|H|^2 with H = 10^(S'/20), S' = dB above per-frequency modal noise",
        "frame_cols": frame_cols if frame_cols > 0 else "Nh (one frame per box)",
        "dct": "as printed: N_p cos(pi/Nu (i-1)(k-0.5)), k = 1..24",
        "f_peak": "row of H with the largest sum_j |H_ij|^2", "f_min_f_max": "box low_hz / high_hz",
        "scaling": scaling,
        "normalisation": ("min-max over all boxes of the run (paper)" if scaling == "minmax" else
                          "NOT in the paper: rank (percentile) over all boxes of the run, ties averaged"),
        "feature_min": dict(zip(feat.FEATURES, lo.round(6).tolist(), strict=True)),
        "feature_max": dict(zip(feat.FEATURES, hi.round(6).tolist(), strict=True)),
        "lamda": {"mad": "binomial", "gad": "3pi", "nic": NIC, "iterations": iterations,
                  "source": "LAMDA_General.m (Bedoya), unsupervised mode, ported"},
        "order": "run order (recording, time)" if shuffle_seed is None else f"shuffled, seed {shuffle_seed}",
        "clusters": int(len(cp)), "clusters_nonempty": int((sizes > 0).sum()),
        "clusters_ge10": int((sizes >= 10).sum()),
        "largest_cluster_share": float(sizes.max() / len(b)),
        "boxes_in_singletons": int(sizes[sizes == 1].sum()),
        "AMI": {k: float(ami(b[k], b.cluster)) for k in DIAGNOSTICS},
        "ARI": {k: float(ari(b[k], b.cluster)) for k in DIAGNOSTICS},
        "seconds_features": round(t_feat, 1), "seconds_lamda": round(t_lamda, 1),
    }
    ref = clus.parent / "boxes_clustered.csv.gz"
    if shuffle_seed is not None and ref.exists():
        info["AMI_vs_run_order"] = float(ami(pd.read_csv(ref).cluster, b.cluster))
    (clus / "run_info.json").write_text(json.dumps(info, indent=1))
    info["output"] = str(clus)
    return info
