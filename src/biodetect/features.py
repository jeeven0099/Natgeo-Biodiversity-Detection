"""Box features: Guerrero et al. (2023), section 2.3.3.

Each box G cuts a segment H (Nu frequency rows x Nh time columns) out of the noise-reduced
spectrogram S' of its 60 s piece, recomputed exactly as in `segmentation.segment`. Then

  eq. 1   Q_{i,m} = log sum_{j in frame m} |H_{i,j}|^2          (frames of w columns)
  eq. 2   Y_{k,m} = N_p sum_{i=1..Nu} Q_{i,m} cos(pi/Nu (i-1)(k-0.5)),
          N_p = sqrt(1/Nu) for k = 1 and sqrt(2/Nu) otherwise, k = 1..24

and the feature vector is y = [f_peak, f_min, f_max, Y_1..Y_24] (27 values).

Choices the paper leaves open:
  * |H|^2   S' is in dB above the per-frequency noise, so H = 10^(S'/20): the magnitude
            relative to the noise (1 where S' was clipped to 0), so the log in eq. 1 is finite.
  * w       not given; y has one value per coefficient, so the default is one frame spanning
            the box (w = Nh). frame_cols > 0 instead averages Y over frames of w columns.
  * eq. 2   used as printed (the usual DCT-II would be cos(pi/Nu (i-0.5)(k-1))); k runs to 24
            even when Nu < 24.
  * f_peak  frequency of the row of H with the most energy, sum_j |H_ij|^2.
  * f_min, f_max  the box's low and high edge.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from . import segmentation as seg

N_CEPS = 24
FEATURES = ["f_peak", "f_min", "f_max"] + [f"c{k}" for k in range(1, N_CEPS + 1)]


def cepstra(q: np.ndarray, n: int = N_CEPS) -> np.ndarray:
    """Eq. 2 as printed, for Q of shape (Nu, frames) -> (n, frames)."""
    nu = q.shape[0]
    i = np.arange(1, nu + 1)[:, None]
    k = np.arange(1, n + 1)[None]
    basis = np.cos(np.pi / nu * (i - 1) * (k - 0.5))                     # (Nu, n)
    norm = np.where(k == 1, np.sqrt(1.0 / nu), np.sqrt(2.0 / nu))
    return (basis * norm).T @ q


def box_features(s1: np.ndarray, f: np.ndarray, r0: int, r1: int, c0: int, c1: int,
                 frame_cols: int = 0) -> np.ndarray:
    """[f_peak, Y_1..Y_24] of the segment H = S'[r0:r1, c0:c1] (f_min and f_max come from the box)."""
    p = 10.0 ** (s1[r0:r1, c0:c1] / 10.0)                                # |H|^2 with H = 10^(S'/20)
    nh = p.shape[1]
    w = nh if frame_cols <= 0 else frame_cols
    edges = list(range(0, nh, w)) + [nh]
    q = np.log(np.stack([p[:, a:b].sum(1) for a, b in zip(edges[:-1], edges[1:], strict=True)], 1))   # eq. 1
    y = cepstra(q).mean(1)
    return np.concatenate([[f[r0 + int(np.argmax(p.sum(1)))]], y])


def box_cells(box, start_s: float, t0: float, dt: float, df: float) -> tuple[int, int, int, int]:
    """Row and column range of a box in the S' of the piece that starts at start_s."""
    c0 = int(round((box.begin_s - start_s + dt / 2 - t0) / dt))
    c1 = int(round((box.end_s - start_s - dt / 2 - t0) / dt)) + 1
    r0 = int(np.ceil(box.low_hz / df - 1e-6))
    r1 = int(np.floor(box.high_hz / df + 1e-6)) + 1
    return r0, r1, c0, c1


def file_features(job: tuple) -> tuple[str, np.ndarray, np.ndarray]:
    """Features of every box of one recording, in the box table's order, and the (Nu, Nh) of each H."""
    name, path, chunk_s, boxes, frame_cols = job
    info = sf.info(str(path))
    sr, duration = info.samplerate, info.frames / info.samplerate
    df, dt = sr / seg.N_FFT, (seg.WIN - seg.OVERLAP) / sr
    out = np.full((len(boxes), 1 + N_CEPS), np.nan)
    shape = np.zeros((len(boxes), 2), int)
    edges = seg.piece_edges(duration, chunk_s)
    piece = np.searchsorted(edges, boxes.begin_s.to_numpy() + dt / 2, side="right") - 1
    for p in np.unique(piece):
        start, stop = edges[p], edges[p + 1]
        x, _ = seg.read_mono(path, start, stop)
        f, t, s1 = seg.noise_reduced(x, sr)
        for j in np.flatnonzero(piece == p):
            r0, r1, c0, c1 = box_cells(boxes.iloc[j], start, t[0], dt, df)
            out[j] = box_features(s1, f, r0, r1, c0, c1, frame_cols)
            shape[j] = r1 - r0, c1 - c0
    return name, out, shape


def extract(box_dir: Path, files: list[str] | None = None, frame_cols: int = 0, workers: int = 1,
            log=print) -> pd.DataFrame:
    """Features of every box of a segmentation run: the box table plus FEATURES, nu and nh."""
    import json

    box_dir = Path(box_dir)
    chunk_s = float(json.loads((box_dir / "run_info.json").read_text())["chunk_s"])
    boxes = pd.read_csv(box_dir / "boxes.csv.gz")
    paths = pd.read_csv(box_dir / "files.csv").set_index("file")["path"]
    if files:
        boxes = boxes[boxes.file.isin(files)]
    if boxes.empty:
        raise ValueError(f"no boxes to describe in {box_dir}")
    groups = dict(tuple(boxes.groupby("file", sort=False)))
    jobs = [(name, paths[name], chunk_s, g, frame_cols) for name, g in groups.items()]
    res = {}
    for k, (name, y, shape) in enumerate(seg._map(file_features, jobs, workers), 1):
        res[name] = (y, shape)
        log(f"{k}/{len(jobs)} {name}: {len(y)} boxes")
    parts = []
    for name, g in groups.items():
        y, shape = res[name]
        parts.append(g.assign(f_peak=y[:, 0], f_min=g.low_hz.to_numpy(), f_max=g.high_hz.to_numpy(),
                              **{f"c{k}": y[:, k] for k in range(1, N_CEPS + 1)},
                              nu=shape[:, 0], nh=shape[:, 1]))
    return pd.concat(parts, ignore_index=True)
