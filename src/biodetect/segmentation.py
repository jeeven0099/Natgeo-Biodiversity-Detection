"""Noise reduction and segmentation: Guerrero et al. (2023), sections 2.3.1-2.3.2.

Guerrero, Bedoya, Lopez, Daza & Isaza (2023). Acoustic animal identification using
unsupervised learning. Methods in Ecology and Evolution 14:1500-1514. Their steps:

  1. magnitude spectrogram S of each recording
  2. convolve S with a 3x3 Gaussian kernel ("to remove small gaps and graininess")
  3. spectral subtraction (Xie et al. 2017), background = broadband geophony -> S'
  4. Otsu threshold on S'
  5. morphological opening and closing
  6. connected segments in ALL frequency bands -> bounding boxes

The paper was implemented in MATLAB R2022a. Where its text gives no number, the
MATLAB reading of the text or co-author Bedoya's published MATLAB code is used:

  * STFT        spectrogram(x, hann(1024), 768, 1024, fs), from Noise_Reduction_Example_GH.m.
  * Gaussian    imgaussfilt's default sigma 0.5, the only default whose kernel is 3x3
                (2*ceil(2*sigma)+1); replicate padding.
  * Subtraction the QUT/Towsey noise removal that Xie et al. build on: each frequency
                bin's modal dB level over the recording is subtracted, negatives set to 0.
  * Otsu        graythresh(mat2gray(S')) with 256 levels, then imbinarize (S' > level).
  * Morphology  imopen then imclose; element size unstated, 3x3 square.
  * Segments    bwconncomp default (8-connected) + regionprops BoundingBox.
  * Unit        the paper's recordings are 1 min long, so a longer file is processed as
                consecutive 60 s pieces, each with its own noise profile and threshold.

`min_cells` is the one step NOT in the paper: it drops segments smaller than N cells
(see `biodetect.speckle_null` for how to choose N without labels).
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import binary_closing, binary_opening, convolve, find_objects, label
from scipy.ndimage import mean as label_mean
from scipy.ndimage import sum as label_sum
from scipy.signal import stft

N_FFT, WIN, OVERLAP = 1024, 1024, 768
SIGMA = 0.5
SE_OPEN = np.ones((3, 3), bool)
SE_CLOSE = np.ones((3, 3), bool)
CONN8 = np.ones((3, 3), bool)
MODE_BINS = 100
EPS = np.finfo(np.float64).tiny
BOX_COLS = ["begin_s", "end_s", "low_hz", "high_hz", "n_cells", "mean_db"]
RAVEN_SUFFIX = ".Guerrero.Table.1.selections.txt"


# ---------------------------------------------------------------- steps 1-5
def gaussian_kernel(sigma: float = SIGMA) -> np.ndarray:
    """MATLAB imgaussfilt kernel: size 2*ceil(2*sigma)+1, normalised to sum 1."""
    r = int(np.ceil(2 * sigma))
    x = np.arange(-r, r + 1)
    g = np.exp(-(x[:, None] ** 2 + x[None] ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def spectrogram_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steps 1-2: |STFT| smoothed by the 3x3 Gaussian, in dB. Returns (freqs_hz, times_s, S_dB)."""
    f, t, z = stft(x, fs=sr, window="hann", nperseg=WIN, noverlap=OVERLAP, nfft=N_FFT,
                   boundary=None, padded=False, detrend=False, scaling="spectrum")
    s = convolve(np.abs(z), gaussian_kernel(), mode="nearest")
    return f, t, 20.0 * np.log10(s + EPS)


def modal_noise_db(s_db: np.ndarray, bins: int = MODE_BINS) -> np.ndarray:
    """Background level of each frequency bin: the mode of its dB values over time."""
    out = np.empty(s_db.shape[0])
    for f, row in enumerate(s_db):
        h, e = np.histogram(row, bins=bins)
        h = np.convolve(h, np.ones(3) / 3, mode="same")
        k = int(np.argmax(h))
        out[f] = 0.5 * (e[k] + e[k + 1])
    return out


def noise_reduced(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steps 1-3: S', the spectrogram in dB above each bin's modal noise level, clipped at 0."""
    f, t, s_db = spectrogram_db(x, sr)
    return f, t, np.maximum(s_db - modal_noise_db(s_db)[:, None], 0.0)


def graythresh_binarize(s1: np.ndarray) -> tuple[np.ndarray, float]:
    """Step 4: MATLAB imbinarize(I, graythresh(I)) with I = mat2gray(s1).

    Returns the mask and the threshold in the units of s1 (dB above noise)."""
    lo, hi = float(s1.min()), float(s1.max())
    if hi <= lo:
        return np.zeros(s1.shape, bool), np.nan
    i = (s1 - lo) / (hi - lo)
    counts = np.bincount(np.round(i * 255).astype(np.int64).ravel(), minlength=256).astype(np.float64)
    p = counts / counts.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(1, 257))
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    sb[~np.isfinite(sb)] = -np.inf
    idx = np.mean(np.flatnonzero(sb == sb.max())) + 1          # 1-based, ties averaged as MATLAB does
    level = (idx - 1) / 255.0
    return i > level, lo + level * (hi - lo)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Step 5: opening, then closing, with 3x3 squares."""
    return binary_closing(binary_opening(mask, structure=SE_OPEN), structure=SE_CLOSE)


# ---------------------------------------------------------------- step 6
def segment(x: np.ndarray, sr: int, t0: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """One recording (or 60 s piece) -> bounding boxes in seconds and Hz, plus piece statistics."""
    f, t, s1 = noise_reduced(x, sr)
    mask, thr_db = graythresh_binarize(s1)
    mask = clean_mask(mask)
    lab, n = label(mask, structure=CONN8)
    info = {"otsu_db_above_noise": thr_db, "cells_on": float(mask.mean()), "segments": int(n)}
    if n == 0:
        return pd.DataFrame(columns=BOX_COLS), info
    sl = find_objects(lab)
    ids = np.arange(1, n + 1)
    fs0 = np.array([a[0].start for a in sl])
    fs1 = np.array([a[0].stop for a in sl])
    ts0 = np.array([a[1].start for a in sl])
    ts1 = np.array([a[1].stop for a in sl])
    df, dt = f[1] - f[0], (WIN - OVERLAP) / sr
    return pd.DataFrame({
        "begin_s": t0 + t[ts0] - dt / 2, "end_s": t0 + t[ts1 - 1] + dt / 2,
        "low_hz": np.maximum(f[fs0] - df / 2, 0.0), "high_hz": np.minimum(f[fs1 - 1] + df / 2, sr / 2),
        "n_cells": label_sum(mask, lab, ids).astype(int),
        "mean_db": label_mean(s1, lab, ids),
    }), info


def piece_edges(duration_s: float, chunk_s: float) -> list[float]:
    """Edges of the consecutive pieces of a file; a tail shorter than half a piece joins the one before."""
    step = chunk_s if 0 < chunk_s < duration_s else duration_s
    edges = list(np.arange(0.0, duration_s, step)) + [duration_s]
    if len(edges) > 2 and edges[-1] - edges[-2] < step / 2:
        del edges[-2]
    return edges


def read_mono(path: str | Path, start_s: float, stop_s: float) -> tuple[np.ndarray, int]:
    """Samples [start_s, stop_s) of a file, channels averaged, as float64."""
    info = sf.info(str(path))
    a = max(0, int(round(start_s * info.samplerate)))
    z = min(info.frames, int(round(stop_s * info.samplerate)))
    x, sr = sf.read(str(path), start=a, stop=max(z, a + 1), dtype="float64", always_2d=True)
    return x.mean(1), sr


def segment_file(path: str | Path, chunk_s: float = 60.0) -> tuple[pd.DataFrame, list[dict]]:
    """All boxes of one file, piece by piece, and one statistics row per piece."""
    info = sf.info(str(path))
    duration = info.frames / info.samplerate
    boxes, pieces = [], []
    edges = piece_edges(duration, chunk_s)
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        x, sr = read_mono(path, start, stop)
        if len(x) >= WIN:
            b, i = segment(x, sr, start)
            boxes.append(b)
            pieces.append({"start_s": start, "duration_s": len(x) / sr, **i})
    out = pd.concat(boxes, ignore_index=True) if boxes else pd.DataFrame(columns=BOX_COLS)
    return out, pieces


# ---------------------------------------------------------------- outputs
def raven_table(boxes: pd.DataFrame, path: Path) -> None:
    """Write boxes as a Raven Pro selection table (tab-separated, CRLF)."""
    pd.DataFrame({
        "Selection": np.arange(1, len(boxes) + 1), "View": "Spectrogram 1", "Channel": 1,
        "Begin Time (s)": boxes["begin_s"].round(4), "End Time (s)": boxes["end_s"].round(4),
        "Low Freq (Hz)": boxes["low_hz"].round(1), "High Freq (Hz)": boxes["high_hz"].round(1),
        "Delta Time (s)": (boxes["end_s"] - boxes["begin_s"]).round(4),
        "Delta Freq (Hz)": (boxes["high_hz"] - boxes["low_hz"]).round(1),
        "Cells": boxes["n_cells"], "Mean dB above noise": boxes["mean_db"].round(2),
    }).to_csv(path, sep="\t", index=False, lineterminator="\r\n")


def find_wavs(inputs: Iterable[str | Path]) -> dict[str, Path]:
    """Recording name (file stem) -> path, for .wav files and folders searched recursively."""
    paths: list[Path] = []
    for p in map(Path, inputs):
        if p.is_dir():
            paths += sorted(q for q in p.rglob("*") if q.suffix.lower() == ".wav")
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(p)
    dupes = sorted(s for s, n in Counter(p.stem for p in paths).items() if n > 1)
    if dupes:
        raise ValueError(f"recordings must have unique file names; repeated: {', '.join(dupes[:10])}")
    return {p.stem: p.resolve() for p in paths}


def _job(job: tuple[str, str, float]) -> tuple[str, pd.DataFrame, list[dict]]:
    name, path, chunk_s = job
    boxes, pieces = segment_file(path, chunk_s)
    return name, boxes, [{"file": name, **p} for p in pieces]


def _map(fn, jobs: list, workers: int):
    if workers <= 1 or len(jobs) <= 1:
        yield from map(fn, jobs)
        return
    with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
        yield from ex.map(fn, jobs)


def run(files: dict[str, Path], out: Path, chunk_s: float = 60.0, min_cells: int = 0,
        workers: int = 1, log=print) -> pd.DataFrame:
    """Segment every recording and write the run folder. Returns all kept boxes.

    Writes to `out`: boxes.csv.gz, per_piece.csv, files.csv, run_info.json and one Raven
    selection table per recording in raven_tables/.
    """
    if not files:
        raise ValueError("no .wav files to segment")
    out = Path(out)
    (out / "raven_tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"file": list(files), "path": [str(p) for p in files.values()]}).to_csv(
        out / "files.csv", index=False)
    jobs = [(n, str(p), chunk_s) for n, p in sorted(files.items())]
    all_boxes, all_pieces = [], []
    for k, (name, b, pieces) in enumerate(_map(_job, jobs, workers), 1):
        for p in pieces:
            in_piece = (b["begin_s"] >= p["start_s"]) & (b["begin_s"] < p["start_s"] + p["duration_s"])
            p["segments_kept"] = int((in_piece & (b["n_cells"] >= min_cells)).sum())
        b = b[b["n_cells"] >= min_cells].sort_values(["begin_s", "low_hz"]).reset_index(drop=True)
        raven_table(b, out / "raven_tables" / f"{name}{RAVEN_SUFFIX}")
        all_boxes.append(b.assign(file=name, selection=np.arange(1, len(b) + 1)))
        all_pieces += pieces
        log(f"{k}/{len(jobs)} {name}: {len(b)} boxes")
    boxes = pd.concat(all_boxes, ignore_index=True)
    boxes = boxes[["file", "selection", *BOX_COLS]]
    boxes.to_csv(out / "boxes.csv.gz", index=False)
    pd.DataFrame(all_pieces).to_csv(out / "per_piece.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({
        "paper": "Guerrero et al. 2023, Methods Ecol Evol 14:1500, sections 2.3.1-2.3.2",
        "stft": {"window": "hann", "win": WIN, "overlap": OVERLAP, "nfft": N_FFT},
        "gaussian": {"size": "3x3", "sigma": SIGMA, "padding": "replicate"},
        "spectral_subtraction": "per-frequency modal dB level subtracted, negatives set to 0",
        "otsu": "graythresh(mat2gray(S')), imbinarize", "opening": "3x3 square", "closing": "3x3 square",
        "connectivity": 8, "chunk_s": chunk_s, "min_cells": min_cells,
        "min_cells_note": ("paper as written" if min_cells <= 0 else
                           "post-step NOT in the paper; choose it with `biodetect speckle-null`"),
        "files": len(jobs), "boxes": int(len(boxes))}, indent=1))
    return boxes
