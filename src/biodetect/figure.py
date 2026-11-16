"""Four-panel figure of the segmentation of one 60 s piece: S, S', Otsu mask, boxes.

Every panel is recomputed with the functions in `biodetect.segmentation`, so the boxes
drawn are exactly the ones that `biodetect segment` writes for that piece.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import segmentation as seg


def plot_piece(x: np.ndarray, sr: int, t0: float = 0.0, top_khz: float | None = None, min_cells: int = 0,
               title: str = "", view: tuple[float, float] | None = None):
    """Return a matplotlib Figure for samples x that start at t0 seconds."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    f, t, s_db = seg.spectrogram_db(x, sr)
    s1 = np.maximum(s_db - seg.modal_noise_db(s_db)[:, None], 0.0)
    raw, thr = seg.graythresh_binarize(s1)
    mask = seg.clean_mask(raw)
    boxes, _ = seg.segment(x, sr, t0)
    small = boxes["n_cells"] < min_cells

    top = sr / 2000 if top_khz is None else min(top_khz, sr / 2000)
    keep = f <= top * 1000
    ext = [t0, t0 + len(x) / sr, 0, f[keep][-1] / 1000]
    lo, hi = np.percentile(s_db[keep], [5, 99.5])
    grey = dict(origin="lower", aspect="auto", extent=ext, cmap="Greys")
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    ax[0].imshow(s_db[keep], vmin=lo, vmax=hi, **grey)
    ax[0].set_title("1-2. |STFT| (Hann 1024, overlap 768) after a 3x3 Gaussian (sigma 0.5), dB")
    ax[1].imshow(s1[keep], vmin=0, vmax=max(np.percentile(s1[keep], 99.5), 1e-6), **grey)
    ax[1].set_title("3. Spectral subtraction: dB above each frequency bin's modal level, clipped at 0")
    ax[2].imshow(raw[keep], interpolation="nearest", **grey)
    ax[2].set_title(f"4. Otsu threshold: {thr:.1f} dB above noise, {raw.mean():.1%} of cells on")
    ax[3].imshow(s_db[keep], alpha=0.45, vmin=lo, vmax=hi, **grey)
    ax[3].imshow(np.ma.masked_where(~mask[keep], mask[keep]), origin="lower", aspect="auto", extent=ext,
                 cmap="autumn", alpha=0.35, interpolation="nearest")
    for r, s in zip(boxes.itertuples(), small, strict=True):
        ax[3].add_patch(Rectangle((r.begin_s, r.low_hz / 1000), r.end_s - r.begin_s, (r.high_hz - r.low_hz) / 1000,
                                  fill=False, ec="#9a9a9a" if s else "#1f5fbf", lw=0.5 if s else 0.8,
                                  ls=":" if s else "-"))
    kept = f"{int((~small).sum())} kept at min_cells {min_cells} (dotted: dropped)" if min_cells > 0 else ""
    ax[3].set_title(f"5-6. Open + close 3x3, 8-connected segments: {len(boxes)} boxes" + (f", {kept}" if kept else ""))
    for a in ax:
        a.set_ylabel("kHz")
        a.set_ylim(0, ext[3])
    if view is not None:
        ax[3].set_xlim(*view)
    ax[3].set_xlabel(f"{title}   time (s)".strip())
    return fig


def render(wav: str | Path, start_s: float = 0.0, chunk_s: float = 60.0, top_khz: float | None = None,
           min_cells: int = 0, view: tuple[float, float] | None = None, out: str | Path | None = None) -> Path:
    """Plot the piece [start_s, start_s + chunk_s) of a recording and save it as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wav = Path(wav)
    x, sr = seg.read_mono(wav, start_s, start_s + chunk_s)
    fig = plot_piece(x, sr, start_s, top_khz, min_cells, wav.stem, view)
    png = Path(out) if out else Path("results") / "figures" / f"{wav.stem}_{int(start_s)}s.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png
