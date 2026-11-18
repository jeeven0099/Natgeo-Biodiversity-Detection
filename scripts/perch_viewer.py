"""Spectrogram viewer for the Perch 2.0 zero-shot group scores (scripts/zeroshot_windows.py).

One page = one minute of one recording. A Perch box is a run of consecutive 2.5 s
segments whose summed group probability is >= that group's threshold. Perch gives
no frequency range, so its boxes span Perch's band (60 Hz - 16 kHz), each group
slightly inset so overlapping groups stay visible. Raven boxes are drawn with their
own frequency range.

Hybrid lanes: when data/perch_windows/earthchirp_segments.csv exists (scripts/
earthchirp_windows.py), two EarthChirp Insecta lanes are added and drawn like the Perch
groups: EC-Insecta = EarthChirp insect cosine, EC-margin = that cosine minus the cosine to
the ESC-50 urban / geophony outgroup. The default lanes are then the hybrid: Perch for
Aves, Amphibia, Anthropogenic and Weather, EarthChirp (cosine) for insects. Perch's own
Insecta lane stays available, unticked, for comparison.

Reads data/perch_windows/segment_scores.csv (+ earthchirp_segments.csv if present),
dataset/annotaions/*.selections.txt and the wavs; writes nothing. Used by
notebooks/perch_zeroshot_viewer.ipynb.
"""
from __future__ import annotations

import glob
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zeroshot_windows as z  # noqa: E402

ROOT = z.ROOT
SEG_S = 2.5
PAGE_S = 60.0
PERCH_BAND = (0.06, 16.0)                    # kHz

# Label-free choice: max(segment recall - share of audio flagged), stable across held-out
# days, except Anthropogenic, where the 10% budget (0.043) keeps 85% of boxes for a third
# of the flags of the 0.009 optimum. Insecta and Mammalia have no useful threshold.
# Confirmed by listening in this viewer: Aves 0.94, Insecta 0.032, Anthropogenic 0.043 (2026-11-14);
# Amphibia 0.128 and, for the hybrid, EC-Insecta 0.42 with the three Perch ones (2026-11-15)
THRESHOLDS = {"Aves": 0.94, "Amphibia": 0.128, "Insecta": 0.032, "Mammalia": 0.013,
              "Anthropogenic": 0.043, "Weather": 0.011,
              # EarthChirp cosine: 0.42 by listening (the clip-matched start was 0.402).
              # EarthChirp margin: the cut that flags the same 68% of clip windows as Perch Insecta
              # >= 0.032 (insecta_compare.py); not listened to
              "EC-Insecta": 0.42, "EC-margin": 0.280}
RELIABILITY = {"Aves": "good (listened)", "Anthropogenic": "good (listened)", "Amphibia": "good (listened)",
               "Weather": "tentative", "Insecta": "good (listened)",
               "Mammalia": "untestable (1 recording)",
               "EC-Insecta": "good (listened)", "EC-margin": "untested (clip-matched)"}
EC_GROUPS = {"EC-Insecta": "ec_cos", "EC-margin": "ec_margin"}   # lane -> earthchirp_segments.csv column
EC_FILE = ROOT / "data" / "perch_windows" / "earthchirp_segments.csv"
HAS_EC = EC_FILE.exists()
VIEW_GROUPS = z.GROUPS + (tuple(EC_GROUPS) if HAS_EC else ())
# hybrid when the EarthChirp scores exist: EarthChirp for insects, Perch for the rest
DEFAULT_ON = (("Aves", "Amphibia", "EC-Insecta", "Anthropogenic", "Weather") if HAS_EC
              else ("Aves", "Amphibia", "Insecta", "Anthropogenic", "Weather"))
MIN_TAIL_S = 3.0                             # a last page shorter than this is not shown
MAX_CLIP_S = 30.0                # longest clip the Perch-box player plays
COLORS = {"Aves": "#e6194b", "Amphibia": "#3cb44b", "Insecta": "#ffe119", "Mammalia": "#f58231",
          "Anthropogenic": "#42d4f4", "Weather": "#f032e6", "Unknown": "#bfbfbf",
          "EC-Insecta": "#9a6324", "EC-margin": "#469990"}


def lane_name(g: str) -> str:
    return g if g in EC_GROUPS else f"Perch {g}"


# ---- tightened boxes (scripts/cell_maps.py) ---------------------------------------------
CELL_FILE = ROOT / "data" / "perch_windows" / "cell_maps.npy"
CELL_LANES = tuple(z.GROUPS) + ("EC-Insecta", "EC-margin")     # last axis of cell_maps.npy
CELL_S = 0.3125                              # 16 time cells per 5 s window
SLOTS_PER_SEG = int(round(SEG_S / CELL_S))   # 8: window k covers slots 8k .. 8k + 15
ROW_EDGES_KHZ = (0.06, 0.955, 2.864, 6.971, 16.0)   # Perch's 4 frequency rows, low to high
KAPPA = 0.5                                  # keep cells >= this share of the way from box median to box max
ROW_SUPPORT = 0.25                           # a row needs >= 1/4 of the kept cells of the strongest row
BOX_MODES = ("both", "tightened", "full band")


def slot_grid(m: np.ndarray) -> np.ndarray:
    """(n_windows, 16, 4, lanes) -> (8 (n - 1) + 16, 4, lanes): each 0.3125 s slot takes the max of
    the two windows covering it, as a segment does in zeroshot_windows.segments()."""
    n = len(m)
    g = np.full((SLOTS_PER_SEG * (n - 1) + 16,) + m.shape[2:], -np.inf, np.float32)
    for off in (0, 1):                       # even (odd) windows do not overlap each other
        k = np.arange(off, n, 2)
        idx = (SLOTS_PER_SEG * k[:, None] + np.arange(16)).ravel()
        g[idx] = np.maximum(g[idx], m[k].reshape((-1,) + m.shape[2:]))
    return g


def tighten(sub: np.ndarray, kappa: float,
            row_support: float = ROW_SUPPORT) -> tuple[int, int, int, int, np.ndarray]:
    """sub (slots, 4) of one box -> (slot_lo, slot_hi, row_lo, row_hi, kept cells): the bounding
    box of the cells at or above kappa of the way from the box's MEDIAN cell (its background) to
    its highest cell, in the rows that hold at least row_support of the kept cells of the
    strongest row. kappa 0 keeps the whole box. (From the box minimum instead, scattered
    background cells pass; and 1-3 stray cells in another row would stretch the box to full
    band. Both seen on the smoke recordings' Aves boxes.)"""
    lo, hi = float(np.median(sub)), float(sub.max())
    if kappa <= 0 or hi - lo < 1e-6:
        keep = np.ones(sub.shape, bool)
    else:
        keep = (sub - lo) / (hi - lo) >= kappa
        n = keep.sum(0)
        keep &= (n >= row_support * n.max())[None, :]
    ts, fs = np.flatnonzero(keep.any(1)), np.flatnonzero(keep.any(0))
    return int(ts[0]), int(ts[-1]) + 1, int(fs[0]), int(fs[-1]) + 1, keep


class CellMaps:
    """Per-cell lane scores of every window (cell_maps.npy), as one slot grid per recording."""

    def __init__(self, outdir: Path = ROOT / "data" / "perch_windows", name: str = CELL_FILE.name):
        self.M = np.load(outdir / name, mmap_mode="r")
        w = pd.read_csv(outdir / "window_scores.csv", usecols=["recording", "start_s"])
        if len(self.M) > len(w) or self.M.shape[1:] != (16, 4, len(CELL_LANES)):
            raise SystemExit(f"{name} has shape {self.M.shape}; rerun cell_maps.py")
        n = len(self.M)                      # a *_limit file holds the first recordings
        if n < len(w) and w["recording"].iloc[n] == w["recording"].iloc[n - 1]:
            raise SystemExit(f"{name} ends inside recording {w['recording'].iloc[n - 1]}")
        w = w.iloc[:n]
        codes, recs = pd.factorize(w["recording"])
        first = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
        self.rows = {r: (a, b - a) for r, a, b in zip(recs, first, np.r_[first[1:], n])}
        self._grids: dict[str, np.ndarray] = {}

    def grid(self, rec: str) -> np.ndarray | None:
        if rec not in self.rows:
            return None
        if rec not in self._grids:
            a, n = self.rows[rec]
            self._grids = {rec: slot_grid(np.asarray(self.M[a:a + n], np.float32))}   # keep one
        return self._grids[rec]

    def tighten_boxes(self, rec: str, pb: pd.DataFrame, kappa: float,
                      row_support: float = ROW_SUPPORT) -> pd.DataFrame:
        """Adds t_begin, t_end, lo_khz, hi_khz, area (share of the full box's cells) and kept
        (bool cells x 4 rows, slot 0 = t_begin) to each Perch box; NaN when no map."""
        pb = pb.copy()
        cols = {c: [] for c in ("t_begin", "t_end", "lo_khz", "hi_khz", "area", "kept")}
        g = self.grid(rec)
        for b in pb.itertuples():
            s0, s1 = int(round(b.begin_s / CELL_S)), int(round(b.end_s / CELL_S))
            if g is None or b.group not in CELL_LANES or s1 > len(g):
                for c in cols:
                    cols[c].append(np.nan if c != "kept" else None)
                continue
            sub = g[s0:s1, :, CELL_LANES.index(b.group)]
            t0, t1, r0, r1, keep = tighten(sub, kappa, row_support)
            cols["t_begin"].append((s0 + t0) * CELL_S)
            cols["t_end"].append((s0 + t1) * CELL_S)
            cols["lo_khz"].append(ROW_EDGES_KHZ[r0])
            cols["hi_khz"].append(ROW_EDGES_KHZ[r1])
            cols["area"].append((t1 - t0) * (r1 - r0) / sub.size)
            cols["kept"].append(keep[t0:t1])
        for c, v in cols.items():
            pb[c] = pd.Series(v, index=pb.index, dtype=object if c == "kept" else float)
        return pb


# ---- data ------------------------------------------------------------------------------
def load_segments(outdir: Path = ROOT / "data" / "perch_windows") -> pd.DataFrame:
    seg = pd.read_csv(outdir / "segment_scores.csv", dtype={"day": str})
    if not np.allclose(np.diff(seg["start_s"])[np.diff(pd.factorize(seg["recording"])[0]) == 0], SEG_S):
        raise SystemExit("segment_scores.csv is not on a 2.5 s grid; rerun zeroshot_windows.py with --hop 2.5")
    ec_file = outdir / EC_FILE.name
    if ec_file.exists():
        e = pd.read_csv(ec_file)
        if len(e) != len(seg) or not (e["recording"].equals(seg["recording"])
                                      and np.allclose(e["start_s"], seg["start_s"])):
            raise SystemExit(f"{ec_file.name} does not line up with segment_scores.csv; "
                             "rerun earthchirp_windows.py --stage score")
        for g, c in EC_GROUPS.items():
            seg[f"prob_{g}"] = e[c].to_numpy()
    return seg


def load_raven(ann_dir: Path = ROOT / "dataset" / "annotaions") -> pd.DataFrame:
    """All Raven boxes with time, frequency (kHz), label and group."""
    rows = []
    for f in sorted(ann_dir.glob("*.selections.txt")):
        t = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
        rec = f.name.split(".")[0]
        for _, r in t.iterrows():
            rows.append((rec, float(r["Begin Time (s)"]), float(r["End Time (s)"]),
                         float(r["Low Freq (Hz)"]) / 1000, float(r["High Freq (Hz)"]) / 1000,
                         r["Annotation"].strip()))
    b = pd.DataFrame(rows, columns=["recording", "begin_s", "end_s", "low_khz", "high_khz", "label"])
    b["group"] = b["label"].map(z.norm_label).map(z.LABEL_TO_GROUP)
    if b["group"].isna().any():
        raise SystemExit(f"unmapped labels: {sorted(b.loc[b['group'].isna(), 'label'].unique())}")
    return b


def wav_paths(audio_dir: Path = ROOT / "dataset") -> dict[str, str]:
    return {Path(p).stem: p for p in glob.glob(str(audio_dir / "S01634_*" / "*" / "*.wav"))}


def audio_options(recs: list[str], wavs: dict[str, str], raven: pd.DataFrame,
                  scored: set[str]) -> list[tuple[str, str]]:
    """Dropdown entries: wav file name, start date and time, length, Raven box count."""
    nbox = raven.groupby("recording").size()
    out = []
    for r in recs:
        info = sf.info(wavs[r])
        dur = info.frames / info.samplerate
        span = f"{pd.to_datetime(r.split('_', 1)[1], format='%Y%m%d_%H%M%S'):%Y-%m-%d %H:%M}"
        note = f"{nbox[r]} Raven boxes" if r in scored else "not annotated, no Perch scores"
        out.append((f"{Path(wavs[r]).name}   {span}   ({dur / 60:.1f} min, {note})", r))
    return out


def perch_boxes(seg_rec: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Runs of consecutive flagged 2.5 s segments of one recording, per group."""
    seg_rec = seg_rec.sort_values("start_s")
    st = seg_rec["start_s"].to_numpy()
    out = []
    for g, t in thresholds.items():
        if f"prob_{g}" not in seg_rec:          # an EarthChirp lane with no EarthChirp scores
            continue
        s = seg_rec[f"prob_{g}"].to_numpy()
        f = s >= t
        edge = np.diff(np.r_[0, f.astype(np.int8), 0])
        for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
            out.append((g, st[a], st[b - 1] + SEG_S, float(s[a:b].max()), b - a))
    return pd.DataFrame(out, columns=["group", "begin_s", "end_s", "max_prob", "n_segments"])


# ---- audio -----------------------------------------------------------------------------
def read_page(path: str, t0: float, dur: float = PAGE_S) -> tuple[np.ndarray, int]:
    info = sf.info(path)
    a = int(round(t0 * info.samplerate))
    b = min(info.frames, int(round((t0 + dur) * info.samplerate)))
    x, sr = sf.read(path, start=a, stop=b, dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def n_pages(path: str) -> int:
    """One-minute pages; a tail shorter than MIN_TAIL_S joins no page (most files are 1200.04 s)."""
    info = sf.info(path)
    return max(1, int(np.ceil((info.frames / info.samplerate - MIN_TAIL_S) / PAGE_S)))


def spectrogram(x: np.ndarray, sr: int, n_fft: int = 1024, hop: int | None = None):
    """Hann STFT in dB. Returns (freq_khz, time_s, db) with time at frame starts."""
    hop = hop or n_fft // 2
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    fr = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft).astype(np.float32)
    db = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)).T + 1e-10)
    return np.fft.rfftfreq(n_fft, 1 / sr) / 1000, np.arange(fr.shape[0]) * hop / sr, db


PLAY_MODES = ("original", "band-passed", "cell mask")
STFT_N = 2048                                # 43 ms at 48 kHz; hop STFT_N // 4
SMOOTH = (5, 5)                              # gain smoothing, bins x frames (~120 Hz x ~50 ms): no clicks


def cell_gain(f_khz: np.ndarray, t_s: np.ndarray, b, mode: str) -> np.ndarray:
    """0/1 gain (freq, time) for STFT bins at f_khz and absolute frame-centre times t_s, from one
    tightened box b (lo_khz, hi_khz, t_begin, kept). 'band-passed': the tightened box's rows at every
    time. 'cell mask': only the kept cells (0.3125 s x one row), muted elsewhere, pad included."""
    if mode == "band-passed":
        return np.repeat(((f_khz >= b.lo_khz) & (f_khz < b.hi_khz)).astype(np.float32)[:, None], len(t_s), 1)
    row = np.searchsorted(ROW_EDGES_KHZ, f_khz, side="right") - 1      # -1 below 60 Hz, 4 from 16 kHz
    inband = ((row >= 0) & (row < 4)).astype(np.float32)
    kept = np.asarray(b.kept, bool)
    slot = np.floor((t_s - b.t_begin) / CELL_S + 1e-9).astype(int)
    ok = (slot >= 0) & (slot < len(kept))
    g = np.zeros((len(f_khz), len(t_s)), np.float32)
    g[:, ok] = kept[slot[ok]][:, np.clip(row, 0, 3)].T * inband[:, None]
    return g


def filter_clip(x: np.ndarray, sr: int, a: float, b, mode: str) -> np.ndarray:
    """Clip x (starting at a s in the recording) filtered to box b's cells by an STFT mask."""
    if mode == "original":
        return x
    from scipy.ndimage import uniform_filter
    from scipy.signal import istft, stft
    kw = dict(fs=sr, nperseg=STFT_N, noverlap=STFT_N - STFT_N // 4)
    f, t, Z = stft(x, **kw)
    g = uniform_filter(cell_gain(f / 1000, a + t, b, mode), size=SMOOTH, mode="nearest")
    _, y = istft(Z * g, **kw)
    return np.pad(y, (0, max(0, len(x) - len(y))))[:len(x)].astype(np.float32)


def wav_bytes(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x / max(1e-9, np.abs(x).max()) * 0.9, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---- drawing ---------------------------------------------------------------------------
def plot_page(rec: str, t0: float, seg: pd.DataFrame, raven: pd.DataFrame, wavs: dict,
              thresholds: dict[str, float] | None = None, groups=None, show_raven: bool = True,
              show_perch: bool = True, max_khz: float = 24.0, range_db: float = 80.0, page_s: float = PAGE_S,
              cells: CellMaps | None = None, box_mode: str = "full band", kappa: float = KAPPA):
    """Spectrogram of [t0, t0 + page_s) with Perch and Raven boxes, plus a per-group score track.
    With `cells` and box_mode 'tightened' / 'both', each Perch box is also drawn tightened to its
    cells (solid; kept cells shaded), the full-band box dashed ('both') or not at all."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    thresholds = dict(THRESHOLDS if thresholds is None else thresholds)
    groups = list(DEFAULT_ON if groups is None else groups)
    x, sr = read_page(wavs[rec], t0, page_s)
    f, t, db = spectrogram(x, sr)
    t1 = t0 + page_s
    sr_rec = seg[seg["recording"] == rec]
    pb = perch_boxes(sr_rec, thresholds)
    pb = pb[pb["group"].isin(groups) & (pb["end_s"] > t0) & (pb["begin_s"] < t1)]
    # numbered in time order; the viewer's play list uses the same numbers
    pb = pb.assign(k=pb["group"].map(groups.index)).sort_values(["begin_s", "k"]).drop(columns="k")
    pb = pb.reset_index(drop=True)
    tight = cells is not None and box_mode != "full band"
    if tight:
        pb = cells.tighten_boxes(rec, pb, kappa)
    full = not tight or box_mode == "both"
    rv = raven[(raven["recording"] == rec) & (raven["end_s"] > t0) & (raven["begin_s"] < t1)]

    fig, (ax, tr) = plt.subplots(2, 1, figsize=(18, 8.5), sharex=True, constrained_layout=True,
                                 gridspec_kw={"height_ratios": [3.2, 1]})
    vmax = np.percentile(db, 99.9)
    ax.imshow(db, origin="lower", aspect="auto", cmap="gray_r", vmin=vmax - range_db, vmax=vmax,
              extent=(t0, t0 + len(x) / sr, f[0], f[-1]), interpolation="nearest")
    ax.set_ylim(0, max_khz)
    ax.set_xlim(t0, t1)
    ax.set_ylabel("kHz")
    if max_khz > PERCH_BAND[1]:
        ax.axhline(PERCH_BAND[1], color="k", lw=0.8, ls=":")
        ax.text(t0 + 0.3, PERCH_BAND[1] + 0.15, "Perch hears below 16 kHz", fontsize=8, va="bottom")

    if show_perch:
        for b in pb.itertuples():
            k = groups.index(b.group)
            inset = 0.25 * k                      # kHz, keeps overlapping groups visible
            lo, hi = PERCH_BAND[0] + inset, min(PERCH_BAND[1], max_khz) - inset
            has_tight = tight and np.isfinite(b.t_begin)
            if full or not has_tight:
                ax.add_patch(Rectangle((b.begin_s, lo), b.end_s - b.begin_s, hi - lo, fill=False,
                                       ec=COLORS[b.group], lw=1.2 if has_tight else 2.2, ls="--"))
            if has_tight:
                tlo, thi = b.lo_khz + 0.05 * k, b.hi_khz - 0.05 * k
                for r in range(b.kept.shape[1]):  # kept cells, one rectangle per run in each row
                    row = b.kept[:, r]
                    if not row.any():
                        continue
                    edge = np.diff(np.r_[0, row.astype(np.int8), 0])
                    for a, e in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
                        ax.add_patch(Rectangle((b.t_begin + a * CELL_S, ROW_EDGES_KHZ[r]), (e - a) * CELL_S,
                                               ROW_EDGES_KHZ[r + 1] - ROW_EDGES_KHZ[r], fill=True,
                                               fc=COLORS[b.group], alpha=0.12, ec="none"))
                ax.add_patch(Rectangle((b.t_begin, tlo), b.t_end - b.t_begin, thi - tlo, fill=False,
                                       ec=COLORS[b.group], lw=2.4, ls="-"))
                lx, ly = max(b.t_begin, t0) + 0.15, min(thi, max_khz) - 0.1 - k * 0.028 * max_khz
            else:
                lx, ly = max(b.begin_s, t0) + 0.15, hi - 0.1 - k * 0.028 * max_khz
            ax.text(lx, ly, f"#{b.Index + 1} {b.group} {b.max_prob:.2f}", color="k",
                    fontsize=7.5, va="top", bbox=dict(fc=COLORS[b.group], ec="none", alpha=0.85, pad=1.2))
    if show_raven:
        for b in rv.itertuples():
            ax.add_patch(Rectangle((b.begin_s, b.low_khz), b.end_s - b.begin_s, b.high_khz - b.low_khz,
                                   fill=True, fc=COLORS[b.group], alpha=0.18, ec=COLORS[b.group], lw=1.6))
            ax.add_patch(Rectangle((b.begin_s, b.low_khz), b.end_s - b.begin_s, b.high_khz - b.low_khz,
                                   fill=False, ec="k", lw=0.6))
            ax.text(max(b.begin_s, t0) + 0.1, min(b.high_khz, max_khz) - 0.05, b.label, fontsize=7,
                    va="top", color="k", bbox=dict(fc="white", ec=COLORS[b.group], alpha=0.85, pad=1))

    handles = [Patch(fc="none", ec=COLORS[g], ls="-" if tight else "--", lw=2,
                     label=f"{lane_name(g)} >= {thresholds[g]:g}") for g in groups if show_perch]
    if show_perch and tight:
        handles += [Patch(fc="0.5", alpha=0.25, ec="0.3", label=f"tightened: cells >= {kappa:g} of box range"
                          + ("; dashed = full band" if full else ""))]
    if show_raven and len(rv):
        handles += [Patch(fc=COLORS[g], alpha=0.4, ec="k", label=f"Raven {g}")
                    for g in dict.fromkeys(rv["group"])]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7.5, ncol=2, framealpha=0.9)
    ax.set_title(f"{rec}   {t0:.0f}-{t1:.0f} s (minute {int(t0 // 60) + 1})   "
                 f"Perch boxes: {len(pb)}   Raven boxes: {len(rv)}", fontsize=11)

    # score track: one lane per group; lane middle = threshold, top = 2 x threshold
    w = sr_rec[(sr_rec["start_s"] + SEG_S > t0) & (sr_rec["start_s"] < t1)]
    for k, g in enumerate(groups):
        y0 = len(groups) - 1 - k
        v = np.clip(w[f"prob_{g}"].to_numpy() / thresholds[g], 0, 2) / 2 * 0.9
        on = w[f"prob_{g}"].to_numpy() >= thresholds[g]
        for s0, o in zip(w["start_s"], on):
            if o:
                tr.add_patch(Rectangle((s0, y0), SEG_S, 0.9, fc=COLORS[g], alpha=0.35, ec="none"))
        tr.step(np.r_[w["start_s"], w["start_s"].iloc[-1] + SEG_S] if len(w) else [],
                np.r_[y0 + v, y0 + v[-1]] if len(w) else [], where="post", color="k", lw=1)
        tr.axhline(y0 + 0.45, color=COLORS[g], lw=0.8, ls=":")
        tr.axhline(y0, color="0.6", lw=0.6)
        if show_raven:
            raven_g = "Insecta" if g in EC_GROUPS else g     # EarthChirp lanes are insect lanes
            for b in rv[rv["group"] == raven_g].itertuples():
                tr.plot([b.begin_s, b.end_s], [y0 + 0.15, y0 + 0.15], color="k", lw=3, solid_capstyle="butt")
    tr.set_yticks([len(groups) - 1 - k + 0.45 for k in range(len(groups))])
    tr.set_yticklabels([f"{g} ({RELIABILITY[g].split(' ')[0]})" for g in groups], fontsize=8)
    tr.set_ylim(0, len(groups))
    if sr_rec.empty:
        tr.text(t0 + page_s / 2, len(groups) / 2, "no Perch scores for this file: it has no Raven table, "
                "so zeroshot_windows.py did not score it", ha="center", va="center", fontsize=11,
                bbox=dict(fc="lightyellow", ec="0.5"))
    tr.set_xlabel("time in recording (s)")
    tr.set_title("score per 2.5 s segment (dotted = threshold, shaded = box)"
                 + ("; black bar = Raven box" if show_raven else ""), fontsize=9, loc="left")
    return fig, pb, rv


# ---- GUI -------------------------------------------------------------------------------
class PerchViewer:
    """ipywidgets viewer: one minute per page, Perch boxes per group, Raven overlay by radio button."""

    def __init__(self, outdir: Path = ROOT / "data" / "perch_windows"):
        import ipywidgets as W
        self.W = W
        self.seg = load_segments(outdir)
        self.cells = CellMaps(outdir) if (outdir / CELL_FILE.name).exists() else None
        self.groups = [g for g in VIEW_GROUPS if f"prob_{g}" in self.seg]
        self.raven = load_raven()
        self.wavs = wav_paths()
        self.recs = sorted(self.wavs)          # every wav, scored or not
        self.npages = {r: n_pages(self.wavs[r]) for r in self.recs}
        lay = W.Layout
        self.rec = W.Dropdown(options=audio_options(self.recs, self.wavs, self.raven, set(self.seg["recording"])),
                              description="Audio file", layout=lay(width="560px"),
                              style={"description_width": "70px"})
        self.minute = W.IntSlider(min=1, max=20, value=1, description="Minute", continuous_update=False,
                                  layout=lay(width="380px"))
        self.raven_radio = W.RadioButtons(options=["Show", "Hide"], value="Show", description="Raven labels")
        self.perch_radio = W.RadioButtons(options=["Show", "Hide"], value="Show", description="Perch boxes")
        # Mammalia starts unticked (untestable); with EarthChirp scores, so does Perch's Insecta
        self.group_on = {g: W.Checkbox(value=g in DEFAULT_ON, description=g, indent=False,
                                       layout=lay(width="125px")) for g in self.groups}
        self.thr = {g: W.BoundedFloatText(value=THRESHOLDS[g], min=-1 if g in EC_GROUPS else 0, max=1,
                                          step=0.001, layout=lay(width="125px")) for g in self.groups}
        self.max_khz = W.FloatSlider(min=2, max=24, step=1, value=24, description="Max kHz",
                                     continuous_update=False)
        self.range_db = W.FloatSlider(min=30, max=120, step=5, value=80, description="Range dB",
                                      continuous_update=False)
        self.find_group = W.Dropdown(options=self.groups, value="Aves", layout=lay(width="150px"))
        # tightened boxes from the per-cell maps (cell_maps.py); full band only without them
        self.box_mode = W.RadioButtons(options=BOX_MODES if self.cells else ["full band"],
                                       value="both" if self.cells else "full band", description="Box extent")
        self.kappa = W.FloatSlider(min=0.0, max=0.95, step=0.05, value=KAPPA, description="keep cells ≥",
                                   continuous_update=False, readout_format=".2f",
                                   disabled=self.cells is None, style={"description_width": "80px"})

        def button(text, width="110px"):
            return W.Button(description=text, layout=lay(width=width))
        self.prev_rec, self.next_rec = button("◀ prev rec"), button("next rec ▶")
        self.prev_min, self.next_min = button("◀ minute"), button("minute ▶")
        self.find_prev, self.find_next = button("◀ prev hit"), button("next hit ▶")
        self.reset = button("reset thresholds", "140px")
        self.play = button("▶ load audio of this minute", "220px")
        self.status = W.HTML()
        self.out, self.audio = W.Output(), W.Output()
        # listen to Perch boxes: click a row to play it (box audio, padded, up to MAX_CLIP_S)
        self.box_list = W.Select(options=[], rows=7, layout=lay(width="520px"))
        self.pad = W.BoundedFloatText(value=0.5, min=0, max=5, step=0.5, description="pad (s)",
                                      layout=lay(width="150px"), style={"description_width": "50px"})
        self.play_box = button("▶ play again", "120px")
        # play the box as recorded, or filtered to its tightened band / kept cells (needs cell maps)
        self.play_filter = W.RadioButtons(options=PLAY_MODES if self.cells else ["original"],
                                          value="original", description="Play")
        self.box_audio = W.Output()
        self.box_list.observe(lambda _: self._play_box(), "value")
        self.play_filter.observe(lambda _: self._play_box(), "value")
        self.play_box.on_click(lambda _: self._play_box())
        self.page_boxes = pd.DataFrame()

        self.prev_rec.on_click(lambda _: self._step_rec(-1))
        self.next_rec.on_click(lambda _: self._step_rec(+1))
        self.prev_min.on_click(lambda _: self._step_min(-1))
        self.next_min.on_click(lambda _: self._step_min(+1))
        self.find_prev.on_click(lambda _: self._find(-1))
        self.find_next.on_click(lambda _: self._find(+1))
        self.reset.on_click(self._reset)
        self.play.on_click(self._play)
        self._busy = False
        self.rec.observe(self._rec_changed, "value")
        for wdg in [self.minute, self.raven_radio, self.perch_radio, self.max_khz, self.range_db,
                    self.box_mode, self.kappa, *self.group_on.values(), *self.thr.values()]:
            wdg.observe(lambda _: self.draw(), "value")
        self._rec_changed()

    # navigation
    def _rec_changed(self, *_, minute: int = 1):
        self._busy = True                      # one redraw for the recording + minute change
        self.minute.max = self.npages[self.rec.value]
        self.minute.value = min(minute, self.minute.max)
        self._busy = False
        self.draw()

    def _step_rec(self, d):
        i = self.recs.index(self.rec.value) + d
        if 0 <= i < len(self.recs):
            self.rec.value = self.recs[i]

    def _step_min(self, d):
        self.minute.value = int(np.clip(self.minute.value + d, 1, self.minute.max))

    def thresholds(self) -> dict[str, float]:
        return {g: float(self.thr[g].value) for g in self.groups}

    def _find(self, d):
        """Jump to the previous / next minute (across recordings) with a Perch box of the chosen group."""
        g = self.find_group.value
        hit = self.seg[self.seg[f"prob_{g}"] >= self.thresholds()[g]]
        pages = sorted({(r, min(int(s // PAGE_S) + 1, self.npages[r]))
                        for r, s in zip(hit["recording"], hit["start_s"]) if r in self.npages})
        here = (self.rec.value, self.minute.value)
        cand = [p for p in pages if (p > here if d > 0 else p < here)]
        if not cand:
            self.status.value = f"<b>no {'later' if d > 0 else 'earlier'} minute with a Perch {g} box</b>"
            return
        r, m = cand[0] if d > 0 else cand[-1]
        self._busy = True                      # tick the group, move, then draw once
        self.group_on[g].value = True
        if r != self.rec.value:
            self.rec.unobserve(self._rec_changed, "value")
            self.rec.value = r
            self.rec.observe(self._rec_changed, "value")
            self.minute.max = self.npages[r]
        self.minute.value = m
        self._busy = False
        self.draw()

    def _reset(self, _):
        self._busy = True
        for g, wdg in self.thr.items():
            wdg.value = THRESHOLDS[g]
        self._busy = False
        self.draw()

    def _play(self, _):
        from IPython.display import Audio, display
        x, sr = read_page(self.wavs[self.rec.value], (self.minute.value - 1) * PAGE_S)
        with self.audio:
            self.audio.clear_output()
            display(Audio(data=wav_bytes(x, sr), rate=sr))

    def box_clip(self, i: int) -> tuple[np.ndarray, int, float, float]:
        """Audio of Perch box i on this page, padded and capped at MAX_CLIP_S."""
        b0, b1 = self.box_span(i)
        a = max(0.0, b0 - self.pad.value)
        e = min(b1 + self.pad.value, a + MAX_CLIP_S)
        x, sr = read_page(self.wavs[self.rec.value], a, e - a)
        return x, sr, a, a + len(x) / sr

    def box_span(self, i: int) -> tuple[float, float]:
        """The tightened span of box i when one is drawn, else the full box."""
        b = self.page_boxes.iloc[i]
        if self.box_mode.value != "full band" and "t_begin" in b and np.isfinite(b.t_begin):
            return float(b.t_begin), float(b.t_end)
        return float(b.begin_s), float(b.end_s)

    def _play_box(self):
        from IPython.display import Audio, display
        i = self.box_list.value
        if i is None or self._busy:
            return
        x, sr, a, e = self.box_clip(i)
        b = self.page_boxes.iloc[i]
        b0, b1 = self.box_span(i)
        mode, note = self.play_filter.value, ""
        if mode != "original":
            # box extent 'full band' draws no tightened boxes: tighten this one for playback only
            tb = b if "kept" in b else self.cells.tighten_boxes(self.rec.value, self.page_boxes.iloc[[i]],
                                                                 self.kappa.value).iloc[0]
            if np.isfinite(tb.t_begin):
                x = filter_clip(x, sr, a, tb, mode)
                note = (f", {mode} {tb.lo_khz:.2g}-{tb.hi_khz:.2g} kHz" if mode == "band-passed"
                        else f", cell mask ({int(np.asarray(tb.kept).sum())} cells kept)")
            else:
                note = ", unfiltered (no cell map for this box)"
        with self.box_audio:
            self.box_audio.clear_output()
            print(f"#{i + 1} {b.group}: playing {a:.1f}-{e:.1f} s"
                  + (" (tightened span)" if (b0, b1) != (b.begin_s, b.end_s) else "") + note
                  + (f" (box is {b1 - b0:.0f} s; first {MAX_CLIP_S:.0f} s only)"
                     if b1 - b0 + 2 * self.pad.value > MAX_CLIP_S else ""))
            display(Audio(data=wav_bytes(x, sr), rate=sr, autoplay=True))

    def _fill_box_list(self, pb: pd.DataFrame):
        self.page_boxes = pb
        self._busy = True                      # replacing options must not trigger a play
        def row(i, b):
            txt = (f"#{i + 1:<3} {b.group:<14} {b.begin_s:7.1f}-{b.end_s:7.1f} s  "
                   f"({b.end_s - b.begin_s:4.1f} s, max {b.max_prob:.2f})")
            if "t_begin" in pb and np.isfinite(b.t_begin):
                txt += (f"  tight {b.t_begin:.1f}-{b.t_end:.1f} s, {b.lo_khz:.2g}-{b.hi_khz:.2g} kHz "
                        f"({b.area:.0%} of box)")
            return txt, i
        self.box_list.options = [row(i, b) for i, b in enumerate(pb.itertuples())]
        self.box_list.value = None
        self._busy = False
        with self.box_audio:
            self.box_audio.clear_output()
            if not len(pb):
                print("no Perch boxes on this minute for the ticked groups")

    def draw(self):
        if self._busy:
            return
        import matplotlib.pyplot as plt
        from IPython.display import display
        t0 = (self.minute.value - 1) * PAGE_S
        groups = [g for g in self.groups if self.group_on[g].value]
        with self.audio:
            self.audio.clear_output()
        with self.out:
            self.out.clear_output(wait=True)
            if not groups:
                print("tick at least one group")
            else:
                fig, pb, rv = plot_page(self.rec.value, t0, self.seg, self.raven, self.wavs,
                                        thresholds=self.thresholds(), groups=groups,
                                        show_raven=self.raven_radio.value == "Show",
                                        show_perch=self.perch_radio.value == "Show",
                                        max_khz=self.max_khz.value, range_db=self.range_db.value,
                                        cells=self.cells, box_mode=self.box_mode.value,
                                        kappa=self.kappa.value)
                display(fig)
                plt.close(fig)
        if not groups:
            self._fill_box_list(pd.DataFrame(columns=["group", "begin_s", "end_s", "max_prob", "n_segments"]))
            return
        self._fill_box_list(pb)
        rows, tight = [], "area" in pb
        for g in groups:
            p, r = pb[pb["group"] == g], rv[rv["group"] == ("Insecta" if g in EC_GROUPS else g)]
            area = ""
            if tight:
                area = f"<td>{p['area'].mean():.0%}</td>" if p["area"].notna().any() else "<td>-</td>"
            rows.append(f"<tr><td style='color:{COLORS[g]}'>■</td><td>{g}</td><td>{len(p)}</td>"
                        f"<td>{p['n_segments'].sum() * SEG_S:.1f} s</td>{area}<td>{len(r)}</td>"
                        f"<td>{RELIABILITY[g]}</td></tr>")
        self.status.value = ("<table style='font-size:12px'><tr><th></th><th>group</th><th>Perch boxes</th>"
                             "<th>Perch time</th>" + ("<th>tightened area (mean)</th>" if tight else "")
                             + "<th>Raven boxes</th><th>threshold reliability</th></tr>"
                             + "".join(rows) + "</table>")

    def show(self):
        from IPython.display import display
        W = self.W
        thr_row = W.HBox([W.VBox([self.group_on[g], self.thr[g]]) for g in self.groups])
        display(W.VBox([
            W.HBox([self.rec, self.prev_rec, self.next_rec]),
            W.HBox([self.minute, self.prev_min, self.next_min,
                    W.HTML("&nbsp;&nbsp;find Perch box of&nbsp;"), self.find_group,
                    self.find_prev, self.find_next]),
            W.HBox([self.raven_radio, self.perch_radio, W.VBox([self.max_khz, self.range_db]),
                    self.box_mode, self.kappa]),
            W.HTML("<b>Box extent</b>: 'tightened' keeps, inside each Perch box, the 0.31 s x one-of-4-row "
                   "cells whose lane score is at least 'keep cells ≥' of the way from the box's median to "
                   "its highest cell (rows with only a few such cells dropped), and draws their bounding "
                   "box (solid; kept cells shaded). Rows: 0.06-0.95, 0.95-2.9, 2.9-7.0, 7.0-16 kHz. The "
                   "player plays the tightened span."
                   if self.cells else "<i>no cell_maps.npy yet (scripts/cell_maps.py): boxes are full band</i>"),
            W.HTML("<b>Groups to draw / thresholds</b> (summed Perch probability per 2.5 s segment"
                   + ("; EC-Insecta = EarthChirp insect cosine, EC-margin = that cosine minus the "
                      "urban/geophony cosine)" if HAS_EC else ")")),
            W.HBox([thr_row, self.reset]),
            self.status, self.out,
            W.HTML("<b>Listen to a Perch box</b>: click a row to play it (numbers match the "
                   f"#labels on the spectrogram; the whole box plus padding, even past this minute, "
                   f"up to {MAX_CLIP_S:.0f} s)"
                   + ("; <b>Play</b>: 'band-passed' keeps only the tightened box's frequency rows, "
                      "'cell mask' only its kept cells (0.31 s x one row; padding is muted). Both cut "
                      "time and frequency, they do not separate sources: other sounds in the same rows "
                      "stay. Clips are peak-normalised, so a filtered clip plays as loud as the original"
                      if self.cells else "")),
            W.HBox([self.box_list, W.VBox([self.pad, self.play_box]), self.play_filter]),
            self.box_audio,
            self.play, self.audio]))
        return self
