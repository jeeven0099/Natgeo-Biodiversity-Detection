"""Interactive check of segmentation boxes: spectrogram + box overlay + audio (Jupyter).

Reads a run folder written by `biodetect segment` (raven_tables/, files.csv, per_piece.csv).
Nothing is written.

    from biodetect.viewer import BoxViewer
    BoxViewer("results").show()            # every run folder under results/

Views
  Spectrogram  |STFT| (the pipeline's Hann 1024 / overlap 768) after the 3x3 Gaussian, dB
  Denoised S'  after the spectral subtraction of the 60 s piece the window lies in
  Otsu mask    the piece's mask after opening + closing: exactly what became the boxes
The min-cells / min-dB fields only hide boxes on screen; the tables are not changed.
"""
from __future__ import annotations

from functools import cache, lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.signal import butter, sosfiltfilt

from .. import segmentation as seg

BOX_COLOUR = "#e8590c"
MAX_PIECE_S = 300.0          # longer pieces (--chunk-s 0) are too big to hold for the S'/mask views
RAVEN_COLS = {"Begin Time (s)": "begin_s", "End Time (s)": "end_s", "Low Freq (Hz)": "low_hz",
              "High Freq (Hz)": "high_hz", "Cells": "n_cells", "Mean dB above noise": "mean_db"}


# ---------------------------------------------------------------- data access
def find_runs(root: str | Path) -> dict[str, Path]:
    """Run folders under root (or root itself), keyed by their path relative to root."""
    root = Path(root)
    runs = sorted(p.parent for p in root.rglob("raven_tables") if (p.parent / "files.csv").exists())
    return {(str(r.relative_to(root)) if r != root else r.name): r for r in runs}


@cache
def recording_paths(run: Path) -> dict[str, Path]:
    """Recording name -> audio path, for every recording with a table in this run."""
    run = Path(run)
    m = pd.read_csv(run / "files.csv").set_index("file")["path"].to_dict()
    names = sorted(p.name[: -len(seg.RAVEN_SUFFIX)] for p in (run / "raven_tables").glob(f"*{seg.RAVEN_SUFFIX}"))
    return {n: Path(m[n]) for n in names if n in m}


@lru_cache(maxsize=8)
def load_table(run: Path, name: str) -> pd.DataFrame:
    t = pd.read_csv(Path(run) / "raven_tables" / f"{name}{seg.RAVEN_SUFFIX}", sep="\t")
    return t.rename(columns=RAVEN_COLS).set_index("Selection")


@cache
def load_pieces(run: Path) -> pd.DataFrame:
    p = Path(run) / "per_piece.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["file", "start_s", "duration_s"])


def piece_at(run: Path, name: str, t: float) -> pd.Series | None:
    p = load_pieces(run)
    p = p[(p["file"] == name) & (p["start_s"] <= t + 1e-6)]
    return None if p.empty else p.sort_values("start_s").iloc[-1]


def window_spectrogram(path: Path, t0: float, t1: float):
    """Pipeline spectrogram (steps 1-2) of [t0, t1]; frame times are absolute seconds."""
    pad = seg.WIN / 2 / sf.info(str(path)).samplerate
    a = max(0.0, t0 - pad)
    x, sr = seg.read_mono(path, a, t1 + pad)
    if len(x) < seg.WIN:
        return None
    f, t, s_db = seg.spectrogram_db(x, sr)
    return f, a + t, s_db.astype(np.float32)


@lru_cache(maxsize=3)
def piece_layers(path: Path, start: float, duration: float):
    """S' and the final mask of one piece, recomputed exactly as the pipeline did."""
    x, sr = seg.read_mono(path, start, start + duration)
    f, t, s1 = seg.noise_reduced(x, sr)
    raw, thr = seg.graythresh_binarize(s1)
    return f, start + t, s1.astype(np.float32), seg.clean_mask(raw), thr


def bandpass(x: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    lo, hi = max(low_hz, 20.0), min(high_hz, 0.98 * sr / 2)
    if hi - lo > 20 and len(x) > 64:
        x = sosfiltfilt(butter(6, [lo, hi], btype="bandpass", fs=sr, output="sos"), x)
    return x


def box_audio(path: Path, row, pad: float = 0.5, band: bool = True) -> tuple[np.ndarray, int]:
    """The box's time span +- pad, optionally band-passed to the box."""
    x, sr = seg.read_mono(path, max(0.0, row.begin_s - pad), row.end_s + pad)
    return (bandpass(x, sr, row.low_hz, row.high_hz) if band else x), sr


# ---------------------------------------------------------------- drawing
def boxes_in(tab: pd.DataFrame, t0: float, t1: float, min_cells: int = 0, min_db: float = 0.0) -> pd.DataFrame:
    w = tab[(tab["end_s"] > t0) & (tab["begin_s"] < t1)]
    return w[(w["n_cells"] >= min_cells) & (w["mean_db"] >= min_db)]


def draw(ax, run: Path, name: str, t0: float, t1: float, view: str = "Spectrogram", fmax_khz: float = 16.0,
         range_db: float = 60.0, cmap: str = "Greys", boxes: pd.DataFrame | None = None,
         numbers: bool = False) -> str:
    """Draw one window on ax; returns a one-line note about the layer shown."""
    path = recording_paths(run)[name]
    note = ""
    if view == "Spectrogram":
        r = window_spectrogram(path, t0, t1)
        if r is None:
            return "window too short"
        f, t, img = r
        keep_t = (t >= t0 - 0.01) & (t <= t1 + 0.01)
        img, t = img[:, keep_t], t[keep_t]
        vmax = np.percentile(img[f <= fmax_khz * 1000], 99.8)
        vmin, interp = vmax - range_db, "antialiased"
    else:
        pc = piece_at(run, name, t0)
        if pc is None or pc["duration_s"] > MAX_PIECE_S:
            ax.text(0.5, 0.5, "S' / mask need a piece of at most 5 min (run used --chunk-s 0?)",
                    ha="center", va="center", transform=ax.transAxes)
            return ""
        f, t, s1, mask, thr = piece_layers(path, float(pc["start_s"]), float(pc["duration_s"]))
        keep_t = (t >= t0 - 0.01) & (t <= t1 + 0.01)
        t = t[keep_t]
        if view == "Otsu mask":
            img, vmin, vmax, interp = mask[:, keep_t].astype(np.float32), 0, 1, "nearest"
        else:
            img = s1[:, keep_t]
            vmin, vmax, interp = 0, max(np.percentile(img[f <= fmax_khz * 1000], 99.8), 1.0), "antialiased"
        note = (f"piece {pc['start_s']:.0f}-{pc['start_s'] + pc['duration_s']:.0f} s: Otsu {thr:.1f} dB above "
                f"noise, {mask.mean():.1%} of cells on after opening/closing")
        if t1 > pc["start_s"] + pc["duration_s"] + 0.01:
            note += " (window runs into the next piece; shown up to its edge)"
    if len(t) == 0:
        return "nothing to show"
    keep_f = f <= fmax_khz * 1000
    dt, df = (t[1] - t[0]) if len(t) > 1 else 0.0053, f[1] - f[0]
    ax.imshow(img[keep_f], origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interp,
              extent=[t[0] - dt / 2, t[-1] + dt / 2, (f[0] - df / 2) / 1000, (f[keep_f][-1] + df / 2) / 1000])
    if boxes is not None and len(boxes):
        ax.add_collection(PatchCollection(
            [Rectangle((b, lo / 1000), e - b, (hi - lo) / 1000)
             for b, e, lo, hi in boxes[["begin_s", "end_s", "low_hz", "high_hz"]].itertuples(index=False)],
            facecolor="none", edgecolor=BOX_COLOUR, linewidth=1.4 if len(boxes) < 200 else 0.8))
        if numbers:                                # at most the 150 largest, or the labels hide everything
            for sel, r in boxes.nlargest(150, "n_cells").iterrows():
                if r.low_hz / 1000 < fmax_khz:
                    ax.text(max(r.begin_s, t0), min(r.high_hz / 1000, fmax_khz), str(sel), color=BOX_COLOUR,
                            fontsize=7, va="bottom", clip_on=True)
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, fmax_khz)
    ax.set_ylabel("kHz")
    ax.set_xlabel("time (s)")
    return note


def plot_window(run: str | Path, name: str, t0: float, window_s: float = 10.0, view: str = "Spectrogram",
                fmax_khz: float = 16.0, min_cells: int = 0, min_db: float = 0.0, numbers: bool = False):
    """Static version of one viewer frame, for a figure or a quick look."""
    run = Path(run)
    tab = load_table(run, name)
    t1 = t0 + window_s
    b = boxes_in(tab, t0, t1, min_cells, min_db)
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    note = draw(ax, run, name, t0, t1, view, fmax_khz, boxes=b, numbers=numbers)
    ax.set_title(f"{name}  {t0:.0f}-{t1:.0f} s   {len(b)} boxes   {note}", fontsize=10)
    return fig


# ---------------------------------------------------------------- GUI
def _btn(label: str, width: str = "90px", **kw):
    import ipywidgets as W

    return W.Button(description=label, layout=W.Layout(width=width), **kw)


class BoxViewer:
    """Browse a recording window by window with its boxes; play the window or one box."""

    def __init__(self, root: str | Path = "results", run: str | None = None, recording: str | None = None,
                 window_s: float = 10.0):
        import ipywidgets as W

        self.runs = find_runs(root)
        if not self.runs:
            raise FileNotFoundError(f"no segmentation runs under {root}; run `biodetect segment` first")
        names = list(self.runs)
        self._quiet = False
        S = {"description_width": "initial"}
        self.w_run = W.Dropdown(options=names, value=run if run in self.runs else names[0], description="Run",
                                style=S, layout=W.Layout(width="260px"))
        self.w_rec = W.Dropdown(description="Recording", style=S, layout=W.Layout(width="360px"))
        self.b_prev_rec, self.b_next_rec = _btn("◀ prev rec"), _btn("next rec ▶")
        self.w_start = W.FloatSlider(description="Start (s)", min=0, max=1, step=0.5, continuous_update=False,
                                     readout_format=".1f", style=S, layout=W.Layout(width="620px"))
        self.w_win = W.Dropdown(options=[2, 5, 10, 20, 30, 60], value=window_s, description="Window (s)",
                                style=S, layout=W.Layout(width="150px"))
        self.b_back, self.b_fwd = _btn("◀ back"), _btn("forward ▶")
        self.w_view = W.ToggleButtons(options=["Spectrogram", "Denoised S′", "Otsu mask"], value="Spectrogram",
                                      style={"button_width": "110px"})
        self.w_fmax = W.FloatSlider(description="Max kHz", min=1, max=24, step=0.5, value=16,
                                    continuous_update=False, style=S, layout=W.Layout(width="260px"))
        self.w_range = W.IntSlider(description="Range dB", min=20, max=100, value=60, continuous_update=False,
                                   style=S, layout=W.Layout(width="260px"))
        self.w_cmap = W.Dropdown(options=["Greys", "magma", "viridis"], value="Greys", description="Colours",
                                 style=S, layout=W.Layout(width="170px"))
        self.w_boxes = W.Checkbox(value=True, description="Show boxes", indent=False, layout=W.Layout(width="110px"))
        self.w_min_cells = W.BoundedIntText(value=0, min=0, max=10 ** 8, description="Hide < cells", style=S,
                                            layout=W.Layout(width="170px"))
        self.w_min_db = W.BoundedFloatText(value=0, min=0, max=300, step=0.5, description="Hide < dB", style=S,
                                           layout=W.Layout(width="150px"))
        self.w_numbers = W.Checkbox(value=False, description="Selection numbers", indent=False,
                                    layout=W.Layout(width="150px"))
        self.info = W.HTML()
        self.out_plot, self.out_audio = W.Output(), W.Output()
        self.w_sel = W.BoundedIntText(value=1, min=1, max=1, description="Selection", style=S,
                                      layout=W.Layout(width="170px"))
        self.w_pad = W.BoundedFloatText(value=0.5, min=0, max=10, step=0.25, description="Pad (s)", style=S,
                                        layout=W.Layout(width="130px"))
        self.w_bp = W.Checkbox(value=True, description="Band-pass to the box", indent=False,
                               layout=W.Layout(width="170px"))
        self.b_play, self.b_goto = _btn("Play box", button_style="info"), _btn("Go to box")
        self.out_box = W.Output()
        self.w_table = W.Checkbox(value=False, description="Table of boxes in window (largest 30)", indent=False)
        self.out_table = W.Output()

        self.w_run.observe(lambda _: self._set_run(), "value")
        self.w_rec.observe(lambda _: self._set_rec(), "value")
        self.b_prev_rec.on_click(lambda _: self._step_rec(-1))
        self.b_next_rec.on_click(lambda _: self._step_rec(+1))
        self.b_back.on_click(lambda _: self._step_win(-1))
        self.b_fwd.on_click(lambda _: self._step_win(+1))
        self.w_win.observe(lambda _: self._set_win(), "value")
        for w in (self.w_start, self.w_view, self.w_fmax, self.w_range, self.w_cmap, self.w_boxes,
                  self.w_min_cells, self.w_min_db, self.w_numbers, self.w_table):
            w.observe(lambda _: self.redraw(), "value")
        self.b_play.on_click(lambda _: self.play_box())
        self.b_goto.on_click(lambda _: self.goto_box())

        self.ui = W.VBox([
            W.HBox([self.w_run, self.w_rec, self.b_prev_rec, self.b_next_rec]),
            W.HBox([self.w_start, self.w_win, self.b_back, self.b_fwd]),
            W.HBox([self.w_view, self.w_fmax, self.w_range, self.w_cmap]),
            W.HBox([self.w_boxes, self.w_min_cells, self.w_min_db, self.w_numbers]),
            self.info, self.out_plot, self.out_audio,
            W.HTML("<b>One box</b>"),
            W.HBox([self.w_sel, self.w_pad, self.w_bp, self.b_play, self.b_goto]),
            self.out_box, self.w_table, self.out_table,
        ])
        self._set_run(recording)

    @property
    def run(self) -> Path:
        return self.runs[self.w_run.value]

    # -- state
    def _set_run(self, recording: str | None = None):
        names = list(recording_paths(self.run))
        self._quiet = True
        self.w_rec.options = names
        self.w_rec.value = recording if recording in names else (names[0] if names else None)
        self._quiet = False
        self._set_rec()

    def _set_rec(self):
        if self._quiet or self.w_rec.value is None:
            return
        self.path = recording_paths(self.run)[self.w_rec.value]
        info = sf.info(str(self.path))
        self.duration = info.frames / info.samplerate
        self.tab = load_table(self.run, self.w_rec.value)
        self._quiet = True
        self.w_fmax.max = info.samplerate / 2000
        self.w_fmax.value = min(self.w_fmax.value, self.w_fmax.max)
        self.w_start.max = max(self.duration - self.w_win.value, 0.5)
        self.w_start.value = 0.0
        self.w_sel.max = max(len(self.tab), 1)
        self._quiet = False
        self.out_box.clear_output()
        self.redraw()

    def _set_win(self):
        self._quiet = True
        self.w_start.max = max(self.duration - self.w_win.value, 0.5)
        self._quiet = False
        self.redraw()

    def _step_rec(self, k: int):
        names = list(self.w_rec.options)
        if names:
            self.w_rec.value = names[(names.index(self.w_rec.value) + k) % len(names)]

    def _step_win(self, k: int):
        self.w_start.value = float(np.clip(self.w_start.value + k * self.w_win.value, 0, self.w_start.max))

    # -- drawing
    def redraw(self):
        from IPython.display import Audio, display

        if self._quiet or self.w_rec.value is None:
            return
        name = self.w_rec.value
        t0 = float(self.w_start.value)
        t1 = min(t0 + float(self.w_win.value), self.duration)
        all_w = boxes_in(self.tab, t0, t1)
        shown = boxes_in(self.tab, t0, t1, self.w_min_cells.value, self.w_min_db.value)
        self.out_plot.clear_output(wait=True)
        with self.out_plot:
            fig, ax = plt.subplots(figsize=(15, 5.2), constrained_layout=True)
            note = draw(ax, self.run, name, t0, t1, self.w_view.value, self.w_fmax.value, self.w_range.value,
                        self.w_cmap.value, shown if self.w_boxes.value else None, self.w_numbers.value)
            ax.set_title(f"{name}   {t0:.1f}-{t1:.1f} s", fontsize=10)
            plt.show()
        hidden = len(all_w) - len(shown)
        self.info.value = (f"<b>{name}</b> &nbsp; {t0:.1f}-{t1:.1f} s of {self.duration:.0f} s &nbsp;|&nbsp; "
                           f"boxes in window: <b>{len(shown)}</b> shown"
                           + (f", {hidden} hidden by the filters" if hidden else "")
                           + f" &nbsp;|&nbsp; {len(self.tab):,} boxes in this recording"
                           + (f"<br><span style='color:#666'>{note}</span>" if note else ""))
        self.out_audio.clear_output(wait=True)
        with self.out_audio:
            x, sr = seg.read_mono(self.path, t0, t1)
            display(Audio(x, rate=sr, normalize=True))
        self.out_table.clear_output()
        if self.w_table.value:
            with self.out_table:
                cols = ["begin_s", "end_s", "low_hz", "high_hz", "n_cells", "mean_db"]
                display(shown.sort_values("n_cells", ascending=False)[cols].head(30).round(3))

    def play_box(self):
        from IPython.display import Audio, display

        sel = int(self.w_sel.value)
        self.out_box.clear_output(wait=True)
        with self.out_box:
            if sel not in self.tab.index:
                print(f"no selection {sel}")
                return
            r = self.tab.loc[sel]
            print(f"Selection {sel}: {r.begin_s:.3f}-{r.end_s:.3f} s, {r.low_hz:.0f}-{r.high_hz:.0f} Hz, "
                  f"{int(r.n_cells)} cells, {r.mean_db:.1f} dB above noise")
            x, sr = box_audio(self.path, r, self.w_pad.value, self.w_bp.value)
            display(Audio(x, rate=sr, normalize=True))

    def goto_box(self):
        sel = int(self.w_sel.value)
        if sel in self.tab.index:
            r = self.tab.loc[sel]
            mid = 0.5 * (r.begin_s + r.end_s)
            self.w_start.value = float(np.clip(mid - self.w_win.value / 2, 0, self.w_start.max))
        self.play_box()

    def show(self):
        from IPython.display import display

        display(self.ui)
        return self
