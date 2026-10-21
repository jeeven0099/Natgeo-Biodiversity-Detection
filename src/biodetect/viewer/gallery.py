"""Look at LAMDA clusters: member boxes per cluster as spectrogram tiles, with audio (Jupyter).

Reads a clustering folder written by `biodetect cluster` (boxes_clustered.csv.gz and
run_info.json, whose box_dir points at the segmentation run and its files.csv). The
only thing written is cluster_names.csv in that folder, when you name a cluster: the
paper's expert-labelling step.

    from biodetect.viewer import ClusterGallery
    ClusterGallery("results/demo/boxes_min50/lamda/rank").show()
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from matplotlib.patches import Rectangle

from .. import segmentation as seg
from ..clustering import BAND_NAMES, DUR_NAMES
from .box_viewer import BOX_COLOUR, _btn, bandpass, box_audio, window_spectrogram


# ---------------------------------------------------------------- data
@cache
def load_info(clus: Path) -> dict:
    return json.loads((Path(clus) / "run_info.json").read_text())


@cache
def recording_paths(clus: Path) -> dict[str, Path]:
    box_dir = Path(load_info(clus)["box_dir"])
    return {k: Path(v) for k, v in pd.read_csv(box_dir / "files.csv").set_index("file")["path"].items()}


@cache
def load_boxes(clus: Path) -> pd.DataFrame:
    b = pd.read_csv(Path(clus) / "boxes_clustered.csv.gz")
    b["dur_ms"] = (b.end_s - b.begin_s) * 1000
    b["centre_khz"] = (b.low_hz + b.high_hz) / 2000
    b["bandwidth_khz"] = (b.high_hz - b.low_hz) / 1000
    return b


def _top(s: pd.Series) -> tuple[str, float]:
    v = s.value_counts(normalize=True)
    return v.index[0], float(v.iloc[0])


def cluster_summary(clus: str | Path) -> pd.DataFrame:
    """One row per non-empty cluster, largest first. Shares are of the cluster's boxes."""
    g = load_boxes(Path(clus)).groupby("cluster")
    out = pd.DataFrame({
        "boxes": g.size(),
        "recordings": g.file.nunique(),
        "top_recording_share": g.file.agg(lambda s: _top(s)[1]),
        "median_ms": g.dur_ms.median(),
        "median_centre_khz": g.centre_khz.median(),
        "median_bandwidth_khz": g.bandwidth_khz.median(),
        "median_cells": g.n_cells.median(),
        "median_db": g.mean_db.median(),
        "main_band": g.band.agg(lambda s: _top(s)[0]),
        "main_band_share": g.band.agg(lambda s: _top(s)[1]),
        "main_duration": g.duration.agg(lambda s: _top(s)[0]),
        "main_duration_share": g.duration.agg(lambda s: _top(s)[1]),
    })
    return out.sort_values("boxes", ascending=False)


def composition(clus: str | Path, column: str) -> pd.DataFrame:
    """Share of each cluster's boxes per category of `column` (band or duration), largest cluster first."""
    b = load_boxes(Path(clus))
    order = {"band": BAND_NAMES, "duration": DUR_NAMES}.get(column)
    t = pd.crosstab(b.cluster, b[column], normalize="index")
    if order:
        t = t.reindex(columns=[c for c in order if c in t.columns])
    return t.loc[b.cluster.value_counts().index]


def plot_composition(clus: str | Path, max_clusters: int = 60):
    """Cluster x band and cluster x duration shares; rows = the largest clusters, largest at the top."""
    sizes = load_boxes(Path(clus)).cluster.value_counts().head(max_clusters)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.22 * len(sizes) + 1.5), sharey=True, constrained_layout=True)
    for ax, col in zip(axes, ["band", "duration"], strict=True):
        t = composition(clus, col).loc[sizes.index]
        im = ax.imshow(t.values, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(t.shape[1]), t.columns, rotation=30, ha="right")
        ax.set_title(f"share of the cluster's boxes per {col}", fontsize=10)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(sizes)), [f"{c}  ({n:,})" for c, n in sizes.items()], fontsize=7)
    axes[0].set_ylabel("cluster  (boxes)")
    fig.colorbar(im, ax=axes, shrink=0.5, label="share")
    return fig


def sample_cluster(clus: str | Path, cluster: int, n: int = 12, seed: int = 0,
                   recording: str | None = None) -> pd.DataFrame:
    """Tile [0] is the cluster's representative (its highest-GAD box, as the paper's expert sees it),
    then a random sample of the other members."""
    b = load_boxes(Path(clus))
    b = b[b.cluster == cluster]
    if recording:
        b = b[b.file == recording]
    if b.empty:
        return b.reset_index(drop=True)
    rep = b.loc[[b.gad.idxmax()]]
    rest = b.drop(rep.index)
    return pd.concat([rep, rest.sample(min(n - 1, len(rest)), random_state=seed)]).reset_index(drop=True)


def names_path(clus: str | Path) -> Path:
    return Path(clus) / "cluster_names.csv"


def load_names(clus: str | Path) -> dict[int, str]:
    """The names a person gave the clusters; {} if none yet."""
    p = names_path(clus)
    if not p.exists():
        return {}
    t = pd.read_csv(p, keep_default_na=False)
    return {int(c): str(n) for c, n in zip(t.cluster, t.name, strict=True) if str(n).strip()}


def save_name(clus: str | Path, cluster: int, label: str) -> None:
    names = load_names(clus)
    if label.strip():
        names[int(cluster)] = label.strip()
    else:
        names.pop(int(cluster), None)
    pd.DataFrame({"cluster": list(names), "name": list(names.values())}, columns=["cluster", "name"]).sort_values(
        "cluster").to_csv(names_path(clus), index=False)


def plot_tiles(clus: str | Path, sample: pd.DataFrame, cols: int = 4, pad_s: float = 0.3, min_view_s: float = 1.0,
               range_db: float = 60.0, title: str = ""):
    """One spectrogram crop per box (box +- pad_s, at least min_view_s wide), box outlined.

    All tiles share one frequency axis: up to 1.15 x the highest box top in the sample."""
    paths = recording_paths(Path(clus))
    rows = max(1, int(np.ceil(len(sample) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.5 * rows), squeeze=False, constrained_layout=True)
    if sample.empty:
        for ax in axes.flat:
            ax.axis("off")
        return fig
    nyq = min(sf.info(str(paths[f])).samplerate for f in sample.file.unique()) / 2000
    fmax = float(np.clip(sample.high_hz.max() * 1.15 / 1000, 4.0, nyq))
    for i, ax in enumerate(axes.flat):
        if i >= len(sample):
            ax.axis("off")
            continue
        r = sample.iloc[i]
        mid, half = (r.begin_s + r.end_s) / 2, max((r.end_s - r.begin_s) / 2 + pad_s, min_view_s / 2)
        t0, t1 = max(0.0, mid - half), mid + half
        spec = window_spectrogram(paths[r.file], t0, t1)
        if spec is not None:
            f, t, img = spec
            keep = f <= fmax * 1000
            vmax = np.percentile(img[keep], 99.8)
            ax.imshow(img[keep], origin="lower", aspect="auto", cmap="Greys", vmin=vmax - range_db, vmax=vmax,
                      extent=[t[0], t[-1], 0, f[keep][-1] / 1000])
        ax.add_patch(Rectangle((r.begin_s, r.low_hz / 1000), r.end_s - r.begin_s, (r.high_hz - r.low_hz) / 1000,
                               facecolor="none", edgecolor=BOX_COLOUR, linewidth=1.4))
        ax.set_xlim(t0, t1)
        ax.set_ylim(0, fmax)
        ax.set_title(f"[{i}] {str(r.file)[-15:]} #{int(r.selection)}\n{r.dur_ms:.0f} ms, "
                     f"{r.low_hz / 1000:.1f}-{r.high_hz / 1000:.1f} kHz", fontsize=8)
        ax.tick_params(labelsize=7)
        if i % cols == 0:
            ax.set_ylabel("kHz", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


HEAR_MODES = ["box only", "box band ± pad", "everything ± pad"]


def box_only_audio(path: Path, row, context_s: float = 0.2, fade_ms: float = 5.0) -> tuple[np.ndarray, int]:
    """Only the sound inside the box: band-passed to [low_hz, high_hz] and cut to [begin_s, end_s].

    The filter runs on the box +- context_s, which is then cropped off (no filter edge
    transient inside the box), and fade_ms ramps at both ends avoid clicks."""
    t0 = max(0.0, row.begin_s - context_s)
    x, sr = seg.read_mono(path, t0, row.end_s + context_s)
    x = bandpass(x, sr, row.low_hz, row.high_hz)
    i0 = int(round((row.begin_s - t0) * sr))
    x = np.array(x[i0:i0 + max(1, int(round((row.end_s - row.begin_s) * sr)))], dtype=np.float64)
    n = min(int(fade_ms / 1000 * sr), len(x) // 2)
    if n:
        ramp = np.linspace(0.0, 1.0, n)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x, sr


def hear(path: Path, row, mode: str, pad: float, repeat: int = 1, gap_s: float = 0.3) -> tuple[np.ndarray, int]:
    """Audio for one box. mode is one of HEAR_MODES; repeat > 1 loops it with gap_s of silence."""
    if mode == "box only":
        x, sr = box_only_audio(path, row)
    else:
        x, sr = box_audio(path, row, pad, band=mode == "box band ± pad")
    if repeat > 1:
        x = np.tile(np.concatenate([x, np.zeros(int(gap_s * sr))]), repeat)
    return x, sr


# ---------------------------------------------------------------- GUI
class ClusterGallery:
    """Pick a cluster, see its representative and a random sample of its boxes, play any of them, name it."""

    def __init__(self, clus: str | Path, n: int = 12):
        import ipywidgets as W

        self.clus = Path(clus).resolve()
        if not (self.clus / "boxes_clustered.csv.gz").exists():
            raise FileNotFoundError(f"no boxes_clustered.csv.gz in {self.clus}; run `biodetect cluster` first")
        self.summ = cluster_summary(self.clus)
        S = {"description_width": "initial"}
        self._quiet = True
        self.w_cluster = W.Dropdown(options=self._options(), description="Cluster", style=S,
                                    layout=W.Layout(width="480px"))
        self.b_prev, self.b_next = _btn("◀ prev"), _btn("next ▶")
        self.w_n = W.Dropdown(options=[8, 12, 16, 24], value=n, description="Boxes", style=S,
                              layout=W.Layout(width="120px"))
        self.w_rec = W.Dropdown(description="Recording", style=S, layout=W.Layout(width="330px"))
        self.b_resample = _btn("new sample", width="110px", button_style="info")
        self.w_range = W.IntSlider(description="Range dB", min=20, max=100, value=60, continuous_update=False,
                                   style=S, layout=W.Layout(width="260px"))
        self.info, self.out_plot = W.HTML(), W.Output()
        self.w_tile = W.Dropdown(description="Tile", style=S, layout=W.Layout(width="120px"))
        self.w_hear = W.Dropdown(options=HEAR_MODES, value="box only", description="Hear", style=S,
                                 layout=W.Layout(width="210px"))
        self.w_pad = W.BoundedFloatText(value=0.5, min=0, max=10, step=0.25, description="Pad (s)", style=S,
                                        layout=W.Layout(width="130px"), disabled=True)
        self.w_repeat = W.Dropdown(options=[1, 3, 5], value=1, description="Repeat", style=S,
                                   layout=W.Layout(width="120px"))
        self.w_hear.observe(lambda ch: setattr(self.w_pad, "disabled", ch["new"] == "box only"), "value")
        self.b_play = _btn("Play box", button_style="info")
        self.out_audio = W.Output()
        self.w_name = W.Text(description="Name", placeholder="e.g. cicada A, frog 2, rain, unsure",
                             style=S, layout=W.Layout(width="330px"))
        self.b_save = _btn("Save name", width="100px", button_style="success")
        self.b_save.on_click(lambda _: self._save_name())
        self.name_msg = W.HTML()
        self.seed = 0

        self.w_cluster.observe(lambda _: self._set_cluster(), "value")
        self.b_prev.on_click(lambda _: self._step(-1))
        self.b_next.on_click(lambda _: self._step(+1))
        self.b_resample.on_click(lambda _: self._resample())
        for w in (self.w_n, self.w_rec, self.w_range):
            w.observe(lambda _: self.redraw(), "value")
        self.b_play.on_click(lambda _: self.play())

        self.ui = W.VBox([
            W.HBox([self.w_cluster, self.b_prev, self.b_next, self.w_n, self.b_resample]),
            W.HBox([self.w_rec, self.w_range]),
            self.info, self.out_plot,
            W.HTML("<b>Listen</b>"),
            W.HBox([self.w_tile, self.w_hear, self.w_pad, self.w_repeat, self.b_play]),
            self.out_audio,
            W.HTML(f"<b>Name this cluster</b> (saved to {names_path(self.clus)})"),
            W.HBox([self.w_name, self.b_save, self.name_msg]),
        ])
        self._quiet = False
        self._set_cluster()

    def _options(self):
        names = load_names(self.clus)
        return [(f"{c}: {int(r.boxes):,} boxes, {r.recordings} rec, {r.median_ms:.0f} ms, "
                 f"{r.median_centre_khz:.1f} kHz" + (f"  [{names[c]}]" if c in names else ""), int(c))
                for c, r in self.summ.iterrows()]

    def _save_name(self):
        c = self.w_cluster.value
        save_name(self.clus, c, self.w_name.value)
        self._quiet = True
        self.w_cluster.options = self._options()
        self.w_cluster.value = c
        self._quiet = False
        self.name_msg.value = f"saved cluster {c}; {len(load_names(self.clus))} of {len(self.summ)} clusters named"

    def _set_cluster(self):
        if self._quiet:
            return
        b = load_boxes(self.clus)
        recs = b[b.cluster == self.w_cluster.value].file.value_counts()
        self._quiet = True
        self.w_rec.options = [("all recordings", "")] + [(f"{f}  ({n:,})", f) for f, n in recs.items()]
        self.w_rec.value = ""
        self._quiet = False
        self.seed = 0
        self.w_name.value = load_names(self.clus).get(self.w_cluster.value, "")
        self.name_msg.value = ""
        self.redraw()

    def _step(self, k: int):
        vals = [v for _, v in self.w_cluster.options]
        self.w_cluster.value = vals[(vals.index(self.w_cluster.value) + k) % len(vals)]

    def _resample(self):
        self.seed += 1
        self.redraw()

    def redraw(self):
        if self._quiet:
            return
        c = self.w_cluster.value
        self.sample = sample_cluster(self.clus, c, self.w_n.value, self.seed, self.w_rec.value or None)
        r = self.summ.loc[c]
        self.info.value = (
            f"<b>cluster {c}</b>: {int(r.boxes):,} boxes from {r.recordings} recordings "
            f"(largest recording {r.top_recording_share:.0%}) &nbsp;|&nbsp; median {r.median_ms:.0f} ms, "
            f"centre {r.median_centre_khz:.1f} kHz, bandwidth {r.median_bandwidth_khz:.1f} kHz, "
            f"{r.median_db:.1f} dB above noise &nbsp;|&nbsp; mostly {r.main_band} ({r.main_band_share:.0%}), "
            f"{r.main_duration} ({r.main_duration_share:.0%}) &nbsp;|&nbsp; sample #{self.seed}"
            " &nbsp;|&nbsp; tile [0] = representative (highest GAD)")
        self.w_tile.options = list(range(len(self.sample)))
        self.out_plot.clear_output(wait=True)
        with self.out_plot:
            plot_tiles(self.clus, self.sample, range_db=self.w_range.value)
            plt.show()
        self.out_audio.clear_output()

    def play(self):
        from IPython.display import Audio, display

        i = self.w_tile.value
        self.out_audio.clear_output(wait=True)
        with self.out_audio:
            if i is None:
                return
            r = self.sample.iloc[i]
            path = recording_paths(self.clus)[r.file]
            print(f"[{i}] {r.file} selection {int(r.selection)}: {r.begin_s:.3f}-{r.end_s:.3f} s, "
                  f"{r.low_hz:.0f}-{r.high_hz:.0f} Hz, {int(r.n_cells)} cells, {r.mean_db:.1f} dB above noise")
            x, sr = hear(path, r, self.w_hear.value, self.w_pad.value, self.w_repeat.value)
            if r.high_hz > 16000:
                print("note: part of this box is above 16 kHz, which many people cannot hear")
            display(Audio(x, rate=sr, normalize=True))

    def show(self):
        from IPython.display import display

        display(self.ui)
        return self
