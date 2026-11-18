"""Box-pooled Perch embeddings: the mean of Perch's spatial cells under a box, on the CPU, cached.

    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\box_embed.py --rec S01634_20250228_012000 --box 10.2 11.0 2.0 4.5

Perch 2.0's spatial_embedding gives each 5 s window a grid of 16 time cells (CELL_S = 0.3125 s) by
4 frequency rows (ROW_EDGES_KHZ: 0.06-0.955-2.864-6.971-16 kHz), each a 1536-d vector; the window
embedding is the mean of all 64. A box's embedding is the mean of the cells it covers, over every
window review_app.windows_for_span picks for its time span (the same windows the app's top 5 uses):
  time  a cell counts when its overlap with the box is at least half of the shorter of the two
        (cell 0.3125 s, box length), so a sliver at an edge does not pull a neighbouring call in;
  rows  the same rule in kHz.
A box outside every row (above 16 kHz) or no cell passing falls back to the cells of largest
overlap (the nearest row when none overlaps). Two calls in the same row at the same time stay mixed:
nothing finer than a cell exists. The audio is not band-passed before Perch (band-passed Perch
clusters follow the filter, not the species).

Used by review_species.py (prototypes from named boxes) and review_app.py (suggestions for the box
on screen). Cached in data/review/embeddings/: vectors.npy (float32) + index.csv keyed by
recording|begin|end|lo|hi (3 decimals) and EMBED_VERSION, so a box is embedded once.
CPU speed (measured 2026-11-16): ~10 windows/s at batch 1, a box of 1-3 windows ~0.1-0.3 s.
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import insecta_compare as IC  # noqa: E402
import perch_viewer as pv  # noqa: E402
import zeroshot_windows as z  # noqa: E402

ROOT = pv.ROOT
PDIR = ROOT / "data" / "perch_windows"
MODEL = ROOT / "external/bacpipe_model_checkpoints/perch_v2/perch_v2_no_dft.onnx"
CACHE_DIR = ROOT / "data" / "review" / "embeddings"
EMBED_VERSION = "perch_v2-box-cells-1"      # change when the pooling rule changes: the cache ignores old rows
CELL_S = pv.CELL_S
N_T, N_F, DIM = 16, 4, 1536
ROW_EDGES = np.array(pv.ROW_EDGES_KHZ)
LRU_WINDOWS = 256                            # window grids kept in memory (~0.4 MB each)
INDEX_COLS = ["key", "recording", "begin_s", "end_s", "lo_khz", "hi_khz", "n_windows", "n_cells", "version"]


def _app():
    import review_app  # here, not at the top: review_app imports this module
    return review_app


def box_key(rec: str, box) -> str:
    a, b, lo, hi = (float(x) for x in box)
    return f"{rec}|{a:.3f}|{b:.3f}|{lo:.3f}|{hi:.3f}"


def picked(lo_e: np.ndarray, hi_e: np.ndarray, a: float, b: float) -> np.ndarray:
    """Which intervals [lo_e, hi_e] a box side [a, b] covers: overlap >= half the shorter of the two;
    none passing -> the largest overlap; none overlapping -> the nearest."""
    ov = np.minimum(hi_e, b) - np.maximum(lo_e, a)
    need = 0.5 * np.minimum(hi_e - lo_e, b - a)
    m = (ov > 0) & (ov >= need - 1e-9)
    if not m.any():
        m = ov >= ov.max() - 1e-9
    return m


def cell_mask(starts: np.ndarray, box) -> np.ndarray:
    """(len(starts), N_T, N_F) bool: the cells of each window the box covers."""
    a, b, lo, hi = (float(x) for x in box)
    rows = picked(ROW_EDGES[:-1], ROW_EDGES[1:], lo, hi)
    out = np.zeros((len(starts), N_T, N_F), bool)
    j = np.arange(N_T)
    for i, s in enumerate(starts):
        t0 = s + j * CELL_S
        ov = np.minimum(t0 + CELL_S, b) - np.maximum(t0, a)
        cols = (ov > 0) & (ov >= 0.5 * min(CELL_S, b - a) - 1e-9)
        out[i] = cols[:, None] & rows[None, :]
    if not out.any():                          # the box misses every chosen window's cells (it should not)
        i = int(np.argmin(np.abs(np.asarray(starts) + 2.5 - (a + b) / 2)))
        t0 = starts[i] + j * CELL_S
        ov = np.minimum(t0 + CELL_S, b) - np.maximum(t0, a)
        out[i] = (ov >= ov.max() - 1e-9)[:, None] & rows[None, :]
    return out


def pool(spatial: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean of the masked cells (float64 sum, float32 out)."""
    return spatial[mask].astype(np.float64).mean(0).astype(np.float32)


def window_starts(pdir: Path = PDIR) -> dict[str, np.ndarray]:
    w = pd.read_csv(pdir / "window_scores.csv", usecols=["recording", "start_s"])
    return {r: g["start_s"].to_numpy() for r, g in w.groupby("recording", sort=False)}


class PerchSpatial:
    """Perch on the CPU (or CUDA), one window per call on the CPU (fastest there, measured):
    rec, starts -> (n, N_T, N_F, DIM) float32."""

    def __init__(self, device: str = "cpu", wavs: dict[str, str] | None = None, session=None):
        self.device, self.wavs, self._sess = device, wavs, session

    def session(self):
        if self._sess is None:
            try:
                self._sess = IC.open_session(MODEL, self.device)
            except SystemExit:
                self._sess = IC.open_session(MODEL, "cpu")
        return self._sess

    def __call__(self, rec: str, starts: np.ndarray) -> np.ndarray:
        R = _app()
        if self.wavs is None:
            self.wavs = pv.wav_paths()
        starts = np.asarray(starts, float)
        x, x0 = R.resampled_slice(self.wavs[rec], float(starts.min()), float(starts.max()) + R.WIN_S)
        n = int(round(R.WIN_S * z.PERCH_SR))
        sess, out = self.session(), []
        for s in starts:
            o = int(round((s - x0) * z.PERCH_SR))
            w = np.zeros((1, n), np.float32)
            seg = x[o:o + n]
            w[0, :len(seg)] = seg
            sp, = sess.run(["spatial_embedding"], {"inputs": IC.normalize_windows(w, 0.25)})
            out.append(np.asarray(sp, np.float32)[0])
        return np.stack(out)


class BoxEmbedder:
    """Box embeddings from window grids. run(rec, starts) computes grids; put() lets a caller that
    already ran Perch on some windows (the app's top 5) hand them over. Keeps LRU_WINDOWS grids."""

    def __init__(self, run=None, starts: dict[str, np.ndarray] | None = None, lru: int = LRU_WINDOWS):
        self.run = run or PerchSpatial()
        self.starts = starts if starts is not None else window_starts()
        self.lru, self._grids = lru, OrderedDict()
        self.n_run = 0                               # windows Perch ran on (tests read it)

    def put(self, rec: str, starts, grids: np.ndarray) -> None:
        for s, g in zip(starts, grids, strict=True):
            self._grids[(rec, round(float(s), 3))] = np.asarray(g, np.float32)
            self._grids.move_to_end((rec, round(float(s), 3)))
        while len(self._grids) > self.lru:
            self._grids.popitem(last=False)

    def grids(self, rec: str, starts: np.ndarray) -> np.ndarray:
        keys = [(rec, round(float(s), 3)) for s in starts]
        got = {k: self._grids[k] for k in keys if k in self._grids}   # read before put() can evict them
        todo = [s for s, k in zip(starts, keys, strict=True) if k not in got]
        if todo:
            g = self.run(rec, np.asarray(todo, float))
            got.update({(rec, round(float(s), 3)): np.asarray(x, np.float32) for s, x in zip(todo, g, strict=True)})
            self.put(rec, todo, g)
            self.n_run += len(todo)
        for k in keys:                              # a box longer than the cache keeps only its last grids
            if k in self._grids:
                self._grids.move_to_end(k)
        return np.stack([got[k] for k in keys])

    def windows(self, rec: str, box) -> np.ndarray:
        st = self.starts[rec]
        return st[_app().windows_for_span(st, float(box[0]), float(box[1]))]

    def embed(self, rec: str, box) -> tuple[np.ndarray, int, int]:
        """(embedding, windows, cells) of one box [begin_s, end_s, lo_khz, hi_khz]."""
        if not (float(box[0]) < float(box[1]) and float(box[2]) < float(box[3])):
            raise ValueError(f"box needs begin < end and low < high: {box}")
        st = self.windows(rec, box)
        m = cell_mask(st, box)
        return pool(self.grids(rec, st), m), len(st), int(m.sum())


class EmbeddingCache:
    """Disk cache of box embeddings (vectors.npy + index.csv); rows of another EMBED_VERSION are ignored."""

    def __init__(self, folder: Path = CACHE_DIR):
        self.folder = Path(folder)
        self.rows: list[dict] = []
        self.vecs: list[np.ndarray] = []
        self.pos: dict[str, int] = {}
        idx, vec = self.folder / "index.csv", self.folder / "vectors.npy"
        if idx.exists() and vec.exists():
            df = pd.read_csv(idx, keep_default_na=False)
            V = np.load(vec)
            if len(df) != len(V):
                raise SystemExit(f"{idx} has {len(df)} rows, {vec} {len(V)}: delete both to rebuild")
            for r, v in zip(df.to_dict("records"), V, strict=True):
                if str(r["version"]) == EMBED_VERSION:
                    self._add(r, v)
        self.dirty = False

    def _add(self, row: dict, v: np.ndarray) -> None:
        self.pos[row["key"]] = len(self.rows)
        self.rows.append(row)
        self.vecs.append(np.asarray(v, np.float32))

    def get(self, key: str) -> np.ndarray | None:
        i = self.pos.get(key)
        return None if i is None else self.vecs[i]

    def add(self, rec: str, box, v: np.ndarray, n_windows: int, n_cells: int) -> None:
        k = box_key(rec, box)
        if k in self.pos:
            return
        self._add(dict(key=k, recording=rec, begin_s=round(float(box[0]), 3), end_s=round(float(box[1]), 3),
                       lo_khz=round(float(box[2]), 3), hi_khz=round(float(box[3]), 3), n_windows=n_windows,
                       n_cells=n_cells, version=EMBED_VERSION), v)
        self.dirty = True

    def save(self) -> None:
        if not self.dirty:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp_v, tmp_i = self.folder / "vectors.partial.npy", self.folder / "index.partial.csv"
        np.save(tmp_v, np.stack(self.vecs) if self.vecs else np.zeros((0, DIM), np.float32))
        pd.DataFrame(self.rows, columns=INDEX_COLS).to_csv(tmp_i, index=False)
        tmp_v.replace(self.folder / "vectors.npy")
        tmp_i.replace(self.folder / "index.csv")
        self.dirty = False


def embed_boxes(items: list[tuple[str, tuple]], embedder: BoxEmbedder, cache: EmbeddingCache | None = None,
                progress: bool = False) -> np.ndarray:
    """(n, DIM) float32 for (recording, box) pairs, from the cache where present; new ones are
    embedded in recording / time order (so neighbouring boxes share window grids) and cached."""
    out = np.zeros((len(items), DIM), np.float32)
    todo = []
    for i, (rec, box) in enumerate(items):
        v = cache.get(box_key(rec, box)) if cache is not None else None
        if v is None:
            todo.append(i)
        else:
            out[i] = v
    todo.sort(key=lambda i: (items[i][0], float(items[i][1][0])))
    for n, i in enumerate(todo):
        rec, box = items[i]
        v, nw, nc = embedder.embed(rec, box)
        out[i] = v
        if cache is not None:
            cache.add(rec, box, v, nw, nc)
        if progress and (n + 1) % 25 == 0:
            print(f"  embedded {n + 1}/{len(todo)} boxes ({embedder.n_run} Perch windows)", flush=True)
    if cache is not None:
        cache.save()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--box", type=float, nargs=4, required=True, metavar=("BEGIN_S", "END_S", "LO_KHZ", "HI_KHZ"))
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = ap.parse_args()
    E = BoxEmbedder(PerchSpatial(args.device))
    v, nw, nc = E.embed(args.rec, args.box)
    print(f"{args.rec} {args.box}: {nw} windows, {nc} cells, |e| {np.linalg.norm(v):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
