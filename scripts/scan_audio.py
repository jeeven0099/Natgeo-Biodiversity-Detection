"""Detect calls on a new audio file: the zero-shot chain behind the long-form caches, run on one file.

The review app's boxes come from caches built over dataset/ (zeroshot_windows.py, earthchirp_windows.py,
cell_maps.py). A file added through the app (or with this script) gets the same numbers from one pass:
per 5 s window, 2.5 s hop, cut exactly as zeroshot_windows.load_windows cuts them (whole file to
32 kHz with resample_poly, mono mean, zero-padded past the end):
  Perch 2.0     perch_v2_no_dft.onnx (TF32 off): group probabilities (softmax over the 14,795 classes,
                summed per group), the group max logits, the 1536-d embedding, the 16 x 4 cell grid;
  EarthChirp    ec_cos and ec_margin of the float16-rounded embedding, as earthchirp_windows.py
                --stage score reads earthchirp_embeddings.npy;
  cell maps     perch_cell_head.onnx on the cell grid + cell_maps.ec_cells: the 8 lanes of cell_maps.npy;
  silent        the raw window is all zeros.
Segments are the rolling max over 2 windows (zeroshot_windows.segments). Nothing here looks at labels.

Added audio lives in data/uploads (--uploads-dir): audio/<rec>.<ext> + audio/<rec>.json (original name,
sha256, format) and scans/<rec>.npz (the arrays above, SCAN_VERSION). The review app lists each such
recording beside the dataset's; the learners fit on their log rows too once the scan is complete
(review_sources.py), except for deleted recordings (deleted.csv). Website guests' uploads go to
uploads/guests/<folder>/ (the review app's own Uploads there, kept only while the guest's page is open):
never in audio/, so nothing here or in the learners reads them.

    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\scan_audio.py --check 3              # rebuild check
    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\scan_audio.py my_recording.wav       # add a file
    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\scan_audio.py --delete REC --reason "..."   # app stopped

--check N rescans N dataset recordings (evenly spread over window_scores.csv) and compares every output
with the caches; it writes nothing. Limits: group probabilities <= 0.01, EarthChirp <= 1e-3, cell lanes
<= 0.01 (Perch) / 2e-3 (EarthChirp, float16), embedding cosine >= 0.999, the same silent flags.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_maps as cm  # noqa: E402
import earthchirp_windows as EW  # noqa: E402
import insecta_compare as IC  # noqa: E402
import perch_viewer as pv  # noqa: E402
import zeroshot_windows as z  # noqa: E402

ROOT = z.ROOT
PDIR = ROOT / "data" / "perch_windows"
UPLOADS_DIR = ROOT / "data" / "uploads"
REVIEW_DIR = ROOT / "data" / "review"
MODEL = ROOT / "external" / "bacpipe_model_checkpoints" / "perch_v2" / "perch_v2_no_dft.onnx"
HEAD = PDIR / cm.HEAD_NAME
LIB = ROOT / "external" / "earthchirp_lib"
ESC50_DIR = ROOT / "data" / "insecta_compare"    # EarthChirp's ESC-50 outgroup (earthchirp_windows.earthchirp)
SCAN_VERSION = "1"
HOP_S = EW.HOP_S                     # 2.5 s
BATCH = 16                           # Perch windows per call on the GPU (zero rows pad the last: cuDNN)
HEAD_BATCH = 4                       # windows per cell-head call (its similarity tensor is 15 MB a window)
MAX_UPLOAD_BYTES = 1 << 30           # 1 GiB
AUDIO_EXT = (".wav", ".flac", ".ogg", ".oga", ".mp3", ".aif", ".aiff", ".w64", ".rf64", ".caf")
DELETED_FILE = "deleted.csv"         # uploads deleted by their owner: their answers stay out of learning
DELETED_COLS = ["recording", "sha256", "deleted_at", "reason"]
# --check limits
MAX_PROB_DIFF, MAX_EC_DIFF = cm.MAX_PROB_DIFF, cm.MAX_EC_DIFF
MAX_CELL_PROB_DIFF, MAX_CELL_EC_DIFF, MIN_EMB_COS = 0.01, 2e-3, 0.999


# ---- windows --------------------------------------------------------------------------------
def load_signal(path: str) -> tuple[np.ndarray, int]:
    """(x, n_windows): the 32 kHz mono signal zero-padded so window k is x[k * hop : k * hop + 5 s].
    The same arithmetic as zeroshot_windows.load_windows, without copying every overlapping window."""
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != z.PERCH_SR:
        g = int(np.gcd(sr, z.PERCH_SR))
        x = resample_poly(x, z.PERCH_SR // g, sr // g).astype(np.float32)
    hop_n = int(round(HOP_S * z.PERCH_SR))
    n_win = max(1, int(np.ceil(len(x) / hop_n - 1e-9)))
    return np.pad(x, (0, (n_win - 1) * hop_n + z.WINDOW_N - len(x))), n_win


def windows_of(x: np.ndarray, k0: int, k1: int) -> np.ndarray:
    hop_n = int(round(HOP_S * z.PERCH_SR))
    return np.stack([x[k * hop_n:k * hop_n + z.WINDOW_N] for k in range(k0, k1)])


def missing_files() -> list[Path]:
    """The files a scan reads besides the audio (Perch, the cell head, EarthChirp's library and its
    ESC-50 outgroup) that are not there. The review app turns adding off when any is missing, so an
    upload is refused at once instead of failing after it was stored."""
    need = [MODEL, HEAD, LIB / "global_reflib_protos.npy", LIB / "global_reflib_index.parquet",
            ESC50_DIR / "esc50_emb.npy", ESC50_DIR / "esc50_meta.csv", ROOT / "config" / "perch_class_index.csv"]
    return [p for p in need if not p.exists()]


def open_session(model: Path, device: str):
    """IC.open_session, falling back to the CPU when CUDA cannot start."""
    if device == "cuda":
        try:
            return IC.open_session(model, "cuda")
        except (SystemExit, Exception) as e:          # SystemExit: no CUDA provider
            print(f"CUDA unavailable for {model.name} ({e}): CPU", flush=True)
    return IC.open_session(model, "cpu")


def uses_cuda(sess) -> bool:
    return "CUDAExecutionProvider" in sess.get_providers()


class Scanner:
    """Perch + EarthChirp + the cell head on every window of a file. perch: an open Perch session to
    share (the review app's); else one is opened on `device`."""

    def __init__(self, device: str = "cuda", perch=None):
        if not HEAD.exists():
            raise SystemExit(f"{HEAD} missing: run cell_maps.py --stage build-head first")
        self.perch = perch if perch is not None else open_session(MODEL, device)
        self.cuda = uses_cuda(self.perch)
        self.head = open_session(HEAD, "cuda" if self.cuda else "cpu")
        self.labels, self.groups = IC.labels_and_groups()
        self.ec = EW.earthchirp(SimpleNamespace(lib=LIB))

    def providers(self) -> str:
        return f"Perch {self.perch.get_providers()[0]}, cell head {self.head.get_providers()[0]}"

    def scan(self, path: str, progress=None) -> dict:
        """-> start_s (n,), prob / maxlogit (n, 6 groups), silent (n,), emb (n, 1536) float16,
        ec_cos / ec_margin (n,), cells (n, 16, 4, 8) float16, meta (dict). progress(done, total)."""
        x, n = load_signal(path)
        bs = BATCH if self.cuda else 1               # on the CPU one window per call is fastest
        ng = len(z.GROUPS)
        out = {"prob": np.empty((n, ng)), "maxlogit": np.empty((n, ng)), "silent": np.empty(n, bool),
               "emb": np.empty((n, EW.EMB_DIM), np.float16),
               "cells": np.empty((n, cm.N_T, cm.N_F, len(cm.LANES)), np.float16)}
        t0 = time.monotonic()
        if progress:
            progress(0, n)
        for k in range(0, n, bs):
            m = min(bs, n - k)
            w = windows_of(x, k, k + m)
            xb = np.zeros((bs, z.WINDOW_N), np.float32)
            xb[:m] = w
            e, sp, lab = self.perch.run(["embedding", "spatial_embedding", "label"],
                                        {"inputs": IC.normalize_windows(xb, 0.25)})
            e, sp, lab = np.asarray(e).reshape(bs, -1)[:m], np.asarray(sp, np.float32)[:m], np.asarray(lab)[:m]
            out["prob"][k:k + m] = cm.group_probs(lab, self.groups)
            lg = lab.astype(np.float64)
            out["maxlogit"][k:k + m] = np.stack([lg[:, self.groups[g]].max(1) for g in z.GROUPS], 1)
            out["silent"][k:k + m] = np.abs(w).max(axis=1) == 0
            out["emb"][k:k + m] = e.astype(np.float16)
            for h in range(0, m, HEAD_BATCH):
                s = sp[h:h + HEAD_BATCH]
                (cg,) = self.head.run(["cell_group_prob"], {"x": s})
                ins, mar = cm.ec_cells(s, self.ec)
                out["cells"][k + h:k + h + len(s)] = np.concatenate(
                    [cg[..., :ng], ins[..., None], mar[..., None]], -1).astype(np.float16)
            if progress:
                progress(k + m, n)
        X = out["emb"].astype(np.float32)             # as earthchirp_windows.py scores the float16 cache
        cos, og = self.ec.cos(X), self.ec.og(X)
        info = sf.info(path)
        out.update(start_s=np.arange(n) * HOP_S, ec_cos=cos, ec_margin=cos - og,
                   meta={"version": SCAN_VERSION, "windows": n, "seconds": round(time.monotonic() - t0, 1),
                         "providers": self.providers(), "scanned_at": dt.datetime.now().isoformat(timespec="seconds"),
                         "samplerate": info.samplerate, "channels": info.channels, "duration_s": float(info.duration)})
        return out


def segments(scan: dict, rec: str) -> pd.DataFrame:
    """The 2.5 s segments of one scanned file, columns as pv.load_segments gives them (no truth_*):
    segment k = max over windows k-1 and k; silent only when both are."""
    r = int(round(z.WINDOW_S / HOP_S))
    cols = {}
    for j, g in enumerate(z.GROUPS):
        cols[f"prob_{g}"] = scan["prob"][:, j]
        cols[f"maxlogit_{g}"] = scan["maxlogit"][:, j]
    for g, c in pv.EC_GROUPS.items():
        cols[f"prob_{g}"] = scan[c]
    seg = pd.DataFrame(cols).rolling(r, min_periods=1).max()
    seg.insert(0, "recording", rec)
    seg.insert(1, "day", "")
    seg.insert(2, "start_s", scan["start_s"])
    silent = pd.Series(scan["silent"].astype(int)).rolling(r, min_periods=1).min()
    seg.insert(3, "silent", silent.astype(bool).to_numpy())
    return seg


ARRAYS = ("start_s", "prob", "maxlogit", "silent", "emb", "ec_cos", "ec_margin", "cells")


def save_scan(path: Path, scan: dict, extra: dict) -> None:
    """scans/<rec>.npz, written whole then renamed (a half-written scan is never read)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    with open(tmp, "wb") as f:
        np.savez(f, meta=np.array(json.dumps({**scan["meta"], **extra})), **{k: scan[k] for k in ARRAYS})
    os.replace(tmp, path)


def load_scan(path: Path) -> dict:
    with np.load(path) as d:
        out = {k: d[k] for k in ARRAYS}
        out["meta"] = json.loads(str(d["meta"]))
    return out


# ---- rows of the dataset caches followed by added files' rows ------------------------------------
class RowStack:
    """A (memmapped) array with more rows appended behind it, indexed along axis 0 like one array:
    ints, slices and integer arrays. append() swaps in new lists whole, so a reader never sees half."""

    def __init__(self, base):
        self._p = ((base,), (0,), len(base))
        self.dtype, self.row_shape = base.dtype, tuple(base.shape[1:])

    def __len__(self) -> int:
        return self._p[2]

    @property
    def shape(self) -> tuple:
        return (len(self), *self.row_shape)

    def append(self, a: np.ndarray) -> int:
        """Adds rows; returns the index of the first."""
        a = np.ascontiguousarray(a)
        if a.dtype != self.dtype or tuple(a.shape[1:]) != self.row_shape:
            raise ValueError(f"rows {a.dtype} {a.shape[1:]} do not fit {self.dtype} {self.row_shape}")
        parts, offs, n = self._p
        self._p = (parts + (a,), offs + (n,), n + len(a))
        return n

    def release(self, first: int, n: int) -> bool:
        """Frees the rows one append() added (first, n: what it returned and its length): they read as
        zeros from then on (a broadcast view, no memory), and every other row keeps its index. The base
        array is never released. -> False when no appended part starts at first with n rows."""
        parts, offs, total = self._p
        j = bisect.bisect_left(offs, first)
        if j == 0 or j >= len(offs) or offs[j] != first or len(parts[j]) != n:
            return False
        stub = np.broadcast_to(np.zeros((1, *self.row_shape), self.dtype), (n, *self.row_shape))
        self._p = (parts[:j] + (stub,) + parts[j + 1:], offs, total)
        return True

    def __getitem__(self, idx):
        parts, offs, n = self._p
        if isinstance(idx, slice):
            a, b, step = idx.indices(n)
            if step == 1:
                j = max(0, bisect.bisect_right(offs, a) - 1)
                if b <= offs[j] + len(parts[j]) or a >= b:
                    return parts[j][a - offs[j]:max(a, b) - offs[j]]
            idx = np.arange(a, b, step)
        elif isinstance(idx, int | np.integer):
            i = int(idx) + (n if int(idx) < 0 else 0)
            if not 0 <= i < n:
                raise IndexError(f"row {idx} of {n}")
            j = bisect.bisect_right(offs, i) - 1
            return parts[j][i - offs[j]]
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        idx = np.where(idx < 0, idx + n, idx).astype(np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise IndexError(f"rows outside 0..{n - 1}")
        out = np.empty(idx.shape + self.row_shape, self.dtype)
        j = np.searchsorted(np.asarray(offs), idx, side="right") - 1
        for p in np.unique(j):
            m = j == p
            out[m] = parts[p][idx[m] - offs[p]]
        return out


# ---- the uploads folder ---------------------------------------------------------------------------
def clean_name(name: str) -> str:
    """A recording id from a file name: letters, digits, . _ - only ('|' separates id fields)."""
    stem = Path(str(name).replace("\\", "/")).stem
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:80]
    return s or "added"


class Uploads:
    """data/uploads: audio/<rec>.<ext>, audio/<rec>.json (written last: a recording exists once it
    is there), scans/<rec>.npz."""

    def __init__(self, root: Path = UPLOADS_DIR):
        self.root = Path(root)
        self.audio, self.scans = self.root / "audio", self.root / "scans"

    def recordings(self) -> dict[str, dict]:
        """rec -> info, in the order the files were added."""
        out = []
        for j in self.audio.glob("*.json") if self.audio.exists() else []:
            info = json.loads(j.read_text(encoding="utf-8"))
            p = self.audio / info["file"]
            if p.exists():
                out.append({**info, "path": str(p)})
        return {i["rec"]: i for i in sorted(out, key=lambda i: (i["added_at"], i["rec"]))}

    def names(self) -> set[str]:
        """Every stem in audio/, finished or not (lower case: Windows file names ignore case)."""
        return {p.name.split(".")[0].lower() for p in self.audio.iterdir()} if self.audio.exists() else set()

    def scan_file(self, rec: str) -> Path:
        return self.scans / f"{rec}.npz"

    def scan_ok(self, rec: str, info: dict) -> bool:
        p = self.scan_file(rec)
        if not p.exists():
            return False
        with np.load(p) as d:
            meta = json.loads(str(d["meta"]))
        return meta.get("version") == SCAN_VERSION and meta.get("sha256") == info["sha256"]

    def receive(self, read, length: int, filename: str, taken: set[str],
                limit: int | None = MAX_UPLOAD_BYTES, max_duration_s: float | None = None,
                extra: dict | None = None, too_big: str = "add longer files with scan_audio.py",
                rec_id: str | None = None) -> tuple[str, dict, bool]:
        """Streams `length` bytes from read(n) into audio/; -> (rec, info, created). The same bytes
        added before -> that recording (created False). Names in `taken` (lower case) get -2, -3, ...
        limit None: no size cap (the command line). max_duration_s: longer audio is refused and not
        kept. extra: more fields for the new recording's audio/<rec>.json (the website's private key).
        rec_id: the new recording's id instead of one made from the file name (a guest's random id)."""
        if length <= 0:
            raise ValueError("the upload is empty")
        if limit is not None and length > limit:
            raise ValueError(f"the file is {length / 2**20:.1f} MB; the limit is {limit / 2**20:g} MB "
                             f"({too_big})")
        ext = Path(str(filename)).suffix.lower()
        ext = ext if ext in AUDIO_EXT else ".wav"
        self.audio.mkdir(parents=True, exist_ok=True)
        tmp = self.audio / f".incoming-{uuid.uuid4().hex}{ext}"
        h, got = hashlib.sha256(), 0
        try:
            with open(tmp, "wb") as f:
                while got < length:
                    b = read(min(1 << 20, length - got))
                    if not b:
                        raise ValueError(f"the upload stopped after {got} of {length} bytes")
                    f.write(b)
                    h.update(b)
                    got += len(b)
            sha = h.hexdigest()
            for rec, info in self.recordings().items():
                if info["sha256"] == sha:
                    tmp.unlink()
                    return rec, info, False
            try:                                   # the header, and the first and last frames (a cut file
                si = sf.info(str(tmp))             # can have a sound header and fail later)
                with sf.SoundFile(str(tmp)) as fh:
                    fh.read(min(fh.frames, 4096), dtype="float32")
                    fh.seek(max(0, fh.frames - 1))
                    if len(fh.read(1, dtype="float32")) != min(1, fh.frames):
                        raise RuntimeError("the last frame is missing")
            except Exception as e:
                raise ValueError(f"cannot read {filename!r} as audio ({type(e).__name__}): open it in the audio "
                                 "editor and press detect calls there (the browser decodes it)") from None
            if si.frames <= 0 or si.samplerate <= 0:
                raise ValueError(f"{filename!r} holds no samples")
            if max_duration_s is not None and si.duration > max_duration_s + 1e-6:
                raise ValueError(f"{filename!r} is {si.duration / 60:.2f} min long; the limit is "
                                 f"{max_duration_s / 60:g} min: cut it into parts")
            base = clean_name(rec_id or filename)
            rec, k = base, 1
            gone = {r.lower() for r in self.deleted()}     # a deleted id is never reused: its log rows stay
            while rec.lower() in taken or rec.lower() in self.names() or rec.lower() in gone:
                k += 1
                rec = f"{base}-{k}"
            dst = self.audio / f"{rec}{ext}"
            os.replace(tmp, dst)
        finally:
            if tmp.exists():
                tmp.unlink()
        info = {"rec": rec, "file": dst.name, "name": Path(str(filename).replace("\\", "/")).name, "sha256": sha,
                "bytes": got, "added_at": dt.datetime.now().isoformat(timespec="seconds"),
                "samplerate": si.samplerate, "channels": si.channels, "duration_s": float(si.duration),
                "format": f"{si.format} {si.subtype}", **(extra or {})}
        self._write_info(rec, info)
        return rec, {**info, "path": str(dst)}, True

    def _write_info(self, rec: str, info: dict) -> None:
        side = self.audio / f"{rec}.json"
        part = side.with_suffix(".json.partial")
        part.write_text(json.dumps({k: v for k, v in info.items() if k != "path"}, indent=1), encoding="utf-8")
        os.replace(part, side)

    def update_info(self, rec: str, **fields) -> dict:
        """Adds fields to a recording's audio/<rec>.json (an upload made before the website had no key)."""
        info = {**json.loads((self.audio / f"{rec}.json").read_text(encoding="utf-8")), **fields}
        self._write_info(rec, info)
        return info

    def bytes_used(self) -> int:
        """Bytes under the uploads folder (audio, sidecars, scans, half-received files)."""
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file()) if self.root.exists() else 0

    def purge(self, rec: str, info: dict) -> tuple[list[str], list[str]]:
        """Removes a recording's audio and scan files -> (removed, left): a file still open elsewhere
        (Windows: the scan worker reading it) is left, for the caller to remove later."""
        gone, left = [], []
        for p in [self.audio / info.get("file", f"{rec}.missing"), self.scan_file(rec),
                  self.scans / f"{rec}.partial.npz"]:
            try:
                if p.exists():
                    p.unlink()
                    gone.append(p.name)
            except OSError:
                left.append(p.name)
        return gone, left

    def delete(self, rec: str, reason: str = "", log: bool = True) -> tuple[list[str], list[str]]:
        """Removes a recording's sidecar (first, so a half-deleted recording is never listed), audio and
        scan; appends rec, sha256 and the time to deleted.csv, which the learners read to leave the
        recording's answers out (log False: no row, for a guest's upload, which no learner reads).
        -> (file names removed, file names left open: see purge)."""
        side = self.audio / f"{rec}.json"
        info = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
        gone = []
        if side.exists():
            side.unlink()
            gone.append(side.name)
        more, left = self.purge(rec, info)
        gone += more
        if not log:
            return gone, left
        p = self.root / DELETED_FILE
        new = not p.exists()
        with open(p, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(DELETED_COLS)
            w.writerow([rec, info.get("sha256", ""), dt.datetime.now().isoformat(timespec="seconds"), reason])
        return gone, left

    def deleted(self) -> set[str]:
        """Recordings deleted from the uploads folder (deleted.csv)."""
        p = self.root / DELETED_FILE
        if not p.exists():
            return set()
        with open(p, encoding="utf-8", newline="") as f:
            return {r["recording"] for r in csv.DictReader(f)}

    def remove_all(self) -> bool:
        """Removes the whole folder (a guest's: nothing in it is kept, nothing goes to deleted.csv)
        -> True when it is gone (a file still open elsewhere is left for a later call)."""
        shutil.rmtree(self.root, ignore_errors=True)
        return not self.root.exists()


def dataset_names() -> set[str]:
    return {r.lower() for r in pv.wav_paths()}


def logged_names(review_dir: Path = REVIEW_DIR) -> set[str]:
    """Recordings named anywhere in the review folder's logs and prediction cache (lower case): a new
    file must never take a name that old rows already point at (a deleted upload, say)."""
    out = set()
    for f in ("annotation_log.csv", "audit_windows.csv"):
        p = Path(review_dir) / f
        if p.exists():
            out |= {str(r).lower() for r in pd.read_csv(p, usecols=["recording"], dtype=str)["recording"].dropna()}
    p = Path(review_dir) / "prediction_cache.jsonl"
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                m = re.match(r'\{"key": "([^"|]*)\|([^"|]*)', line)
                if m:
                    out.add((m.group(2) if m.group(1) == "new" else m.group(1)).lower())
    return out


# ---- --check: rescan dataset recordings, compare with the caches -----------------------------------
def check(args) -> int:
    w = pd.read_csv(PDIR / "window_scores.csv", dtype={"day": str})
    ecw = pd.read_csv(PDIR / "earthchirp_windows.csv")
    E = np.load(PDIR / "earthchirp_embeddings.npy", mmap_mode="r")
    M = np.load(PDIR / "cell_maps.npy", mmap_mode="r")
    if not (len(w) == len(ecw) == len(E) == len(M)):
        raise SystemExit("the caches differ in length: rebuild them first")
    seg = pv.load_segments(PDIR)
    wavs = pv.wav_paths()
    recs = list(pd.unique(w["recording"]))
    pick = args.recs or [recs[i] for i in np.unique(np.linspace(0, len(recs) - 1, args.check).round().astype(int))]
    codes, names = pd.factorize(w["recording"])
    first = dict(zip(names, np.r_[0, np.flatnonzero(np.diff(codes)) + 1], strict=True))
    sc = Scanner(args.device)
    print(f"--check on {len(pick)} recordings ({sc.providers()})", flush=True)
    prob_cols = [f"prob_{g}" for g in z.GROUPS]
    logit_cols = [f"maxlogit_{g}" for g in z.GROUPS]
    lanes = [g for g in pv.VIEW_GROUPS if f"prob_{g}" in seg]
    thr = {g: pv.THRESHOLDS[g] for g in lanes}
    rows = []
    for i, rec in enumerate(pick):
        a = int(first[rec])
        want = w[w["recording"] == rec]
        n = len(want)
        if i == 0:                                     # the windows themselves, once
            x, nw = load_signal(wavs[rec])
            ref = z.load_windows(wavs[rec], HOP_S)
            if nw != len(ref) or not np.array_equal(windows_of(x, 0, nw), ref):
                raise SystemExit(f"{rec}: load_signal windows differ from zeroshot_windows.load_windows")
            del x, ref
            print(f"  {rec}: windows identical to zeroshot_windows.load_windows ({nw})", flush=True)
        s = sc.scan(wavs[rec])
        if len(s["start_s"]) != n or not np.allclose(s["start_s"], want["start_s"]):
            raise SystemExit(f"{rec}: {len(s['start_s'])} windows, the cache has {n}")
        Ef = np.asarray(E[a:a + n], np.float32)
        emb = s["emb"].astype(np.float32)
        cos = (emb * Ef).sum(1) / np.linalg.norm(emb, axis=1).clip(1e-9) / np.linalg.norm(Ef, axis=1).clip(1e-9)
        Mc = np.asarray(M[a:a + n], np.float32)
        cells = s["cells"].astype(np.float32)
        ng = len(z.GROUPS)
        e = ecw.iloc[a:a + n]
        new_seg, old_seg = segments(s, rec), seg[seg["recording"] == rec].reset_index(drop=True)
        seg_cols = [f"prob_{g}" for g in lanes]
        nb, ob = pv.perch_boxes(new_seg, thr), pv.perch_boxes(old_seg, thr)
        key = lambda b: set(zip(b["group"], b["begin_s"].round(3), b["end_s"].round(3), strict=True))   # noqa: E731
        kn, ko = key(nb), key(ob)
        rows.append({"recording": rec, "windows": n, "seconds": s["meta"]["seconds"],
                     "prob": float(np.abs(s["prob"] - want[prob_cols].to_numpy()).max()),
                     "maxlogit": float(np.abs(s["maxlogit"] - want[logit_cols].to_numpy()).max()),
                     "silent_same": bool((s["silent"] == want["silent"].to_numpy(bool)).all()),
                     "ec": float(max(np.abs(s["ec_cos"] - e["ec_cos"].to_numpy()).max(),
                                     np.abs(s["ec_margin"] - e["ec_margin"].to_numpy()).max())),
                     "emb_min_cos": float(cos.min()),
                     "cell_prob": float(np.abs(cells[..., :ng] - Mc[..., :ng]).max()),
                     "cell_ec": float(np.abs(cells[..., ng:] - Mc[..., ng:]).max()),
                     "seg": float(np.abs(new_seg[seg_cols].to_numpy() - old_seg[seg_cols].to_numpy()).max()),
                     "boxes_cache": len(ko), "boxes_scan": len(kn), "boxes_differ": len(kn ^ ko)})
        r = rows[-1]
        print(f"  {i + 1}/{len(pick)} {rec}: {n} windows in {r['seconds']} s; max diff prob {r['prob']:.1e}, "
              f"EC {r['ec']:.1e}, cells {r['cell_prob']:.1e}/{r['cell_ec']:.1e}, emb cos >= {r['emb_min_cos']:.5f}, "
              f"boxes {r['boxes_scan']} vs {r['boxes_cache']} ({r['boxes_differ']} differ)", flush=True)
    c = pd.DataFrame(rows)
    lim = [("prob", MAX_PROB_DIFF, "max"), ("ec", MAX_EC_DIFF, "max"), ("cell_prob", MAX_CELL_PROB_DIFF, "max"),
           ("cell_ec", MAX_CELL_EC_DIFF, "max"), ("seg", MAX_PROB_DIFF, "max"), ("emb_min_cos", MIN_EMB_COS, "min")]
    bad = [f"{k} {c[k].max() if how == 'max' else c[k].min():.3g} (limit {v})" for k, v, how in lim
           if (c[k].max() > v if how == "max" else c[k].min() < v)]
    if not c["silent_same"].all():
        bad.append("silent flags differ")
    print(f"over {len(c)} recordings / {c['windows'].sum()} windows: prob {c['prob'].max():.2e}, maxlogit "
          f"{c['maxlogit'].max():.2e}, EC {c['ec'].max():.2e}, cells {c['cell_prob'].max():.2e} / "
          f"{c['cell_ec'].max():.2e}, "
          f"segments {c['seg'].max():.2e}, embedding cosine >= {c['emb_min_cos'].min():.5f}; model boxes "
          f"{c['boxes_scan'].sum()} vs {c['boxes_cache'].sum()} in the cache, {c['boxes_differ'].sum()} differ; "
          f"{c['windows'].sum() / c['seconds'].sum():.1f} windows/s", flush=True)
    print("check " + ("FAILED: " + "; ".join(bad) if bad else "passed"), flush=True)
    return 1 if bad else 0


def add_files(args) -> int:
    up = Uploads(args.uploads_dir)
    sc = None
    for f in args.files:
        with open(f, "rb") as fh:            # no size cap here: the app's cap is for browser uploads
            rec, info, created = up.receive(fh.read, os.path.getsize(f), f,
                                            dataset_names() | logged_names(args.review_dir), limit=None)
        if not created and up.scan_ok(rec, info):
            print(f"{f}: already added as {rec} (scanned)")
            continue
        sc = sc or Scanner(args.device)
        t0 = time.monotonic()
        s = sc.scan(info["path"],
                    progress=lambda d, t, rec=rec: print(f"\r  {rec}: {d}/{t} windows", end="", flush=True))
        save_scan(up.scan_file(rec), s, {"sha256": info["sha256"], "rec": rec})
        seg = segments(s, rec)
        nb = pv.perch_boxes(seg, {g: pv.THRESHOLDS[g] for g in pv.VIEW_GROUPS if f"prob_{g}" in seg})
        print(f"\r{f} -> {rec}: {len(s['start_s'])} windows in {time.monotonic() - t0:.0f} s; model boxes per lane: "
              + (", ".join(f"{g} {n}" for g, n in nb["group"].value_counts().items()) or "none")
              + f" -> {up.scan_file(rec)}", flush=True)
    return 0


def delete_files(args) -> int:
    """--delete: removes added recordings as the website's delete does (Uploads.delete: files gone, a
    deleted.csv row, so the learners leave out their answers). Stop the review app first: it holds the
    recordings in memory and would keep serving them until its next start."""
    up = Uploads(args.uploads_dir)
    have = up.recordings()
    bad = [r for r in args.delete if r not in have]
    if bad:
        print("not in " + str(up.audio) + ": " + ", ".join(bad) + " (nothing deleted)")
        return 1
    for rec in args.delete:
        gone, left = up.delete(rec, args.reason)
        print(f"{rec}: removed {', '.join(gone)}" + (f"; still open, not removed: {', '.join(left)}" if left else "")
              + f"; logged in {up.root / DELETED_FILE}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="audio files to add to --uploads-dir and scan")
    ap.add_argument("--check", type=int, default=0, metavar="N",
                    help="rescan N dataset recordings, compare, write nothing")
    ap.add_argument("--recs", nargs="*", default=None, help="with --check: these recordings instead")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--uploads-dir", type=Path, default=UPLOADS_DIR)
    ap.add_argument("--review-dir", type=Path, default=REVIEW_DIR, help="whose logged names a new file avoids")
    ap.add_argument("--delete", nargs="+", metavar="REC",
                    help="delete these added recordings (stop the review app first); needs --reason")
    ap.add_argument("--reason", default="", help="with --delete: the reason written to deleted.csv")
    args = ap.parse_args()
    if args.delete:
        if args.files or args.check or args.recs or not args.reason.strip():
            ap.error("--delete takes recording ids and --reason, nothing else")
        return delete_files(args)
    if args.check or args.recs:
        args.check = args.check or len(args.recs)
        return check(args)
    if not args.files:
        ap.error("give audio files to add, or --check N")
    return add_files(args)


if __name__ == "__main__":
    raise SystemExit(main())
