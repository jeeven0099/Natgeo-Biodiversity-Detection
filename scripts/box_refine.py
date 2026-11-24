"""Box refiner: moves the edges of a Perch-grid box onto the call in the fine spectrogram,
fitted to the boxes the reviewers accept.

    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\box_refine.py

Perch's boxes snap to its grid (0.3125 s x 4 bands). The refiner works on the STFT the review app
draws (1024-point Hann, 10 ms hop: ~21 ms x 47 Hz at 48 kHz):

  1. patch   the coarse box (the tightened box the app suggests) +- MARGIN_S in time, and from
             low / FREQ_FACTOR to high x FREQ_FACTOR in frequency; dB smoothed (Gaussian, 1 bin),
             minus each frequency bin's FLOOR_Q quantile over the box +- CONTEXT_S (the background).
  2. mask    cells at least delta dB above the background, kept only in the 8-connected regions
             that touch the coarse box (a call may run past the grid box; other sounds in the
             margin are dropped).
  3. edges   the first and last time column and frequency row holding at least rho x the
             busiest one's mask cells, widened by pad_t s and pad_f kHz.
  The setting (delta, rho, pad_t, pad_f) is fitted per group, and for all groups together, to the
  reviewers' accepted boxes: the highest mean time x frequency IoU (kHz linear) over the GRID.

Confidence of one box: the lowest IoU between its refined box and the boxes from the neighbouring
settings (delta one step up, the stricter threshold) and from the settings fitted on each half of
the recordings, every box widened by EDGE_TOL first (so a one-bin shift on a narrow tonal call is
not read as instability). A call well above the noise gives the same box under all of them; no
region touching the coarse box gives confidence 0.

Which boxes count:
  training    every "yes" box the pipeline suggested (model and below-threshold boxes), edited or not;
  evaluation  only those shown as the plain grid box (all of them before a refiner is adopted, then
              the CONTROL_SHARE of confident boxes the app shows plain on purpose): a refined box
              accepted unedited cannot show that it was right.
  Raven       a model box fitted to its Raven box (raven_autolabel.py, shown_box "raven-auto") counts
              for both: the Raven box was drawn without the refiner. Model boxes merged into one Raven
              box ("raven-auto merge") count for neither.
A refiner is adopted when, leaving each recording out in turn (setting and confidence cut both
chosen on the other recordings only), what the app would show (refined when confident, else the
grid box) is closer to the reviewer's box than the grid box: bootstrap over recordings, 95%
interval above 0. That nested test needs NESTED_EXTRA recordings beyond a fit's minimum (5 at the
defaults); fewer are reported as too few labels. The cut is the lowest confidence at which at
least P_WIN of the held-out refined boxes are no worse than the grid box, over at least MIN_SHOWN
boxes.

Recordings (review_sources.py, --uploads-dir): the survey's and the scanned audio added in the app.
The refined boxes of each coarse box at every GRID setting are cached (data/review/refine_cache/,
BoxCache, --cache): they need the audio, which the server has only for added recordings, so survey
boxes come from the cache built where the audio is; a box with neither is left out, with a count.

Writes data/review/calibration/refiner.json (review_app.py reads it at start: adopted groups get
refined boxes; every group's confidence fills the "box uncertain" queue), refine_settings.csv,
refine_boxes.csv and refine_grid.csv.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perch_viewer as pv  # noqa: E402

N_FFT, HOP_S = 1024, 0.010        # the review app's spectrogram (its hop grows with the view; this one does not)
MARGIN_S = 0.5                    # the refined box may reach this far past the coarse box in time
FREQ_FACTOR = 1.5                 # ... and from low / 1.5 to high x 1.5 in frequency
CONTEXT_S = 5.0                   # background: each bin's FLOOR_Q quantile over the box +- this
FLOOR_Q = 0.25
FLOOR_FRAMES = 500              # ... estimated on at most this many evenly spaced frames
MAX_SPAN_S = 30.0                 # longer boxes (bouts, choruses) are not refined
SMOOTH_SIGMA = 1.0                # Gaussian smoothing of the dB spectrogram, in bins
DELTAS = (3.0, 6.0, 9.0, 12.0, 15.0, 18.0)   # dB above the background
RHOS = (0.0, 0.05, 0.1, 0.2, 0.3)             # edge support, share of the busiest column / row
PADS_T = (0.0, 0.025, 0.05)                   # s added each side
PADS_F = (0.0, 0.1, 0.25)                     # kHz added below and above
GRID = tuple((d, r, pt, pf) for d in DELTAS for r in RHOS for pt in PADS_T for pf in PADS_F)
DEFAULT = (9.0, 0.1, 0.0, 0.0)    # before any fit (confidence for the uncertain queue only)
UNCERTAIN_BELOW = 0.5             # before a cut is fitted: boxes below this go to "box uncertain"
EDGE_TOL = (PADS_T[1], PADS_F[1])  # confidence compares boxes widened by the smallest pad steps (s, kHz)
NESTED_EXTRA = 2                  # the nested held-out test needs this many recordings beyond a fit's minimum
                                  # (at DEFAULT on the busiest recording: 58% of boxes; 0.8 would be 90%)
CONTROL_SHARE = 0.1               # confident boxes shown plain anyway, to keep an unanchored test set
P_WIN = 0.9                       # cut: this share of shown refined boxes no worse than the grid box
MIN_SHOWN = 10                    # ... over at least this many held-out boxes
MIN = dict(boxes=20, recordings=3)
BOOT, SEED = 2000, 0
OUT_DIR = pv.ROOT / "data" / "review" / "calibration"
REFINER_FILE = OUT_DIR / "refiner.json"
CACHE_DIR = pv.ROOT / "data" / "review" / "refine_cache"   # BoxCache
CONTROL_SHOWN = ("tight", "tight (control)",   # shown_box values whose accepted box is not anchored on a refined one
                 "raven-auto")                # (a model box fitted to its Raven box, raven_autolabel.py)
MERGED_SHOWN = "raven-auto merge"            # several model boxes merged into one Raven box: not a box label
EDGES = ("begin_s", "end_s", "lo_khz", "hi_khz")

SET_COLS = ["group", "delta", "rho", "pad_t", "pad_f", "half_a", "half_b", "conf_min", "decision", "reason",
            "n_boxes", "n_control", "n_recordings", "n_edited", "iou_grid_all", "iou_refined_all",
            "loro_iou_grid", "loro_iou_app", "loro_iou_diff", "ci_lo", "ci_hi", "loro_coverage", "loro_no_worse",
            "loro_n_eval"]
BOX_OUT_COLS = ["box_id", "recording", "split", "group", "shown_box", "control", "box_edited",
                "grid_begin_s", "grid_end_s", "grid_lo_khz", "grid_hi_khz",
                "accepted_begin_s", "accepted_end_s", "accepted_lo_khz", "accepted_hi_khz",
                "refined_begin_s", "refined_end_s", "refined_lo_khz", "refined_hi_khz",
                "spec_group", "conf", "iou_grid", "iou_refined"]


# ---- the refiner -------------------------------------------------------------------------
@dataclass
class Patch:
    t: np.ndarray            # frame centres, s (recording time)
    f: np.ndarray            # bin centres, kHz
    r: np.ndarray            # dB above the background, (freq, time), search region only
    inside: np.ndarray       # cells inside the coarse box
    dt: float
    df: float
    dur: float
    nyq: float


def iou(a, b) -> float:
    """Time x frequency IoU of two (begin_s, end_s, lo_khz, hi_khz) boxes, kHz linear."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    union = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return float(inter / union) if union > 0 else 0.0


def patch_from_db(f: np.ndarray, t: np.ndarray, db: np.ndarray, coarse, dt_s: float, dur: float,
                  nyq: float) -> Patch | None:
    """f (kHz), t (frame centres, s), db (freq, time) of the box +- CONTEXT_S -> the search patch.
    Only the search band (plus the smoothing's reach) is smoothed; the background quantile uses at
    most FLOOR_FRAMES evenly spaced frames."""
    t0, t1, lo, hi = (float(v) for v in coarse)
    tm = (t >= t0 - MARGIN_S) & (t <= t1 + MARGIN_S)
    fm = (f >= lo / FREQ_FACTOR) & (f <= min(nyq, hi * FREQ_FACTOR))
    if not tm.any() or not fm.any():
        return None
    reach = int(np.ceil(4 * SMOOTH_SIGMA))
    r0, r1 = max(0, np.flatnonzero(fm)[0] - reach), min(len(f), np.flatnonzero(fm)[-1] + 1 + reach)
    s = ndimage.gaussian_filter(db[r0:r1].astype(np.float32), SMOOTH_SIGMA, mode="nearest")
    cols = np.unique(np.linspace(0, s.shape[1] - 1, min(s.shape[1], FLOOR_FRAMES)).round().astype(int))
    floor = np.quantile(s[:, cols], FLOOR_Q, axis=1)
    tt, ff = t[tm], f[fm]
    inside = ((ff >= lo) & (ff <= hi))[:, None] & ((tt >= t0) & (tt <= t1))[None, :]
    if not inside.any():
        return None
    return Patch(tt, ff, (s - floor[:, None])[np.ix_(fm[r0:r1], tm)], inside, float(dt_s), float(f[1] - f[0]),
                 float(dur), float(nyq))


def make_patch(path: str, coarse) -> Patch | None:
    """The patch of a coarse box, read from the wav (the box +- CONTEXT_S only); None for boxes
    longer than MAX_SPAN_S (bouts and choruses: not refined)."""
    if float(coarse[1]) - float(coarse[0]) > MAX_SPAN_S:
        return None
    info = sf.info(path)
    sr, dur = info.samplerate, info.frames / info.samplerate
    a, b = max(0.0, float(coarse[0]) - CONTEXT_S), min(dur, float(coarse[1]) + CONTEXT_S)
    if b <= a:
        return None
    x, sr = pv.read_page(path, a, b - a)
    hop = max(1, int(round(HOP_S * sr)))
    f, ts, db = stft_db(x, sr, hop)
    return patch_from_db(f, a + ts + N_FFT / (2 * sr), db, coarse, hop / sr, dur, sr / 2000)


def stft_db(x: np.ndarray, sr: int, hop: int):
    """perch_viewer.spectrogram (Hann N_FFT, dB, frame-start times) in float32 via scipy.fft
    (numpy's FFT works in float64: ~3x slower here)."""
    from scipy import fft
    x = np.asarray(x, np.float32)
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    fr = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::hop] * np.hanning(N_FFT).astype(np.float32)
    db = 20 * np.log10(np.abs(fft.rfft(fr, axis=1, workers=-1)).T + 1e-10)
    return np.fft.rfftfreq(N_FFT, 1 / sr) / 1000, np.arange(fr.shape[0]) * hop / sr, db.astype(np.float32)


def boxes_for(P: Patch, settings) -> dict[tuple, tuple | None]:
    """Refined box (begin_s, end_s, lo_khz, hi_khz) at each setting; None when no region above
    delta touches the coarse box."""
    out: dict[tuple, tuple | None] = {}
    by_delta: dict[float, list[tuple]] = {}
    for s in settings:
        by_delta.setdefault(float(s[0]), []).append(tuple(s))
    for d, ss in by_delta.items():
        m = P.r >= d
        lab, _ = ndimage.label(m, structure=np.ones((3, 3), bool))
        ids = np.unique(lab[P.inside & m])
        ids = ids[ids > 0]
        if not len(ids):
            out.update({s: None for s in ss})
            continue
        k = np.isin(lab, ids)
        nt, nf = k.sum(0), k.sum(1)
        for s in ss:
            _, rho, pt, pf = s
            ct = np.flatnonzero(nt >= max(1.0, rho * nt.max()))
            cf = np.flatnonzero(nf >= max(1.0, rho * nf.max()))
            out[s] = (round(max(0.0, float(P.t[ct[0]]) - P.dt / 2 - pt), 3),
                      round(min(P.dur, float(P.t[ct[-1]]) + P.dt / 2 + pt), 3),
                      round(max(0.0, float(P.f[cf[0]]) - P.df / 2 - pf), 3),
                      round(min(P.nyq, float(P.f[cf[-1]]) + P.df / 2 + pf), 3))
    return out


def neighbours(setting, halves=()) -> list[tuple]:
    """delta one step up (one step down at the top of DELTAS), and the half-fit settings (not the
    setting itself). Only the stricter threshold: a soft-edged call shrinks under it, and noise
    attached to the call at this delta drops off under it. A looser threshold that attaches
    background noise says nothing against the box at this one."""
    setting = tuple(float(x) for x in setting)
    i = DELTAS.index(setting[0])
    j = i + 1 if i + 1 < len(DELTAS) else i - 1
    nb = [(DELTAS[j], *setting[1:])] if 0 <= j < len(DELTAS) else []
    nb += [tuple(float(x) for x in h) for h in halves if h is not None]
    return list(dict.fromkeys(s for s in nb if s != setting))


def iou_tol(a, b) -> float:
    """IoU after widening both boxes by EDGE_TOL on every edge: a one-bin edge shift on a narrow
    tonal call (0.3 kHz tall) is not instability."""
    w = lambda x: (x[0] - EDGE_TOL[0], x[1] + EDGE_TOL[0], x[2] - EDGE_TOL[1], x[3] + EDGE_TOL[1])  # noqa: E731
    return iou(w(a), w(b))


def confidence(boxes: dict, setting, nb) -> float:
    b = boxes.get(tuple(setting))
    if b is None:
        return 0.0
    return min((iou_tol(b, boxes[n]) if boxes.get(n) is not None else 0.0 for n in nb), default=1.0)


def default_spec() -> dict:
    return dict(group="default", setting=list(DEFAULT), neighbours=[list(s) for s in neighbours(DEFAULT)],
                conf_min=None, decision="keep")


def is_control(box_id: str) -> bool:
    """The fixed CONTROL_SHARE of boxes shown plain even when the refiner is confident."""
    return zlib.crc32(box_id.encode()) % 1000 < CONTROL_SHARE * 1000


class Refiner:
    """refiner.json as the app uses it. A group uses its own spec when adopted, else the adopted
    'all' spec; with neither, nothing is shown refined and confidence comes from its own fit,
    'all', or DEFAULT."""

    def __init__(self, spec: dict | None = None):
        self.spec = spec or {}
        self.version = self.spec.get("version", "default")
        self.groups: dict[str, dict] = self.spec.get("groups", {})

    @classmethod
    def load(cls, path: Path | None = REFINER_FILE) -> Refiner:
        if path is not None and Path(path).exists():
            return cls(json.loads(Path(path).read_text(encoding="utf-8")))
        return cls()

    def adopted(self) -> list[str]:
        return [g for g, s in self.groups.items() if s.get("decision") == "adopt"]

    def group_spec(self, group: str) -> dict:
        own, every = self.groups.get(group), self.groups.get("all")
        for s in (own, every):
            if s and s.get("decision") == "adopt":
                return s
        return own or every or default_spec()

    def apply(self, P: Patch | None, group: str) -> dict:
        s = self.group_spec(group)
        st, nb = tuple(float(x) for x in s["setting"]), [tuple(float(x) for x in n) for n in s["neighbours"]]
        bx = boxes_for(P, [st, *nb]) if P is not None else {}
        b = bx.get(st)
        conf = confidence(bx, st, nb) if P is not None else 0.0
        cut = s.get("conf_min")
        show = s.get("decision") == "adopt" and cut is not None and b is not None and conf >= cut
        below = cut if cut is not None else UNCERTAIN_BELOW
        return dict(refined=list(b) if b is not None else None, conf=round(float(conf), 4), show=bool(show),
                    refinable=P is not None, uncertain=bool(P is not None and (b is None or conf < below)),
                    spec_group=s.get("group", ""), version=self.version)


# ---- fitting -----------------------------------------------------------------------------
GRID_INDEX = {s: k for k, s in enumerate(GRID)}
_POS = np.array([(DELTAS.index(d), RHOS.index(r), PADS_T.index(pt), PADS_F.index(pf)) for d, r, pt, pf in GRID])
GRID_DIST = np.abs(_POS - _POS[GRID_INDEX[DEFAULT]]).sum(1)     # ties go to the setting nearest DEFAULT


@dataclass
class Table:
    """Every pipeline yes box at every GRID setting."""
    info: pd.DataFrame       # one row per box (BOX_OUT_COLS subset)
    B: np.ndarray            # (boxes, settings, 4) refined boxes, NaN where none
    I: np.ndarray            # (boxes, settings) IoU with the accepted box (0 where none)  # noqa: E741
    grid_iou: np.ndarray     # (boxes,) IoU of the grid (coarse) box with the accepted box


def half_of(rec: str) -> int:
    return zlib.crc32(rec.encode()) % 2


def cache_version() -> str:
    """Changes with anything that changes a refined box: the STFT, the patch and the GRID."""
    c = [N_FFT, HOP_S, MARGIN_S, FREQ_FACTOR, CONTEXT_S, FLOOR_Q, FLOOR_FRAMES, MAX_SPAN_S, SMOOTH_SIGMA, GRID]
    return f"refine-{zlib.crc32(json.dumps(c).encode()):08x}"


class BoxCache:
    """The refined boxes of a coarse box at every GRID setting, (len(GRID), 4) float64 (NaN where none),
    keyed by recording|coarse box (3 decimals): boxes.npy + index.csv. They depend on the audio and
    the coarse box only (not on the accepted box), so the server, which has no dataset/ audio, fits
    on the survey boxes cached where the audio is. Rows of another cache_version() are ignored."""

    def __init__(self, folder: Path | None = CACHE_DIR):
        self.folder = Path(folder) if folder is not None else None
        self.version = cache_version()
        self.keys: list[str] = []
        self.rows: list[np.ndarray] = []
        self.pos: dict[str, int] = {}
        self.dirty = False
        if self.folder is not None and (self.folder / "index.csv").exists() and (self.folder / "boxes.npy").exists():
            idx = pd.read_csv(self.folder / "index.csv", keep_default_na=False)
            B = np.load(self.folder / "boxes.npy")
            if len(idx) != len(B):
                raise SystemExit(f"{self.folder} index.csv has {len(idx)} rows, boxes.npy {len(B)}: delete both")
            for k, v, b in zip(idx["key"], idx["version"], B, strict=True):
                if v == self.version and B.shape[1:] == (len(GRID), 4):
                    self._add(k, b)

    @staticmethod
    def key(rec: str, coarse) -> str:
        a, b, lo, hi = (float(x) for x in coarse)
        return f"{rec}|{a:.3f}|{b:.3f}|{lo:.3f}|{hi:.3f}"

    def _add(self, k: str, b: np.ndarray) -> None:
        self.pos[k] = len(self.keys)
        self.keys.append(k)
        self.rows.append(np.asarray(b, float))

    def get(self, rec: str, coarse) -> np.ndarray | None:
        i = self.pos.get(self.key(rec, coarse))
        return None if i is None else self.rows[i]

    def add(self, rec: str, coarse, b: np.ndarray) -> None:
        if self.key(rec, coarse) not in self.pos:
            self._add(self.key(rec, coarse), b)
            self.dirty = True

    def save(self) -> None:
        if not self.dirty or self.folder is None:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        tb, ti = self.folder / "boxes.partial.npy", self.folder / "index.partial.csv"
        np.save(tb, np.stack(self.rows) if self.rows else np.zeros((0, len(GRID), 4)))
        pd.DataFrame(dict(key=self.keys, version=self.version)).to_csv(ti, index=False)
        tb.replace(self.folder / "boxes.npy")
        ti.replace(self.folder / "index.csv")
        self.dirty = False


def build_table(box: pd.DataFrame, patch_of, cache: BoxCache | None = None) -> Table:
    """box: pipeline yes boxes (review_labels box rows, with shown_box). patch_of(row, coarse) -> Patch,
    called only for boxes the cache does not hold."""
    info, Bs, Is, G = [], [], [], []
    for b in box.itertuples():
        coarse = tuple(float(getattr(b, f"tight_{e}")) for e in EDGES)
        acc = tuple(float(getattr(b, f"accepted_{e}")) for e in EDGES)
        Bb = cache.get(b.recording, coarse) if cache is not None else None
        if Bb is None:
            P = patch_of(b, coarse)
            bx = boxes_for(P, GRID) if P is not None else {}
            Bb = np.array([bx.get(s) if bx.get(s) is not None else (np.nan,) * 4 for s in GRID], float)
            if cache is not None:
                cache.add(b.recording, coarse, Bb)
        Bs.append(Bb)
        Is.append(np.array([iou(tuple(r), acc) if not np.isnan(r).any() else 0.0 for r in Bb]))
        G.append(iou(coarse, acc))
        shown = getattr(b, "shown_box", "tight") or "tight"
        info.append(dict(box_id=b.box_id, recording=b.recording, split=b.split, group=b.group, shown_box=shown,
                         control=shown in CONTROL_SHOWN, box_edited=b.box_edited,
                         **{f"grid_{e}": v for e, v in zip(EDGES, coarse, strict=True)},
                         **{f"accepted_{e}": v for e, v in zip(EDGES, acc, strict=True)}))
    n = len(info)
    return Table(pd.DataFrame(info), np.array(Bs).reshape(n, len(GRID), 4), np.array(Is).reshape(n, len(GRID)),
                 np.array(G, float))


def too_few(T: Table, idx: np.ndarray, mins: dict[str, int]) -> str:
    n_rec = T.info["recording"].to_numpy()[idx]
    low = [f"{k} {v} < {mins[k]}" for k, v in (("boxes", len(idx)), ("recordings", len(set(n_rec)))) if v < mins[k]]
    return "too few labels: " + ", ".join(low) if low else ""


def fit_k(T: Table, idx: np.ndarray) -> int:
    m = T.I[idx].mean(0)
    best = np.flatnonzero(m >= m.max() - 1e-12)
    return int(best[np.argmin(GRID_DIST[best])])


def fit_spec(T: Table, idx: np.ndarray) -> dict:
    """Setting fitted on idx, the settings fitted on each half of its recordings, the neighbours."""
    k = fit_k(T, idx)
    recs = T.info["recording"].to_numpy()[idx]
    halves = []
    for h in (0, 1):
        sub = idx[np.array([half_of(r) == h for r in recs], bool)]
        halves.append(GRID[fit_k(T, sub)] if len(sub) else None)
    nb = neighbours(GRID[k], halves)
    return dict(k=k, setting=GRID[k], halves=halves, nb=[GRID_INDEX[s] for s in nb])


def box_conf(T: Table, i: int, spec: dict) -> float:
    b = T.B[i, spec["k"]]
    if np.isnan(b).any():
        return 0.0
    vals = [iou_tol(b, T.B[i, n]) if not np.isnan(T.B[i, n]).any() else 0.0 for n in spec["nb"]]
    return min(vals, default=1.0)


def loro(T: Table, idx: np.ndarray, mins: dict[str, int]) -> pd.DataFrame:
    """Held-out predictions on the control boxes of idx, one recording left out at a time
    (folds below the minimum support give none)."""
    rec, ctrl = T.info["recording"].to_numpy(), T.info["control"].to_numpy(bool)
    out = []
    for r in sorted(set(rec[idx])):
        tr, te = idx[rec[idx] != r], idx[(rec[idx] == r) & ctrl[idx]]
        if not len(te) or too_few(T, tr, mins):
            continue
        spec = fit_spec(T, tr)
        for i in te:
            out.append((i, r, box_conf(T, i, spec), T.I[i, spec["k"]], T.grid_iou[i]))
    return pd.DataFrame(out, columns=["i", "recording", "conf", "iou_refined", "iou_grid"])


def choose_cut(pred: pd.DataFrame, p_win: float = P_WIN, min_shown: int = MIN_SHOWN) -> float | None:
    """Lowest confidence at which >= p_win of the shown refined boxes are no worse than the grid box
    (over >= min_shown boxes); None when no confidence qualifies."""
    if not len(pred):
        return None
    conf = pred["conf"].to_numpy()
    ok = (pred["iou_refined"].to_numpy() >= pred["iou_grid"].to_numpy() - 1e-9)
    for c in np.unique(conf[conf > 0]):
        m = conf >= c
        if m.sum() >= min_shown and ok[m].mean() >= p_win:
            return float(c)
    return None


def evaluate(T: Table, idx: np.ndarray, mins: dict[str, int], n_boot: int, seed: int) -> dict:
    """Nested leave-one-recording-out: per held-out recording, the setting and the cut come from
    the other recordings only. Returns pooled held-out IoU of what the app would show vs the grid box."""
    rec, ctrl = T.info["recording"].to_numpy(), T.info["control"].to_numpy(bool)
    per_rec, shown_ok = [], []
    n_eval = n_shown = 0
    for r in sorted(set(rec[idx])):
        tr, te = idx[rec[idx] != r], idx[(rec[idx] == r) & ctrl[idx]]
        if not len(te):
            continue
        app, grid = T.grid_iou[te].copy(), T.grid_iou[te]
        if not too_few(T, tr, mins):
            spec, cut = fit_spec(T, tr), choose_cut(loro(T, tr, mins))
            if cut is not None:
                for j, i in enumerate(te):
                    if not np.isnan(T.B[i, spec["k"]]).any() and box_conf(T, i, spec) >= cut:
                        app[j] = T.I[i, spec["k"]]
                        n_shown += 1
                        shown_ok.append(app[j] >= grid[j] - 1e-9)
        n_eval += len(te)
        per_rec.append(((app.sum(), len(te)), (grid.sum(), len(te))))
    if not per_rec:
        return dict(n_eval=0)
    new = np.array([a for a, _ in per_rec], float)
    old = np.array([b for _, b in per_rec], float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(new), (n_boot, len(new)))
    ns, os_ = new[draw].sum(1), old[draw].sum(1)
    d = ns[:, 0] / ns[:, 1] - os_[:, 0] / os_[:, 1]
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    fa, fg = new.sum(0), old.sum(0)
    return dict(n_eval=n_eval, loro_iou_app=fa[0] / fa[1], loro_iou_grid=fg[0] / fg[1],
                ci_lo=float(lo), ci_hi=float(hi), coverage=n_shown / n_eval,
                no_worse=float(np.mean(shown_ok)) if shown_ok else np.nan)


def fit_group(T: Table, idx: np.ndarray, group: str, mins: dict[str, int], n_boot: int, seed: int) -> tuple[dict, dict]:
    """(refiner.json entry, refine_settings.csv row) of one group (or 'all')."""
    info = T.info.iloc[idx]
    row = dict(group=group, n_boxes=len(idx), n_control=int(info["control"].sum()),
               n_recordings=info["recording"].nunique(), n_edited=int((info["box_edited"].astype(str) == "1").sum()),
               iou_grid_all=float(T.grid_iou[idx].mean()) if len(idx) else np.nan)
    why = too_few(T, idx, mins)
    need = mins["recordings"] + NESTED_EXTRA
    if not why and row["n_recordings"] < need:
        why = (f"too few labels: recordings {row['n_recordings']} < {need} (a fit needs {mins['recordings']}, "
               f"the nested held-out test {NESTED_EXTRA} more)")
    spec = fit_spec(T, idx) if len(idx) else dict(k=GRID_INDEX[DEFAULT], setting=DEFAULT, halves=[None, None],
                                                  nb=[GRID_INDEX[s] for s in neighbours(DEFAULT)])
    cut = None if why else choose_cut(loro(T, idx, mins))
    ev = {} if why else evaluate(T, idx, mins, n_boot, seed)
    if why:
        decision, reason = "keep", why
    elif not ev.get("n_eval"):
        decision, reason = "keep", "no control boxes to evaluate on"
    elif ev["ci_lo"] > 0 and cut is not None:
        decision, reason = "adopt", "held-out IoU higher, interval above 0"
    elif cut is None:
        decision, reason = "keep", f"no confidence cut keeps {P_WIN:.0%} of refined boxes no worse"
    else:
        decision, reason = "keep", "held-out gain not shown (interval reaches 0)"
    d, rho, pt, pf = spec["setting"]
    row.update(delta=d, rho=rho, pad_t=pt, pad_f=pf, conf_min=cut, decision=decision, reason=reason,
               half_a=str(spec["halves"][0] or ""), half_b=str(spec["halves"][1] or ""),
               iou_refined_all=float(T.I[idx, spec["k"]].mean()) if len(idx) else np.nan,
               loro_iou_grid=ev.get("loro_iou_grid", np.nan), loro_iou_app=ev.get("loro_iou_app", np.nan),
               loro_iou_diff=ev.get("loro_iou_app", np.nan) - ev.get("loro_iou_grid", np.nan),
               ci_lo=ev.get("ci_lo", np.nan), ci_hi=ev.get("ci_hi", np.nan), loro_coverage=ev.get("coverage", np.nan),
               loro_no_worse=ev.get("no_worse", np.nan), loro_n_eval=ev.get("n_eval", 0))
    entry = dict(group=group, setting=list(spec["setting"]), neighbours=[list(GRID[n]) for n in spec["nb"]],
                 halves=[list(h) if h else None for h in spec["halves"]], conf_min=cut, decision=decision,
                 reason=reason, n_boxes=len(idx), n_recordings=row["n_recordings"])
    return entry, row


def fit_all(T: Table, mins: dict[str, int], n_boot: int = BOOT, seed: int = SEED) -> tuple[dict, pd.DataFrame]:
    groups = sorted(set(T.info["group"])) if len(T.info) else []
    entries, rows = {}, []
    for g in [*groups, "all"]:
        idx = np.arange(len(T.info)) if g == "all" else np.flatnonzero(T.info["group"].to_numpy() == g)
        entries[g], row = fit_group(T, idx, g, mins, n_boot, seed)
        rows.append(row)
    return entries, pd.DataFrame(rows, columns=SET_COLS)


def box_rows(T: Table, ref: Refiner) -> pd.DataFrame:
    """Every box with the box the fitted refiner gives it (its group's spec as the app picks it)."""
    out = []
    for i, b in enumerate(T.info.to_dict("records")):
        s = ref.group_spec(b["group"])
        k, nb = GRID_INDEX[tuple(s["setting"])], [GRID_INDEX[tuple(n)] for n in s["neighbours"]]
        r = T.B[i, k]
        out.append({**b, **{f"refined_{e}": v for e, v in zip(EDGES, r, strict=True)},
                    "spec_group": s["group"], "conf": box_conf(T, i, dict(k=k, nb=nb)),
                    "iou_grid": T.grid_iou[i], "iou_refined": T.I[i, k]})
    return pd.DataFrame(out, columns=BOX_OUT_COLS)


def grid_rows(T: Table) -> pd.DataFrame:
    out = []
    for g in [*sorted(set(T.info["group"])), "all"] if len(T.info) else []:
        idx = np.arange(len(T.info)) if g == "all" else np.flatnonzero(T.info["group"].to_numpy() == g)
        m = T.I[idx].mean(0)
        out += [(g, *s, float(m[k]), len(idx)) for k, s in enumerate(GRID)]
    return pd.DataFrame(out, columns=["group", "delta", "rho", "pad_t", "pad_f", "mean_iou", "n_boxes"])


def pipeline_yes(box: pd.DataFrame) -> pd.DataFrame:
    """Yes boxes the pipeline suggested, less model boxes merged into one Raven box (their accepted
    box spans the other model boxes too)."""
    keep = box["proposal"] == "pipeline"
    if "shown_box" in box:
        keep &= box["shown_box"] != MERGED_SHOWN
    return box[keep].reset_index(drop=True)


def main() -> int:
    import review_labels as L  # here, not at the top: review_app imports this module
    R = L.R
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=R.LOG_FILE)
    ap.add_argument("--audit", type=Path, default=R.AUDIT_FILE)
    ap.add_argument("--shown", type=Path, default=R.SHOWN_FILE)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    for k, v in MIN.items():
        ap.add_argument(f"--min-{k}", type=int, default=v, help=f"support needed before fitting (default {v})")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--uploads-dir", type=Path, default=L.SRC.SA.UPLOADS_DIR, help="audio added in the app")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR, help="refined boxes per coarse box (BoxCache)")
    args = ap.parse_args()
    mins = {k: getattr(args, f"min_{k}") for k in MIN}

    src = L.SRC.Sources(args.uploads_dir)
    print(src.describe())
    lab = L.labels_from_logs(L.read_csv_rows(args.log), L.read_csv_rows(args.audit), L.read_csv_rows(args.shown),
                             sources=src)
    box = pipeline_yes(pd.DataFrame(lab["box"], columns=L.BOX_COLS))
    audio, cache = src.audio(), BoxCache(args.cache)
    here = np.array([t["recording"] in audio or cache.get(t["recording"], [t[f"tight_{e}"] for e in EDGES]) is not None
                     for t in box.to_dict("records")], bool).reshape(-1)
    if (~here).any():
        print(f"left out: {int((~here).sum())} boxes with no audio here and not in {args.cache}")
    box = box[here].sort_values(["recording", "box_begin_s"]).reset_index(drop=True)
    T = build_table(box, lambda b, coarse: make_patch(audio[b.recording], coarse), cache)
    cache.save()
    entries, sets = fit_all(T, mins, args.boot, args.seed)
    spec = dict(version=dt.datetime.now().isoformat(timespec="seconds"), kind="rule",
                constants=dict(N_FFT=N_FFT, HOP_S=HOP_S, MARGIN_S=MARGIN_S, FREQ_FACTOR=FREQ_FACTOR,
                               CONTEXT_S=CONTEXT_S, FLOOR_Q=FLOOR_Q, SMOOTH_SIGMA=SMOOTH_SIGMA,
                               CONTROL_SHARE=CONTROL_SHARE, P_WIN=P_WIN, MIN_SHOWN=MIN_SHOWN,
                               EDGE_TOL=list(EDGE_TOL), NESTED_EXTRA=NESTED_EXTRA),
                minimum_support=mins, groups=entries, log=str(args.log), added_recordings=len(src.uploads),
                left_out_boxes=int((~here).sum()))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "refiner.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    sets.to_csv(args.out / "refine_settings.csv", index=False)
    box_rows(T, Refiner(spec)).to_csv(args.out / "refine_boxes.csv", index=False)
    grid_rows(T).to_csv(args.out / "refine_grid.csv", index=False)

    print(f"{len(box)} pipeline yes boxes ({int(T.info['control'].sum()) if len(box) else 0} shown plain)")
    for r in sets.itertuples():
        extra = (f"  held-out IoU {r.loro_iou_grid:.3f} -> {r.loro_iou_app:.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}],"
                 f" refined {r.loro_coverage:.0%} of boxes, {r.loro_no_worse:.0%} no worse"
                 if np.isfinite(r.loro_iou_diff) else "")
        cut = "" if r.conf_min is None or pd.isna(r.conf_min) else f" conf >= {r.conf_min:.3f}"
        print(f"  {r.group:<14} ({r.delta:g} dB, rho {r.rho:g}, +{r.pad_t:g} s, +{r.pad_f:g} kHz){cut} "
              f"{r.decision:<6} {r.reason}{extra}")
    print(f"written to {args.out}; review_app.py reads refiner.json at start "
          f"(adopted: {', '.join(Refiner(spec).adopted()) or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
