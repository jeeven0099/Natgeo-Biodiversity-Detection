"""Local review app for the zero-shot boxes: listen, judge, name the species, log the decision.

    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\review_app.py            # opens http://127.0.0.1:8765
    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\review_app.py --device cpu --port 8766 --no-browser

The review queue of a recording mixes five sources (the page has a checkbox for each):
  model         Perch / EarthChirp zero-shot boxes at the viewer's thresholds (perch_viewer.THRESHOLDS),
                tightened to their cells (cell_maps.npy, kappa adjustable);
  near          runs the model would flag a little lower: segments in the next NEAR_BAND (5%) of all
                2.5 s segments by score below a lane's threshold, as runs with no segment at the
                threshold (so not the edges of a model box); tightened the same way;
  raven         Raven boxes not matched to a model box (raven_match.py: same group, time IoU of the
                matched model boxes >= 0.3, 1:1); raven_autolabel.py logs these as new boxes;
  reviewer      boxes the reviewer drew where the model had none ("new box"), rebuilt from the log;
  audit         AUDIT_PER_REC random AUDIT_S windows per recording (fixed seed): the reviewer draws a box
                around every call in the window, then states the window is complete for the ticked
                groups. That row goes to data/review/audit_windows.csv; only these windows measure
                recall without the model choosing where to look.
  box uncertain model and below-threshold boxes whose refined box (box_refine.py) is unstable: confidence
                below the fitted cut (0.5 before a fit); listed least confident first when ticked alone.
  taxon check   only for groups review_taxon.py adopted (the checkbox is hidden otherwise): runs of
                2.5 s segments where the taxon head's probability is >= its cut and no model box of the
                group covers the segment, on the group's default lane, tightened like model boxes;
                logged with model "taxon head". Answered time is never asked again: a lane's answered
                taxon-check spans are taken out of new runs (a refitted head's runs have other edges, so
                the same time would come back under a new id), and every answered item is listed from
                the log at its logged span. A recording uses the head that was live when it was added
                (review_taxon.PinnedHeads, data/review/taxon_versions/; the survey's: the first), so a
                retrained head proposes on new audio only. Model boxes of an adopted group also carry the head's
                probability (taxon_p, the highest over the box's 2.5 s segments), and taxon_doubt when
                it is below the cut (a tag on the page).
Learners (data/review/calibration/, read at start; --no-learners runs on zero-shot only):
  species.json + species_protos.npy (review_species.py): for an adopted species learner, up to 3
  learned names (prototypes of reviewed boxes, cosine >= tau) are listed above the zero-shot top 5,
  except a fixed 10% control share of boxes (crc32). Each species answer gets a row in
  data/review/species_shown.csv naming what was shown, so evaluation can use unanchored rows only.
  taxon.json + taxon.npz (review_taxon.py): the taxon head above. Nothing adopted = the app as before.
Box refiner: data/review/calibration/refiner.json (box_refine.py fits it). For groups it adopted, a
confident box opens on its refined box (dotted), except a fixed 10% shown plain as a control. Each
log row gets a row in data/review/box_shown.csv naming the box the reviewer started from.
For each box the app shows:
  * the spectrogram around it, with the model box, the tightened box, its kept cells and the Raven boxes;
  * the model's top-5 names for the box's group:
      Perch lanes      Perch's softmax probability, restricted to that group's classes, max over the
                       5 s windows overlapping the span (Perch is run on those windows on demand);
      EarthChirp lanes cosine of the cached Perch embedding to EarthChirp's species / genus /
                       group prototypes, max over the same windows;
  * playback of the box only: its time span and, by default, only its frequency band
    (perch_viewer.filter_clip); also the box's time at all frequencies, or its kept cells.
The reviewer answers yes / no / unknown (is there a sound of this group in the box?), ticks any of the
top 5 and/or adds other species (autocomplete over Perch, EarthChirp, Raven and earlier-added names;
free text allowed), may redraw the box by dragging on the spectrogram (or drag its edges), and
presses Accept. Every Accept appends one row to
data/review/annotation_log.csv (nothing is ever rewritten; the newest row of a box is its current
state). The log's `model` column names the box's source: Perch / EarthChirp (model), "Perch below
threshold" / "EarthChirp below threshold" (near), Raven, reviewer, "taxon head". No learning happens
here: the learners are fitted offline from the log and only read here.

Added audio (scan_audio.py): "add audio..." on the page, or "detect calls" in the editor, uploads a file
to data/uploads (--uploads-dir); one background worker runs the zero-shot chain on it (Perch, EarthChirp,
the cell head: the numbers behind the caches, checked by scan_audio.py --check) and the recording joins
the list with its model / below-threshold / audit / taxon-check items like any other. Added recordings
have no Raven boxes; their answers are logged as usual, but review_labels.py leaves them out of every
learner (the learners are fitted and tested on the dataset's recordings only).

Website (--public, behind the proxy in deploy/): anyone can upload audio at / (public_app.html) and see
the model boxes, their top 5 and playback for their own uploads only. A visitor is a guest: their files
go to data/uploads/guests/<folder>/ (folder = sha256 of the session cookie, which has no Max-Age), under
random ids, and exist only while their page is open: the page checks in every GUEST_PING_S (/api/alive)
and says when it closes (/api/leave); a session with no open page is ended GUEST_GRACE_S after its last
page closed (a reload keeps it), or GUEST_QUIET_S after anything was heard, and its folder is removed.
A guest's uploads are never listed for reviewers (every reviewer request naming one is not found),
never logged, never in prediction_cache.jsonl or deleted.csv, never flagged, never trained on. Uploads
from before guests keep their private key: the link /u/<key> and an HttpOnly cookie hold it. Visitors
cannot answer; on such an older upload they can flag a box ("this looks wrong": flags.csv, shown to
reviewers, never a label); they can delete any of their uploads (an older one: deleted.csv; the
learners skip it). Reviewers' own uploads (at /review/) are shared and kept, as in the local app.
The review app moves to /review/, which the proxy guards with a login; the reviewer name in the log
is the login's (REVIEWER_HEADER), not the page's field. Limits: --max-upload-mb, --max-upload-min,
--daily-uploads per address (counted in memory only), --disk-quota-gb, --min-free-gb. Without
--public the app is the local review app as before.

Reads segment_scores.csv, earthchirp_segments.csv, window_scores.csv, earthchirp_embeddings.npy,
cell_maps.npy, the Raven tables, the wavs, external/earthchirp_lib, config/*.csv and
data/review/calibration/. Writes only data/review/annotation_log.csv, box_shown.csv, species_shown.csv,
audit_windows.csv, flags.csv and prediction_cache.jsonl (box embeddings for the species learner stay in
memory), and added audio with its scans under data/uploads (guests' only while their page is open).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import queue
import secrets
import shutil
import sys
import threading
import time
import traceback
import uuid
import webbrowser
import zlib
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perch_viewer as pv  # noqa: E402
import zeroshot_windows as z  # noqa: E402
import insecta_compare as IC  # noqa: E402
import box_refine as BR  # noqa: E402
import box_embed as BE  # noqa: E402
import review_species as RS  # noqa: E402
import review_taxon as RT  # noqa: E402
import raven_match as RM  # noqa: E402
import scan_audio as SA  # noqa: E402
import frozen_test as FT  # noqa: E402

ROOT = pv.ROOT
PDIR = ROOT / "data" / "perch_windows"
REVIEW_DIR = ROOT / "data" / "review"
LOG_FILE = REVIEW_DIR / "annotation_log.csv"
CACHE_FILE = REVIEW_DIR / "prediction_cache.jsonl"
AUDIT_FILE = REVIEW_DIR / "audit_windows.csv"
SHOWN_FILE = REVIEW_DIR / "box_shown.csv"
SPECIES_SHOWN_FILE = REVIEW_DIR / "species_shown.csv"
HTML_FILE = Path(__file__).resolve().parent / "review_app.html"
COMMON_FILE = ROOT / "config" / "common_names.csv"
FROZEN_TEST = frozenset(FT.recordings())   # the frozen test recordings: not listed on the reviewer page
APP_VERSION = "5"            # 2: queue sources, reviewer boxes, audit windows; 3: box refiner, box_shown.csv;
                             # 4: species learner (species_shown.csv), taxon check source;
                             # 5: website (--public): reviewer name from the proxy's login, visitor flags
TOP_K = 5
CONTEXT_S = 2.0              # spectrogram shows the box +- this
MIN_VIEW_S, MAX_VIEW_S = 10.0, 120.0
MINUTE_S = 60.0              # minute view: the page shows a recording a minute at a time
SPEC_COLS = 3000             # spectrogram columns at most (hop grows with the view)
SPEC_WINDOWS = (256, 512, 1024, 2048, 4096)   # STFT sizes the page offers (window select); 1024 by default
SPEC_RANGE_DB = 120.0        # the PNG's grey 255 lies this far below the view's top; the page shows a part
VIEW_PLAY_S = 60.0           # 'play view': the longest stretch served in one go
MAX_PRED_WINDOWS = 64        # longer spans: evenly spaced windows
BATCH = 16                   # Perch windows per ONNX call
AUDIO_MARGIN_S = 0.25        # playback: audio filtered beyond the box each side, then cut off
FADE_S = 0.01                # playback: fade in/out at the box edges (no clicks)
SILENT_REL = 1e-4            # playback: filtered peak below this share of the raw peak plays as silence
WIN_S, HOP_S = z.WINDOW_S, pv.SEG_S
MARGIN_S = 1.0               # audio read around the windows before resampling (filter edges)
LOG_COLS = ["log_id", "logged_at", "reviewer", "recording", "lane", "model", "threshold",
            "box_begin_s", "box_end_s", "box_max_score", "kappa",
            "tight_begin_s", "tight_end_s", "tight_lo_khz", "tight_hi_khz",
            "accepted_begin_s", "accepted_end_s", "accepted_lo_khz", "accepted_hi_khz", "box_edited",
            "decision", "species", "species_source", "top5", "prediction_span", "comment", "app_version"]
DECISIONS = ("yes", "no", "unknown")
SOURCES = {"model": "model boxes", "near": "below threshold", "raven": "Raven, not matched",
           "reviewer": "reviewer boxes", "audit": "audit windows", "uncertain": "box uncertain",
           "taxon": "taxon check"}
TAXON_MODEL = "taxon head"           # the log's model for a box the taxon head found
DEFAULT_SOURCES = ("model", "reviewer")
# the review page's groups leave these out (2026-11-21: insects come from EarthChirp's cosine lane only;
# no Weather); the public page and the learners still have every lane
REVIEW_HIDDEN_LANES = ("Insecta", "EC-margin", "Weather")
REFINED_SOURCES = ("model", "near")    # boxes the refiner works on (the others are a person's box)
# box_shown.csv: one row per log row (joined on log_id) naming the box the reviewer started from:
# refined | tight | tight (control) | previous (the box's last accepted box) | raven | drawn
SHOWN_COLS = ["log_id", "logged_at", "box_id", "shown_box", "shown_begin_s", "shown_end_s", "shown_lo_khz",
              "shown_hi_khz", "refined_begin_s", "refined_end_s", "refined_lo_khz", "refined_hi_khz",
              "refine_conf", "refiner_version", "app_version"]
# species_shown.csv: one row per log row while the species learner is adopted: the learned names the
# reviewer saw above the zero-shot top 5 ("; "), and whether the box is a control (shown none)
SPECIES_SHOWN_COLS = ["log_id", "logged_at", "box_id", "learned_shown", "control", "species_version", "app_version"]
NEAR_BAND = 0.05             # near source: the next 5% of all segments by score below the threshold
AUDIT_S = 15.0               # audit window length
AUDIT_PER_REC = 4            # audit windows per recording (non-overlapping AUDIT_S slots)
AUDIT_SEED = 20261116
AUDIT_GROUPS = tuple(z.GROUPS)
AUDIT_STATUS = ("complete", "skipped")
AUDIT_COLS = ["audit_id", "logged_at", "reviewer", "recording", "window_begin_s", "window_end_s", "status",
              "groups", "n_reviewer_boxes", "comment", "app_version"]
SOURCE_ORDER = {s: k for k, s in enumerate(SOURCES)}
# Raven selection table export (the page's "export" button, /api/raven_table.txt, --export-raven)
RAVEN_DIR = ROOT / "dataset" / "annotaions"      # the Raven tables read as labels: never exported into
RAVEN_TABLE_COLS = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)",
                    "High Freq (Hz)", "Delta Time (s)", "Delta Freq (Hz)", "Annotation", "Group", "Decision",
                    "Source", "Reviewer", "Logged At", "Log ID"]
RAVEN_EXPORT = {"yes": ("yes",), "all": DECISIONS}   # which answers an export keeps
EDITOR_FILE = Path(__file__).resolve().parent / "review_editor.html"
UPLOADS_DIR = SA.UPLOADS_DIR                     # added audio and its scans (--uploads-dir)
# ---- website (--public): anyone uploads and sees the calls found in their own files; only invited
# reviewers (a login at the proxy, which passes their name in REVIEWER_HEADER under /review/) answer
PUBLIC_FILE = Path(__file__).resolve().parent / "public_app.html"
FLAGS_FILE = REVIEW_DIR / "flags.csv"            # visitors' "this looks wrong": a hint for reviewers, never a label
FLAG_COLS = ["flag_id", "flagged_at", "box_id", "recording", "lane", "begin_s", "end_s", "comment", "app_version"]
REVIEWER_HEADER = "X-Reviewer"                   # set by the proxy after its login; the proxy strips it elsewhere
REVIEW_PREFIX = "/review"                        # the reviewer pages and API live under it in --public mode
KEY_COOKIE = "uploads"                           # a visitor's private keys ('.'-joined), one per upload
MAX_KEYS = 40                                    # keys a cookie keeps (the newest)
COOKIE_AGE_S = 365 * 24 * 3600
MAX_JSON_BYTES = 64 * 1024                       # a JSON POST body
FLAGS_PER_DAY = 200                              # per address
MAX_COMMENT = 500
PUBLIC_MAX_VIEW_S = MAX_VIEW_S                   # a visitor's spectrogram / view audio: at most this long
HEAVY_AT_ONCE = 3                                # --public: spectrograms, audio and top 5s computed at once
# guests (--public): a visitor's uploads live in their own folder, only while their page is open. The
# session cookie (no Max-Age: the browser drops it on exit) names the session; on disk the folder is
# sha256(cookie)[:32], so the cookie is never written anywhere.
GUEST_COOKIE = "guest"
GUEST_DIR = "guests"                             # under the uploads folder: guests/<folder>/{audio,scans}
GUEST_PREFIX = "guest-"                          # a guest recording's id: this + 12 random hex digits
GUEST_QUIET_S = 600.0                            # a tab that has not checked in for this long is gone
GUEST_GRACE_S = 30.0                             # after a tab says it closed: time for a reload to check in
GUEST_REAP_S = 15.0                              # how often sessions are checked
GUEST_PING_S = 60                                # how often the pages check in (sent to them in /api/meta)
GUEST_MAX_TABS = 20                              # tab ids a session keeps (the newest)


def model_of(lane: str) -> str:
    return "EarthChirp" if lane in pv.EC_GROUPS else "Perch"


def lane_group(lane: str) -> str:
    """The taxonomic group of a lane: the EarthChirp lanes are insects."""
    return "Insecta" if lane in pv.EC_GROUPS else lane


def item_id(rec: str, model: str, lane: str, b0: float, b1: float, lo: float = 0.0, hi: float = 0.0) -> str:
    """Raven and reviewer boxes are named by their box (their group can be changed when reviewed);
    model and below-threshold boxes by their group and span."""
    if model in ("Raven", "reviewer"):
        return f"{rec}|{model}|{b0:.3f}|{b1:.3f}|{lo:.3f}|{hi:.3f}"
    return f"{rec}|{model}|{lane}|{b0:.3f}|{b1:.3f}"


def row_id(r: dict) -> str:
    """The id of the box a log row belongs to."""
    lohi = ((float(r["tight_lo_khz"]), float(r["tight_hi_khz"])) if r["model"] in ("Raven", "reviewer")
            else (0.0, 0.0))
    return item_id(r["recording"], r["model"], r["lane"], float(r["box_begin_s"]), float(r["box_end_s"]), *lohi)


def audit_id(rec: str, a: float, b: float) -> str:
    return f"{rec}|audit|{a:.3f}|{b:.3f}"


def audit_windows(rec: str, duration: float) -> list[tuple[float, float]]:
    """AUDIT_PER_REC random non-overlapping AUDIT_S slots of a recording, the same on every start
    (seeded by AUDIT_SEED and the recording name)."""
    n = int(duration // AUDIT_S)
    rng = np.random.default_rng([AUDIT_SEED, zlib.crc32(rec.encode())])
    k = sorted(rng.choice(n, min(AUDIT_PER_REC, n), replace=False)) if n else []
    return [(float(s * AUDIT_S), float((s + 1) * AUDIT_S)) for s in k]


def read_rows(path: Path) -> list[dict]:
    """A CSV log's rows in file order (none when it does not exist yet)."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def raven_file(rec: str) -> str:
    """The name Raven gives a recording's first selection table."""
    return f"{rec}.Table.1.selections.txt"


def raven_cell(x) -> str:
    """One field of a tab-separated table: tabs and line breaks become single spaces."""
    return " ".join(str(x).split())


def fixed(x: float, d: int) -> str:
    """x to d decimals, with no '-0.000' from a value that rounds to zero."""
    return f"{round(x, d) + 0.0:.{d}f}"


def raven_table(rows: list[dict], rec: str, keep: str = "yes") -> tuple[str, int]:
    """A Raven selection table of one recording's reviewed boxes, and how many selections it has.
    Each box is its newest log row (its current state) as its accepted box, kept when its answer is
    in RAVEN_EXPORT[keep], by begin time. Annotation = the species named ("; "), else the group.
    Tab-separated with CRLF line ends and the leading columns of the tables in dataset/annotaions."""
    if keep not in RAVEN_EXPORT:
        raise ValueError(f"keep must be one of {tuple(RAVEN_EXPORT)}")
    latest = {}
    for r in rows:
        if r["recording"] == rec:
            latest[row_id(r)] = r
    acc = ("accepted_begin_s", "accepted_end_s", "accepted_lo_khz", "accepted_hi_khz")
    sel = sorted((r for r in latest.values() if r["decision"] in RAVEN_EXPORT[keep] and all(r[k] for k in acc)),
                 key=lambda r: (*(float(r[k]) for k in acc), r["log_id"]))
    out = ["\t".join(RAVEN_TABLE_COLS)]
    for i, r in enumerate(sel, 1):
        b0, b1, lo, hi = (float(r[k]) for k in acc)
        lo, hi = lo * 1000, hi * 1000
        out.append("\t".join(raven_cell(x) for x in (
            i, "Spectrogram 1", 1, fixed(b0, 3), fixed(b1, 3), fixed(lo, 1), fixed(hi, 1), fixed(b1 - b0, 3),
            fixed(hi - lo, 1), r["species"] or r["lane"], r["lane"], r["decision"], r["model"], r["reviewer"],
            r["logged_at"], r["log_id"])))
    return "\r\n".join(out) + "\r\n", len(sel)


def export_raven(out_dir: Path, keep: str = "yes") -> dict[str, int]:
    """--export-raven: one selection table per recording with a kept box in the log; reads the log only."""
    out_dir = out_dir.resolve()
    if out_dir == RAVEN_DIR.resolve():
        raise SystemExit(f"{rel(out_dir)} holds the Raven tables the app reads as labels: export elsewhere")
    rows = read_rows(LOG_FILE)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for rec in dict.fromkeys(r["recording"] for r in rows):
        txt, n = raven_table(rows, rec, keep)
        if n:
            (out_dir / raven_file(rec)).write_bytes(txt.encode("utf-8"))
            counts[rec] = n
    return counts


def score_or_none(x) -> float | None:
    """JSON has no NaN: missing scores are null."""
    return None if x is None or x == "" or not np.isfinite(float(x)) else float(x)


def windows_for_span(starts: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Indices of the 5 s windows overlapping [t0, t1] by at least min(2.5 s, span length);
    the best-overlapping window when none does. At most MAX_PRED_WINDOWS, evenly spaced."""
    ov = np.minimum(t1, starts + WIN_S) - np.maximum(t0, starts)
    k = np.flatnonzero(ov >= min(HOP_S, t1 - t0) - 1e-6)
    if not len(k):
        k = np.array([int(np.argmax(ov))])
    if len(k) > MAX_PRED_WINDOWS:
        k = k[np.unique(np.linspace(0, len(k) - 1, MAX_PRED_WINDOWS).round().astype(int))]
    return k


def resampled_slice(path: str, t0: float, t1: float) -> tuple[np.ndarray, float]:
    """Mono audio at 32 kHz covering [t0 - MARGIN_S, t1 + MARGIN_S]; returns (x, start_s of x).
    The read starts on a sample that maps onto the 32 kHz grid, so inside the margin the result
    equals resampling the whole file (zeroshot_windows.load_windows)."""
    from scipy.signal import resample_poly
    info = sf.info(path)
    sr = info.samplerate
    g = int(np.gcd(sr, z.PERCH_SR))
    up, down = z.PERCH_SR // g, sr // g
    a = max(0, int(np.floor((t0 - MARGIN_S) * sr / down)) * down)
    b = min(info.frames, int(np.ceil((t1 + MARGIN_S) * sr)))
    x, _ = sf.read(path, start=a, stop=b, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != z.PERCH_SR:
        x = resample_poly(x, up, down).astype(np.float32)
    return x, a / sr


class Store:
    """Everything the app reads, loaded once."""

    def __init__(self, device: str, refiner: BR.Refiner | None = None, species: RS.SpeciesLearner | None = None,
                 taxon: RT.TaxonHead | None = None, pins: RT.PinnedHeads | None = None):
        self.device = device
        self.refiner = refiner                 # None: no refined boxes, no uncertain queue
        self.species = species or RS.SpeciesLearner()   # suggests names only when adopted
        self.taxon_head = taxon or RT.TaxonHead()           # taxon check source only for adopted groups
        self.pins = pins if pins is not None and pins.rows else None   # per recording: the head live when added
        self._ref: dict[tuple[str, float], dict] = {}
        self._ref_key = None                   # the (recording, kappa) the background refiner works on
        self.seg = pv.load_segments(PDIR)
        self.raven = pv.load_raven()
        self.wavs = pv.wav_paths()
        self.cells = pv.CellMaps(PDIR) if pv.CELL_FILE.exists() else None
        self.lanes = [g for g in pv.VIEW_GROUPS if f"prob_{g}" in self.seg]
        self.thresholds = {g: pv.THRESHOLDS[g] for g in self.lanes}
        self.default_on = [g for g in pv.DEFAULT_ON if g in self.lanes]
        # near source: per lane, the score below which the next NEAR_BAND of all segments starts
        self.near_low = {}
        for g in self.lanes:
            s = self.seg[f"prob_{g}"].to_numpy()
            self.near_low[g] = float(np.quantile(s, max(0.0, 1 - (s >= self.thresholds[g]).mean() - NEAR_BAND)))
        self.recs = [r for r in pd.unique(self.seg["recording"]) if r in self.wavs]
        self.seg_by_rec = {r: g for r, g in self.seg.groupby("recording", sort=False)}
        w = pd.read_csv(PDIR / "window_scores.csv",
                        usecols=["recording", "start_s"] + [f"prob_{g}" for g in z.GROUPS])
        codes, names = pd.factorize(w["recording"])
        first = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
        self.win_rows = {r: (a, b - a) for r, a, b in zip(names, first, np.r_[first[1:], len(w)])}
        self.win_starts = w["start_s"].to_numpy()
        self.win_probs = w[[f"prob_{g}" for g in z.GROUPS]].to_numpy()
        emb = PDIR / "earthchirp_embeddings.npy"
        self.E = np.load(emb, mmap_mode="r") if emb.exists() else None
        if self.E is not None and len(self.E) != len(w):
            raise SystemExit(f"{emb.name} has {len(self.E)} rows, window_scores.csv {len(w)}")
        # added recordings' window rows go behind the caches' (scan_audio.RowStack), so every reader
        # indexing by win_rows keeps working; the cell maps the same way
        self.can_add = self.E is not None and self.cells is not None and len(self.cells.M) == len(w)
        self.scan_missing = SA.missing_files() if self.can_add else []
        if self.scan_missing:
            print("adding audio is off: a scan needs " + ", ".join(rel(p) for p in self.scan_missing), flush=True)
            self.can_add = False
        if self.can_add:
            self.E = SA.RowStack(self.E)
            self.cells.M = SA.RowStack(self.cells.M)
        # learners: the taxon head reads the cached window embeddings; the species learner pools Perch's
        # cells over the box (box_embed.py), from the windows the top 5 ran on or a run of its own
        if self.E is None:
            self.taxon_head, self.pins = RT.TaxonHead(), None
        self.twin = RT.Windows(self.E, self.win_rows, self.win_starts, self.seg_by_rec)
        self.embedder = BE.BoxEmbedder(
            run=lambda rec, st: self.perch_run(rec, st, ["spatial_embedding"])["spatial_embedding"],
            starts={r: self.win_starts[a:a + n] for r, (a, n) in self.win_rows.items()})
        self._learned: dict[str, list[dict]] = {}
        self._emb_lock = threading.Lock()
        # Perch classes and groups
        self.labels, self.groups = IC.labels_and_groups()
        idx = pd.read_csv(ROOT / "config" / "perch_class_index.csv", keep_default_na=False)
        self.kind, self.ebird, self.taxon = idx["kind"].tolist(), idx["ebird"].tolist(), idx["class"].tolist()
        # English common names (scripts/build_common_names.py): scientific name, lower case -> (name, others)
        self.common: dict[str, tuple[str, str]] = {}
        if COMMON_FILE.exists():
            cn = pd.read_csv(COMMON_FILE, keep_default_na=False)
            self.common = {s.lower(): (c, o) for s, c, o in
                           zip(cn["scientific_name"], cn["common_name"], cn["other_english"], strict=True)}
        # EarthChirp prototypes
        lib = ROOT / "external" / "earthchirp_lib"
        self.ec_P = IC.l2(np.load(lib / "global_reflib_protos.npy"))
        ec_idx = pd.read_parquet(lib / "global_reflib_index.parquet")
        self.ec_names, self.ec_levels = ec_idx["name"].tolist(), ec_idx["level"].tolist()
        self._sess = None
        self._lock = threading.Lock()
        self._boxes: dict[tuple[str, float], list[dict]] = {}
        self._rec_info: dict[str, dict] = {}
        self._pred: dict[str, dict] = {}
        self._log_lock = threading.Lock()
        REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        if CACHE_FILE.exists():
            for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self._pred[d["key"]] = d
        self.vocab = self._vocab()
        self.box_counts = {r: {g: int(n) for g, n in
                               pv.perch_boxes(self.seg_by_rec[r], self.thresholds)["group"].value_counts().items()}
                           for r in self.recs}
        # added audio (scan_audio.py): start_uploads() (the server) registers the scanned files and
        # starts the one worker that scans the rest in turn; a Store built by a script has none
        self.uploads = SA.Uploads(UPLOADS_DIR)
        self.added: dict[str, dict] = {}
        self.jobs: dict[str, dict] = {}
        self._scanner = None
        self._scan_q: queue.Queue = queue.Queue()
        self._add_lock = threading.Lock()          # registration (rows appended, dicts extended)
        self._upload_lock = threading.Lock()       # one upload names itself at a time
        self._del_lock = threading.Lock()          # a delete and a finished scan's save + registration
        self._uploads_on = False
        self.key_rec: dict[str, str] = {}          # private key -> added recording (the website's links)
        self.rec_key: dict[str, str] = {}
        # guests (--public, start_uploads(guests=True)): each session's uploads in its own folder
        self.guests_on = False
        self.sessions: dict[str, dict] = {}        # folder name -> {uploads, recs, tabs, seen, left, busy}
        self.guest_of: dict[str, str] = {}         # guest recording -> folder name (kept after a delete)
        self._guest_lock = threading.Lock()

    # ---- added audio (scan_audio.py) ------------------------------------------------------
    def add_off_reason(self) -> str:
        if self.scan_missing:
            return "a scan needs " + ", ".join(rel(p) for p in self.scan_missing)
        return "the app runs without earthchirp_embeddings.npy / a full cell_maps.npy"

    def start_uploads(self, guests: bool = False) -> None:
        """Starts the scan worker and loads the uploads folder; guests (--public): also the guest
        folders a last run left, and the thread that ends quiet guest sessions."""
        if self._uploads_on:
            return
        self._uploads_on = True
        threading.Thread(target=self._scan_worker, name="scan-worker", daemon=True).start()
        if guests:
            self.guests_on = True
            self._load_guests()
            threading.Thread(target=self._guest_reaper, name="guest-reaper", daemon=True).start()
        recs = self.uploads.recordings()
        if not recs:
            return
        if not self.can_add:
            print(f"{len(recs)} added recordings in {rel(UPLOADS_DIR)} not shown: {self.add_off_reason()}", flush=True)
            return
        n_scan = 0
        for rec, info in recs.items():
            if rec in self.wavs:
                print(f"added recording {rec} has a dataset recording's name: skipped", flush=True)
                continue
            self._set_key(rec, info.get("key"))
            try:
                if self.uploads.scan_ok(rec, info):
                    self._register(rec, info, SA.load_scan(self.uploads.scan_file(rec)))
                    continue
            except Exception:
                traceback.print_exc()
            self._queue(rec, info)
            n_scan += 1
        print(f"added recordings: {len(self.added)} loaded from {rel(UPLOADS_DIR)}"
              + (f", {n_scan} to scan" if n_scan else ""), flush=True)

    def _queue(self, rec: str, info: dict, uploads: SA.Uploads | None = None) -> dict:
        """Queues a scan; uploads: the folder the recording lives in (a guest's; default the shared one)."""
        job = {"rec": rec, "name": info["name"], "state": "queued", "done": 0,
               "total": int(np.ceil(info["duration_s"] / SA.HOP_S - 1e-9)) or 1, "error": "", "counts": {},
               "queued_at": dt.datetime.now().isoformat(timespec="seconds")}
        with self._add_lock:
            self.jobs[rec] = job
        self._scan_q.put((rec, info, uploads or self.uploads))
        return job

    def scanner(self) -> SA.Scanner:
        if self._scanner is None:
            self._scanner = SA.Scanner(perch=self.session())     # shares the top 5's Perch session
            print(f"scanner ready ({self._scanner.providers()})", flush=True)
        return self._scanner

    def _scan_worker(self) -> None:
        while True:
            rec, info, up = self._scan_q.get()
            job = self.jobs.get(rec)
            with self._add_lock:
                if job is None or job["state"] == "deleted":    # deleted while queued
                    self._deleted_while_scanning(rec, info, up)
                    continue
                job["state"] = "running"
            try:
                def progress(d, t, job=job):
                    job["done"], job["total"] = d, t
                scan = self.scanner().scan(info["path"], progress)
                with self._del_lock:
                    if job["state"] == "deleted":      # deleted while scanning: files the delete could not
                        self._deleted_while_scanning(rec, info, up)   # remove (open on Windows) go now
                        continue
                    SA.save_scan(up.scan_file(rec), scan, {"sha256": info["sha256"], "rec": rec})
                    self._register(rec, info, scan)
                job.update(state="done", counts=self.box_counts[rec], seconds=scan["meta"]["seconds"])
                print(("scanned a guest upload" if rec in self.guest_of else f"scanned {rec}")
                      + f": {len(scan['start_s'])} windows in {scan['meta']['seconds']} s, model boxes "
                      f"{sum(self.box_counts[rec].values())}", flush=True)
            except Exception as e:
                if job["state"] == "deleted":          # the file went away under the scan
                    self._deleted_while_scanning(rec, info, up)
                    continue
                traceback.print_exc()
                job.update(state="error", error=f"{type(e).__name__}: {e}")

    def _deleted_while_scanning(self, rec: str, info: dict, up: SA.Uploads) -> None:
        """A recording deleted while queued or scanning: nothing is saved, files left open go now; a
        guest's job is forgotten, and so is the folder once its session has ended."""
        up.purge(rec, info)
        h = self.guest_of.get(rec)
        if h is not None:
            self.jobs.pop(rec, None)
            with self._guest_lock:
                ended = h not in self.sessions
            if ended:
                up.remove_all()

    def _register(self, rec: str, info: dict, scan: dict) -> None:
        """Makes a scanned file a recording like the dataset's: its rows go behind the caches', then
        every per-recording table learns it; the list (self.recs) last, so nothing half-made is shown."""
        n = len(scan["start_s"])
        seg = SA.segments(scan, rec)
        counts = {g: int(c) for g, c in pv.perch_boxes(seg, self.thresholds)["group"].value_counts().items()}
        with self._add_lock:
            if rec in self.added or self.jobs.get(rec, {}).get("state") == "deleted":
                return
            a = len(self.win_starts)
            if len(self.E) != a or len(self.cells.M) != a:
                raise RuntimeError(f"window rows out of step: {len(self.E)} / {len(self.cells.M)} / {a}")
            self.E.append(scan["emb"])
            self.cells.M.append(scan["cells"])
            self.win_starts = np.r_[self.win_starts, scan["start_s"]]
            self.win_probs = np.vstack([self.win_probs, scan["prob"]])
            self.twin.win_starts = self.win_starts
            self.win_rows[rec] = (a, n)              # the same dict as self.twin.win_rows
            self.cells.rows[rec] = (a, n)
            self.seg_by_rec[rec] = seg               # the same dict as self.twin.seg_by_rec
            self.embedder.starts[rec] = scan["start_s"]
            self.wavs[rec] = info["path"]
            self.box_counts[rec] = counts
            self.added[rec] = {"name": info["name"], "file": rel(Path(info["path"])), "added_at": info["added_at"]}
            self.recs.append(rec)

    def taken_names(self) -> set[str]:
        """Names a new file may not take (lower case): dataset and added recordings, and any recording
        the review folder's logs or prediction cache name."""
        return {r.lower() for r in self.wavs} | SA.logged_names(REVIEW_DIR)

    def _set_key(self, rec: str, key: str | None) -> None:
        if key:
            with self._add_lock:
                self.key_rec[key], self.rec_key[rec] = rec, key

    def add_upload(self, read, length: int, name: str, limit: int | None = SA.MAX_UPLOAD_BYTES,
                   max_duration_s: float | None = None, via: str = "local",
                   too_big: str = "add longer files with scan_audio.py") -> dict:
        """POST /api/upload: stores the file, queues its scan; -> {rec, created, key, job}. Every new
        upload gets a private key (the website's link to it); the same bytes again give the key of the
        recording they already are (made now if that recording has none)."""
        if not (self.can_add and self._uploads_on):
            raise ValueError(f"adding audio is off on this server: {self.add_off_reason()}")
        with self._upload_lock:
            rec, info, created = self.uploads.receive(
                read, length, name, self.taken_names(), limit=limit, max_duration_s=max_duration_s,
                extra={"key": secrets.token_urlsafe(18), "via": via}, too_big=too_big)
            if not info.get("key"):
                info = {**info, **self.uploads.update_info(rec, key=secrets.token_urlsafe(18))}
            self._set_key(rec, info["key"])
            job = self.jobs.get(rec)
            if rec in self.added:
                job = job or {"rec": rec, "name": info["name"], "state": "done", "done": 1, "total": 1,
                              "error": "", "counts": self.box_counts[rec]}
            elif job is None or job["state"] == "error":
                job = self._queue(rec, info)
        return {"rec": rec, "created": created, "key": info["key"], "job": dict(job)}

    def queue_position(self, rec: str) -> int | None:
        """1 + the scans queued or running ahead of a recording's (None when it is not waiting)."""
        ahead = 0
        for r, j in list(self.jobs.items()):         # insertion order = the queue's order
            if r == rec:
                return ahead + 1 if j["state"] in ("queued", "running") else None
            ahead += j["state"] in ("queued", "running")
        return None

    def delete_upload(self, rec: str) -> dict:
        """An added recording's owner deletes it: files removed, every table forgets it (_forget); a
        queued or running scan is dropped. A shared upload's delete is logged in deleted.csv
        (scan_audio.Uploads.delete): log rows naming it stay (append-only), and the learners leave out
        recordings in deleted.csv. A guest's upload leaves no trace (no row anywhere)."""
        with self._del_lock:
            with self._add_lock:
                if rec not in self.added and rec not in self.jobs:
                    raise KeyError(rec)
            self._forget(rec)
            h = self.guest_of.get(rec)
            if h is None:
                gone, left = self.uploads.delete(rec, "owner")
            else:
                with self._guest_lock:
                    s = self.sessions.get(h)
                    if s is not None and rec in s["recs"]:
                        s["recs"].remove(rec)
                gone, left = SA.Uploads(self.guest_root(h)).delete(rec, log=False)
        return {"rec": rec, "deleted": True, "removed": gone, "left": left}

    def _forget(self, rec: str) -> None:
        """Every table forgets an added recording (the caller holds _del_lock), and the window rows its
        scan appended are freed (RowStack.release: they read as zeros), so a deleted upload holds no
        memory until the next start. A queued or running scan is marked deleted (the worker drops it)."""
        with self._add_lock:
            j = self.jobs.get(rec)
            if j is not None:
                busy = j["state"] in ("queued", "running")
                j["state"] = "deleted"
                if rec in self.guest_of and not busy:     # a guest's finished job: nothing waits on it
                    self.jobs.pop(rec, None)
            rows = self.win_rows.get(rec) if rec in self.added else None
            for d in (self.added, self.wavs, self.box_counts, self.seg_by_rec, self.win_rows,
                      self.embedder.starts, self._rec_info, self.twin._rec):
                d.pop(rec, None)
            if self.cells is not None:
                self.cells.rows.pop(rec, None)
            if rec in self.recs:
                self.recs.remove(rec)
            key = self.rec_key.pop(rec, None)
            self.key_rec.pop(key, None)
            if rows is not None:
                self.E.release(*rows)
                self.cells.M.release(*rows)
        with self._lock:
            self._boxes = {k: v for k, v in self._boxes.items() if k[0] != rec}
            self._pred = {k: v for k, v in self._pred.items() if k.split("|")[0] != rec}
            self._ref = {k: v for k, v in self._ref.items() if k[0].split("|")[0] != rec}
            self._learned = {k: v for k, v in self._learned.items() if k.split("|")[1:2] != [rec]}
        with self._emb_lock:
            for k in [k for k in self.embedder._grids if k[0] == rec]:
                self.embedder._grids.pop(k, None)

    # ---- guests (--public): uploads kept only while the guest's page is open ---------------
    @staticmethod
    def guest_folder(sid: str) -> str:
        """A session's folder name: the cookie itself is never written to disk."""
        return hashlib.sha256(sid.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def guest_root(h: str) -> Path:
        return UPLOADS_DIR / GUEST_DIR / h

    def _new_session(self, h: str) -> dict:
        """A session for folder h (the caller holds _guest_lock or runs before the server)."""
        s = {"folder": h, "uploads": SA.Uploads(self.guest_root(h)), "recs": [], "tabs": {},
             "seen": time.monotonic(), "left": None, "busy": 0}
        self.sessions[h] = s
        return s

    def _load_guests(self) -> None:
        """Guest folders a last run left: each is a session again, with a fresh GUEST_QUIET_S, so a
        guest whose page stayed open checks in with the same cookie and keeps their files; the others
        are removed when that time is up."""
        base = UPLOADS_DIR / GUEST_DIR
        n_rec = n_scan = 0
        dirs = sorted(d for d in base.iterdir() if d.is_dir()) if base.exists() else []
        for d in dirs:
            if len(d.name) != 32 or any(c not in "0123456789abcdef" for c in d.name):
                continue                               # not a session folder: left alone
            with self._guest_lock:
                s = self._new_session(d.name)
            for rec, info in s["uploads"].recordings().items():
                if not rec.startswith(GUEST_PREFIX) or rec in self.wavs:
                    continue
                with self._guest_lock:
                    s["recs"].append(rec)
                    self.guest_of[rec] = d.name
                n_rec += 1
                if not self.can_add:
                    continue
                try:
                    if s["uploads"].scan_ok(rec, info):
                        self._register(rec, info, SA.load_scan(s["uploads"].scan_file(rec)))
                        continue
                except Exception:
                    traceback.print_exc()
                self._queue(rec, info, s["uploads"])
                n_scan += 1
        if dirs:
            print(f"guest sessions from the last run: {len(self.sessions)} ({n_rec} uploads, {n_scan} to scan); "
                  f"each is removed {GUEST_QUIET_S / 60:g} min from now unless its page checks in", flush=True)

    def guest_session(self, sid: str | None, create: bool = False) -> tuple[str | None, dict | None]:
        """-> (sid, session) for a cookie's sid, or (None, None). create: a new session (and sid) when
        the cookie names none; a sid the server did not make is never adopted."""
        with self._guest_lock:
            s = self.sessions.get(self.guest_folder(sid)) if sid else None
            if s is None and create:
                sid = secrets.token_urlsafe(24)
                s = self._new_session(self.guest_folder(sid))
            return (sid if s is not None else None), s

    def guest_recs(self, sid: str | None) -> list[str]:
        """A session's uploads, oldest first (none for an unknown or ended session)."""
        with self._guest_lock:
            s = self.sessions.get(self.guest_folder(sid)) if sid else None
            return list(s["recs"]) if s is not None else []

    def guest_ping(self, sid: str | None, tab: str, leave: bool = False) -> bool:
        """POST /api/alive (leave False) or /api/leave from one of the session's tabs -> the session
        is still there."""
        tab = str(tab)[:64]
        with self._guest_lock:
            s = self.sessions.get(self.guest_folder(sid)) if sid else None
            if s is None:
                return False
            now = time.monotonic()
            if leave:
                s["tabs"].pop(tab, None)
                s["left"] = now
            elif tab:
                s["tabs"].pop(tab, None)
                s["tabs"][tab] = now
                while len(s["tabs"]) > GUEST_MAX_TABS:
                    s["tabs"].pop(next(iter(s["tabs"])))
                s["seen"] = now
            return True

    def reap_guests(self, now: float | None = None) -> int:
        """Ends the sessions no page keeps open: no tab checked in within GUEST_QUIET_S, no upload
        arriving, and either a tab said it closed over GUEST_GRACE_S ago or nothing was heard for
        GUEST_QUIET_S. Their uploads are forgotten and their folder removed. -> sessions ended."""
        now = time.monotonic() if now is None else now
        ended = []
        with self._guest_lock:
            for h, s in list(self.sessions.items()):
                s["tabs"] = {t: v for t, v in s["tabs"].items() if now - v <= GUEST_QUIET_S}
                if s["busy"] or s["tabs"]:
                    continue
                left = s["left"] is not None and now - max(s["left"], s["seen"]) > GUEST_GRACE_S
                if left or now - s["seen"] > GUEST_QUIET_S:
                    del self.sessions[h]
                    ended.append(s)
        for s in ended:
            with self._del_lock:
                for rec in s["recs"]:
                    self._forget(rec)
                s["uploads"].remove_all()          # a file still open (a scan) goes when the worker drops it
            print(f"guest session ended: {len(s['recs'])} upload(s) removed", flush=True)
        return len(ended)

    def _guest_reaper(self) -> None:
        while True:
            time.sleep(GUEST_REAP_S)
            try:
                self.reap_guests()
            except Exception:
                traceback.print_exc()

    def add_guest_upload(self, sid: str | None, tab: str, read, length: int, name: str, limit: int | None,
                         max_duration_s: float | None, too_big: str) -> tuple[str, dict]:
        """POST /api/upload from the website: the file goes to the guest's own folder under a random id
        (a new session when the cookie names none) -> (sid, {rec, created, key None, job}). Nothing of
        it reaches the shared folder, the logs or the learners; no other visitor or reviewer sees it."""
        if not (self.can_add and self._uploads_on and self.guests_on):
            raise ValueError(f"adding audio is off on this server: {self.add_off_reason()}")
        sid0 = sid
        sid, s = self.guest_session(sid, create=True)
        with self._guest_lock:
            s["busy"] += 1                         # the session is not ended while the file arrives
        try:
            self.guest_ping(sid, tab)
            taken = {r.lower() for r in list(self.wavs)} | {r.lower() for r in list(self.guest_of)}
            rec, info, created = s["uploads"].receive(
                read, length, name, taken, limit=limit, max_duration_s=max_duration_s, extra={"via": "guest"},
                too_big=too_big, rec_id=GUEST_PREFIX + secrets.token_hex(6))
            with self._guest_lock:
                if rec not in s["recs"]:
                    s["recs"].append(rec)
                self.guest_of[rec] = s["folder"]
            job = self.jobs.get(rec)
            if rec in self.added:
                job = job or {"rec": rec, "name": info["name"], "state": "done", "done": 1, "total": 1,
                              "error": "", "counts": self.box_counts[rec]}
            elif job is None or job["state"] in ("error", "deleted"):
                job = self._queue(rec, info, s["uploads"])
            return sid, {"rec": rec, "created": created, "key": None, "job": dict(job)}
        except BaseException:
            if sid != sid0:                        # refused, and the session was made for it: no cookie
                with self._guest_lock:             # names it, so it ends now (not after GUEST_QUIET_S)
                    gone = self.sessions.get(s["folder"]) is s and not s["recs"]
                    if gone:
                        del self.sessions[s["folder"]]
                if gone:
                    s["uploads"].remove_all()
            raise
        finally:
            with self._guest_lock:
                s["busy"] -= 1
                s["seen"] = time.monotonic()
            self.guest_ping(sid, tab)

    def is_guest(self, *refs) -> bool:
        """Does any of these (a recording, a box id 'rec|...', 'new|rec') name a guest's recording?"""
        return any(p in self.guest_of for r in refs if r for p in str(r).split("|"))

    # ---- visitor flags (--public) ----------------------------------------------------------
    def append_flag(self, body: dict) -> dict:
        """A visitor's "this looks wrong" on a model box: one row in FLAGS_FILE. Reviewers see the count;
        no learner reads it. A guest's boxes cannot be flagged: no reviewer ever sees them."""
        d = self.box(str(body.get("id", "")), float(body.get("kappa", pv.KAPPA)))
        if d["recording"] in self.guest_of:
            raise ValueError("this upload is yours only: reviewers do not see it, so it cannot be flagged")
        row = {"flag_id": uuid.uuid4().hex[:12], "flagged_at": dt.datetime.now().isoformat(timespec="seconds"),
               "box_id": d["id"], "recording": d["recording"], "lane": d["lane"], "begin_s": d["begin_s"],
               "end_s": d["end_s"], "comment": " ".join(str(body.get("comment", "")).split())[:MAX_COMMENT],
               "app_version": APP_VERSION}
        with self._log_lock:
            new = not FLAGS_FILE.exists()
            with FLAGS_FILE.open("a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FLAG_COLS)
                if new:
                    w.writeheader()
                w.writerow(row)
        return {"flag_id": row["flag_id"], "id": d["id"]}

    def flag_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in read_rows(FLAGS_FILE):
            out[r["box_id"]] = out.get(r["box_id"], 0) + 1
        return out

    # ---- queue items (boxes of every source, and audit windows) ---------------------------
    def items(self, rec: str, kappa: float) -> list[dict]:
        """Every queue item of a recording, all sources, by begin time; n numbers them."""
        if rec not in self.seg_by_rec:
            raise KeyError(rec)
        key = (rec, round(kappa, 3))
        with self._lock:
            fixed = self._boxes.get(key)
        if fixed is None:
            fixed = (self._model_items(rec, kappa, near=False) + self._model_items(rec, kappa, near=True)
                     + self._raven_items(rec) + self._audit_items(rec) + self._taxon_items(rec, kappa))
            with self._lock:
                self._boxes = {k: v for k, v in self._boxes.items() if k[0] == rec}   # keep one recording
                self._boxes[key] = fixed
        if self.refiner is not None and self._ref_key != key:
            self._start_refining(key, [d for d in fixed if d["source"] in REFINED_SOURCES])
        out = sorted(fixed + self._reviewer_items(rec),
                     key=lambda d: (d["begin_s"], SOURCE_ORDER[d["source"]], d["lane"], d["end_s"]))
        return [{**d, "n": i + 1} for i, d in enumerate(out)]

    def boxes(self, rec: str, kappa: float) -> list[dict]:
        """The model boxes only (the first source)."""
        return [d for d in self.items(rec, kappa) if d["source"] == "model"]

    # ---- box refiner (box_refine.py) -----------------------------------------------------
    def _start_refining(self, key: tuple[str, float], todo: list[dict]) -> None:
        """Refines a recording's model and below-threshold boxes in the background (~70 ms each),
        so the list can show confidence and the uncertain queue; stops when another recording or
        kappa is opened. A box opened before its turn is refined on the spot."""
        self._ref_key = key

        def run():
            for d in todo:
                if self._ref_key != key:
                    return
                try:
                    self.refinement(d)
                except Exception:                  # one bad box must not stop the rest
                    traceback.print_exc()
        threading.Thread(target=run, daemon=True).start()

    def refinement(self, d: dict, compute: bool = True) -> dict | None:
        """The refiner's result for a model / below-threshold box (refined, conf, show, uncertain,
        shown_box, ...); None for other boxes, with the refiner off, or (compute=False) not done yet."""
        if self.refiner is None or d["source"] not in REFINED_SOURCES:
            return None
        k = (d["id"], round(float(d["kappa"]), 3))
        r = self._ref.get(k)
        if r is None and compute:
            P = BR.make_patch(self.wavs[d["recording"]], (d["t_begin"], d["t_end"], d["lo_khz"], d["hi_khz"]))
            r = self.refiner.apply(P, lane_group(d["lane"]))     # fitted per group, as review_labels writes it
            r["shown_box"] = ("tight (control)" if BR.is_control(d["id"]) else "refined") if r["show"] else "tight"
            self._ref[k] = r
        return r

    def shown_of(self, d: dict | None, previous: dict | None) -> tuple[str, list[float], dict | None]:
        """(shown_box, the box shown, refinement) of the box a reviewer starts from: the box's last
        accepted box if it has a log row, else the refined box when the refiner shows it, else the
        tightened (Raven, first drawn) box."""
        if d is None:
            return "drawn", [], None
        ref = self.refinement(d)
        tight = [d["t_begin"], d["t_end"], d["lo_khz"], d["hi_khz"]]
        if previous is not None:
            return "previous", [float(previous[f"accepted_{e}"]) for e in ("begin_s", "end_s", "lo_khz", "hi_khz")], ref
        if d["source"] in REFINED_SOURCES:
            kind = ref["shown_box"] if ref else "tight"
            return kind, (list(ref["refined"]) if kind == "refined" else tight), ref
        # audit windows open on the whole window and never log a box row (they close via /api/audit)
        return {"raven": "raven", "reviewer": "drawn", "audit": "window", "taxon": "tight"}[d["source"]], tight, ref

    def _item(self, rec: str, source: str, model: str, lane: str, b0: float, b1: float,
              tight: list[float] | None = None, kept=None, max_score=None, kappa=None,
              label: str = "", area: float = 1.0) -> dict:
        t = [float(x) for x in (tight or [b0, b1, *pv.PERCH_BAND])]
        d = {"id": (audit_id(rec, b0, b1) if source == "audit" else item_id(rec, model, lane, b0, b1, t[2], t[3])),
             "source": source, "recording": rec, "lane": lane, "model": model,
             "predictor": model_of(lane) if lane else None,
             "threshold": self.thresholds.get(lane), "begin_s": float(b0), "end_s": float(b1),
             "max_score": score_or_none(max_score), "kappa": kappa, "tightened": kept is not None,
             "t_begin": t[0], "t_end": t[1], "lo_khz": t[2], "hi_khz": t[3], "area": float(area),
             "label": label, "lane_editable": source in ("raven", "reviewer"), "_kept": kept,
             "taxon_p": None, "taxon_doubt": False}
        d["kept_rects"] = self._kept_rects(d)
        return d

    def _model_items(self, rec: str, kappa: float, near: bool) -> list[dict]:
        """Model boxes at the listening thresholds; with near, the runs at the lower near_low cut
        that hold no segment at the threshold."""
        head = self.head_of(rec)
        pb = pv.perch_boxes(self.seg_by_rec[rec], self.near_low if near else self.thresholds)
        if near:
            pb = pb[pb["max_prob"] < pb["group"].map(self.thresholds)]
        pb = pb.sort_values(["begin_s", "group"]).reset_index(drop=True)
        if self.cells is not None and len(pb):
            pb = self.cells.tighten_boxes(rec, pb, kappa)
        out = []
        for b in pb.itertuples():
            tight = self.cells is not None and np.isfinite(b.t_begin)
            model = model_of(b.group) + (" below threshold" if near else "")
            out.append(self._item(rec, "near" if near else "model", model, b.group, b.begin_s, b.end_s,
                                  [b.t_begin, b.t_end, b.lo_khz, b.hi_khz] if tight else None,
                                  np.asarray(b.kept, bool) if tight else None, b.max_prob, kappa,
                                  area=float(b.area) if tight else 1.0))
            g = lane_group(b.group)
            if not near and g in head.groups:                # taxon check: the head doubts this model box
                d = out[-1]
                p = head.span_prob(self.twin, g, rec, d["t_begin"], d["t_end"])
                d["taxon_p"], d["taxon_doubt"] = round(p, 4), p < head.cut(g)
        return out

    def head_of(self, rec: str) -> RT.TaxonHead:
        """The taxon head a recording's taxon check uses: with pinned heads, the one live when the
        recording was added (the survey's: the first), so a retrained head proposes on new audio only."""
        if self.pins is None:
            return self.taxon_head
        return self.pins.head_for(self.added.get(rec, {}).get("added_at"))

    def taxon_adopted(self) -> list[str]:
        """Groups any head in use adopted (the taxon check source is listed when there is one)."""
        heads = self.pins.heads() if self.pins is not None else [self.taxon_head]
        return sorted({g for h in heads for g in h.adopted()})

    def _answered_taxon_spans(self, rec: str) -> dict[str, list[tuple[float, float]]]:
        """lane -> the spans of the recording's logged taxon-check items (any answer, unknown included),
        first logged first."""
        out: dict[str, list[tuple[float, float]]] = {}
        seen = set()
        for r in self.log_rows():
            if r["recording"] == rec and r["model"] == TAXON_MODEL and row_id(r) not in seen:
                seen.add(row_id(r))
                out.setdefault(r["lane"], []).append((float(r["box_begin_s"]), float(r["box_end_s"])))
        return out

    def _taxon_boxes(self, rec: str, kappa: float, lane: str, rows: list[tuple]) -> list[dict]:
        """Taxon-check items from (begin_s, end_s, max_prob, head p or None) rows, tightened like model boxes."""
        if not rows:
            return []
        fb = pd.DataFrame([(lane, b0, b1, mp, round((b1 - b0) / pv.SEG_S), hp) for b0, b1, mp, hp in rows],
                          columns=["group", "begin_s", "end_s", "max_prob", "n_segments", "head"])
        if self.cells is not None:
            fb = self.cells.tighten_boxes(rec, fb, kappa)
        out = []
        for b in fb.itertuples():
            tight = self.cells is not None and np.isfinite(b.t_begin)
            head = None if b.head is None or not np.isfinite(b.head) else float(b.head)
            d = self._item(rec, "taxon", TAXON_MODEL, lane, b.begin_s, b.end_s,
                           [b.t_begin, b.t_end, b.lo_khz, b.hi_khz] if tight else None,
                           np.asarray(b.kept, bool) if tight else None, b.max_prob, kappa,
                           label=f"p={head:.2f}" if head is not None else "", area=float(b.area) if tight else 1.0)
            d["taxon_p"] = round(head, 4) if head is not None else None
            out.append(d)
        return out

    def _taxon_items(self, rec: str, kappa: float) -> list[dict]:
        """Taxon check, for groups the taxon head was adopted for (review_taxon.py): runs of 2.5 s
        segments the head scores >= its cut and no model box of the group's lanes covers (calls the
        thresholds miss; model-box time is taken out of a run, not the run dropped, since a group
        like Aves can score high for most of a recording), on the group's default lane, tightened
        like model boxes. The time of the lane's answered taxon-check items is taken out the same way,
        so a refitted head, whose runs have other edges, never asks about answered time again; each
        answered item is listed from the log (its logged span, any group, adopted or not), with the
        recording's head's probability over it where the group is adopted. The head is head_of(rec):
        with pinned heads, the one live when the recording was added."""
        answered = self._answered_taxon_spans(rec)
        head = self.head_of(rec)
        out = []
        for g in head.adopted():
            lane = head.lane(g)
            if lane not in self.lanes or rec not in self.win_rows:
                continue
            st, sc, p = head.segments(self.twin, g, rec)
            pb = pv.perch_boxes(self.seg_by_rec[rec], {x: self.thresholds[x] for x in self.lanes if lane_group(x) == g})
            spans = ([(float(a), float(b)) for a, b in zip(pb["begin_s"], pb["end_s"])]
                     + answered.get(lane, []))
            b0s = np.array([a for a, _ in spans], float)
            b1s = np.array([b for _, b in spans], float)
            covered = np.array([((b0s < a + pv.SEG_S - 1e-6) & (b1s > a + 1e-6)).any() for a in st], bool)
            rows = []
            for i, j in RT.runs((p >= head.cut(g)) & ~covered):
                rows.append((float(st[i]), float(st[j - 1] + pv.SEG_S), float(np.nanmax(sc[i:j])), float(p[i:j].max())))
            out += self._taxon_boxes(rec, kappa, lane, rows)
        latest = self.latest()
        for lane, spans in answered.items():
            if lane not in self.lanes:
                continue
            g = lane_group(lane)
            rows = []
            for b0, b1 in spans:
                r = latest[item_id(rec, TAXON_MODEL, lane, b0, b1)]
                hp = (head.span_prob(self.twin, g, rec, b0, b1)
                      if g in head.groups and rec in self.win_rows else None)
                rows.append((b0, b1, score_or_none(r["box_max_score"]), hp))
            out += self._taxon_boxes(rec, kappa, lane, rows)
        return out

    def raven_lanes(self, group: str) -> list[str]:
        """The lanes whose boxes count as finding a Raven box of this group: the default (hybrid)
        lanes of the group, else every lane of it; Unknown: every default lane."""
        if group == "Unknown":
            return list(self.default_on)
        return ([g for g in self.default_on if lane_group(g) == group]
                or [g for g in self.lanes if lane_group(g) == group])

    def raven_boxes(self, rec: str) -> list[dict]:
        """A recording's distinct Raven boxes by begin time (a box in two tables counts once, the first
        label kept): key (rounded edges), group, label, begin_s, end_s, lo_khz, hi_khz."""
        r = self.raven[self.raven["recording"] == rec].sort_values(["begin_s", "end_s"], kind="stable")
        seen, out = set(), []
        for x in r.itertuples():
            key = (round(x.begin_s, 3), round(x.end_s, 3), round(x.low_khz, 3), round(x.high_khz, 3))
            if key not in seen:
                seen.add(key)
                out.append(dict(key=key, group=x.group, label=x.label, begin_s=float(x.begin_s),
                                end_s=float(x.end_s), lo_khz=float(x.low_khz), hi_khz=float(x.high_khz)))
        return out

    def raven_plan(self, rec: str) -> list[dict]:
        """raven_match.match of a recording's Raven boxes against its model boxes (the model source, at
        the listening thresholds; the 2.5 s segment runs, so kappa does not change the match)."""
        pb = pv.perch_boxes(self.seg_by_rec[rec], self.thresholds)
        model = [dict(id=item_id(rec, model_of(b.group), b.group, b.begin_s, b.end_s), lane=b.group,
                      begin_s=float(b.begin_s), end_s=float(b.end_s)) for b in pb.itertuples()]
        return RM.match(model, self.raven_boxes(rec), self.raven_lanes)

    def _raven_items(self, rec: str) -> list[dict]:
        """Raven boxes not matched to a model box (raven_match.py): new boxes, below the IoU cut,
        and the labels it skips (Unknown)."""
        matched = {p["raven"]["key"] for p in self.raven_plan(rec) if p["kind"] != "new"}
        out = []
        for x in self.raven_boxes(rec):
            if x["key"] in matched:
                continue
            lanes = self.raven_lanes(x["group"])
            lane = lanes[0] if len(lanes) == 1 and x["group"] != "Unknown" else ""
            out.append(self._item(rec, "raven", "Raven", lane, x["begin_s"], x["end_s"],
                                  [x["begin_s"], x["end_s"], x["lo_khz"], x["hi_khz"]],
                                  max_score=self.seg_max(rec, lane, x["begin_s"], x["end_s"]), label=x["label"]))
        return out

    def _audit_items(self, rec: str) -> list[dict]:
        top = sf.info(self.wavs[rec]).samplerate / 2000
        return [self._item(rec, "audit", "audit", "", a, b, [a, b, 0.0, top])
                for a, b in audit_windows(rec, self.duration(rec))]

    def _reviewer_items(self, rec: str) -> list[dict]:
        """Boxes the reviewer drew, from the log: the box as first drawn, the group as last logged."""
        first, last = {}, {}
        for r in self.log_rows():
            if r["recording"] == rec and r["model"] == "reviewer":
                i = row_id(r)
                first.setdefault(i, r)
                last[i] = r
        out = []
        for i, r in first.items():
            t = [float(r[k]) for k in ("tight_begin_s", "tight_end_s", "tight_lo_khz", "tight_hi_khz")]
            out.append(self._item(rec, "reviewer", "reviewer", last[i]["lane"], float(r["box_begin_s"]),
                                  float(r["box_end_s"]), t, max_score=last[i]["box_max_score"]))
        return out

    def seg_max(self, rec: str, lane: str, a: float, b: float) -> float | None:
        """A lane's highest 2.5 s segment score over [a, b] (how close the model came)."""
        if not lane:
            return None
        s = self.seg_by_rec[rec]
        m = (s["start_s"] < b) & (s["start_s"] + pv.SEG_S > a)
        return float(s.loc[m, f"prob_{lane}"].max()) if m.any() else None

    def lookup(self, q: dict) -> dict:
        """The item a request names: id=new with rec and lane is a box being drawn (not logged yet);
        lane= on a Raven or reviewer box asks about another group."""
        if q["id"] == "new":
            rec, lane = q.get("rec", ""), q.get("lane", "")
            if rec not in self.seg_by_rec or rec not in self.wavs:
                raise ValueError(f"unknown recording {rec!r}")
            if lane and lane not in self.lanes:
                raise ValueError(f"unknown group {lane!r}")
            return {**self._item(rec, "reviewer", "reviewer", lane, 0.0, 0.0), "id": f"new|{rec}"}
        d = self.box(q["id"], float(q.get("kappa", pv.KAPPA)))
        lane = q.get("lane")
        if lane and lane != d["lane"]:
            if not d["lane_editable"] or lane not in self.lanes:
                raise ValueError(f"the group of this box cannot be {lane!r}")
            d = {**d, "lane": lane, "predictor": model_of(lane), "threshold": self.thresholds[lane]}
        return d

    @staticmethod
    def _kept_rects(d: dict) -> list[list[float]]:
        """Kept cells as [t0, t1, lo_khz, hi_khz] runs, one list per row."""
        kept = d["_kept"]
        if kept is None:
            return []
        out = []
        for r in range(4):
            on = np.r_[0, kept[:, r].astype(np.int8), 0]
            e = np.diff(on)
            for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
                out.append([d["t_begin"] + a * pv.CELL_S, d["t_begin"] + b * pv.CELL_S,
                            pv.ROW_EDGES_KHZ[r], pv.ROW_EDGES_KHZ[r + 1]])
        return out

    def box(self, bid: str, kappa: float) -> dict:
        rec = bid.split("|")[0]
        for d in self.items(rec, kappa):
            if d["id"] == bid:
                return d
        raise KeyError(bid)

    def duration(self, rec: str) -> float:
        return float(sf.info(self.wavs[rec]).duration)

    def rec_info(self, rec: str) -> dict:
        """Length and top frequency of a recording (the minute view needs them with no box open)."""
        i = self._rec_info.get(rec)
        if i is None:
            f = sf.info(self.wavs[rec])
            i = self._rec_info[rec] = {"duration_s": float(f.duration), "fmax_khz": f.samplerate / 2000}
        return i

    def view(self, d: dict) -> tuple[float, float]:
        """Default spectrogram range for a box: the model box +- CONTEXT_S, 10-MAX_VIEW_S s
        (the page can pan and zoom from there)."""
        dur = self.duration(d["recording"])
        a, b = d["begin_s"] - CONTEXT_S, d["end_s"] + CONTEXT_S
        if b - a < MIN_VIEW_S:
            c = (a + b) / 2
            a, b = c - MIN_VIEW_S / 2, c + MIN_VIEW_S / 2
        if b - a > MAX_VIEW_S:
            b = a + MAX_VIEW_S
        if a < 0:
            a, b = 0.0, b - a
        return max(0.0, a), min(dur, b)

    def raven_in(self, rec: str, a: float, b: float) -> list[dict]:
        r = self.raven[(self.raven["recording"] == rec) & (self.raven["end_s"] > a) & (self.raven["begin_s"] < b)]
        return [{"begin_s": float(x.begin_s), "end_s": float(x.end_s), "lo_khz": float(x.low_khz),
                 "hi_khz": float(x.high_khz), "label": x.label, "group": x.group} for x in r.itertuples()]

    # ---- predictions ---------------------------------------------------------------------
    def session(self):
        with self._lock:
            if self._sess is None:
                model = ROOT / "external/bacpipe_model_checkpoints/perch_v2/perch_v2_no_dft.onnx"
                try:
                    self._sess = IC.open_session(model, self.device)
                except SystemExit:
                    print("CUDA unavailable: Perch runs on the CPU", flush=True)
                    self._sess = IC.open_session(model, "cpu")
            return self._sess

    def perch_run(self, rec: str, starts: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
        """Perch's outputs `names` for the 5 s windows starting at `starts`."""
        starts = np.asarray(starts, float)
        x, x0 = resampled_slice(self.wavs[rec], float(starts.min()), float(starts.max()) + WIN_S)
        n = int(round(WIN_S * z.PERCH_SR))
        wins = np.zeros((len(starts), n), np.float32)
        for i, s in enumerate(starts):
            o = int(round((s - x0) * z.PERCH_SR))
            seg = x[o:o + n]
            wins[i, :len(seg)] = seg
        # fixed batches of BATCH (zero rows pad the last): cuDNN finds no plan for some small batches;
        # on the CPU one window per call is fastest (box_embed.py)
        sess = self.session()
        bs = 1 if "CUDAExecutionProvider" not in sess.get_providers() else BATCH
        out = {k: [] for k in names}
        for k in range(0, len(wins), bs):
            xb = np.zeros((bs, n), np.float32)
            m = len(wins[k:k + bs])
            xb[:m] = wins[k:k + bs]
            for name, v in zip(names, sess.run(names, {"inputs": IC.normalize_windows(xb, 0.25)}), strict=True):
                out[name].append(np.asarray(v)[:m])
        return {k: np.concatenate(v) for k, v in out.items()}

    def perch_logits(self, rec: str, starts: np.ndarray) -> np.ndarray:
        """Perch's logits; with the species learner adopted, the cell embeddings of the same run go to
        the box embedder, so the learned names need no second run."""
        if not self.species.adopted:
            return self.perch_run(rec, starts, ["label"])["label"].astype(np.float64)
        r = self.perch_run(rec, starts, ["label", "spatial_embedding"])
        with self._emb_lock:
            self.embedder.put(rec, starts, r["spatial_embedding"].astype(np.float32))
        return r["label"].astype(np.float64)

    def learned(self, d: dict, t0: float, t1: float, lo: float | None, hi: float | None) -> dict:
        """The species learner's names for a box (review_species.py): {learned, learned_control,
        species_version}. Nothing when not adopted, for a control box, or without the band."""
        out = {"learned": [], "learned_control": False, "species_version": self.species.version}
        g = lane_group(d["lane"])
        if not self.species.adopted or g not in self.species.groups() or lo is None or hi is None:
            return out
        ck = d["id"] if not str(d["id"]).startswith("new|") else f"{d['recording']}|{t0:.2f}|{t1:.2f}"
        if RS.is_control(ck):
            return {**out, "learned_control": True}
        box = [round(float(v), 3) for v in (t0, t1, lo, hi)]
        if not (box[0] < box[1] and box[2] < box[3]):
            return out
        key = f"{g}|{BE.box_key(d['recording'], box)}"
        if key not in self._learned:
            with self._emb_lock:
                emb, _, _ = self.embedder.embed(d["recording"], box)
            self._learned[key] = [{**s, "common": self.common_of(s["name"])} for s in self.species.suggest(g, emb)]
        return {**out, "learned": self._learned[key]}

    def predict(self, d: dict, t0: float, t1: float) -> dict:
        if not d["lane"]:
            raise ValueError("pick the box's group first")
        if not t0 < t1:
            raise ValueError("the span needs t0 < t1")
        key = f"{d['id']}|{d['lane']}|{t0:.2f}|{t1:.2f}"
        with self._lock:
            if key in self._pred:
                return self._pred[key]
        rec, lane = d["recording"], d["lane"]
        a, n = self.win_rows[rec]
        starts = self.win_starts[a:a + n]
        k = windows_for_span(starts, t0, t1)
        res = {"key": key, "box": d["id"], "lane": lane, "model": d["predictor"], "span": [t0, t1],
               "windows": [float(s) for s in starts[k]]}
        if d["predictor"] == "Perch":
            lg = self.perch_logits(rec, starts[k])
            p = np.exp(lg - lg.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            # self-check: the group sums equal the cached window scores
            got = np.stack([p[:, self.groups[g]].sum(1) for g in z.GROUPS], 1)
            res["check_max_diff"] = float(np.abs(got - self.win_probs[a + k]).max())
            cls = self.groups[lane]
            pc = p[:, cls]
            best = pc.max(0)
            order = np.argsort(-best)[:TOP_K]
            res["top"] = [self._perch_entry(cls[j], best[j], starts[k][int(pc[:, j].argmax())]) for j in order]
            # Perch's own top over all 14,795 classes, for boxes whose group is wrong
            ball = p.max(0)
            res["top_all"] = [self._perch_entry(j, ball[j], starts[k][int(p[:, j].argmax())])
                              for j in np.argsort(-ball)[:TOP_K]]
            res["score_name"] = "Perch probability (max over windows)"
        else:
            if self.E is None:
                raise RuntimeError("earthchirp_embeddings.npy is missing")
            c = IC.l2(np.asarray(self.E[a + k], np.float32)) @ self.ec_P.T
            best = c.max(0)
            order = np.argsort(-best)[:TOP_K]
            res["top"] = [{"name": self.ec_names[j], "score": float(best[j]),
                           "at_s": float(starts[k][int(c[:, j].argmax())]),
                           "kind": self.ec_levels[j], "ebird": "", "taxon": "Insecta"} for j in order]
            res["top_all"] = []
            res["score_name"] = "EarthChirp cosine (max over windows)"
        with self._lock:
            self._pred[key] = res
            if rec not in self.guest_of:              # a guest's top 5s stay in memory only
                with CACHE_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(res) + "\n")
        return res

    def common_of(self, name: str) -> str:
        return self.common.get(str(name).lower(), ("", ""))[0]

    def with_common(self, res: dict) -> dict:
        """A prediction with each top-5 entry's English common name ('' if none). Added when served,
        not stored, so cached predictions get it too."""
        return {**res, **{k: [{**t, "common": self.common_of(t["name"])} for t in res.get(k, [])]
                          for k in ("top", "top_all")}}

    def _perch_entry(self, j: int, score: float, at_s: float) -> dict:
        return {"name": self.labels[j], "score": float(score), "at_s": float(at_s), "kind": self.kind[j],
                "ebird": self.ebird[j], "taxon": self.taxon[j]}

    # ---- species vocabulary --------------------------------------------------------------
    def _vocab(self) -> list[dict]:
        seen, out = set(), []

        def add(name, source, note=""):
            name = str(name).strip()
            if name and (name.lower(), source) not in seen:
                seen.add((name.lower(), source))
                common, others = self.common.get(name.lower(), ("", ""))
                out.append({"name": name, "source": source, "note": note, "common": common, "others": others})
        for n, kd, cl, eb in zip(self.labels, self.kind, self.taxon, self.ebird):
            add(n, "Perch", " ".join(x for x in (cl or kd, eb) if x))
        for n, lv in zip(self.ec_names, self.ec_levels):
            add(n, "EarthChirp", lv)
        for n in sorted(self.raven["label"].unique()):
            add(n, "Raven")
        tx = ROOT / "config" / "taxonomy.csv"
        if tx.exists():
            for n in pd.read_csv(tx, keep_default_na=False)["label_raw"]:
                add(n, "Raven")
        tg = ROOT / "config" / "perch_targets.csv"
        if tg.exists():
            for q in pd.read_csv(tg, keep_default_na=False)["query"]:
                for n in q.split(";"):
                    add(n, "target list")
        for r in self.log_rows():
            for n, s in zip(r["species"].split("; "), r["species_source"].split("; ")):
                if s == "added":
                    add(n, "added earlier")
        return out

    def search(self, q: str, limit: int = 25) -> list[dict]:
        q = q.strip().lower()
        if not q:
            return []
        hits = [v for v in self.vocab if any(q in v[k].lower() for k in ("name", "note", "common", "others"))]
        hits.sort(key=lambda v: (not (v["name"].lower().startswith(q) or v["common"].lower().startswith(q)),
                                 not (q in v["name"].lower() or q in v["common"].lower()), len(v["name"])))
        return hits[:limit]

    # ---- log -----------------------------------------------------------------------------
    def log_rows(self) -> list[dict]:
        return read_rows(LOG_FILE)

    def latest(self) -> dict[str, dict]:
        """box id -> its newest log row."""
        return {row_id(r): r for r in self.log_rows()}

    def audit_rows(self) -> list[dict]:
        return read_rows(AUDIT_FILE)

    def audit_latest(self) -> dict[str, dict]:
        """audit window id -> its newest row."""
        return {audit_id(r["recording"], float(r["window_begin_s"]), float(r["window_end_s"])): r
                for r in self.audit_rows()}

    def append(self, body: dict, shown_as: str | None = None) -> dict:
        """One log row. body["new"]: a box the reviewer drew (body["recording"], body["lane"]);
        else body["id"] names a queue box. Raven and reviewer boxes take body["lane"] as their group.
        shown_as: box_shown.csv's shown_box for rows written by raven_autolabel.py (never from the page)."""
        if body.get("new"):
            rec, d = str(body.get("recording", "")), None
            if rec not in self.seg_by_rec or rec not in self.wavs:
                raise ValueError(f"unknown recording {rec!r}")
        else:
            d = self.box(body["id"], float(body["kappa"]))
            rec = d["recording"]
            if d["source"] == "audit":
                raise ValueError("an audit window is closed with /api/audit; draw a new box to log a call")
        if body.get("decision") not in DECISIONS:
            raise ValueError("decision must be yes, no or unknown")
        acc = [float(body[k]) for k in ("begin_s", "end_s", "lo_khz", "hi_khz")]
        if not (acc[0] < acc[1] and acc[2] < acc[3]):
            raise ValueError("box must have begin < end and low < high")
        if d is None or d["lane_editable"]:
            lane = str(body.get("lane") or (d["lane"] if d else ""))
            if lane not in self.lanes:
                raise ValueError("pick the box's group")
        else:
            lane = d["lane"]
        if d is None:            # the drawn box is the reviewer's proposal and its first answer
            model, kappa = "reviewer", ""
            b0, b1 = round(acc[0], 3), round(acc[1], 3)
            tight = [round(x, 3) for x in acc]
        else:
            model, kappa, b0, b1 = d["model"], d["kappa"] if d["kappa"] is not None else "", d["begin_s"], d["end_s"]
            tight = [d["t_begin"], d["t_end"], d["lo_khz"], d["hi_khz"]]
        if d is None or d["lane_editable"]:
            threshold, score = self.thresholds[lane], self.seg_max(rec, lane, b0, b1)
        else:
            threshold, score = d["threshold"], d["max_score"]
        species = [s for s in body.get("species", []) if s.get("name", "").strip()]
        shown, shown_box, ref = self.shown_of(d, self.latest().get(d["id"]) if d else None)
        shown = shown_as or shown
        if shown == "drawn" and d is None:
            shown_box = [round(x, 3) for x in acc]
        row = {"log_id": uuid.uuid4().hex[:12], "logged_at": dt.datetime.now().isoformat(timespec="seconds"),
               "reviewer": str(body.get("reviewer", "")).strip(), "recording": rec, "lane": lane,
               "model": model, "threshold": threshold, "box_begin_s": b0,
               "box_end_s": b1, "box_max_score": "" if score is None else round(score, 6), "kappa": kappa,
               "tight_begin_s": tight[0], "tight_end_s": tight[1], "tight_lo_khz": tight[2], "tight_hi_khz": tight[3],
               "accepted_begin_s": round(acc[0], 3), "accepted_end_s": round(acc[1], 3),
               "accepted_lo_khz": round(acc[2], 3), "accepted_hi_khz": round(acc[3], 3),
               "box_edited": int(not np.allclose(acc, tight, atol=1e-3)),
               "decision": body["decision"],
               "species": "; ".join(s["name"].strip() for s in species),
               "species_source": "; ".join(s.get("source", "added") for s in species),
               "top5": " | ".join(f"{t['name']}={t['score']:.4f}" for t in body.get("top", [])),
               "prediction_span": body.get("prediction_span", ""),
               "comment": str(body.get("comment", "")).replace("\n", " ").strip(), "app_version": APP_VERSION}
        rb = (ref or {}).get("refined") or [""] * 4
        side = {"log_id": row["log_id"], "logged_at": row["logged_at"], "box_id": row_id(row), "shown_box": shown,
                **dict(zip(SHOWN_COLS[4:8], shown_box, strict=True)), **dict(zip(SHOWN_COLS[8:12], rb, strict=True)),
                "refine_conf": ref["conf"] if ref else "",
                "refiner_version": self.refiner.version if self.refiner is not None else "",
                "app_version": APP_VERSION}
        files = [(LOG_FILE, LOG_COLS, row), (SHOWN_FILE, SHOWN_COLS, side)]
        if self.species.adopted:          # the learned names the reviewer saw (none: an unanchored row)
            shown_names = [str(s.get("name", "")).strip() for s in body.get("learned", []) or []]
            files.append((SPECIES_SHOWN_FILE, SPECIES_SHOWN_COLS, {
                "log_id": row["log_id"], "logged_at": row["logged_at"], "box_id": row_id(row),
                "learned_shown": "; ".join(n for n in shown_names if n),
                "control": int(bool(body.get("learned_control"))), "species_version": self.species.version,
                "app_version": APP_VERSION}))
        with self._log_lock:
            for path, cols, r in files:
                new = not path.exists()
                with path.open("a", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=cols)
                    if new:
                        w.writeheader()
                    w.writerow(r)
        for s in species:
            if s.get("source") == "added":
                n = s["name"].strip()
                if not any(v["name"].lower() == n.lower() for v in self.vocab):
                    common, others = self.common.get(n.lower(), ("", ""))
                    self.vocab.append({"name": n, "source": "added earlier", "note": "",
                                       "common": common, "others": others})
        return {**row, "id": row_id(row), "shown_box": shown}

    def relabel(self, body: dict) -> dict:
        """A model box whose call is of another group (the page's group buttons): the two rows a reviewer
        would log by hand, No on the model box and the box drawn anew with the other group. The new box
        (model reviewer) carries body's answer, species, top 5 and comment; the model box's row is No,
        with no species, the same accepted box, and a comment naming the new box."""
        d = self.box(body["id"], float(body["kappa"]))
        lane = str(body.get("lane", ""))
        if d["lane_editable"] or d["source"] == "audit":
            raise ValueError("only a model box changes its group this way (a Raven or reviewer box takes lane=)")
        if lane not in self.lanes or lane == d["lane"]:
            raise ValueError(f"the call's group must be another group than {d['lane']!r}, not {lane!r}")
        new = self.append({**body, "new": True, "recording": d["recording"], "lane": lane})
        c = str(body.get("comment", "")).strip()
        no = self.append({**{k: body[k] for k in ("kappa", "begin_s", "end_s", "lo_khz", "hi_khz")},
                          "reviewer": body.get("reviewer", ""), "id": d["id"], "decision": "no", "species": [], "top": [], "prediction_span": "",
                          "learned": [], "comment": f"the call is {lane}: logged as new box {new['log_id']}"
                          + (f" · {c}" if c else "")})
        return {**new, "model_row": no}

    def append_audit(self, body: dict) -> dict:
        """Closes an audit window: status complete (every call of the ticked groups has a box) or
        skipped (cannot be marked fully). One row in AUDIT_FILE; the newest row of a window wins."""
        d = self.box(body["id"], float(body.get("kappa", pv.KAPPA)))
        if d["source"] != "audit":
            raise ValueError("not an audit window")
        status = body.get("status")
        if status not in AUDIT_STATUS:
            raise ValueError("status must be complete or skipped")
        groups = [g for g in AUDIT_GROUPS if g in body.get("groups", [])]
        if status == "complete" and not groups:
            raise ValueError("tick the groups whose every call you marked")
        a, b = d["begin_s"], d["end_s"]
        n = sum(1 for r in self.latest().values()
                if r["recording"] == d["recording"] and r["model"] == "reviewer" and r["decision"] == "yes"
                and float(r["accepted_begin_s"]) < b and float(r["accepted_end_s"]) > a)
        row = {"audit_id": uuid.uuid4().hex[:12], "logged_at": dt.datetime.now().isoformat(timespec="seconds"),
               "reviewer": str(body.get("reviewer", "")).strip(), "recording": d["recording"],
               "window_begin_s": a, "window_end_s": b, "status": status,
               "groups": "; ".join(groups) if status == "complete" else "", "n_reviewer_boxes": n,
               "comment": str(body.get("comment", "")).replace("\n", " ").strip(), "app_version": APP_VERSION}
        with self._log_lock:
            new = not AUDIT_FILE.exists()
            with AUDIT_FILE.open("a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=AUDIT_COLS)
                if new:
                    w.writeheader()
                w.writerow(row)
        return {**row, "id": d["id"]}

    # ---- media ---------------------------------------------------------------------------
    def spectrogram_png(self, rec: str, a: float, b: float, n_fft: int = 1024) -> tuple[bytes, float]:
        """8-bit grey PNG, 0 Hz at the bottom: 0 = the view's 99.9th percentile, 255 = SPEC_RANGE_DB
        below it or quieter. The page colours it (brightness, contrast, colour map). The hop grows with
        the view but never past the window, so no audio falls between frames; when that gives more
        than SPEC_COLS frames, each column is the loudest of its run of frames (short clicks stay)."""
        from PIL import Image
        if n_fft not in SPEC_WINDOWS:
            raise ValueError(f"window must be one of {SPEC_WINDOWS}")
        x, sr = pv.read_page(self.wavs[rec], a, b - a)
        hop = min(n_fft, max(128, int(np.ceil(len(x) / SPEC_COLS))))
        f, t, db = pv.spectrogram(x, sr, n_fft=n_fft, hop=hop)
        k = int(np.ceil(db.shape[1] / SPEC_COLS))
        if k > 1:
            db = np.pad(db, ((0, 0), (0, -db.shape[1] % k)), mode="edge").reshape(len(db), -1, k).max(2)
        vmax = np.percentile(db, 99.9)
        g = (np.clip((vmax - db) / SPEC_RANGE_DB, 0, 1) * 255).astype(np.uint8)[::-1]
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(g), "L").save(buf, format="PNG", compress_level=3)
        return buf.getvalue(), float(f[-1])

    def audio_wav(self, d: dict, t0: float, t1: float, lo: float, hi: float, pad: float, mode: str) -> bytes:
        """Audio of [t0 - pad, t1 + pad] (capped at MAX_CLIP_S), and nothing outside it.
        'band-passed': only lo..hi kHz (the drawn box). 'cell mask': only the kept cells, and of those
        only the part inside lo..hi (no cell map: band-passed). 'original': every frequency.
        The filter runs on AUDIO_MARGIN_S more audio each side, which is then cut off, so the STFT
        edges fall outside the clip; FADE_S raised-cosine fades stop clicks at the cut."""
        a = max(0.0, t0 - pad)
        e = min(t1 + pad, a + pv.MAX_CLIP_S)
        ra = max(0.0, a - AUDIO_MARGIN_S)
        x, sr = pv.read_page(self.wavs[d["recording"]], ra, e + AUDIO_MARGIN_S - ra)
        band, peak0 = SimpleNamespace(lo_khz=lo, hi_khz=hi), float(np.abs(x).max()) if len(x) else 0.0
        if mode == "cell mask" and d["_kept"] is not None:
            x = pv.filter_clip(x, sr, ra, SimpleNamespace(t_begin=d["t_begin"], kept=d["_kept"]), mode)
            x = pv.filter_clip(x, sr, ra, band, "band-passed")
        elif mode in ("band-passed", "cell mask"):
            x = pv.filter_clip(x, sr, ra, band, "band-passed")
        i0 = int(round((a - ra) * sr))
        x = x[i0:i0 + int(round((e - a) * sr))].copy()
        if not len(x) or np.abs(x).max() <= SILENT_REL * peak0:
            x[:] = 0                     # nothing passes (e.g. no kept cell in the band): silence, not
                                         # filter residue peak-normalised up to full scale
        return pv.wav_bytes(faded(x, sr), sr)

    def view_wav(self, rec: str, a: float, b: float) -> bytes:
        """'play view': the recording's audio of [a, b], all frequencies, at most VIEW_PLAY_S."""
        if rec not in self.wavs:
            raise ValueError(f"unknown recording {rec!r}")
        a = max(0.0, a)
        b = min(b, a + VIEW_PLAY_S)
        if not a < b:
            raise ValueError("the range needs a < b")
        x, sr = pv.read_page(self.wavs[rec], a, b - a)
        return pv.wav_bytes(faded(x.copy(), sr), sr)


def faded(x: np.ndarray, sr: int) -> np.ndarray:
    """FADE_S raised-cosine fades at both ends (no click at a cut), in place."""
    n = min(int(FADE_S * sr), len(x) // 2)
    if n > 1:
        w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n, dtype=np.float32))
        x[:n] *= w
        x[-n:] *= w[::-1]
    return x


def public(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)


class HTTPError(Exception):
    """An answer with this status and message (403 not a reviewer, 404 not yours, 429 daily limit, ...)."""

    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


class Redirect(Exception):
    def __init__(self, location: str):
        super().__init__(location)
        self.location = location


class Site:
    """--public: the website's limits, and per-address daily counts kept in memory only (no address is
    written to disk; a restart clears them)."""

    def __init__(self, max_upload_mb: float = 700, max_upload_min: float = 60, daily_uploads: int = 10,
                 disk_quota_gb: float = 50, min_free_gb: float = 2):
        self.max_bytes = int(max_upload_mb * 2**20)
        self.max_duration_s = float(max_upload_min * 60)
        self.daily_uploads = int(daily_uploads)
        self.quota = int(disk_quota_gb * 2**30)
        self.min_free = int(min_free_gb * 2**30)
        self._n: dict[tuple[str, str, str], int] = {}
        self._lock = threading.Lock()

    def count(self, kind: str, addr: str) -> int:
        with self._lock:
            return self._n.get((dt.date.today().isoformat(), kind, addr), 0)

    def add(self, kind: str, addr: str) -> None:
        day = dt.date.today().isoformat()
        with self._lock:
            self._n = {k: v for k, v in self._n.items() if k[0] == day}     # yesterday's counts go
            self._n[(day, kind, addr)] = self._n.get((day, kind, addr), 0) + 1

    def room(self, uploads: SA.Uploads, length: int) -> None:
        """Refuses an upload that would take the uploads folder past the quota or the disk below min_free."""
        if uploads.bytes_used() + length > self.quota:
            raise HTTPError(507, "the site's storage for uploads is full: please try again later")
        p = uploads.root
        while not p.exists():
            p = p.parent
        if shutil.disk_usage(p).free - length < self.min_free:
            raise HTTPError(507, "the server's disk is nearly full: please try again later")


class Handler(BaseHTTPRequestHandler):
    store: Store = None
    site: Site | None = None                     # None: the local app (no login, no limits, no public side)
    survey = False                               # --survey: the review page lists the survey's recordings too
    heavy = threading.BoundedSemaphore(HEAVY_AT_ONCE)

    def log_message(self, fmt, *args):          # quiet
        pass

    def send(self, code: int, body: bytes, ctype: str, headers: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")       # private links never leak in a Referer
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code: int = 200, headers: dict | None = None):
        self.send(code, json.dumps(obj).encode("utf-8"), "application/json", headers)

    def do_GET(self):
        try:
            path, q, role = self.route()
            (self.get_public if role == "public" else self.get_reviewer)(path, q)
        except (BrokenPipeError, ConnectionError):  # the page moved on (e.g. a newer spectrogram)
            pass
        except Redirect as r:
            self.send(301, b"", "text/plain", {"Location": r.location})
        except HTTPError as e:
            self.json({"error": str(e)}, e.code)
        except ValueError as e:                   # the page's input errors
            self.json({"error": self.input_error(e)}, 400)
        except Exception as e:                    # report to the page and the console, keep serving
            traceback.print_exc()
            self.json({"error": self.shown_error(e)}, 500)

    def input_error(self, e: Exception) -> str:
        """An input error's text: a visitor gets the message alone, the review app its type too."""
        if self.site is not None and getattr(self, "role", "public") == "public":
            return f"missing {e}" if isinstance(e, KeyError) else str(e)
        return f"{type(e).__name__}: {e}"

    def shown_error(self, e: Exception) -> str:
        """A server error's text for the page; a visitor sees no detail (it can hold file paths)."""
        return "server error" if getattr(self, "role", "public") == "public" and self.site is not None \
            else f"{type(e).__name__}: {e}"

    def get_reviewer(self, path: str, q: dict) -> None:
        """The review app (the local app, or /review/ behind the login in --public mode). A guest's
        recording does not exist here: not listed, and every request naming it is not found."""
        S = self.store
        if S.is_guest(q.get("rec"), q.get("id")):
            self.send(404, b"not found", "text/plain")
            return
        if path == "/":
            self.send(200, HTML_FILE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/editor":
            self.send(200, EDITOR_FILE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/raven_table.txt":
            rec = q.get("rec", "")
            if rec not in S.wavs:
                raise ValueError(f"unknown recording {rec!r}")
            txt, n = raven_table(S.log_rows(), rec, q.get("keep", "yes"))
            self.send(200, txt.encode("utf-8"), "text/plain; charset=utf-8",
                      {"Content-Disposition": f'attachment; filename="{raven_file(rec)}"',
                       "X-Selections": str(n)})
        elif path == "/api/meta":
            latest = S.latest()
            done = {}
            for bid, r in latest.items():
                done[r["recording"]] = done.get(r["recording"], 0) + 1
            lanes = [g for g in S.lanes if g not in REVIEW_HIDDEN_LANES]
            # added audio only, unless the app runs with --survey (the survey's own recordings); never the
            # frozen test recordings (config/frozen_test.csv, read at start): no new reviews of them
            recs = [r for r in list(S.recs) if (self.survey or r in S.added) and r not in S.guest_of
                    and r not in FROZEN_TEST]
            self.json({"recordings": [{"id": r, "counts": S.box_counts[r], "reviewed": done.get(r, 0),
                                       "added": S.added.get(r), **S.rec_info(r)} for r in recs],
                       "can_add": S.can_add and S._uploads_on, "uploads_dir": rel(UPLOADS_DIR),
                       "max_upload_mb": (SA.MAX_UPLOAD_BYTES if self.site is None else self.site.max_bytes) // 2**20,
                       "max_upload_min": None if self.site is None else self.site.max_duration_s / 60,
                       "reviewer": self.reviewer() if self.site is not None else None,   # set by the login
                       "public": False, "app_version": APP_VERSION,
                       "minute_s": MINUTE_S, "spec_windows": list(SPEC_WINDOWS),
                       "spec_range_db": SPEC_RANGE_DB, "view_play_s": VIEW_PLAY_S,
                       "lanes": lanes, "default_on": [g for g in S.default_on if g in lanes],
                       "thresholds": S.thresholds, "near_low": S.near_low, "colors": pv.COLORS, "kappa": pv.KAPPA,
                       "has_cells": S.cells is not None, "row_edges_khz": pv.ROW_EDGES_KHZ,
                       "sources": {k: v for k, v in SOURCES.items()
                                   if (k != "uncertain" or S.refiner is not None)
                                   and (k != "taxon" or S.taxon_adopted())},
                       "learners": {
                           "species": {"version": S.species.version, "adopted": S.species.adopted,
                                       "groups": S.species.groups(), "tau": S.species.spec.get("tau"),
                                       "n_prototypes": len(S.species.meta)},
                           "taxon": {"version": S.taxon_head.version, "adopted": S.taxon_head.adopted(),
                                     "cut": {g: S.taxon_head.cut(g) for g in S.taxon_head.adopted()},
                                     "pinned": [{"version": r["version"], "live_since": r["live_since"]}
                                                for r in (S.pins.rows if S.pins is not None else [])]}},
                       "default_sources": list(DEFAULT_SOURCES),
                       "refiner": None if S.refiner is None else {
                           "version": S.refiner.version, "adopted": S.refiner.adopted(),
                           "uncertain_below": {g: (s.get("conf_min") if s.get("conf_min") is not None
                                                   else BR.UNCERTAIN_BELOW) for g, s in S.refiner.groups.items()},
                           "default_uncertain_below": BR.UNCERTAIN_BELOW},
                       "audit_groups": list(AUDIT_GROUPS), "audit_s": AUDIT_S,
                       "log_file": rel(LOG_FILE), "audit_file": rel(AUDIT_FILE),
                       "n_logged": len(S.log_rows()),
                       "n_audit_complete": sum(r["status"] == "complete" for r in S.audit_latest().values())})
        elif path == "/api/boxes":
            latest, alatest, flags = S.latest(), S.audit_latest(), S.flag_counts()
            out = []
            for d in S.items(q["rec"], float(q.get("kappa", pv.KAPPA))):
                r = latest.get(d["id"]) if d["source"] != "audit" else None
                a = alatest.get(d["id"]) if d["source"] == "audit" else None
                ref = S.refinement(d, compute=False)
                pending = S.refiner is not None and d["source"] in REFINED_SOURCES and ref is None
                out.append({"uncertain": bool(ref and ref["uncertain"]),
                            "refine_conf": ref["conf"] if ref else None, "refine_pending": pending,
                            **{k: d[k] for k in ("id", "n", "source", "model", "lane", "label", "begin_s", "end_s",
                                                 "t_begin", "t_end", "lo_khz", "hi_khz", "max_score",
                                                 "taxon_p", "taxon_doubt")},
                            "lane": r["lane"] if r else d["lane"],
                            "decision": r["decision"] if r else "", "species": r["species"] if r else "",
                            "acc": [float(r[k]) for k in ("accepted_begin_s", "accepted_end_s", "accepted_lo_khz",
                                                          "accepted_hi_khz")] if r else None,
                            "status": a["status"] if a else "", "groups": a["groups"] if a else "",
                            "flagged": flags.get(d["id"], 0)})
            self.json(out)
        elif path == "/api/box":
            d = S.box(q["id"], float(q.get("kappa", pv.KAPPA)))
            if d["source"] == "audit":
                hist = [x for x in S.audit_rows() if audit_id(
                    x["recording"], float(x["window_begin_s"]), float(x["window_end_s"])) == d["id"]]
            else:
                hist = [x for x in S.log_rows() if row_id(x) == d["id"]]
            ref = S.refinement(d)
            shown, shown_box, _ = S.shown_of(d, hist[-1] if hist and d["source"] != "audit" else None)
            self.json({**public(d), "view": list(S.view(d)), "shown_box": shown, "start_box": shown_box,
                       "refined": ref["refined"] if ref and shown == "refined" else None,
                       "refine_conf": ref["conf"] if ref else None, "uncertain": bool(ref and ref["uncertain"]),
                       "refinable": bool(ref and ref["refinable"]), "duration_s": S.duration(d["recording"]),
                       "fmax_khz": sf.info(S.wavs[d["recording"]]).samplerate / 2000,
                       "history": hist, "latest": hist[-1] if hist else None,
                       "common": {n: S.common_of(n) for x in hist for n in x.get("species", "").split("; ") if n}})
        elif path == "/api/raven":
            self.json(S.raven_in(q["rec"], float(q["a"]), float(q["b"])))
        elif path == "/api/predict":
            d = S.lookup(q)
            t0, t1 = float(q["t0"]), float(q["t1"])
            res = S.with_common(S.predict(d, t0, t1))
            lo, hi = (float(q["lo"]), float(q["hi"])) if "lo" in q and "hi" in q else (None, None)
            self.json({**res, **S.learned(d, t0, t1, lo, hi)})
        elif path == "/api/spec.png":
            png, fmax = S.spectrogram_png(q["rec"], float(q["a"]), float(q["b"]), int(q.get("n", 1024)))
            self.send(200, png, "image/png")
        elif path == "/api/view.wav":
            self.send(200, S.view_wav(q["rec"], float(q["a"]), float(q["b"])), "audio/wav")
        elif path == "/api/audio.wav":
            d = S.lookup(q)
            wav = S.audio_wav(d, float(q["t0"]), float(q["t1"]), float(q["lo"]), float(q["hi"]),
                              float(q.get("pad", 0)), q.get("mode", "band-passed"))
            self.send(200, wav, "audio/wav")
        elif path == "/api/species":
            self.json(S.search(q.get("q", "")))
        elif path == "/api/scans":
            self.json([{**j, "position": S.queue_position(r)} for r, j in list(S.jobs.items())
                       if r not in S.guest_of])
        else:
            self.send(404, b"not found", "text/plain")

    # ---- the website's public side (--public, outside /review/) ----------------------------------
    def get_public(self, path: str, q: dict) -> None:
        """Anyone: upload, and see the model boxes and top 5 of their own uploads only (the private keys
        in their cookie). No answers, no reviewer names, no other recording."""
        S, site = self.store, self.site
        if path == "/":
            self.send(200, PUBLIC_FILE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/editor":
            self.send(200, EDITOR_FILE.read_bytes(), "text/html; charset=utf-8")
        elif path.startswith("/u/"):                  # a private link: remember its key, open the recording
            key = path[3:]
            rec = S.key_rec.get(key)
            if rec is None:
                self.send(404, b"This link names no upload (it may have been deleted).", "text/plain; charset=utf-8")
                return
            self.send(302, b"", "text/plain", {"Location": "/?rec=" + quote(rec),
                                               "Set-Cookie": self.key_cookie(self.cookie_keys() + [key])})
        elif path == "/api/meta":
            recs = []
            for r in self.owned():
                a, key, guest = S.added.get(r), S.rec_key.get(r), r in S.guest_of
                if a is None or (key is None and not guest) or r not in S.box_counts:   # still scanning
                    continue
                recs.append({"id": r, "name": a["name"], "added_at": a["added_at"], "guest": guest,
                             "link": None if guest else f"u/{key}", "counts": S.box_counts[r], **S.rec_info(r)})
            left = site.daily_uploads - site.count("upload", self.addr())
            self.json({"public": True, "app_version": APP_VERSION, "recordings": recs,
                       "guest": S.guest_session(self.guest_sid())[1] is not None, "guest_ping_s": GUEST_PING_S,
                       "guest_quiet_min": GUEST_QUIET_S / 60,
                       "can_add": S.can_add and S._uploads_on, "max_upload_mb": site.max_bytes // 2**20,
                       "max_upload_min": site.max_duration_s / 60, "daily_uploads": site.daily_uploads,
                       "uploads_left_today": max(0, left), "minute_s": MINUTE_S, "max_view_s": PUBLIC_MAX_VIEW_S,
                       "spec_windows": list(SPEC_WINDOWS), "spec_range_db": SPEC_RANGE_DB,
                       "view_play_s": VIEW_PLAY_S, "lanes": S.lanes, "default_on": S.default_on,
                       "groups": {g: lane_group(g) for g in S.lanes}, "models": {g: model_of(g) for g in S.lanes},
                       "thresholds": S.thresholds, "colors": pv.COLORS, "kappa": pv.KAPPA,
                       "has_cells": S.cells is not None, "row_edges_khz": pv.ROW_EDGES_KHZ})
        elif path == "/api/boxes":
            rec = self.own(q.get("rec", ""), scanned=True)
            latest = S.latest()
            out = []
            for d in S.items(rec, float(q.get("kappa", pv.KAPPA))):
                if d["source"] != "model":
                    continue
                r = latest.get(d["id"])
                out.append({**{k: d[k] for k in ("id", "n", "model", "lane", "begin_s", "end_s", "t_begin",
                                                 "t_end", "lo_khz", "hi_khz", "max_score", "threshold")},
                            "group": lane_group(d["lane"]),
                            "reviewed": r["decision"] if r else "",       # a reviewer's answer, no name
                            "species": r["species"] if r and r["decision"] == "yes" else ""})
            self.json(out)
        elif path == "/api/box":
            d = self.public_box(q)
            self.json({**{k: v for k, v in public(d).items() if k not in ("taxon_p", "taxon_doubt")},
                       "view": list(S.view(d)), **S.rec_info(d["recording"])})
        elif path == "/api/predict":                  # the top 5 of the box as found (tightened), only
            d = self.public_box(q)
            with self.heavy:
                res = S.with_common(S.predict(d, d["t_begin"], d["t_end"]))
                learned = S.learned(d, d["t_begin"], d["t_end"], d["lo_khz"], d["hi_khz"])
            self.json({**{k: v for k, v in res.items() if k != "check_max_diff"}, **learned})
        elif path == "/api/spec.png":
            rec = self.own(q.get("rec", ""), scanned=True)
            a, b = float(q["a"]), float(q["b"])
            if b - a > PUBLIC_MAX_VIEW_S + 1e-6:
                raise ValueError(f"a spectrogram covers at most {PUBLIC_MAX_VIEW_S:.0f} s")
            with self.heavy:
                png, _ = S.spectrogram_png(rec, a, b, int(q.get("n", 1024)))
            self.send(200, png, "image/png")
        elif path == "/api/view.wav":
            rec = self.own(q.get("rec", ""), scanned=True)
            with self.heavy:
                wav = S.view_wav(rec, float(q["a"]), float(q["b"]))
            self.send(200, wav, "audio/wav")
        elif path == "/api/audio.wav":
            d = self.public_box(q)
            mode = q.get("mode", "band-passed")
            if mode not in ("band-passed", "cell mask", "original"):
                raise ValueError("mode must be band-passed, cell mask or original")
            with self.heavy:
                wav = S.audio_wav(d, float(q["t0"]), float(q["t1"]), float(q["lo"]), float(q["hi"]),
                                  min(2.0, max(0.0, float(q.get("pad", 0)))), mode)
            self.send(200, wav, "audio/wav")
        elif path == "/api/scans":
            out = []
            for r in self.owned():
                j = S.jobs.get(r)
                if j is not None:
                    out.append({**{k: j.get(k) for k in ("rec", "name", "state", "done", "total", "counts",
                                                         "seconds", "queued_at")},
                                "error": "the scan failed: try again, or tell us" if j["state"] == "error" else "",
                                "position": S.queue_position(r), "link": f"u/{S.rec_key[r]}" if r in S.rec_key else None})
            self.json(out)
        else:
            self.send(404, b"not found", "text/plain")

    # ---- identity and ownership ---------------------------------------------------------------
    def route(self) -> tuple[str, dict, str]:
        """-> (path, query, role). role 'reviewer': the local app (no --public), or /review/... with the
        proxy's login (the prefix is stripped); 'public': the rest of the --public site."""
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        path = u.path
        if path == REVIEW_PREFIX or path.startswith(REVIEW_PREFIX + "/"):
            if path == REVIEW_PREFIX:                 # relative links need the trailing slash
                raise Redirect(REVIEW_PREFIX + "/" + (f"?{u.query}" if u.query else ""))
            if self.site is not None and not self.reviewer():
                raise HTTPError(403, "reviewers sign in at /review/")
            self.role = "reviewer"
            return path[len(REVIEW_PREFIX):], q, self.role
        self.role = "reviewer" if self.site is None else "public"
        return path, q, self.role

    def reviewer(self) -> str:
        """The signed-in reviewer's name: set by the proxy after its login, never by the page."""
        return " ".join(str(self.headers.get(REVIEWER_HEADER, "")).split())[:80]

    def addr(self) -> str:
        """The visitor's address for the daily counts (memory only): the proxy appends it to X-Forwarded-For."""
        fwd = [x.strip() for x in str(self.headers.get("X-Forwarded-For", "")).split(",") if x.strip()]
        return fwd[-1] if fwd else self.client_address[0]

    def cookie_keys(self) -> list[str]:
        try:
            c = SimpleCookie(str(self.headers.get("Cookie", "")))
        except CookieError:
            return []
        m = c.get(KEY_COOKIE)
        return [k for k in (m.value.split(".") if m is not None else []) if k][-MAX_KEYS:]

    def key_cookie(self, keys: list[str]) -> str:
        """Set-Cookie for the keys that still name an upload (newest MAX_KEYS, no duplicates)."""
        keys = [k for k in dict.fromkeys(keys) if k in self.store.key_rec][-MAX_KEYS:]
        secure = "; Secure" if self.headers.get("X-Forwarded-Proto", "") == "https" else ""
        age = COOKIE_AGE_S if keys else 0
        return f"{KEY_COOKIE}={'.'.join(keys)}; Path=/; Max-Age={age}; HttpOnly; SameSite=Lax{secure}"

    def owned(self) -> list[str]:
        """The visitor's uploads: the recordings their cookie's keys name (uploads from before guest
        sessions), then their guest session's, oldest first."""
        out = [self.store.key_rec.get(k) for k in self.cookie_keys()] + self.store.guest_recs(self.guest_sid())
        return list(dict.fromkeys(r for r in out if r is not None))

    def guest_sid(self) -> str | None:
        try:
            m = SimpleCookie(str(self.headers.get("Cookie", ""))).get(GUEST_COOKIE)
        except CookieError:
            return None
        return m.value[:64] if m is not None and m.value else None

    def guest_cookie(self, sid: str) -> str:
        """Set-Cookie for a guest session: no Max-Age, so the browser drops it when it closes."""
        secure = "; Secure" if self.headers.get("X-Forwarded-Proto", "") == "https" else ""
        return f"{GUEST_COOKIE}={sid}; Path=/; HttpOnly; SameSite=Lax{secure}"

    def own(self, rec: str, scanned: bool = False) -> str:
        if rec not in self.owned() or (scanned and rec not in self.store.seg_by_rec):
            raise HTTPError(404, "not found: open your upload's private link (or it is still being scanned)")
        return rec

    def public_box(self, q: dict) -> dict:
        """A model box of the visitor's own upload (no other source, no box being drawn)."""
        bid = str(q.get("id", ""))
        rec = self.own(bid.split("|")[0], scanned=True)
        try:
            d = self.store.box(bid, float(q.get("kappa", pv.KAPPA)))
        except KeyError:
            raise HTTPError(404, "no such box") from None
        if d["source"] != "model" or d["recording"] != rec:
            raise HTTPError(404, "no such box")
        return d

    def do_POST(self):
        S, site = self.store, self.site
        try:
            path, q, role = self.route()
            if path == "/api/upload":                 # raw audio bytes, streamed to disk (not JSON)
                self.upload(q.get("name", "added.wav"), role, q.get("tab", ""))
                return
            n = int(self.headers.get("Content-Length", 0))
            if site is not None and n > MAX_JSON_BYTES:
                raise HTTPError(413, "the request is too large")
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("expected a JSON object")
            if role == "reviewer" and path in ("/api/log", "/api/audit"):
                if S.is_guest(body.get("id"), body.get("recording")):    # never logged: not found
                    self.send(404, b"not found", "text/plain")
                    return
                if site is not None:                  # the login's name, whatever the page sent
                    body["reviewer"] = self.reviewer()
                self.json(S.append_audit(body) if path == "/api/audit"
                          else S.relabel(body) if body.get("relabel") else S.append(body))
            elif role == "public" and path == "/api/flag":
                addr = self.addr()
                if site.count("flag", addr) >= FLAGS_PER_DAY:
                    raise HTTPError(429, "too many reports today: thank you, try again tomorrow")
                d = self.public_box({"id": body.get("id", ""), "kappa": body.get("kappa", pv.KAPPA)})
                res = S.append_flag({**body, "id": d["id"]})
                site.add("flag", addr)
                self.json(res)
            elif role == "public" and path == "/api/delete":
                rec = self.own(str(body.get("rec", "")))
                S.delete_upload(rec)
                self.json({"rec": rec, "deleted": True}, headers={"Set-Cookie": self.key_cookie(self.cookie_keys())})
            elif role == "public" and path in ("/api/alive", "/api/leave"):    # a guest page checks in / closes
                self.json({"guest": S.guest_ping(self.guest_sid(), str(body.get("tab", "")),
                                                 leave=path == "/api/leave")})
            else:
                self.send(404, b"not found", "text/plain")
        except Redirect as r:
            self.send(301, b"", "text/plain", {"Location": r.location})
        except HTTPError as e:
            self.json({"error": str(e)}, e.code)
        except Exception as e:
            if not isinstance(e, (ValueError, KeyError)):   # those are the page's input errors
                traceback.print_exc()
            self.json({"error": self.input_error(e) if isinstance(e, (ValueError, KeyError))
                       else self.shown_error(e)}, 400)

    def upload(self, name: str, role: str, tab: str = "") -> None:
        """POST /api/upload. --public: at most --max-upload-mb and --max-upload-min, inside the disk quota;
        a visitor's address at most --daily-uploads new files a day (reviewers: no daily count). A
        visitor's file is a guest upload (their session's folder; the answer sets the session cookie);
        a reviewer's joins the shared uploads, with a private key as before."""
        S, site = self.store, self.site
        length = int(self.headers.get("Content-Length", 0))
        if site is None:
            self.json(S.add_upload(self.rfile.read, length, name))
            return
        addr = self.addr()
        if role == "public" and site.count("upload", addr) >= site.daily_uploads:
            raise HTTPError(429, f"{site.daily_uploads} uploads a day per address: please try again tomorrow")
        if 0 < length <= site.max_bytes:
            site.room(S.uploads, length)
        if role == "public":
            sid, res = S.add_guest_upload(self.guest_sid(), tab, self.rfile.read, length, name,
                                          limit=site.max_bytes, max_duration_s=site.max_duration_s,
                                          too_big="cut it into parts")
            if res["created"]:
                site.add("upload", addr)
            res["link"] = None
            self.json(res, headers={"Set-Cookie": self.guest_cookie(sid)})
            return
        res = S.add_upload(self.rfile.read, length, name, limit=site.max_bytes, max_duration_s=site.max_duration_s,
                           via="website" if role == "public" else "reviewer", too_big="cut it into parts")
        if res["created"] and role == "public":
            site.add("upload", addr)
        res["link"] = f"u/{res['key']}"
        self.json(res, headers={"Set-Cookie": self.key_cookie(self.cookie_keys() + [res["key"]])})


class Server(ThreadingHTTPServer):
    """socketserver ends each request with shutdown(SHUT_WR); on this laptop that loses the last
    ~25 KB of any response over ~70 KB (spectrograms, audio), and the client hangs. Instead: wait
    (at most 5 s) for the client to close after reading, then close. Measured with curl: 1.2-20 MB
    complete in < 0.2 s this way; with the shutdown, every one stalls 25-40 KB short."""
    daemon_threads = True

    def shutdown_request(self, request):
        try:
            request.settimeout(5)
            while request.recv(65536):
                pass
        except OSError:
            pass
        self.close_request(request)


def main() -> None:
    global REVIEW_DIR, LOG_FILE, CACHE_FILE, AUDIT_FILE, SHOWN_FILE, SPECIES_SHOWN_FILE, UPLOADS_DIR, FLAGS_FILE
    sys.modules.setdefault("review_app", sys.modules[__name__])   # the learners import this module: one copy
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="for Perch's on-demand top 5")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--review-dir", type=Path, default=REVIEW_DIR,
                    help="where annotation_log.csv, audit_windows.csv, box_shown.csv and prediction_cache.jsonl "
                         "live (default data/review); the refiner is read from its calibration/refiner.json")
    ap.add_argument("--uploads-dir", type=Path, default=UPLOADS_DIR,
                    help="where added audio and its scans live (default data/uploads)")
    ap.add_argument("--no-refiner", action="store_true",
                    help="no box refiner: tightened boxes only, no 'box uncertain' queue")
    ap.add_argument("--no-learners", action="store_true",
                    help="no species suggestions from reviews and no 'taxon check' source")
    ap.add_argument("--survey", action="store_true",
                    help="the review page lists the survey's recordings too (default: added audio only)")
    ap.add_argument("--export-raven", type=Path, metavar="DIR",
                    help="write a Raven selection table per reviewed recording into DIR and exit (no server)")
    ap.add_argument("--export-keep", choices=tuple(RAVEN_EXPORT), default="yes",
                    help="with --export-raven: 'yes' boxes only (default) or every answer, Decision column says which")
    web = ap.add_argument_group("website (--public; run behind the proxy in deploy/, which signs reviewers in)")
    web.add_argument("--public", action="store_true",
                     help="/ is the public upload page (own uploads only, via private links); the review app moves "
                          f"to {REVIEW_PREFIX}/ and needs the proxy's {REVIEWER_HEADER} header, which names the reviewer")
    web.add_argument("--max-upload-mb", type=float, default=700)
    web.add_argument("--max-upload-min", type=float, default=60, help="longest upload, minutes")
    web.add_argument("--daily-uploads", type=int, default=10, help="new uploads per visitor address per day")
    web.add_argument("--disk-quota-gb", type=float, default=50, help="the uploads folder's size at most")
    web.add_argument("--min-free-gb", type=float, default=2, help="free disk an upload must leave")
    args = ap.parse_args()
    REVIEW_DIR = args.review_dir.resolve()
    LOG_FILE, CACHE_FILE = REVIEW_DIR / "annotation_log.csv", REVIEW_DIR / "prediction_cache.jsonl"
    AUDIT_FILE, SHOWN_FILE = REVIEW_DIR / "audit_windows.csv", REVIEW_DIR / "box_shown.csv"
    SPECIES_SHOWN_FILE, FLAGS_FILE = REVIEW_DIR / "species_shown.csv", REVIEW_DIR / "flags.csv"
    UPLOADS_DIR = args.uploads_dir.resolve()
    if args.export_raven is not None:
        counts = export_raven(args.export_raven, args.export_keep)
        print(f"{sum(counts.values())} selections in {len(counts)} tables written to {rel(args.export_raven)} "
              f"(answers kept: {', '.join(RAVEN_EXPORT[args.export_keep])}; log: {rel(LOG_FILE)})")
        return
    refiner = None if args.no_refiner else BR.Refiner.load(REVIEW_DIR / "calibration" / "refiner.json")
    if refiner is not None:
        print(f"box refiner {refiner.version}: refined boxes shown for "
              f"{', '.join(refiner.adopted()) or 'no group yet (none adopted)'}", flush=True)
    cal = REVIEW_DIR / "calibration"
    species = RS.SpeciesLearner() if args.no_learners else RS.SpeciesLearner.load(cal / "species.json")
    taxon = RT.TaxonHead() if args.no_learners else RT.TaxonHead.load(cal / "taxon.json")
    pins = None
    if not args.no_learners:
        sp = f"adopted, {len(species.meta)} prototypes" if species.adopted else "not adopted"
        print(f"species learner: {sp}; taxon head: {', '.join(taxon.adopted()) or 'no group adopted'}", flush=True)
        # a retrained head proposes on audio added from now on; earlier recordings keep the head they had
        pins = RT.PinnedHeads(REVIEW_DIR / "taxon_versions")
        if pins.register(cal):
            print(f"taxon head {taxon.version} pinned: live from now", flush=True)
        if pins.rows:
            print(f"pinned taxon heads: {len(pins.rows)}; the survey's recordings use {pins.rows[0]['version']}",
                  flush=True)
    print("loading scores, cell maps and the Raven tables ...", flush=True)
    Handler.store = Store(args.device, refiner, species, taxon, pins)
    Handler.store.start_uploads(guests=args.public)
    Handler.survey = args.survey
    if args.public:
        Handler.site = Site(args.max_upload_mb, args.max_upload_min, args.daily_uploads, args.disk_quota_gb,
                            args.min_free_gb)
    srv = Server(("127.0.0.1", args.port), Handler)        # loopback only: in --public mode the proxy is the way in
    url = f"http://127.0.0.1:{args.port}/"
    if args.public:
        print(f"website at {url} (behind the proxy); reviewers at {url}{REVIEW_PREFIX.strip('/')}/ with "
              f"{REVIEWER_HEADER}; uploads <= {args.max_upload_mb:g} MB, {args.max_upload_min:g} min, "
              f"{args.daily_uploads}/day/address; quota {args.disk_quota_gb:g} GB  (log: {LOG_FILE})", flush=True)
    else:
        print(f"review app at {url}  (log: {LOG_FILE})  Ctrl+C to stop", flush=True)
    if not args.no_browser and not args.public:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
