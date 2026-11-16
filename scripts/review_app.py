"""Local review app for the zero-shot boxes: listen, judge, name the species, log the decision.

    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\review_app.py            # opens http://127.0.0.1:8765
    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\review_app.py --device cpu --port 8766 --no-browser

Each box is a Perch / EarthChirp zero-shot box at the viewer's thresholds (perch_viewer.THRESHOLDS),
tightened to its cells (cell_maps.npy, kappa adjustable). For each box the app shows:
  * the spectrogram around it, with the model box, the tightened box, its kept cells and the Raven boxes;
  * the model's top-5 names for the box's group:
      Perch lanes      Perch's softmax probability, restricted to that group's classes, max over the
                       5 s windows overlapping the span (Perch is run on those windows on demand);
      EarthChirp lanes cosine of the cached Perch embedding to EarthChirp's species / genus /
                       group prototypes, max over the same windows;
  * playback (original / band-passed / cell mask, perch_viewer.filter_clip).
The reviewer answers yes / no / unknown (is there a sound of this group in the box?), ticks any of the
top 5 and/or adds other species (autocomplete over Perch, EarthChirp, Raven and earlier-added names;
free text allowed), may redraw the box, and presses Accept. Every Accept appends one row to
data/review/annotation_log.csv (nothing is ever rewritten; the newest row of a box is its current
state). No learning happens here.

Reads segment_scores.csv, earthchirp_segments.csv, window_scores.csv, earthchirp_embeddings.npy,
cell_maps.npy, the Raven tables, the wavs, external/earthchirp_lib and config/*.csv. Writes only
data/review/annotation_log.csv and data/review/prediction_cache.jsonl.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import sys
import threading
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perch_viewer as pv  # noqa: E402
import zeroshot_windows as z  # noqa: E402
import insecta_compare as IC  # noqa: E402

ROOT = pv.ROOT
PDIR = ROOT / "data" / "perch_windows"
REVIEW_DIR = ROOT / "data" / "review"
LOG_FILE = REVIEW_DIR / "annotation_log.csv"
CACHE_FILE = REVIEW_DIR / "prediction_cache.jsonl"
HTML_FILE = Path(__file__).resolve().parent / "review_app.html"
APP_VERSION = "1"
TOP_K = 5
CONTEXT_S = 2.0              # spectrogram shows the box +- this
MIN_VIEW_S, MAX_VIEW_S = 10.0, 120.0
SPEC_COLS = 3000             # spectrogram columns at most (hop grows with the view)
MAX_PRED_WINDOWS = 64        # longer spans: evenly spaced windows
BATCH = 16                   # Perch windows per ONNX call
WIN_S, HOP_S = z.WINDOW_S, pv.SEG_S
MARGIN_S = 1.0               # audio read around the windows before resampling (filter edges)
LOG_COLS = ["log_id", "logged_at", "reviewer", "recording", "lane", "model", "threshold",
            "box_begin_s", "box_end_s", "box_max_score", "kappa",
            "tight_begin_s", "tight_end_s", "tight_lo_khz", "tight_hi_khz",
            "accepted_begin_s", "accepted_end_s", "accepted_lo_khz", "accepted_hi_khz", "box_edited",
            "decision", "species", "species_source", "top5", "prediction_span", "comment", "app_version"]
DECISIONS = ("yes", "no", "unknown")


def model_of(lane: str) -> str:
    return "EarthChirp" if lane in pv.EC_GROUPS else "Perch"


def box_id(rec: str, lane: str, b0: float, b1: float) -> str:
    return f"{rec}|{lane}|{b0:.1f}|{b1:.1f}"


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

    def __init__(self, device: str):
        self.device = device
        self.seg = pv.load_segments(PDIR)
        self.raven = pv.load_raven()
        self.wavs = pv.wav_paths()
        self.cells = pv.CellMaps(PDIR) if pv.CELL_FILE.exists() else None
        self.lanes = [g for g in pv.VIEW_GROUPS if f"prob_{g}" in self.seg]
        self.thresholds = {g: pv.THRESHOLDS[g] for g in self.lanes}
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
        # Perch classes and groups
        self.labels, self.groups = IC.labels_and_groups()
        idx = pd.read_csv(ROOT / "config" / "perch_class_index.csv", keep_default_na=False)
        self.kind, self.ebird, self.taxon = idx["kind"].tolist(), idx["ebird"].tolist(), idx["class"].tolist()
        # EarthChirp prototypes
        lib = ROOT / "external" / "earthchirp_lib"
        self.ec_P = IC.l2(np.load(lib / "global_reflib_protos.npy"))
        ec_idx = pd.read_parquet(lib / "global_reflib_index.parquet")
        self.ec_names, self.ec_levels = ec_idx["name"].tolist(), ec_idx["level"].tolist()
        self._sess = None
        self._lock = threading.Lock()
        self._boxes: dict[tuple[str, float], list[dict]] = {}
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

    # ---- boxes ---------------------------------------------------------------------------
    def boxes(self, rec: str, kappa: float) -> list[dict]:
        key = (rec, round(kappa, 3))
        with self._lock:
            if key in self._boxes:
                return self._boxes[key]
        pb = pv.perch_boxes(self.seg_by_rec[rec], self.thresholds)
        pb = pb.sort_values(["begin_s", "group"]).reset_index(drop=True)
        if self.cells is not None:
            pb = self.cells.tighten_boxes(rec, pb, kappa)
        out = []
        for i, b in enumerate(pb.itertuples()):
            tight = self.cells is not None and np.isfinite(b.t_begin)
            d = {"id": box_id(rec, b.group, b.begin_s, b.end_s), "n": i + 1, "recording": rec,
                 "lane": b.group, "model": model_of(b.group), "threshold": self.thresholds[b.group],
                 "begin_s": float(b.begin_s), "end_s": float(b.end_s), "max_score": float(b.max_prob),
                 "kappa": kappa, "tightened": bool(tight),
                 "t_begin": float(b.t_begin) if tight else float(b.begin_s),
                 "t_end": float(b.t_end) if tight else float(b.end_s),
                 "lo_khz": float(b.lo_khz) if tight else pv.PERCH_BAND[0],
                 "hi_khz": float(b.hi_khz) if tight else pv.PERCH_BAND[1],
                 "area": float(b.area) if tight else 1.0,
                 "_kept": np.asarray(b.kept, bool) if tight else None}
            d["kept_rects"] = self._kept_rects(d)
            out.append(d)
        with self._lock:
            self._boxes = {k: v for k, v in self._boxes.items() if k[0] == rec}   # keep one recording
            self._boxes[key] = out
        return out

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
        for d in self.boxes(rec, kappa):
            if d["id"] == bid:
                return d
        raise KeyError(bid)

    def duration(self, rec: str) -> float:
        return float(sf.info(self.wavs[rec]).duration)

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

    def perch_logits(self, rec: str, starts: np.ndarray) -> np.ndarray:
        x, x0 = resampled_slice(self.wavs[rec], float(starts.min()), float(starts.max()) + WIN_S)
        n = int(round(WIN_S * z.PERCH_SR))
        wins = np.zeros((len(starts), n), np.float32)
        for i, s in enumerate(starts):
            o = int(round((s - x0) * z.PERCH_SR))
            seg = x[o:o + n]
            wins[i, :len(seg)] = seg
        # fixed batches of BATCH (zero rows pad the last): cuDNN finds no plan for some small batches
        sess, out = self.session(), []
        for k in range(0, len(wins), BATCH):
            xb = np.zeros((BATCH, n), np.float32)
            m = len(wins[k:k + BATCH])
            xb[:m] = wins[k:k + BATCH]
            lg, = sess.run(["label"], {"inputs": IC.normalize_windows(xb, 0.25)})
            out.append(np.asarray(lg, np.float64)[:m])
        return np.concatenate(out)

    def predict(self, d: dict, t0: float, t1: float) -> dict:
        key = f"{d['id']}|{t0:.2f}|{t1:.2f}"
        with self._lock:
            if key in self._pred:
                return self._pred[key]
        rec, lane = d["recording"], d["lane"]
        a, n = self.win_rows[rec]
        starts = self.win_starts[a:a + n]
        k = windows_for_span(starts, t0, t1)
        res = {"key": key, "box": d["id"], "lane": lane, "model": d["model"], "span": [t0, t1],
               "windows": [float(s) for s in starts[k]]}
        if d["model"] == "Perch":
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
            with CACHE_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
        return res

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
                out.append({"name": name, "source": source, "note": note})
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
        hits = [v for v in self.vocab if q in v["name"].lower() or q in v["note"].lower()]
        hits.sort(key=lambda v: (not v["name"].lower().startswith(q), len(v["name"])))
        return hits[:limit]

    # ---- log -----------------------------------------------------------------------------
    def log_rows(self) -> list[dict]:
        if not LOG_FILE.exists():
            return []
        with LOG_FILE.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def latest(self) -> dict[str, dict]:
        """box id -> its newest log row."""
        out = {}
        for r in self.log_rows():
            out[box_id(r["recording"], r["lane"], float(r["box_begin_s"]), float(r["box_end_s"]))] = r
        return out

    def append(self, body: dict) -> dict:
        d = self.box(body["id"], float(body["kappa"]))
        if body.get("decision") not in DECISIONS:
            raise ValueError("decision must be yes, no or unknown")
        acc = [float(body[k]) for k in ("begin_s", "end_s", "lo_khz", "hi_khz")]
        if not (acc[0] < acc[1] and acc[2] < acc[3]):
            raise ValueError("box must have begin < end and low < high")
        tight = [d["t_begin"], d["t_end"], d["lo_khz"], d["hi_khz"]]
        species = [s for s in body.get("species", []) if s.get("name", "").strip()]
        row = {"log_id": uuid.uuid4().hex[:12], "logged_at": dt.datetime.now().isoformat(timespec="seconds"),
               "reviewer": str(body.get("reviewer", "")).strip(), "recording": d["recording"], "lane": d["lane"],
               "model": d["model"], "threshold": d["threshold"], "box_begin_s": d["begin_s"],
               "box_end_s": d["end_s"], "box_max_score": round(d["max_score"], 6), "kappa": d["kappa"],
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
        with self._log_lock:
            new = not LOG_FILE.exists()
            with LOG_FILE.open("a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=LOG_COLS)
                if new:
                    w.writeheader()
                w.writerow(row)
        for s in species:
            if s.get("source") == "added":
                n = s["name"].strip()
                if not any(v["name"].lower() == n.lower() for v in self.vocab):
                    self.vocab.append({"name": n, "source": "added earlier", "note": ""})
        return row

    # ---- media ---------------------------------------------------------------------------
    def spectrogram_png(self, rec: str, a: float, b: float) -> tuple[bytes, float]:
        """8-bit grey PNG, 0 Hz at the bottom, white = quiet, 80 dB below the 99.9th percentile = white."""
        from PIL import Image
        x, sr = pv.read_page(self.wavs[rec], a, b - a)
        hop = max(128, int(np.ceil(len(x) / SPEC_COLS)))
        f, t, db = pv.spectrogram(x, sr, n_fft=1024, hop=hop)
        vmax = np.percentile(db, 99.9)
        g = (np.clip((vmax - db) / 80.0, 0, 1) * 255).astype(np.uint8)[::-1]
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(g), "L").save(buf, format="PNG", compress_level=3)
        return buf.getvalue(), float(f[-1])

    def audio_wav(self, d: dict, t0: float, t1: float, lo: float, hi: float, pad: float, mode: str) -> bytes:
        a = max(0.0, t0 - pad)
        e = min(t1 + pad, a + pv.MAX_CLIP_S)
        x, sr = pv.read_page(self.wavs[d["recording"]], a, e - a)
        if mode == "band-passed":
            x = pv.filter_clip(x, sr, a, SimpleNamespace(lo_khz=lo, hi_khz=hi), mode)
        elif mode == "cell mask" and d["_kept"] is not None:
            x = pv.filter_clip(x, sr, a, SimpleNamespace(t_begin=d["t_begin"], kept=d["_kept"]), mode)
        return pv.wav_bytes(x, sr)


def public(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


class Handler(BaseHTTPRequestHandler):
    store: Store = None

    def log_message(self, fmt, *args):          # quiet
        pass

    def send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code: int = 200):
        self.send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        S = self.store
        try:
            if u.path == "/":
                self.send(200, HTML_FILE.read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/meta":
                latest = S.latest()
                done = {}
                for bid, r in latest.items():
                    done[r["recording"]] = done.get(r["recording"], 0) + 1
                self.json({"recordings": [{"id": r, "counts": S.box_counts[r], "reviewed": done.get(r, 0)}
                                          for r in S.recs],
                           "lanes": S.lanes, "default_on": [g for g in pv.DEFAULT_ON if g in S.lanes],
                           "thresholds": S.thresholds, "colors": pv.COLORS, "kappa": pv.KAPPA,
                           "has_cells": S.cells is not None, "row_edges_khz": pv.ROW_EDGES_KHZ,
                           "log_file": str(LOG_FILE.relative_to(ROOT) if LOG_FILE.is_relative_to(ROOT) else LOG_FILE),
                           "n_logged": len(S.log_rows())})
            elif u.path == "/api/boxes":
                latest = S.latest()
                out = []
                for d in S.boxes(q["rec"], float(q.get("kappa", pv.KAPPA))):
                    r = latest.get(d["id"])
                    out.append({**{k: d[k] for k in ("id", "n", "lane", "begin_s", "end_s", "t_begin", "t_end",
                                                     "lo_khz", "hi_khz", "max_score")},
                                "decision": r["decision"] if r else "", "species": r["species"] if r else ""})
                self.json(out)
            elif u.path == "/api/box":
                d = S.box(q["id"], float(q.get("kappa", pv.KAPPA)))
                hist = [x for x in S.log_rows()
                        if box_id(x["recording"], x["lane"], float(x["box_begin_s"]), float(x["box_end_s"])) == d["id"]]
                self.json({**public(d), "view": list(S.view(d)), "duration_s": S.duration(d["recording"]),
                           "fmax_khz": sf.info(S.wavs[d["recording"]]).samplerate / 2000,
                           "history": hist, "latest": hist[-1] if hist else None})
            elif u.path == "/api/raven":
                self.json(S.raven_in(q["rec"], float(q["a"]), float(q["b"])))
            elif u.path == "/api/predict":
                d = S.box(q["id"], float(q.get("kappa", pv.KAPPA)))
                self.json(S.predict(d, float(q["t0"]), float(q["t1"])))
            elif u.path == "/api/spec.png":
                png, fmax = S.spectrogram_png(q["rec"], float(q["a"]), float(q["b"]))
                self.send(200, png, "image/png")
            elif u.path == "/api/audio.wav":
                d = S.box(q["id"], float(q.get("kappa", pv.KAPPA)))
                wav = S.audio_wav(d, float(q["t0"]), float(q["t1"]), float(q["lo"]), float(q["hi"]),
                                  float(q.get("pad", 0.5)), q.get("mode", "original"))
                self.send(200, wav, "audio/wav")
            elif u.path == "/api/species":
                self.json(S.search(q.get("q", "")))
            else:
                self.send(404, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionError):  # the page moved on (e.g. a newer spectrogram)
            pass
        except Exception as e:                    # report to the page and the console, keep serving
            traceback.print_exc()
            self.json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if u.path == "/api/log":
                self.json(self.store.append(body))
            else:
                self.send(404, b"not found", "text/plain")
        except Exception as e:
            if not isinstance(e, (ValueError, KeyError)):   # those are the page's input errors
                traceback.print_exc()
            self.json({"error": f"{type(e).__name__}: {e}"}, 400)


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
    global REVIEW_DIR, LOG_FILE, CACHE_FILE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="for Perch's on-demand top 5")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--review-dir", type=Path, default=REVIEW_DIR,
                    help="where annotation_log.csv and prediction_cache.jsonl live (default data/review)")
    args = ap.parse_args()
    REVIEW_DIR = args.review_dir.resolve()
    LOG_FILE, CACHE_FILE = REVIEW_DIR / "annotation_log.csv", REVIEW_DIR / "prediction_cache.jsonl"
    print("loading scores, cell maps and the Raven tables ...", flush=True)
    Handler.store = Store(args.device)
    srv = Server(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"review app at {url}  (log: {LOG_FILE})  Ctrl+C to stop", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
