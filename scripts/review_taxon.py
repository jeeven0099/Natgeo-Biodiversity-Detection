"""Taxon learner: per group, a logistic head on Perch window embeddings, fitted to positives, the
reviewers' "no" answers and the unlabelled time.

    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\review_taxon.py
    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\review_taxon.py --positives-only

The question per group (Aves, Amphibia, Insecta, Mammalia, Anthropogenic, Weather): is a sound of
this group in this 2.5 s segment? Positive vs unlabelled (user decision 2026-11-17), with the "no"
answers as negatives (user decision 2026-11-19):
  positive    a segment touching (> EPS_S) a positive taxon row of the group (review_labels.py:
              a yes on a box of the group's lane, a species of the group named in any box, a Raven
              label logged as a review)
  negative    a segment touching a "no" on a box of the group's lane and no positive of the group
              (a positive wins: a "no" is about one box, a call elsewhere in the segment still counts)
  unlabelled  every other segment of a recording holding at least one positive of the group
A recording with negatives but no positive of the group gives its negative segments only.
Audit gaps, Raven gaps, unreviewed boxes and "unknown" answers are not used. --positives-only drops
the negatives (they fall back to unlabelled): the 2026-11-17 fit. If the positives are a random
sample of the group's sounds (Elkan & Noto 2008), a positive-vs-unlabelled score ranks like a
positive-vs-negative one, and AUC(P vs U) = 0.5 x (share of positives among the unlabelled) +
(1 - that share) x AUC(P vs N): the sign of head minus zero-shot is kept.
The head is trained positive (1) vs the rest (0: negatives and unlabelled); negatives are never
dropped by the cap. AUROC P vs N (positives vs the "no" segments) is reported, not used to decide:
the "no" boxes are mostly where the zero-shot score was over its threshold, which biases that
comparison against the zero-shot score.

Recordings (review_sources.py, --uploads-dir): the survey caches' and the scanned audio added in the
app, whose scan rows go behind the caches' as the app puts them there.
Features of a segment, read from caches (no Perch run): the mean of the L2-normalised Perch
embeddings of the 5 s windows over the segment (earthchirp_embeddings.npy is exactly Perch's window
embedding; the windows are review_app.windows_for_span's), L2-normalised again; and the zero-shot
score of the group's default lane (logit for Perch lanes, the cosine as is for EarthChirp lanes).
The head is StandardScaler + LogisticRegression (balanced classes). Training rows: at most CAP per
class per recording (seeded draw); scoring uses every segment. C from CS by the pooled AUROC of a
grouped INNER_FOLDS split over the training recordings (ties: the smaller C).

Test, whole recordings held out: OUTER_FOLDS grouped folds (leave one recording out with fewer
recordings); each held-out segment is scored by a head fitted (C included) on the other folds only.
Gain = pooled held-out AUROC (positive vs the rest) of the head minus that of the zero-shot score
on the same segments; bootstrap over recordings. Adopted when the 95% interval is above 0. Too few
labels (MIN): kept. Also reported, not used for the decision: the mean AUROC within each held-out
recording holding both classes (a pooled AUROC also rewards telling recordings apart).
The cut: the head probability maximising recall - flagged share on the pooled held-out segments
(ties: the higher cut); the app finds calls at >= cut and doubts model boxes whose highest segment
probability is below it.

Writes data/review/calibration/taxon.json + taxon.npz (review_app.py reads both at start),
taxon_eval.csv (every held-out segment) and taxon_segments.csv (every 2.5 s segment of the survey
re-scored by the adopted heads: recording, start_s, group, zero_shot, head).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perch_viewer as pv  # noqa: E402
import zeroshot_windows as z  # noqa: E402

ROOT = pv.ROOT
PDIR = ROOT / "data" / "perch_windows"
OUT_DIR = ROOT / "data" / "review" / "calibration"
SPEC_FILE, PARAM_FILE = OUT_DIR / "taxon.json", OUT_DIR / "taxon.npz"
GROUPS = tuple(z.GROUPS)
CS = (0.001, 0.01, 0.1, 1.0)
EPS_S = 0.05                      # overlaps shorter than this are edge contact, not overlap
CUT = 0.5                         # head probability cut before a fit
MAX_CUTS = 400                    # cut candidates (quantiles of the held-out probabilities)
CAP = 100                         # training segments per class per recording
MIN = dict(pos=30, recordings=5)
POS_NEG_FOLD = 2                  # a training fold needs this many of each class
OUTER_FOLDS, INNER_FOLDS = 10, 3
LOGIT_EPS = 1e-4
BOOT, SEED = 2000, 0
EVAL_COLS = ["recording", "group", "start_s", "label", "negative", "zero_shot", "head", "fold", "C"]
ROW_COLS = ["recording", "start_s", "label", "negative", "zero_shot"]
SEG_COLS = ["recording", "start_s", "group", "zero_shot", "head"]


def lane_group(lane: str) -> str:
    return "Insecta" if lane in pv.EC_GROUPS else lane


def default_lane(group: str, lanes=None, default_on=None) -> str:
    """The lane whose score is the group's zero-shot baseline: the group's default (hybrid) lane,
    else its first lane."""
    lanes = list(pv.VIEW_GROUPS if lanes is None else lanes)
    default_on = list(pv.DEFAULT_ON if default_on is None else default_on)
    for pool in (default_on, lanes):
        for lane in pool:
            if lane in lanes and lane_group(lane) == group:
                return lane
    return ""


def transform(lane: str) -> str:
    return "raw" if lane in pv.EC_GROUPS else "logit"


def zfeat(score, how: str) -> np.ndarray:
    s = np.asarray(score, float)
    if how == "logit":
        s = np.clip(s, LOGIT_EPS, 1 - LOGIT_EPS)
        return np.log(s / (1 - s))
    return s


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9, None)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with tied scores counted half; nan without both classes."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def touches(st: np.ndarray, spans) -> np.ndarray:
    """Segments [st, st + SEG_S) overlapping any (begin, end) span by more than EPS_S."""
    st = np.asarray(st, float)
    hit = np.zeros(len(st), bool)
    for a, b in spans:
        hit |= np.clip(np.minimum(st + pv.SEG_S, b) - np.maximum(st, a), 0, None) > EPS_S
    return hit


# ---- caches ---------------------------------------------------------------------------------
class Windows:
    """Per recording: window starts, L2-normalised window embeddings and 2.5 s segment scores."""

    def __init__(self, E, win_rows: dict, win_starts: np.ndarray, seg_by_rec: dict):
        self.E, self.win_rows, self.win_starts, self.seg_by_rec = E, win_rows, win_starts, seg_by_rec
        self._rec: dict[str, np.ndarray] = {}

    @classmethod
    def from_disk(cls, pdir: Path = PDIR) -> Windows:
        w = pd.read_csv(pdir / "window_scores.csv", usecols=["recording", "start_s"])
        codes, names = pd.factorize(w["recording"])
        first = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
        rows = {r: (int(a), int(b - a)) for r, a, b in zip(names, first, np.r_[first[1:], len(w)], strict=True)}
        E = np.load(pdir / "earthchirp_embeddings.npy", mmap_mode="r")
        if len(E) != len(w):
            raise SystemExit(f"earthchirp_embeddings.npy has {len(E)} rows, window_scores.csv {len(w)}")
        seg = pv.load_segments(pdir)
        return cls(E, rows, w["start_s"].to_numpy(), {r: g for r, g in seg.groupby("recording", sort=False)})

    def starts(self, rec: str) -> np.ndarray:
        a, n = self.win_rows[rec]
        return self.win_starts[a:a + n]

    def emb(self, rec: str) -> np.ndarray:
        if rec not in self._rec:
            if len(self._rec) > 8:
                self._rec.clear()
            a, n = self.win_rows[rec]
            self._rec[rec] = l2(np.asarray(self.E[a:a + n], np.float32))
        return self._rec[rec]

    def seg_max(self, rec: str, lane: str, t0: float, t1: float) -> float:
        s = self.seg_by_rec[rec]
        col = f"prob_{lane}"
        if col not in s:
            return float("nan")
        m = (s["start_s"] < t1 - 1e-6) & (s["start_s"] + pv.SEG_S > t0 + 1e-6)
        return float(s.loc[m, col].max()) if m.any() else float("nan")

    def segment_features(self, rec: str, lane: str, t0: float | None = None, t1: float | None = None
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """2.5 s segments of a recording (all, or those overlapping [t0, t1) as seg_max counts them):
        (start_s, features, raw zero-shot score)."""
        import review_app as R  # windows_for_span; imported here (review_app imports this module)
        s = self.seg_by_rec[rec].sort_values("start_s")
        if t0 is not None:
            s = s[(s["start_s"] < t1 - 1e-6) & (s["start_s"] + pv.SEG_S > t0 + 1e-6)]
        st = s["start_s"].to_numpy(float)
        sc = s[f"prob_{lane}"].to_numpy(float) if f"prob_{lane}" in s else np.full(len(st), np.nan)
        starts, E = self.starts(rec), self.emb(rec)
        X = np.stack([E[R.windows_for_span(starts, a, a + pv.SEG_S)].mean(0) for a in st]) if len(st) else \
            np.zeros((0, E.shape[1]), np.float32)
        return st, np.c_[l2(X), zfeat(sc, transform(lane))].astype(np.float32), sc


# ---- labels -> segment rows ---------------------------------------------------------------------
@dataclass
class GroupData:
    group: str
    lane: str
    rows: pd.DataFrame          # ROW_COLS: label 1 positive / 0 not; negative 1 = a "no" segment (label 0)
    X: np.ndarray               # (rows, D + 1)

    @property
    def y(self) -> np.ndarray:
        return self.rows["label"].to_numpy(int)

    @property
    def neg(self) -> np.ndarray:
        if "negative" not in self.rows:
            return np.zeros(len(self.rows), bool)
        return self.rows["negative"].to_numpy(int).astype(bool)

    @property
    def rec(self) -> np.ndarray:
        return self.rows["recording"].to_numpy()


def spans(d: pd.DataFrame):
    return list(zip(d["begin_s"].astype(float), d["end_s"].astype(float), strict=True))


def group_data(taxon: pd.DataFrame, win: Windows, group: str, lane: str, negatives: bool = True) -> GroupData:
    """Every segment of each recording holding a positive of the group (and in the caches), and the
    negative segments of recordings holding negatives but no positive. Label-0 taxon rows are the
    negatives (main passes the "no" answers only); negatives=False ignores them."""
    t = taxon[taxon["group"] == group]
    lab = t["label"].astype(int)
    pos, neg = t[lab == 1], t[lab == 0]
    if not negatives:
        neg = neg.iloc[:0]
    pos_recs = set(pos["recording"])
    rows, X = [], []
    for rec in sorted(pos_recs | set(neg["recording"])):
        if rec not in win.win_rows or rec not in win.seg_by_rec:
            continue
        st, f, sc = win.segment_features(rec, lane)
        y = touches(st, spans(pos[pos["recording"] == rec]))
        n = touches(st, spans(neg[neg["recording"] == rec])) & ~y
        ok = np.isfinite(sc) & (n if rec not in pos_recs else True)
        if not ok.any():
            continue
        rows.append(pd.DataFrame(dict(recording=rec, start_s=st[ok], label=y[ok].astype(int),
                                      negative=n[ok].astype(int), zero_shot=sc[ok])))
        X.append(f[ok])
    if not rows:
        return GroupData(group, lane, pd.DataFrame(columns=ROW_COLS), np.zeros((0, 1), np.float32))
    return GroupData(group, lane, pd.concat(rows, ignore_index=True), np.concatenate(X))


def cap_rows(y: np.ndarray, rec: np.ndarray, cap: int, seed: int, always: np.ndarray | None = None) -> np.ndarray:
    """Indices keeping at most cap rows per class per recording (seeded draw), in order. Rows marked
    in always (the negatives) are kept whole and not counted against the cap; with none marked, the
    draw is the same as without always."""
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(pd.Series(rec), sort=True)
    always = np.zeros(len(codes), bool) if always is None else np.asarray(always, bool)
    keep = [np.flatnonzero(always)]
    for r in range(codes.max() + 1 if len(codes) else 0):
        for c in (1, 0):
            k = np.flatnonzero((codes == r) & (y == c) & ~always)
            keep.append(np.sort(rng.choice(k, cap, replace=False)) if len(k) > cap else k)
    return np.sort(np.concatenate(keep)) if len(codes) else np.zeros(0, int)


# ---- fitting ----------------------------------------------------------------------------------
def make_model(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000))


def enough(y: np.ndarray) -> bool:
    return int((y == 1).sum()) >= POS_NEG_FOLD and int((y == 0).sum()) >= POS_NEG_FOLD


def choose_C(X: np.ndarray, y: np.ndarray, rec: np.ndarray, cs=CS) -> tuple[float, list[float]]:
    """C by the pooled AUROC of a grouped split over recordings (ties: the smaller C); with fewer
    than two recordings, or no usable fold, the smallest C."""
    from sklearn.model_selection import GroupKFold
    n = len(set(rec))
    if n < 2:
        return float(cs[0]), [float("nan")] * len(cs)
    folds = [(tr, te) for tr, te in GroupKFold(n_splits=min(INNER_FOLDS, n)).split(X, y, rec) if enough(y[tr])]
    if not folds:
        return float(cs[0]), [float("nan")] * len(cs)
    te_all = np.concatenate([te for _, te in folds])
    aucs = []
    for C in cs:
        p = np.concatenate([make_model(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] for tr, te in folds])
        aucs.append(auroc(y[te_all], p))
    a = np.nan_to_num(np.array(aucs), nan=-1.0)
    return float(cs[int(np.flatnonzero(a >= a.max() - 1e-12)[0])]), aucs


def recall_flagged(y: np.ndarray, s: np.ndarray, t) -> tuple[np.ndarray, np.ndarray]:
    """Share of positives with s >= t, share of all rows with s >= t (t may be an array)."""
    sa, sp = np.sort(np.asarray(s, float)), np.sort(np.asarray(s, float)[np.asarray(y) == 1])
    t = np.asarray(t, float)
    r = (len(sp) - np.searchsorted(sp, t, side="left")) / max(len(sp), 1)
    f = (len(sa) - np.searchsorted(sa, t, side="left")) / max(len(sa), 1)
    return r, f


def pu_cut(y: np.ndarray, p: np.ndarray) -> float:
    """The probability cut maximising recall - flagged share (ties: the higher cut)."""
    c = np.unique(np.round(np.quantile(p, np.linspace(0, 1, MAX_CUTS)), 6))
    r, f = recall_flagged(y, p, c)
    g = r - f
    return float(c[np.flatnonzero(g >= g.max() - 1e-12)[-1]])


def support(gd: GroupData) -> dict[str, int]:
    """recordings: those holding a positive (the minimum is on them); negatives may add others."""
    y, n, rec = gd.y, gd.neg, gd.rec
    return dict(pos=int((y == 1).sum()), unlabelled=int(((y == 0) & ~n).sum()), negative=int(n.sum()),
                recordings=int(len(set(rec[y == 1]))), negative_recordings=int(len(set(rec[n]))))


def too_few(sup: dict[str, int], mins: dict[str, int]) -> str:
    low = [f"{k} {sup[k]} < {v}" for k, v in mins.items() if sup[k] < v]
    return "too few labels: " + ", ".join(low) if low else ""


def evaluate(gd: GroupData, cap: int = CAP, n_boot: int = BOOT, seed: int = SEED, cs=CS) -> tuple[dict, pd.DataFrame]:
    """Grouped outer folds: each fold's segments scored by a head (C chosen inside) fitted on the
    other folds' recordings only, training rows capped."""
    from sklearn.model_selection import GroupKFold
    X, y, rec, neg = gd.X, gd.y, gd.rec, gd.neg
    n = len(set(rec))
    parts = []
    for f, (tr, te) in enumerate(GroupKFold(n_splits=min(OUTER_FOLDS, n)).split(X, y, rec)):
        tr = tr[cap_rows(y[tr], rec[tr], cap, seed + f, neg[tr])]
        if not enough(y[tr]):
            continue
        C, _ = choose_C(X[tr], y[tr], rec[tr], cs)
        p = make_model(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        parts.append(gd.rows.iloc[te].assign(head=p, fold=f, C=C))
    if not parts:
        return dict(n_eval=0), pd.DataFrame(columns=EVAL_COLS)
    ev = pd.concat(parts, ignore_index=True).assign(group=gd.group)
    if "negative" not in ev:
        ev["negative"] = 0
    ev = ev[EVAL_COLS]
    yy, h, zz = ev["label"].to_numpy(int), ev["head"].to_numpy(float), ev["zero_shot"].to_numpy(float)
    nn = ev["negative"].to_numpy(int).astype(bool)
    pn = (yy == 1) | nn                                  # positives and "no" segments: AUROC P vs N
    a_head, a_zero = auroc(yy, h), auroc(yy, zz)
    recs = ev["recording"].to_numpy()
    idx = {r: np.flatnonzero(recs == r) for r in sorted(set(recs))}
    keys = list(idx)
    rng = np.random.default_rng(seed)
    diffs, diffs_pn = [], []
    for _ in range(n_boot):
        k = np.concatenate([idx[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        diffs.append(auroc(yy[k], h[k]) - auroc(yy[k], zz[k]))
        kp = k[pn[k]]
        diffs_pn.append(auroc(yy[kp], h[kp]) - auroc(yy[kp], zz[kp]))
    lo, hi = np.nanpercentile(np.array(diffs), [2.5, 97.5]) if np.isfinite(diffs).any() else (np.nan, np.nan)
    lo_pn, hi_pn = (np.nanpercentile(np.array(diffs_pn), [2.5, 97.5]) if np.isfinite(diffs_pn).any()
                    else (np.nan, np.nan))
    within = np.array([(auroc(yy[i], h[i]), auroc(yy[i], zz[i])) for i in idx.values() if 0 < yy[i].sum() < len(i)])
    out = dict(n_eval=len(ev), n_eval_recordings=len(keys), n_eval_pos=int(yy.sum()),
               n_eval_unlabelled=int(((yy == 0) & ~nn).sum()), n_eval_negative=int(nn.sum()),
               n_eval_negative_recordings=int(len(set(recs[nn]))), auc_zero=a_zero, auc_head=a_head,
               ci_lo=float(lo), ci_hi=float(hi),
               auc_pn_zero=auroc(yy[pn], zz[pn]), auc_pn_head=auroc(yy[pn], h[pn]),
               ci_pn_lo=float(lo_pn), ci_pn_hi=float(hi_pn), n_within=len(within),
               auc_within_zero=float(within[:, 1].mean()) if len(within) else float("nan"),
               auc_within_head=float(within[:, 0].mean()) if len(within) else float("nan"),
               within_head_better=int((within[:, 0] > within[:, 1]).sum()) if len(within) else 0)
    if 0 < yy.sum() < len(yy):
        cut = pu_cut(yy, h)
        r, fl = recall_flagged(yy, h, cut)
        thr = pv.THRESHOLDS.get(gd.lane, float("nan"))
        rz, fz = recall_flagged(yy, zz, thr)
        out.update(cut=cut, recall_at_cut=float(r), flagged_at_cut=float(fl), zero_threshold=thr,
                   zero_recall_at_threshold=float(rz), zero_flagged_at_threshold=float(fz),
                   negative_flagged_at_cut=float((h[nn] >= cut).mean()) if nn.any() else float("nan"),
                   zero_negative_flagged_at_threshold=float((zz[nn] >= thr).mean()) if nn.any() else float("nan"))
    return out, ev


def fit_group(gd: GroupData, mins: dict[str, int] = MIN, cap: int = CAP, n_boot: int = BOOT, seed: int = SEED,
              cs=CS) -> tuple[dict, dict | None, pd.DataFrame]:
    """(spec entry, parameters or None, evaluation rows)."""
    sup = support(gd)
    entry = dict(lane=gd.lane, transform=transform(gd.lane), cut=CUT, **sup)
    why = too_few(sup, mins) or ("" if enough(gd.y) else "labels hold one class only")
    if why:
        return {**entry, "decision": "keep", "reason": why, "C": None}, None, pd.DataFrame(columns=EVAL_COLS)
    ev, ev_df = evaluate(gd, cap, n_boot, seed, cs)
    k = cap_rows(gd.y, gd.rec, cap, seed, gd.neg)
    C, aucs = choose_C(gd.X[k], gd.y[k], gd.rec[k], cs)
    m = make_model(C).fit(gd.X[k], gd.y[k])
    sc, lr = m[0], m[-1]
    params = dict(mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32),
                  coef=lr.coef_[0].astype(np.float32), intercept=np.float32(lr.intercept_[0]))
    if not ev.get("n_eval"):
        decision, reason = "keep", "no held-out recording could be scored"
    elif not np.isfinite(ev["auc_zero"]) or not np.isfinite(ev["auc_head"]):
        decision, reason = "keep", "held-out rows hold one class only"
    elif ev["ci_lo"] > 0:
        decision, reason = "adopt", "held-out AUROC (P vs rest) higher than the zero-shot score, interval above 0"
    else:
        decision, reason = "keep", "held-out gain not shown (interval reaches 0)"
    inner = dict(zip(map(str, cs), aucs, strict=True))
    return ({**entry, "decision": decision, "reason": reason, "C": C, "n_train": len(k), "inner_auc": inner, **ev},
            params, ev_df)


# ---- the app's side -----------------------------------------------------------------------------
class TaxonHead:
    """taxon.json + taxon.npz as review_app.py uses them: probabilities for adopted groups only."""

    def __init__(self, spec: dict | None = None, params: dict | None = None):
        self.spec = spec or {}
        self.version = self.spec.get("version", "")
        self.params = params or {}
        self.groups = {g: s for g, s in self.spec.get("groups", {}).items()
                       if s.get("decision") == "adopt" and g in self.params}

    @classmethod
    def load(cls, spec_path: Path = SPEC_FILE, param_path: Path | None = None) -> TaxonHead:
        spec_path = Path(spec_path)
        param_path = Path(param_path) if param_path else spec_path.with_name("taxon.npz")
        if not (spec_path.exists() and param_path.exists()):
            return cls()
        npz = np.load(param_path)
        params = {}
        for k in npz.files:
            g, field = k.split("__", 1)
            params.setdefault(g, {})[field] = npz[k]
        return cls(json.loads(spec_path.read_text(encoding="utf-8")), params)

    def adopted(self) -> list[str]:
        return sorted(self.groups)

    def lane(self, group: str) -> str:
        return self.groups[group]["lane"]

    def cut(self, group: str) -> float:
        return float(self.groups[group].get("cut", CUT))

    def prob(self, group: str, X: np.ndarray) -> np.ndarray:
        p = self.params[group]
        X = np.atleast_2d(np.asarray(X, np.float64))
        logit = ((X - p["mean"]) / p["scale"]) @ p["coef"].astype(np.float64) + float(p["intercept"])
        return 1 / (1 + np.exp(-logit))

    def span_prob(self, win: Windows, group: str, rec: str, t0: float, t1: float) -> float:
        """The highest head probability over the 2.5 s segments the span overlaps (the unit it was fitted on)."""
        st, X, _ = win.segment_features(rec, self.lane(group), t0, t1)
        p = self.prob(group, X) if len(st) else np.zeros(0)
        return float(np.nanmax(p)) if np.isfinite(p).any() else float("nan")

    def segments(self, win: Windows, group: str, rec: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(start_s, zero-shot score, head probability) of every 2.5 s segment of a recording."""
        st, X, sc = win.segment_features(rec, self.lane(group))
        return st, sc, (self.prob(group, X) if len(st) else np.zeros(0))


def runs(on: np.ndarray) -> list[tuple[int, int]]:
    """(first, last + 1) of each run of True."""
    e = np.diff(np.r_[0, np.asarray(on, np.int8), 0])
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist(), strict=True))


# ---- run ----------------------------------------------------------------------------------------
def main() -> int:
    import review_labels as L
    R = L.R
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=R.LOG_FILE)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    for k, v in MIN.items():
        ap.add_argument(f"--min-{k}", type=int, default=v, help=f"default {v}")
    ap.add_argument("--cap", type=int, default=CAP, help=f"training segments per class per recording (default {CAP})")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), choices=GROUPS,
                    help="fit only these (taxon.json then holds only these)")
    ap.add_argument("--no-segments", action="store_true", help="skip taxon_segments.csv")
    ap.add_argument("--positives-only", action="store_true",
                    help="ignore the 'no' answers (they fall back to unlabelled): the 2026-11-17 fit")
    ap.add_argument("--uploads-dir", type=Path, default=L.SRC.SA.UPLOADS_DIR, help="audio added in the app")
    args = ap.parse_args()
    mins = {k: getattr(args, f"min_{k}") for k in MIN}

    src = L.SRC.Sources(args.uploads_dir)
    print(src.describe())
    lab = L.labels_from_logs(L.read_csv_rows(args.log), [], sources=src)   # no audit file: no audit gaps
    taxon = pd.DataFrame(lab["taxon"], columns=L.TAXON_COLS)
    lab01 = taxon["label"].astype(int)
    no = (lab01 == 0) & (taxon["reason"] == "decision")              # a reviewer's "no" on a box of the lane
    taxon = taxon[(lab01 == 1) | (no & (not args.positives_only))]
    n_no = int((taxon["label"].astype(int) == 0).sum())
    print(f"{int((taxon['label'].astype(int) == 1).sum())} positive taxon labels, {n_no} 'no' labels"
          + (" (ignored: --positives-only)" if args.positives_only else ""))
    win = src.windows()
    spec_groups, params, evs = {}, {}, []
    for g in args.groups:
        lane = default_lane(g)
        gd = group_data(taxon, win, g, lane, negatives=not args.positives_only)
        entry, p, ev = fit_group(gd, mins, args.cap, args.boot, args.seed)
        spec_groups[g] = entry
        if p is not None:
            params[g] = p
        evs.append(ev)
        line = (f"  {g:<13} {entry['pos']} positive / {entry['negative']} negative / "
                f"{entry['pos'] + entry['negative'] + entry['unlabelled']} segments, {entry['recordings']} recordings"
                f" (negatives in {entry['negative_recordings']}): {entry['decision']}")
        if entry.get("n_eval"):
            line += (f"; held-out AUROC P vs rest zero-shot {entry['auc_zero']:.3f} -> head {entry['auc_head']:.3f} "
                     f"[{entry['ci_lo']:+.3f}, {entry['ci_hi']:+.3f}]; within recordings {entry['auc_within_zero']:.3f}"
                     f" -> {entry['auc_within_head']:.3f}, head better in {entry['within_head_better']}/"
                     f"{entry['n_within']}")
            if entry.get("n_eval_negative"):
                line += (f"; P vs N ({entry['n_eval_negative']} negative segments in "
                         f"{entry['n_eval_negative_recordings']} recordings) {entry['auc_pn_zero']:.3f} -> "
                         f"{entry['auc_pn_head']:.3f} [{entry['ci_pn_lo']:+.3f}, {entry['ci_pn_hi']:+.3f}]")
        if "cut" in entry and entry.get("recall_at_cut") is not None:
            line += (f"; cut {entry['cut']:.3f}: recall {entry['recall_at_cut']:.2f}, flagged "
                     f"{entry['flagged_at_cut']:.2f} (zero-shot at {entry['zero_threshold']:g}: "
                     f"{entry['zero_recall_at_threshold']:.2f}, {entry['zero_flagged_at_threshold']:.2f})")
            if entry.get("n_eval_negative"):
                line += (f"; 'no' segments flagged: head {entry['negative_flagged_at_cut']:.2f}, zero-shot "
                         f"{entry['zero_negative_flagged_at_threshold']:.2f}")
        print(line + f" ({entry['reason']})", flush=True)
    spec = dict(version=dt.datetime.now().isoformat(timespec="seconds"), kind="logistic head on window embeddings",
                labels=("positive vs unlabelled 2.5 s segments of recordings holding a positive of the group"
                        if args.positives_only else
                        "positive vs the rest (the 'no' segments as negatives, kept whole, and the unlabelled) "
                        "2.5 s segments of recordings holding a positive of the group, plus the 'no' segments "
                        "of other recordings"),
                negatives=not args.positives_only,
                features="mean of L2-normalised Perch window embeddings over the segment, L2-normalised; "
                         "+ the default lane's zero-shot score (logit for Perch lanes)",
                cs=list(CS), cap=args.cap, outer_folds=OUTER_FOLDS, inner_folds=INNER_FOLDS, minimum_support=mins,
                log=str(args.log), added_recordings=len(src.uploads), groups=spec_groups)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "taxon.json").write_text(json.dumps(spec, indent=2, default=float), encoding="utf-8")
    np.savez(args.out / "taxon.npz", **{f"{g}__{k}": v for g, p in params.items() for k, v in p.items()})
    pd.concat([e for e in evs if len(e)] or [pd.DataFrame(columns=EVAL_COLS)]).to_csv(
        args.out / "taxon_eval.csv", index=False)
    head = TaxonHead(spec, {g: {k: np.asarray(v) for k, v in p.items()} for g, p in params.items()})
    if not args.no_segments:
        rows = []
        for g in head.adopted():
            for rec in win.seg_by_rec:
                if rec not in win.win_rows:
                    continue
                st, sc, p = head.segments(win, g, rec)
                rows.append(pd.DataFrame(dict(recording=rec, start_s=st, group=g, zero_shot=sc, head=p.round(5))))
        (pd.concat(rows) if rows else pd.DataFrame(columns=SEG_COLS)).to_csv(args.out / "taxon_segments.csv",
                                                                               index=False)
    print(f"adopted: {', '.join(head.adopted()) or 'none'}; written to {args.out}; "
          "review_app.py reads taxon.json at start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
