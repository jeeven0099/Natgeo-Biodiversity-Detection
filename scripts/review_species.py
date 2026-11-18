"""Species learner: named boxes become prototypes that suggest names ahead of the zero-shot top 5.

    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\review_species.py

Each name a reviewer gives a box (species_labels.csv rows, built by review_labels.py) is an example:
the accepted box's Perch embedding, pooled over the cells the box covers (box_embed.py, CPU, cached
in data/review/embeddings/). Per (group of the box's lane, name) the prototype is the mean of its
boxes' L2-normalised embeddings. The name is review_labels' canonical name, or the text as typed when
it resolves to nothing, so a species Perch has no class for works the same: one example lets the
learner suggest it.

For a box of group G, every G prototype with cosine >= tau is suggested, highest first (at most
MAX_LEARNED), ahead of the zero-shot top 5 the app logged (Perch probability or EarthChirp cosine);
repeats are dropped and the list is cut to TOP_K. That merged list is what is scored.

Fit and test (whole recordings held out, as box_refine.py does):
  tau     from TAUS (inf = suggest nothing = the zero-shot top 5 alone), by the mean reciprocal rank
          of the named names in the merged list (MRR@5), leaving one recording out at a time; ties
          go to the higher tau.
  test    nested: for each recording, prototypes and tau come from the other recordings only. Gain =
          MRR@5 of the merged list minus that of the zero-shot list, on that recording's evaluation
          rows; bootstrap over recordings (BOOT draws). Adopted when the 95% interval is above 0
          and the tau fitted on all recordings is finite.
  ranks   filtered: a box with two names is scored on each with the other taken out of the list.
  rows    training uses every named row. Evaluation uses only rows where the reviewer saw no learned
          suggestion (learned_shown empty: every row before adoption, then the CONTROL_SHARE of boxes
          shown without them, and boxes with nothing above tau), so ticking a suggestion cannot
          prove it right. Positives only: an unticked top-5 name may still be calling.
Too few labels (MIN) or no name named in two recordings: kept, with the reason.

Writes data/review/calibration/species.json + species_protos.npy (review_app.py reads both at start),
species_eval.csv (one row per held-out evaluation row) and species_tau.csv (MRR per tau, all recordings).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import box_embed as BE  # noqa: E402

OUT_DIR = BE.ROOT / "data" / "review" / "calibration"
SPEC_FILE, PROTO_FILE = OUT_DIR / "species.json", OUT_DIR / "species_protos.npy"
TAUS = tuple(round(float(t), 2) for t in np.arange(0.30, 0.951, 0.05)) + (float("inf"),)
MAX_LEARNED = 3                    # learned names shown at most (the zero-shot list keeps >= 2 places)
TOP_K = 5
CONTROL_SHARE = 0.1                # boxes shown without learned names, to keep an unanchored test set
MIN = dict(rows=20, recordings=5, eval_rows=10)
BOOT, SEED = 2000, 0
EVAL_COLS = ["recording", "box_id", "log_id", "group", "name", "key", "tau", "learned", "zero_shot",
             "rank_zero", "rank_app", "rr_zero", "rr_app"]


def is_control(box_key: str) -> bool:
    """The fixed CONTROL_SHARE of boxes shown with the zero-shot list only (salted: not the
    refiner's control boxes)."""
    return zlib.crc32(f"species|{box_key}".encode()) % 1000 < CONTROL_SHARE * 1000


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9, None)


def name_key(name: str, canonical: str = "", rank: str = "") -> str:
    """The name a prototype is filed under: the canonical name when it resolved, else the text."""
    n = canonical if canonical and rank != "unresolved" else name
    return " ".join(str(n).strip().lower().replace("_", " ").split())


def merged(learned: list[str], zero: list[str], k: int = TOP_K) -> list[str]:
    out = []
    for x in [*learned, *zero]:
        if x not in out:
            out.append(x)
    return out[:k]


def rank_of(key: str, lst: list[str], others: set[str]) -> int:
    """1-based filtered rank of key (the box's other names taken out); 0 = not in the list."""
    lst = [x for x in lst if x == key or x not in others]
    return lst.index(key) + 1 if key in lst else 0


# ---- data ---------------------------------------------------------------------------------
@dataclass
class Data:
    rows: pd.DataFrame       # one per named row: box (int), recording, group, key, name, eval, zero (list[str])
    X: np.ndarray            # (boxes, D) L2-normalised box embeddings
    others: list[set]        # per row: the other keys named in its box

    @classmethod
    def make(cls, rows: pd.DataFrame, X: np.ndarray) -> Data:
        rows = rows.reset_index(drop=True)
        by_box = rows.groupby("box")["key"].apply(set).to_dict()
        return cls(rows, l2(X), [by_box[b] - {k} for b, k in zip(rows["box"], rows["key"], strict=True)])


def protos(D: Data, rmask: np.ndarray) -> tuple[list[tuple[str, str]], np.ndarray]:
    """Prototypes from the rows in rmask: ((group, key) list, (m, D) L2-normalised means);
    a box named twice with the same key counts once."""
    r = D.rows[rmask].drop_duplicates(["box", "key"])
    meta, P = [], []
    for (g, k), d in r.groupby(["group", "key"], sort=True):
        meta.append((g, k))
        P.append(D.X[d["box"].to_numpy()].mean(0))
    return meta, (l2(np.array(P)) if P else np.zeros((0, D.X.shape[1]), np.float32))


def learned_for(x: np.ndarray, group: str, meta, P: np.ndarray, tau: float, max_learned: int = MAX_LEARNED
                ) -> list[tuple[str, float]]:
    """(key, cosine) of the group's prototypes with cosine >= tau, best first, at most max_learned."""
    if not np.isfinite(tau) or not len(meta):
        return []
    j = np.array([i for i, (g, _) in enumerate(meta) if g == group], int)
    if not len(j):
        return []
    c = P[j] @ x
    o = np.argsort(-c, kind="stable")
    return [(meta[j[i]][1], float(c[i])) for i in o[:max_learned] if c[i] >= tau]


def rr_table(D: Data, train: np.ndarray, test: np.ndarray, taus=TAUS) -> np.ndarray:
    """(len(test), len(taus)) reciprocal ranks of the test rows' names in the merged list, the
    prototypes built from the train rows (bool masks over D.rows)."""
    meta, P = protos(D, train)
    idx = np.flatnonzero(test)
    out = np.zeros((len(idx), len(taus)))
    for n, i in enumerate(idx):
        row = D.rows.iloc[i]
        lo = min((t for t in taus), default=np.inf)
        cand = learned_for(D.X[row["box"]], row["group"], meta, P, lo)
        for m, t in enumerate(taus):
            lst = merged([k for k, c in cand if c >= t], row["zero"])
            r = rank_of(row["key"], lst, D.others[i])
            out[n, m] = 1.0 / r if r else 0.0
    return out


def choose_tau(D: Data, rmask: np.ndarray, taus=TAUS) -> tuple[float, np.ndarray]:
    """tau with the highest leave-one-recording-out MRR over the evaluation rows in rmask (ties: the
    higher tau); and the MRR of every tau."""
    rec, ev = D.rows["recording"].to_numpy(), D.rows["eval"].to_numpy(bool)
    tot, n = np.zeros(len(taus)), 0
    for r in sorted(set(rec[rmask])):
        te = rmask & ev & (rec == r)
        if not te.any():
            continue
        t = rr_table(D, rmask & (rec != r), te, taus)
        tot += t.sum(0)
        n += len(t)
    if not n:
        return float("inf"), np.full(len(taus), np.nan)
    mrr = tot / n
    best = np.flatnonzero(mrr >= mrr.max() - 1e-12)
    return float(taus[int(best[-1])]), mrr


def too_few(D: Data, rmask: np.ndarray, mins: dict[str, int]) -> str:
    r = D.rows[rmask]
    low = [f"{k} {v} < {mins[k]}" for k, v in (("rows", len(r)), ("recordings", r["recording"].nunique()))
           if v < mins[k]]
    if low:
        return "too few labels: " + ", ".join(low)
    if not (r.groupby(["group", "key"])["recording"].nunique() >= 2).any():
        return "no name is named in two recordings, so no prototype can be tested on a held-out recording"
    return ""


def evaluate(D: Data, mins: dict[str, int], n_boot: int = BOOT, seed: int = SEED, taus=TAUS
             ) -> tuple[dict, pd.DataFrame]:
    """Nested leave-one-recording-out (prototypes and tau from the other recordings only)."""
    rec, ev = D.rows["recording"].to_numpy(), D.rows["eval"].to_numpy(bool)
    per_rec, out = [], []
    for r in sorted(set(rec[ev])):
        tr, te = rec != r, ev & (rec == r)
        tau = float("inf") if too_few(D, tr, mins) else choose_tau(D, tr, taus)[0]
        meta, P = protos(D, tr)
        app, zero = [], []
        for i in np.flatnonzero(te):
            row = D.rows.iloc[i]
            lk = [k for k, _ in learned_for(D.X[row["box"]], row["group"], meta, P, tau)]
            ra = rank_of(row["key"], merged(lk, row["zero"]), D.others[i])
            rz = rank_of(row["key"], merged([], row["zero"]), D.others[i])
            app.append(ra)
            zero.append(rz)
            out.append(dict(recording=r, box_id=row.get("box_id", ""), log_id=row.get("log_id", ""),
                            group=row["group"], name=row["name"], key=row["key"], tau=tau, learned="; ".join(lk),
                            zero_shot="; ".join(row["zero"]), rank_zero=rz, rank_app=ra,
                            rr_zero=1 / rz if rz else 0.0, rr_app=1 / ra if ra else 0.0))
        per_rec.append((sum(1 / x for x in app if x), sum(1 / x for x in zero if x), len(app)))
    ev_df = pd.DataFrame(out, columns=EVAL_COLS)
    if not per_rec:
        return dict(n_eval=0), ev_df
    a = np.array(per_rec, float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(a), (n_boot, len(a)))
    s = a[draw].sum(1)
    diff = (s[:, 0] - s[:, 1]) / s[:, 2]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    ra, rz = ev_df["rank_app"].to_numpy(), ev_df["rank_zero"].to_numpy()
    return dict(n_eval=len(ev_df), n_eval_recordings=len(per_rec), mrr_zero=float(a[:, 1].sum() / a[:, 2].sum()),
                mrr_app=float(a[:, 0].sum() / a[:, 2].sum()), ci_lo=float(lo), ci_hi=float(hi),
                hit1_zero=float(np.mean(rz == 1)), hit1_app=float(np.mean(ra == 1)),
                hit5_zero=float(np.mean(rz > 0)), hit5_app=float(np.mean(ra > 0)),
                shown_share=float(np.mean(ev_df["learned"] != ""))), ev_df


def fit(D: Data, mins: dict[str, int] = MIN, n_boot: int = BOOT, seed: int = SEED, taus=TAUS
        ) -> tuple[dict, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """(spec without version, prototype matrix, evaluation rows, MRR per tau)."""
    allm = np.ones(len(D.rows), bool)
    n_ev = int(D.rows["eval"].sum())
    why = too_few(D, allm, mins)
    if not why and n_ev < mins["eval_rows"]:
        why = f"too few labels: evaluation rows {n_ev} < {mins['eval_rows']} (rows shown no learned name)"
    ev, ev_df = (dict(n_eval=0), pd.DataFrame(columns=EVAL_COLS)) if why else evaluate(D, mins, n_boot, seed, taus)
    tau, mrr = (float("inf"), np.full(len(taus), np.nan)) if why else choose_tau(D, allm, taus)
    if why:
        decision, reason = "keep", why
    elif not ev.get("n_eval"):
        decision, reason = "keep", "no evaluation rows"
    elif not np.isfinite(tau):
        decision, reason = "keep", "no tau beats the zero-shot top 5 on held-out recordings"
    elif ev["ci_lo"] > 0:
        decision, reason = "adopt", "held-out MRR@5 higher, interval above 0"
    else:
        decision, reason = "keep", "held-out gain not shown (interval reaches 0)"
    meta, P = protos(D, allm)
    plist = []
    for g, k in meta:
        d = D.rows[(D.rows["group"] == g) & (D.rows["key"] == k)]
        canon = d["canonical"].iloc[0] if "canonical" in d and d["rank"].iloc[0] != "unresolved" else ""
        plist.append(dict(group=g, key=k, name=canon or d["name"].value_counts().index[0],
                          n_boxes=int(d["box"].nunique()), n_recordings=int(d["recording"].nunique())))
    spec = dict(kind="prototypes", embed_version=BE.EMBED_VERSION, tau=tau if np.isfinite(tau) else None,
                max_learned=MAX_LEARNED, top_k=TOP_K, control_share=CONTROL_SHARE, decision=decision, reason=reason,
                minimum_support=mins, taus=[t if np.isfinite(t) else None for t in taus],
                metrics=dict(n_rows=len(D.rows), n_boxes=int(D.rows["box"].nunique()),
                             n_recordings=int(D.rows["recording"].nunique()), n_eval_rows=n_ev,
                             **{k: v for k, v in ev.items()}),
                prototypes=plist)
    tau_df = pd.DataFrame({"tau": [t if np.isfinite(t) else "inf" for t in taus], "loro_mrr": mrr})
    return spec, P, ev_df, tau_df


# ---- the app's side -------------------------------------------------------------------------
class SpeciesLearner:
    """species.json + species_protos.npy as review_app.py uses them: suggestions only when adopted."""

    def __init__(self, spec: dict | None = None, P: np.ndarray | None = None):
        self.spec = spec or {}
        self.version = self.spec.get("version", "")
        self.adopted = self.spec.get("decision") == "adopt" and self.spec.get("tau") is not None
        self.tau = float(self.spec["tau"]) if self.spec.get("tau") is not None else float("inf")
        self.meta = [(p["group"], p["key"]) for p in self.spec.get("prototypes", [])]
        self.names = {(p["group"], p["key"]): p for p in self.spec.get("prototypes", [])}
        self.P = l2(P) if P is not None and len(P) else np.zeros((0, BE.DIM), np.float32)
        if len(self.P) != len(self.meta):
            raise SystemExit(f"species prototypes: {len(self.meta)} in the json, {len(self.P)} in the npy")
        if self.spec and self.spec.get("embed_version") != BE.EMBED_VERSION:
            self.adopted = False                     # prototypes from another pooling rule do not compare

    @classmethod
    def load(cls, spec_path: Path = SPEC_FILE, proto_path: Path | None = None) -> SpeciesLearner:
        spec_path = Path(spec_path)
        proto_path = Path(proto_path) if proto_path else spec_path.with_name("species_protos.npy")
        if spec_path.exists() and proto_path.exists():
            return cls(json.loads(spec_path.read_text(encoding="utf-8")), np.load(proto_path))
        return cls()

    def groups(self) -> list[str]:
        return sorted({g for g, _ in self.meta}) if self.adopted else []

    def suggest(self, group: str, emb: np.ndarray) -> list[dict]:
        if not self.adopted:
            return []
        out = []
        for k, c in learned_for(l2(emb), group, self.meta, self.P, self.tau,
                                int(self.spec.get("max_learned", MAX_LEARNED))):
            p = self.names[(group, k)]
            out.append(dict(name=p["name"], key=k, score=round(c, 4), n_boxes=p["n_boxes"],
                            n_recordings=p["n_recordings"]))
        return out


# ---- run ------------------------------------------------------------------------------------
def build_rows(species: pd.DataFrame, log_rows: list[dict], lane_group, zero_key) -> tuple[pd.DataFrame, list]:
    """species_labels rows -> learner rows and the (recording, box) list to embed. zero_key(name)
    maps a logged top-5 name to its key."""
    top = {r.get("log_id", ""): r["top5"] for r in log_rows}
    import review_labels as L
    sp = species[species["recording"].astype(str) != ""].reset_index(drop=True)
    boxes, bpos, rows = [], {}, []
    for r in sp.to_dict("records"):
        box = tuple(round(float(r[e]), 3) for e in ("begin_s", "end_s", "lo_khz", "hi_khz"))
        b = bpos.setdefault(r["box_id"], len(boxes))
        if b == len(boxes):
            boxes.append((r["recording"], box))
        zero = [zero_key(n) for n in L.top5_names(top.get(r["log_id"], ""))]
        rows.append(dict(box=b, box_id=r["box_id"], log_id=r["log_id"], recording=r["recording"],
                         group=lane_group(r["lane"]), key=name_key(r["name"], r["canonical"], r["rank"]),
                         name=r["name"], canonical=r["canonical"], rank=r["rank"],
                         eval=str(r.get("learned_shown", "") or "") == "", zero=zero))
    return pd.DataFrame(rows), boxes


def main() -> int:
    import review_labels as L  # here, not at the top: review_app imports this module
    R = L.R
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=R.LOG_FILE)
    ap.add_argument("--audit", type=Path, default=R.AUDIT_FILE)
    ap.add_argument("--shown", type=Path, default=R.SHOWN_FILE)
    ap.add_argument("--species-shown", type=Path, default=R.SPECIES_SHOWN_FILE)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--cache", type=Path, default=BE.CACHE_DIR, help="box embedding cache")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    for k, v in MIN.items():
        ap.add_argument(f"--min-{k.replace('_', '-')}", type=int, default=v, help=f"default {v}")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    mins = {k: getattr(args, f"min_{k}") for k in MIN}

    log_rows = L.read_csv_rows(args.log)
    lab = L.labels_from_logs(log_rows, L.read_csv_rows(args.audit), L.read_csv_rows(args.shown),
                             L.read_csv_rows(args.species_shown))
    species = pd.DataFrame(lab["species"], columns=L.SPECIES_COLS)
    names, memo = L.Names(), {}

    def zero_key(n: str) -> str:
        if n not in memo:
            x = names.resolve(n)
            memo[n] = name_key(n, x["canonical"], x["rank"])
        return memo[n]
    rows, boxes = build_rows(species, log_rows, R.lane_group, zero_key)
    print(f"{len(rows)} named rows on {len(boxes)} boxes, {rows['recording'].nunique() if len(rows) else 0} recordings")
    X = np.zeros((0, BE.DIM), np.float32)
    if boxes:
        X = BE.embed_boxes(boxes, BE.BoxEmbedder(BE.PerchSpatial(args.device)), BE.EmbeddingCache(args.cache),
                           progress=True)
    D = Data.make(rows if len(rows) else pd.DataFrame(columns=["box", "recording", "group", "key", "name", "eval",
                                                               "zero", "canonical", "rank"]), X)
    spec, P, ev_df, tau_df = fit(D, mins, args.boot, args.seed)
    spec = dict(version=dt.datetime.now().isoformat(timespec="seconds"), **spec, log=str(args.log))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "species.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    np.save(args.out / "species_protos.npy", P.astype(np.float32))
    ev_df.to_csv(args.out / "species_eval.csv", index=False)
    tau_df.to_csv(args.out / "species_tau.csv", index=False)
    m = spec["metrics"]
    print(f"{len(spec['prototypes'])} prototypes; {spec['decision']}: {spec['reason']}")
    if m.get("n_eval"):
        print(f"  held-out MRR@5 zero-shot {m['mrr_zero']:.3f} -> with learned names {m['mrr_app']:.3f} "
              f"[{m['ci_lo']:+.3f}, {m['ci_hi']:+.3f}]; hit@1 {m['hit1_zero']:.2f} -> {m['hit1_app']:.2f}, "
              f"hit@5 {m['hit5_zero']:.2f} -> {m['hit5_app']:.2f}; learned names shown on {m['shown_share']:.0%} "
              f"of {m['n_eval']} rows")
    print(f"  tau {spec['tau']}; written to {args.out}; review_app.py reads species.json at start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
