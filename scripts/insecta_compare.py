"""Insect vs non-insect: EarthChirp against Perch 2.0 zero-shot, on the labelled short-form clips.

Both methods read the SAME Perch 2.0 forward pass of the same 5 s windows; they differ only in
what they do with it:

  Perch      summed softmax probability of the 2,045 Insecta classes of Perch's own 14,795-class
             head (every Insecta species + FSD50K 'Insect' and 'Cricket'), exactly as in
             zeroshot_windows.py. Listening-confirmed threshold: 0.032.
  EarthChirp max cosine of the 1536-d embedding to the EarthChirp library's insect prototypes
             (2,012 species, 619 genera, Orthoptera + Cicadidae groups; Melnick 2026,
             Zenodo 10.5281/zenodo.20697426), as earthchirp/recognise.py computes it.
             Label-free operating points, both from the paper and both built on ESC-50 only:
               tau       cosine > 95th percentile on ESC-50 mechanical clips (56_esc50_specificity.py)
               outgroup  nearest prototype is an insect, not the ESC-50 'urban' or 'geophony'
                         prototype (train half of each, RandomState(0); 58_outgroup_reject.py)
             Continuous scores: 'ec_cos' (insect cosine) and 'ec_margin' (insect cosine minus
             outgroup cosine, the paper's AFTER score).
Nothing in the comparison sees our labels except the metrics.

Test set: every window of data/perch/windows.csv (35,424 windows of 4,684 clips cut from the Raven
boxes of 108 long-form sessions). Positive = the clip's coarse label is insect. Negatives are split
by type (bird, anthropogenic, amphibian); geophony (19 windows) and mammal (10) count only in 'all'.
Caveat: a bird or aircraft clip from this forest often has insects behind it, so a "false" insect
flag on a negative clip is not surely wrong. The negatives are equally noisy for both methods, so
read the comparison as relative.

Metrics, per window and per clip (clip score = max over its windows):
  AUROC / AP, insect vs each negative type, with 95% CIs from resampling SESSIONS (1,000 draws),
  plus the paired EarthChirp-minus-Perch AUROC difference on the same draws.
  Operating points: share flagged among insect windows (recall) and among each negative type.
  'matched' = EarthChirp's cosine cut that flags the same share of all windows as Perch >= 0.032.

Stages (each is cached in --outdir and skipped when its file exists):
  esc50   ONNX Perch on the 2,000 ESC-50 clips          -> esc50_emb.npy, esc50_meta.csv   (~2 min GPU)
  clips   ONNX Perch on the 35,424 clip windows          -> clip_perch.npz                  (~20 min GPU)
  score   metrics                                         -> the CSVs below                  (seconds)

    .venv-perch/Scripts/python -X utf8 scripts/insecta_compare.py                 # all stages
    .venv-perch/Scripts/python -X utf8 scripts/insecta_compare.py --stage esc50 --device cpu
    .venv-perch/Scripts/python -X utf8 scripts/insecta_compare.py --stage score

Writes data/insecta_compare/:
  esc50_emb.npy / esc50_meta.csv   ESC-50 embeddings + category / tier / fold
  esc50_check.csv                  our pipeline's EarthChirp numbers on ESC-50 (vs the paper)
  clip_perch.npz                   per window: emb (float16), prob (7 groups), maxlogit, rebuild check
  window_scores.csv                per window: labels, perch_prob, perch_maxlogit, ec_cos, ec_og, ec_margin
  metrics_auc.csv                  AUROC / AP + CIs per unit, negative type and score
  metrics_operating.csv            flag rates per operating point, unit and label type
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_perch import add_nvidia_dll_dirs, load_audio_windows, normalize_windows  # noqa: E402
from zeroshot_windows import GROUPS, class_groups  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PERCH_SR, WIN = 32_000, 160_000
PERCH_INSECTA_THR = 0.032            # listening-confirmed in the viewer, 2026-11-14
NEG_TYPES = ("all", "bird", "anthropogenic", "amphibian")
SCORES = ("perch_prob", "perch_maxlogit", "ec_cos", "ec_margin")
N_BOOT = 1000

# ESC-50 tiers, copied from EarthChirp 56_esc50_specificity.py / 58_outgroup_reject.py
ESC_TIER = {"crickets": "positive", "insects": "positive",
            "chirping_birds": "bird", "crow": "bird", "hen": "bird", "rooster": "bird",
            "frog": "frog", "cat": "mammal", "dog": "mammal", "cow": "mammal", "pig": "mammal",
            "sheep": "mammal", "rain": "geophony", "wind": "geophony", "sea_waves": "geophony",
            "thunderstorm": "geophony", "crackling_fire": "geophony"}


# ---- Perch ------------------------------------------------------------------------------
def open_session(model: Path, device: str):
    import onnxruntime as ort
    if device == "cpu":
        prov = ["CPUExecutionProvider"]
    else:
        add_nvidia_dll_dirs()
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        # TF32 off, as in zeroshot_windows.py: with it on, logits drift up to 3.09 from TF.
        prov = [("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(model), providers=prov)
    got = sess.get_providers()
    print("onnxruntime providers:", ", ".join(got), flush=True)
    if device == "cuda" and not got[0].startswith("CUDA"):
        raise SystemExit("--device cuda but onnxruntime has no CUDA provider")
    return sess


def labels_and_groups():
    idx = pd.read_csv(ROOT / "config" / "perch_class_index.csv", keep_default_na=False)
    labels = idx["label"].tolist()
    return labels, class_groups(labels)


def run_perch(sess, wins: np.ndarray, groups: dict, batch: int):
    """-> embeddings (n, 1536) float32, group probs (n, len(GROUPS)+1: last = other), group max logits."""
    emb, logit = [], []
    for k in range(0, len(wins), batch):
        e, l = sess.run(["embedding", "label"], {"inputs": normalize_windows(wins[k:k + batch].copy(), 0.25)})
        emb.append(np.asarray(e).reshape(len(l), -1))
        logit.append(np.asarray(l, dtype=np.float64))
    emb, logit = np.concatenate(emb), np.concatenate(logit)
    p = np.exp(logit - logit.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    member = np.zeros(p.shape[1], dtype=bool)
    prob, mlog = [], []
    for g in GROUPS:
        prob.append(p[:, groups[g]].sum(axis=1))
        mlog.append(logit[:, groups[g]].max(axis=1))
        member[groups[g]] = True
    prob.append(p[:, ~member].sum(axis=1))
    return emb.astype(np.float32), np.stack(prob, 1), np.stack(mlog, 1)


def stage_esc50(args, sess, groups) -> None:
    meta = pd.read_csv(args.esc50_dir / "meta" / "esc50.csv")
    meta["tier"] = meta["category"].map(lambda c: ESC_TIER.get(c, "mechanical"))
    wins = []
    for f in meta["filename"]:
        x, sr = sf.read(str(args.esc50_dir / "audio" / f), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        g = int(np.gcd(sr, PERCH_SR))
        x = resample_poly(x, PERCH_SR // g, sr // g).astype(np.float32)
        # EarthChirp load5s: loudest 5 s window (ESC-50 clips are 5 s, so this is pad / trim)
        if len(x) > WIN:
            st = range(0, len(x) - WIN + 1, WIN // 2)
            s = max(st, key=lambda a: float(np.abs(x[a:a + WIN]).mean()))
            x = x[s:s + WIN]
        wins.append(np.pad(x, (0, max(0, WIN - len(x))))[:WIN])
    t = time.monotonic()
    emb, _, _ = run_perch(sess, np.stack(wins), groups, args.batch)
    print(f"esc50: {len(emb)} clips embedded in {time.monotonic() - t:.0f} s", flush=True)
    np.save(args.outdir / "esc50_emb.npy", emb)
    meta.to_csv(args.outdir / "esc50_meta.csv", index=False)


class ClipReader:
    """Clip audio by manifest path: from the extracted file if present, else straight from the
    'Labelled Data-*.zip' whose name is the path's first folder (the clips are not unpacked)."""

    def __init__(self):
        self.zips = {}

    def windows(self, rel: str) -> tuple[np.ndarray, np.ndarray]:
        if (ROOT / rel).exists():
            return load_audio_windows(ROOT / rel, 5.0)
        import io
        import zipfile
        top, member = rel.split("/", 1)
        if top not in self.zips:
            self.zips[top] = zipfile.ZipFile(ROOT / (top + ".zip"))
        with self.zips[top].open(member) as fh:
            buf = io.BytesIO(fh.read())
        return windows_from(*sf.read(buf, dtype="float32", always_2d=True))


def windows_from(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """extract_perch.load_audio_windows(path, 5.0) on already-read audio, line for line."""
    x = x.mean(axis=1)
    if sr != PERCH_SR:
        g = int(np.gcd(sr, PERCH_SR))
        x = resample_poly(x, PERCH_SR // g, sr // g).astype(np.float32)
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    hop = int(PERCH_SR * 5.0)
    starts = list(range(0, len(x) - WIN + 1, hop))
    if starts[-1] + WIN < len(x):
        starts.append(len(x) - WIN)
    return np.stack([x[s:s + WIN] for s in starts]), np.array(starts) / PERCH_SR


def stage_clips(args, sess, groups) -> None:
    w = pd.read_csv(ROOT / "data" / "perch" / "windows.csv")
    man = pd.read_csv(ROOT / "data" / "manifest.csv", usecols=["clip_id", "path", "keep"])
    path = dict(zip(man["clip_id"], man["path"]))
    clips = w["clip_id"].drop_duplicates().tolist()
    if args.limit:
        clips = clips[:args.limit]
        w = w[w["clip_id"].isin(clips)]
    n = len(w)
    want_starts = {c: g.to_numpy() for c, g in w.groupby("clip_id", sort=False)["start_s"]}
    reader = ClipReader()
    embs, probs, mlogs = [], [], []
    buf, t0, done = [], time.monotonic(), 0
    for i, c in enumerate(clips):
        wins, starts = reader.windows(path[c])
        want = want_starts[c]
        # the cache's window rows must be exactly this clip's windows, in order. windows.csv holds
        # starts rounded to 0.01 s (Python round; np.round differs on half-way values such as
        # 6.985 s), so compare the unrounded starts within half a rounding step.
        if len(want) != len(starts) or not np.allclose(want, starts, rtol=0, atol=0.0051):
            raise SystemExit(f"{c}: rebuilt windows {starts} != windows.csv {want}")
        buf.append(wins)
        if sum(len(b) for b in buf) >= 256 or i + 1 == len(clips):
            e, p, m = run_perch(sess, np.concatenate(buf), groups, args.batch)
            embs.append(e), probs.append(p), mlogs.append(m)
            done += len(e)
            buf = []
            el = time.monotonic() - t0
            print(f"  {i + 1}/{len(clips)} clips, {done}/{n} windows, {done / el:.1f} win/s, "
                  f"~{(n - done) / (done / el) / 60:.1f} min left", flush=True)
    emb = np.concatenate(embs)
    old = np.load(ROOT / "data" / "perch" / "embeddings.npy", mmap_mode="r")[:len(emb)].astype(np.float32)
    cos = (emb * old).sum(1) / np.linalg.norm(emb, axis=1) / np.linalg.norm(old, axis=1)
    print(f"rebuild check vs data/perch/embeddings.npy: max|diff| {np.abs(emb - old).max():.4f}, "
          f"min cosine {cos.min():.5f}", flush=True)
    np.savez(args.outdir / ("clip_perch.npz" if not args.limit else "clip_perch_limit.npz"),
             emb=emb.astype(np.float16), prob=np.concatenate(probs), maxlogit=np.concatenate(mlogs),
             groups=np.array(list(GROUPS) + ["other"]), cache_cosine=cos.astype(np.float32))


# ---- EarthChirp -------------------------------------------------------------------------
def l2(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)


class EarthChirpScore:
    """recognise.py's insect cosine (max over species, genus and group prototypes) + the outgroup."""

    def __init__(self, lib: Path, esc_emb: np.ndarray, esc_meta: pd.DataFrame):
        self.P = l2(np.load(lib / "global_reflib_protos.npy"))
        self.index = pd.read_parquet(lib / "global_reflib_index.parquet")
        E = l2(esc_emb)
        tier = esc_meta["tier"].to_numpy()
        # 58_outgroup_reject.py: urban = every tier not in its table (= our 'mechanical'), geophony;
        # prototype = mean of a random half of each (RandomState(0), urban drawn first)
        rng = np.random.RandomState(0)
        halves = {}
        for t in ("mechanical", "geophony"):
            i = np.flatnonzero(tier == t)
            rng.shuffle(i)
            halves[t] = (i[:len(i) // 2], i[len(i) // 2:])
        self.outgroup = l2(np.stack([E[halves[t][0]].mean(0) for t in ("mechanical", "geophony")]))
        self.esc_test = np.ones(len(E), dtype=bool)          # ESC-50 clips not used to build the outgroup
        for t in halves:
            self.esc_test[halves[t][0]] = False
        # 56_esc50_specificity.py: tau = 95th percentile on ALL mechanical clips
        self.tau = float(np.percentile(self.cos(E)[tier == "mechanical"], 95))

    def cos(self, E: np.ndarray) -> np.ndarray:
        return (l2(E) @ self.P.T).max(1)

    def og(self, E: np.ndarray) -> np.ndarray:
        return (l2(E) @ self.outgroup.T).max(1)


def esc50_check(ec: EarthChirpScore, E: np.ndarray, meta: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Our ONNX pipeline + scorer on ESC-50, to set beside the paper's ESC-50 numbers."""
    from sklearn.metrics import roc_auc_score
    s, m = ec.cos(E), ec.cos(E) - ec.og(E)
    tier = meta["tier"].to_numpy()
    rows = []
    pos = tier == "positive"
    for t in ("bird", "frog", "mammal", "mechanical", "geophony"):
        neg = tier == t
        sel = pos | neg
        test = sel & ec.esc_test     # margin uses the outgroup, so score it on held-out clips only
        rows.append({"negative": t, "n_neg": int(neg.sum()),
                     "auroc_cos": roc_auc_score(pos[sel], s[sel]),
                     "auroc_margin_heldout": roc_auc_score(pos[test], m[test]),
                     "fire_rate_tau": float((s[neg] > ec.tau).mean()),
                     "fire_rate_outgroup_heldout": float((m[neg & ec.esc_test] > 0).mean())})
    vert = np.isin(tier, ["bird", "frog", "mammal"])
    rows.append({"negative": "vertebrate (bird+frog+mammal)", "n_neg": int(vert.sum()),
                 "auroc_cos": roc_auc_score(pos[pos | vert], s[pos | vert]),
                 "auroc_margin_heldout": np.nan, "fire_rate_tau": float((s[vert] > ec.tau).mean()),
                 "fire_rate_outgroup_heldout": float((m[vert] > 0).mean())})
    rows.append({"negative": "positive controls (kept as insect)", "n_neg": int(pos.sum()),
                 "auroc_cos": np.nan, "auroc_margin_heldout": np.nan,
                 "fire_rate_tau": float((s[pos] > ec.tau).mean()),
                 "fire_rate_outgroup_heldout": float((m[pos] > 0).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(out / "esc50_check.csv", index=False)
    return df


# ---- scoring ----------------------------------------------------------------------------
def boot_auc(y, s, sess, mask, n_boot=N_BOOT, seed=0):
    """AUROC / AP per score + session-bootstrap CIs; paired differences vs perch_prob."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    m = np.flatnonzero(mask)
    y, sess = y[m], sess[m]
    s = {k: v[m] for k, v in s.items()}
    us, inv = np.unique(sess, return_inverse=True)
    members = [np.flatnonzero(inv == j) for j in range(len(us))]
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in s}
    for _ in range(n_boot):
        pick = np.concatenate([members[j] for j in rng.integers(0, len(us), len(us))])
        if y[pick].all() or not y[pick].any():
            continue
        for k in s:
            draws[k].append(roc_auc_score(y[pick], s[k][pick]))
    out = {}
    for k in s:
        d = np.asarray(draws[k])
        out[k] = {"auroc": roc_auc_score(y, s[k]), "auroc_lo": np.percentile(d, 2.5),
                  "auroc_hi": np.percentile(d, 97.5), "ap": average_precision_score(y, s[k]),
                  "prevalence": y.mean(), "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
                  "n_sessions": len(us), "n_draws": len(d)}
        if k != "perch_prob":
            diff = d - np.asarray(draws["perch_prob"])
            out[k].update(d_vs_perch=out[k]["auroc"] - out["perch_prob"]["auroc"],
                          d_lo=np.percentile(diff, 2.5), d_hi=np.percentile(diff, 97.5))
    return out


def stage_score(args) -> None:
    w = pd.read_csv(ROOT / "data" / "perch" / "windows.csv")
    z = np.load(args.outdir / "clip_perch.npz")
    if len(z["prob"]) != len(w):
        raise SystemExit("clip_perch.npz does not cover windows.csv - rerun --stage clips")
    gi = list(z["groups"]).index("Insecta")
    esc_meta = pd.read_csv(args.outdir / "esc50_meta.csv")
    esc_emb = np.load(args.outdir / "esc50_emb.npy")
    ec = EarthChirpScore(args.lib, esc_emb, esc_meta)
    chk = esc50_check(ec, esc_emb, esc_meta, args.outdir)
    print(f"ESC-50 check (tau = {ec.tau:.3f}):\n" + chk.round(3).to_string(index=False), flush=True)

    E = z["emb"].astype(np.float32)
    w["perch_prob"] = z["prob"][:, gi]
    w["perch_maxlogit"] = z["maxlogit"][:, gi]
    w["ec_cos"] = ec.cos(E)
    w["ec_og"] = ec.og(E)
    w["ec_margin"] = w["ec_cos"] - w["ec_og"]
    w.to_csv(args.outdir / "window_scores.csv", index=False)

    flag_share = (w["perch_prob"] >= PERCH_INSECTA_THR).mean()
    matched = float(np.quantile(w["ec_cos"], 1 - flag_share))
    clip = w.groupby("clip_id", sort=False).agg(session_id=("session_id", "first"), coarse=("coarse", "first"),
                                                **{k: (k, "max") for k in SCORES})
    units = {"window": w, "clip": clip}

    auc_rows, op_rows = [], []
    for unit, d in units.items():
        y = (d["coarse"] == "insect").to_numpy()
        sess = d["session_id"].to_numpy()
        sc = {k: d[k].to_numpy() for k in SCORES}
        for neg in NEG_TYPES:
            mask = y | ((d["coarse"] != "insect") if neg == "all" else (d["coarse"] == neg)).to_numpy()
            for k, r in boot_auc(y, sc, sess, mask).items():
                auc_rows.append({"unit": unit, "negative": neg, "score": k, **r})
        ops = {f"Perch prob >= {PERCH_INSECTA_THR}": d["perch_prob"] >= PERCH_INSECTA_THR,
               f"EarthChirp cos > tau ({ec.tau:.3f}, ESC-50)": d["ec_cos"] > ec.tau,
               "EarthChirp outgroup (nearest is insect)": d["ec_margin"] > 0,
               f"EarthChirp cos > {matched:.3f} (Perch's window flag share)": d["ec_cos"] > matched}
        for name, flag in ops.items():
            flag = flag.to_numpy()
            row = {"unit": unit, "rule": name, "flagged_all": flag.mean()}
            for lab in ("insect", "bird", "anthropogenic", "amphibian", "geophony", "mammal"):
                sel = (d["coarse"] == lab).to_numpy()
                row[f"flagged_{lab}"] = flag[sel].mean()
            op_rows.append(row)
    auc = pd.DataFrame(auc_rows)
    op = pd.DataFrame(op_rows)
    auc.to_csv(args.outdir / "metrics_auc.csv", index=False)
    op.to_csv(args.outdir / "metrics_operating.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nAUROC (95% session-bootstrap CI); d = minus perch_prob, paired")
    for (unit, neg), g in auc.groupby(["unit", "negative"], sort=False):
        print(f"  {unit:6s} vs {neg:13s} (n_neg {g.n_neg.iloc[0]:5d}): " + "  ".join(
            f"{r.score} {r.auroc:.3f} [{r.auroc_lo:.3f},{r.auroc_hi:.3f}]"
            + (f" d {r.d_vs_perch:+.3f} [{r.d_lo:+.3f},{r.d_hi:+.3f}]" if r.score != "perch_prob" else "")
            for r in g.itertuples()))
    print("\nshare flagged (insect = recall)")
    print(op.round(3).to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("all", "esc50", "clips", "score"), default="all")
    ap.add_argument("--model", type=Path,
                    default=ROOT / "external" / "bacpipe_model_checkpoints" / "perch_v2" / "perch_v2_no_dft.onnx")
    ap.add_argument("--lib", type=Path, default=ROOT / "external" / "earthchirp_lib")
    ap.add_argument("--esc50-dir", type=Path, default=ROOT / "external" / "esc50" / "ESC-50-master")
    ap.add_argument("--outdir", type=Path, default=ROOT / "data" / "insecta_compare")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="clips stage: first N clips only, to a _limit file")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    need_esc = args.stage == "esc50" or (args.stage == "all" and not (args.outdir / "esc50_emb.npy").exists())
    need_clips = args.stage == "clips" or (args.stage == "all" and not (args.outdir / "clip_perch.npz").exists())
    if need_esc or need_clips:
        _, groups = labels_and_groups()
        sess = open_session(args.model, args.device)
        if need_esc:
            stage_esc50(args, sess, groups)
        if need_clips:
            stage_clips(args, sess, groups)
    if args.stage in ("all", "score") and not args.limit:
        stage_score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
