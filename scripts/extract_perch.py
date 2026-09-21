"""Run Perch 2.0 over the labelled corpus and cache embeddings + target logits.

This is the one compute-heavy stage. It runs in ~20 min on a laptop RTX GPU via
--model-name perch_v2_onnx --device cuda, or ~3.4 h on CPU. It writes to
--outdir, after which scoring needs nothing but numpy:

  windows.csv           one row per 5 s window, keyed back to clip_id
  embeddings.npy        (n_windows, 1536) float16
  target_logits.npy     (n_windows, n_targets) float32, columns = resolved targets
  target_names.txt      column order for the above
  targets_resolved.csv  which Perch class each of our labels mapped onto
  class_lists/<head>.csv  every class list the model exposes, dumped for reference
  topk.npz              top-K species indices/scores per window, for error analysis

Full logits are deliberately not stored: ~15k classes x ~33k windows is 2 GB of
mostly-noise. Target columns plus a top-K tail covers both scoring and the
"what did it actually think this was" diagnostics.

Perch exposes several output heads (species, and typically higher ranks such as
genus/family/order), each with its own class list. Species targets resolve
against the species head and aggregate targets such as Orthoptera against a
higher-rank head. A target never mixes heads, since their logit scales are not
comparable.

Perch expects 5-second windows of 32 kHz mono. Source audio is 48 kHz, so the
resample is an exact 2/3 polyphase - no interpolation error.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

PERCH_SR = 32_000
WINDOW_S = 5.0
WINDOW_N = int(PERCH_SR * WINDOW_S)  # 160_000


def norm(s: str) -> str:
    """Loosely normalise a label so 'Fixed-wing_aircraft' ~ 'fixed wing aircraft'."""
    return " ".join("".join(c if c.isalnum() else " " for c in s.lower()).split())


def load_audio_windows(path: Path, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (n_windows, WINDOW_N) float32 at 32 kHz, plus each window's start time."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)  # corpus is mono already; guard anyway
    if sr != PERCH_SR:
        g = int(np.gcd(sr, PERCH_SR))
        x = resample_poly(x, PERCH_SR // g, sr // g).astype(np.float32)
    if len(x) < WINDOW_N:
        x = np.pad(x, (0, WINDOW_N - len(x)))
    hop = int(PERCH_SR * hop_s)
    starts = list(range(0, len(x) - WINDOW_N + 1, hop))
    # Keep a final flush-right window so the tail of a clip is never silently dropped.
    if starts[-1] + WINDOW_N < len(x):
        starts.append(len(x) - WINDOW_N)
    return np.stack([x[s:s + WINDOW_N] for s in starts]), np.array(starts) / PERCH_SR


def normalize_windows(x: np.ndarray, target_peak: float | None) -> np.ndarray:
    """Peak-normalise [B, T] the way zoo_interface.EmbeddingModel.normalize_audio does.

    The direct-session path below bypasses perch_hoplite's wrapper, which is
    what would otherwise apply this. Skipping it does not fail loudly - it just
    shifts every logit - so the arithmetic is replicated here rather than
    assumed harmless. One deliberate difference: a digitally silent window keeps
    its zeros instead of picking up whatever np.divide left in the unwritten
    output buffer.
    """
    if target_peak is None:
        return x
    x = x - x.mean(axis=-1, keepdims=True)
    peak = np.abs(x).max(axis=-1, keepdims=True)
    return np.divide(x, peak, out=x, where=(peak > 0.0)) * target_peak


def add_nvidia_dll_dirs() -> list[str]:
    """Make the pip-installed NVIDIA runtime loadable by onnxruntime-gpu.

    Two mechanisms are needed, not one: onnxruntime opens cudnn itself, which
    os.add_dll_directory covers, but cuDNN 9 then pulls its own sublibraries in
    by bare name through LoadLibrary, which consults PATH only. With just the
    former, CUDA initialises and then dies at the first convolution with
    CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED.
    """
    base = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
    if not os.path.isdir(base):  # posix layout
        base = os.path.join(sys.prefix, "lib", "site-packages", "nvidia")
    dirs = sorted(glob.glob(os.path.join(base, "*", "bin"))
                  + glob.glob(os.path.join(base, "*", "lib")))
    if not dirs:
        return []
    os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ["PATH"]
    for d in dirs:
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(d)
    return dirs


def open_session(model_path: str, device: str):
    """Open an ONNX Runtime session, returning it with the providers really in use."""
    import onnxruntime as ort

    if device != "cpu":
        add_nvidia_dll_dirs()
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    want = ["CPUExecutionProvider"] if device == "cpu" else [
        "CUDAExecutionProvider", "CPUExecutionProvider"]
    avail = ort.get_available_providers()
    sess = ort.InferenceSession(model_path, providers=[p for p in want if p in avail])
    return sess, sess.get_providers()


def resolve_targets(targets_csv: Path, classes: list[str], index_rows: list[dict]):
    """Map each fine class onto a list of Perch output indices.

    A query is a ';'-separated list of alternatives, unioned. Each is either a
    literal class name, or one of the expansions below. Perch v2 has a single
    flat head of 14,795 classes with no higher-rank outputs, so "any Orthoptera"
    has to become "the 380-odd Orthoptera species in the list".

      Mixornis gularis     literal label (loosely normalised)
      genus:Purana         every species in that genus
      family=Cicadidae     every species in that family   (likewise order=, class=)
      ebird:any            every class carrying an eBird code, i.e. every bird
    """
    by_name: dict[str, list[int]] = {}
    for i, c in enumerate(classes):
        by_name.setdefault(norm(c), []).append(i)

    def expand(q: str) -> list[int]:
        q = q.strip()
        if q == "ebird:any":
            return [i for i, r in enumerate(index_rows) if r["ebird"]]
        if q.startswith("genus:"):
            g = norm(q.split(":", 1)[1])
            return [i for i, c in enumerate(classes) if norm(c).split(" ")[0] == g]
        for rank in ("class", "order", "family"):
            if q.startswith(rank + "="):
                want = norm(q.split("=", 1)[1])
                return [i for i, r in enumerate(index_rows) if norm(r[rank]) == want]
        return by_name.get(norm(q), [])

    resolved: dict[str, list[int]] = {}
    report: list[tuple] = []
    for row in csv.DictReader(targets_csv.open(encoding="utf-8")):
        fine, strategy = row["fine"], row["strategy"]
        cols, matched, unmatched = [], [], []
        for cand in row["query"].split(";"):
            hits = expand(cand)
            if not hits:
                unmatched.append(cand.strip())
            for i in hits:
                if i not in cols:
                    cols.append(i)
                    matched.append(classes[i])
        if cols:
            resolved[fine] = sorted(cols)
            # Long expansions get a sample rather than 10k names in a CSV cell.
            shown = "|".join(matched[:8]) + ("|..." if len(matched) > 8 else "")
            report.append((fine, strategy, "ok", len(cols), shown, ";".join(unmatched)))
        else:
            report.append((fine, strategy, "UNMATCHED", 0, "", ";".join(unmatched)))
    return resolved, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--model-name", default="perch_v2",
                    help="perch_hoplite model name (perch_v2, perch_v2_cpu, birdnet_v2.4, ...)")
    ap.add_argument("--model-path", default=None,
                    help="Unpacked SavedModel dir, for compute nodes without internet")
    ap.add_argument("--species-head", default=None,
                    help="Name of the species class list; default = the largest head")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--hop-s", type=float, default=5.0, help="Window hop in seconds; 2.5 gives 50%% overlap")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"),
                    help="Only applies to ONNX models. 'cuda' fails loudly if the "
                         "GPU is unavailable; 'auto' reports the fallback")
    ap.add_argument("--batch", type=int, default=0,
                    help="0 = 16 on GPU (6 GB cards OOM above ~32), 64 otherwise")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="Process only N clips (smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load model, resolve targets, write the report, then stop")
    args = ap.parse_args()

    # Imported late so --help works without TF installed.
    from perch_hoplite.zoo import model_configs

    root = args.root
    outdir = args.outdir or root / "data" / "perch"
    (outdir / "class_lists").mkdir(parents=True, exist_ok=True)

    if args.model_path:
        from ml_collections import config_dict
        from perch_hoplite.zoo import taxonomy_model_tf
        cfg = config_dict.ConfigDict({
            "model_path": args.model_path, "sample_rate": PERCH_SR,
            "window_size_s": WINDOW_S, "hop_size_s": WINDOW_S,
        })
        model = taxonomy_model_tf.TaxonomyModelTF.from_config(cfg)
    else:
        model = model_configs.load_model_by_name(args.model_name)

    head_classes = {h: list(cl.classes) for h, cl in model.class_list.items()}
    for h, cl in head_classes.items():
        (outdir / "class_lists" / (h + ".csv")).write_text("\n".join(cl), encoding="utf-8")
    print("model heads: " + ", ".join(f"{h}({len(c)})" for h, c in head_classes.items()), flush=True)

    species_head = args.species_head or max(head_classes, key=lambda h: len(head_classes[h]))
    classes = head_classes[species_head]
    print(f"using '{species_head}' as the species head", flush=True)

    # Taxonomy for the flat class list, built once by build_taxonomy_map.py and
    # committed, so this stage needs no network. Alignment is checked rather
    # than assumed: a model version bump would silently shift every index.
    index_rows = list(csv.DictReader((root / "config" / "perch_class_index.csv")
                                     .open(encoding="utf-8")))
    if [r["label"] for r in index_rows] != classes:
        print(f"config/perch_class_index.csv ({len(index_rows)} rows) does not match the "
              f"model's '{species_head}' head ({len(classes)} classes). Rerun "
              "scripts/build_taxonomy_map.py against this model's assets.", file=sys.stderr)
        return 1

    # perch_hoplite keys class_list by asset filename ('labels') but keys logits
    # by the head's own name ('label'); the two are not required to agree. Probe
    # once with 5 s of silence and match on width rather than trusting a spelling.
    probe = model.embed(np.zeros(WINDOW_N, dtype=np.float32))
    cand = [k for k, v in probe.logits.items() if np.asarray(v).shape[-1] == len(classes)]
    if len(cand) != 1:
        print(f"cannot identify the logits key for head '{species_head}' "
              f"({len(classes)} classes); logits offers "
              f"{ {k: np.asarray(v).shape for k, v in probe.logits.items()} }", file=sys.stderr)
        return 1
    logit_key = cand[0]
    print(f"logits key: '{logit_key}'", flush=True)

    # Direct-session fast path for ONNX. perch_hoplite's ONNX wrapper inherits
    # batch_embed_from_embed_fn, which feeds the session one window at a time,
    # so going through it would throw away most of what a GPU buys: measured on
    # this laptop, 3.7 win/s on TF CPU, 7.7 batched on ONNX CPU, 26.7 on CUDA.
    session = None
    onnx_path = str(getattr(model, "model_path", "") or "")
    if onnx_path.endswith(".onnx"):
        session, got = open_session(onnx_path, args.device)
        on_gpu = any(p.startswith("CUDA") for p in got)
        print(f"onnxruntime providers: {', '.join(got)}", flush=True)
        if args.device == "cuda" and not on_gpu:
            print("--device cuda requested but onnxruntime gave us "
                  f"{got}. Check that onnxruntime-gpu matches the driver's CUDA "
                  "version (1.22 for CUDA 12, 1.30 for CUDA 13) and that the "
                  "nvidia-*-cu12 wheels are installed.", file=sys.stderr)
            return 1
        if args.device == "auto" and not on_gpu:
            print("  no CUDA provider - running on CPU, roughly 3.5x slower",
                  file=sys.stderr)
        out_names = [model.output_map["embedding"], model.output_map["logits"]]
        in_name, target_peak = model.input_name, model.target_peak
        print(f"peak normalisation: {target_peak}", flush=True)
    elif args.device == "cuda":
        print(f"--device cuda needs an ONNX model; '{args.model_name}' is a "
              "SavedModel. Rerun with --model-name perch_v2_onnx.", file=sys.stderr)
        return 1
    batch_n = args.batch or (16 if session is not None and
                             any(p.startswith("CUDA") for p in session.get_providers())
                             else 64)

    targets, report = resolve_targets(root / "config" / "perch_targets.csv",
                                      classes, index_rows)
    with (outdir / "targets_resolved.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fine", "strategy", "status", "n_classes", "sample_classes",
                    "unmatched_queries"])
        w.writerows(report)
    n_ok = sum(1 for r in report if r[2] == "ok")
    print(f"resolved {n_ok}/{len(report)} targets -> {outdir / 'targets_resolved.csv'}", flush=True)
    for r in report:
        if r[2] != "ok":
            print(f"  UNMATCHED: {r[0]} ({r[5]})", file=sys.stderr)
        elif r[5]:
            print(f"  partial: {r[0]} -> {r[3]} classes; no match for {r[5]}", file=sys.stderr)
    if not targets:
        print("no targets resolved - check class list dumps in "
              f"{outdir / 'class_lists'}", file=sys.stderr)
        return 1
    if args.dry_run:
        print("dry run: stopping before inference")
        return 0

    target_names = sorted(targets)

    rows = [r for r in csv.DictReader((root / "data" / "manifest.csv").open(encoding="utf-8"))
            if r["keep"] == "1"]
    if args.limit:
        rows = rows[:args.limit]

    win_index: list[tuple] = []
    embs: list[np.ndarray] = []
    tlogits: list[np.ndarray] = []
    topi: list[np.ndarray] = []
    topv: list[np.ndarray] = []
    buf: list[np.ndarray] = []
    buf_meta: list[tuple] = []

    def flush() -> None:
        if not buf:
            return
        batch = np.stack(buf)
        if session is None:
            out = model.batch_embed(batch)
            raw_emb, raw_sp = out.embeddings, out.logits[logit_key]
        else:
            # The session takes framed audio as [Frames, Samples]; our windows
            # are already exactly one frame each, so a batch is the same shape.
            raw_emb, raw_sp = session.run(
                out_names, {in_name: normalize_windows(batch, target_peak)})
        # embeddings arrive as [B, Frames, Channels, Features]; each window is
        # exactly one frame, so collapse everything but the batch axis.
        emb = np.asarray(raw_emb).reshape(len(buf), -1)
        embs.append(emb.astype(np.float16))

        sp = np.asarray(raw_sp).reshape(len(buf), -1)
        # A target may match several classes (synonyms, or every species in an
        # order); take the max as "best evidence for this label".
        tlogits.append(np.stack([sp[:, targets[n]].max(axis=1) for n in target_names],
                                axis=1).astype(np.float32))

        k = min(args.topk, sp.shape[1])
        idx = np.argpartition(-sp, k - 1, axis=1)[:, :k]
        val = np.take_along_axis(sp, idx, axis=1)
        order = np.argsort(-val, axis=1)
        topi.append(np.take_along_axis(idx, order, axis=1).astype(np.int32))
        topv.append(np.take_along_axis(val, order, axis=1).astype(np.float32))

        win_index.extend(buf_meta)
        buf.clear()
        buf_meta.clear()

    def n_windows(duration_s: float) -> int:
        """Window count for a clip, mirroring load_audio_windows exactly."""
        n = max(int(round(duration_s * PERCH_SR)), WINDOW_N)
        hop = int(PERCH_SR * args.hop_s)
        k = (n - WINDOW_N) // hop + 1
        return k + (1 if (k - 1) * hop + WINDOW_N < n else 0)

    # Extrapolate on windows, not clips: the first N rows of the manifest are not
    # length-representative, and cost is per window.
    all_rows = [r for r in csv.DictReader((root / "data" / "manifest.csv").open(encoding="utf-8"))
                if r["keep"] == "1"]
    n_all, w_all = len(all_rows), sum(n_windows(float(r["duration_s"])) for r in all_rows)
    print(f"corpus: {n_all} clips -> {w_all} windows at hop {args.hop_s}s", flush=True)
    t0 = time.monotonic()
    for n, r in enumerate(rows, 1):
        try:
            wins, starts = load_audio_windows(root / r["path"], args.hop_s)
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill a cluster job
            print(f"SKIP {r['path']}: {exc}", file=sys.stderr)
            continue
        for w, s in zip(wins, starts):
            buf.append(w)
            buf_meta.append((r["clip_id"], r["session_id"], r["fine"], r["coarse"], round(float(s), 2)))
            if len(buf) >= batch_n:
                flush()
        if n % 200 == 0:
            el = time.monotonic() - t0
            print(f"  {n}/{len(rows)} clips, {len(win_index) + len(buf)} windows, "
                  f"{(len(win_index) + len(buf)) / el:.1f} win/s", flush=True)
    flush()
    elapsed = time.monotonic() - t0

    np.save(outdir / "embeddings.npy", np.concatenate(embs))
    np.save(outdir / "target_logits.npy", np.concatenate(tlogits))
# Plain unicode, not dtype=object: an object array in an .npz can only
    # be read back with allow_pickle=True, which analysis code should not need.
    np.savez_compressed(outdir / "topk.npz", idx=np.concatenate(topi), val=np.concatenate(topv),
                        labels=np.array(classes))
    with (outdir / "windows.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "session_id", "fine", "coarse", "start_s"])
        w.writerows(win_index)
    (outdir / "target_names.txt").write_text("\n".join(target_names), encoding="utf-8")
    print(f"wrote {len(win_index)} windows x {len(target_names)} targets -> {outdir}")
    # On a subset, extrapolate so a smoke test answers "is CPU fast enough?".
    rate = len(win_index) / elapsed if elapsed > 0 else float("nan")
    print(f"{elapsed / 60:.1f} min for {len(rows)} clips ({rate:.1f} windows/s)")
    if len(win_index) < w_all and rate > 0:
        print(f"extrapolated to all {w_all} windows: ~{w_all / rate / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
