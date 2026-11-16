"""Per-cell score maps inside each 5 s window, to tighten the zero-shot boxes in time and frequency.

Perch 2.0 does not score a window as a whole. Its backbone gives a 16 x 4 grid of 1536-d cells
(`spatial_embedding`: 16 time cells of 312.5 ms x 4 frequency rows of 32 mel bins), and
  embedding = mean of the 64 cells                                                (exact)
  label     = sum_proto w * max_cells cos(cell, prototype) + b                    (ProtoPNet head)
with 4 unit prototypes per class (P: 1536 x 59,180, w: 14,795 x 4 >= 0, b: 14,795). So both
scores split into cells without any training:

  Perch lanes   the same head evaluated on ONE cell (no max over cells):
                  logit_cell = sum_proto w * cos(cell, prototype) + b
                softmax over the 14,795 classes, summed per group as in zeroshot_windows.py
                -> "what Perch would say if this cell were the whole window"
  EarthChirp    the window's insect cosine is cos(e, p*) with e = mean of cells and p* the best
                prototype, so it splits exactly: ec_cos = mean over cells of cell . p* / |e|.
                The lane value per cell is cell . p* / |e| (mean over the 64 cells = ec_cos);
                same for the outgroup, and margin = insect minus outgroup.

Frequency rows (HTK mel 60 Hz - 16 kHz, 128 bins; checked with a tone sweep): 0.06-0.95,
0.95-2.86, 2.86-6.97, 6.97-16 kHz. Perch hears nothing above 16 kHz. The viewer
(perch_viewer.py) turns these maps into tightened boxes; nothing here looks at labels.

Stages:
  build-head  export the ProtoPNet head from the Perch ONNX into a small per-cell model
              -> data/perch_windows/perch_cell_head.onnx (~363 MB). Needs the `onnx` package,
              which .venv-perch does not have; run once (pip install onnx) or keep the file.
  extract     Perch + the cell head on the 64,368 long-form windows -> cell_maps.npy
              (n_windows, 16, 4, 8) float16, rows in window_scores.csv order, + cell_maps_check.csv
              (~25 min GPU). Lanes: the 6 Perch groups, EC-Insecta (cosine), EC-margin.
              Checks per recording, else nothing is written (the .partial.npy is kept):
                group probabilities vs window_scores.csv                  <= 0.01
                head's rebuilt window logits vs the Perch ONNX `label`     <= 1e-3
                mean of the EC cells vs earthchirp_windows.csv ec_cos/margin <= 1e-3

    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\cell_maps.py                       # extract (GPU)
    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\cell_maps.py --limit 2 --device cpu   # smoke
    python -X utf8 scripts\\cell_maps.py --stage build-head                              # needs onnx

Reads window_scores.csv and earthchirp_windows.csv in data/perch_windows (both must exist),
dataset/, external/earthchirp_lib and data/insecta_compare/esc50_*. --limit N -> *_limit files.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import earthchirp_windows as EW  # noqa: E402
import insecta_compare as IC  # noqa: E402
import zeroshot_windows as z  # noqa: E402

ROOT = z.ROOT
HOP_S = EW.HOP_S
N_T, N_F, EMB_DIM = 16, 4, 1536
CELL_S = z.WINDOW_S / N_T                       # 0.3125 s
LANES = tuple(z.GROUPS) + ("EC-Insecta", "EC-margin")
ROW_EDGES_KHZ = (0.06, 0.955, 2.864, 6.971, 16.0)    # rows 0..3, low to high
N_CLASSES, N_PROTO = 14795, 4
EPS = 1e-5                                       # the ONNX head: cell / (sqrt(sum cell^2) + 1e-5)
MAX_PROB_DIFF, MAX_LOGIT_DIFF, MAX_EC_DIFF = 0.01, 1e-3, 1e-3
HEAD_NAME = "perch_cell_head.onnx"


# ---- build-head ---------------------------------------------------------------------------
def head_weights(model: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P (1536, 59180), w (1, 14795, 4), b (1, 14795) from the Perch ONNX, found by walking back
    from `label` (Add <- ReduceSum <- Mul(w) <- ReduceMax <- Reshape <- MatMul(P) <- Div)."""
    import onnx
    from onnx import numpy_helper
    g = onnx.load(str(model)).graph
    prod = {o: n for n in g.node for o in n.output}
    init = {i.name: i for i in g.initializer}

    def step(name, op):
        n = prod[name]
        if n.op_type != op:
            raise SystemExit(f"Perch head is not the expected ProtoPNet: {n.op_type} where {op} was expected")
        return n

    add = step("label", "Add")
    b = numpy_helper.to_array(init[add.input[1]])
    mul = step(step(add.input[0], "ReduceSum").input[0], "Mul")
    w = numpy_helper.to_array(init[mul.input[1]])
    resh = step(step(mul.input[0], "ReduceMax").input[0], "Reshape")
    mm = step(resh.input[0], "MatMul")
    P = numpy_helper.to_array(init[mm.input[1]])
    div = step(mm.input[0], "Div")
    if div.input[0] != "spatial_embedding":
        raise SystemExit("the prototype MatMul does not read spatial_embedding")
    eps = numpy_helper.to_array(init[step(div.input[1], "Add").input[1]])
    want = {"P": (P.shape, (EMB_DIM, N_CLASSES * N_PROTO)), "w": (w.shape, (1, N_CLASSES, N_PROTO)),
            "b": (b.shape, (1, N_CLASSES))}
    for k, (got, exp) in want.items():
        if tuple(got) != exp:
            raise SystemExit(f"head weight {k} has shape {got}, expected {exp}")
    if abs(float(eps) - EPS) > 1e-12:
        raise SystemExit(f"normalisation eps {float(eps)} != {EPS}")
    return P.astype(np.float32), w.astype(np.float32), b.astype(np.float32)


def stage_build_head(args) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    P, w, b = head_weights(args.model)
    _, groups = IC.labels_and_groups()
    G = np.zeros((N_CLASSES, len(z.GROUPS) + 1), np.float32)      # last column = other
    for j, g in enumerate(z.GROUPS):
        G[groups[g], j] = 1
    G[G.sum(1) == 0, -1] = 1
    if not (G.sum(1) == 1).all():
        raise SystemExit("a class belongs to more than one group")
    c = lambda name, a: numpy_helper.from_array(np.asarray(a), name)   # noqa: E731
    inits = [c("P", P), c("w", w), c("b", b), c("G", G), c("eps", np.float32(EPS)),
             c("ax_last", np.array([-1], np.int64)),
             c("shape5", np.array([-1, N_T, N_F, N_CLASSES, N_PROTO], np.int64))]
    nodes = [
        helper.make_node("Mul", ["x", "x"], ["x2"]),
        helper.make_node("ReduceSum", ["x2", "ax_last"], ["ss"], keepdims=1),
        helper.make_node("Sqrt", ["ss"], ["nrm"]),
        helper.make_node("Add", ["nrm", "eps"], ["nrm_e"]),
        helper.make_node("Div", ["x", "nrm_e"], ["xn"]),
        helper.make_node("MatMul", ["xn", "P"], ["sim"]),                  # (b, 16, 4, 59180)
        helper.make_node("Reshape", ["sim", "shape5"], ["sim5"]),          # (b, 16, 4, 14795, 4)
        # window logits exactly as the Perch graph: max over cells, then the prototype weights
        helper.make_node("ReduceMax", ["sim5"], ["smax"], axes=[1, 2], keepdims=0),
        helper.make_node("Mul", ["smax", "w"], ["wm"]),
        helper.make_node("ReduceSum", ["wm", "ax_last"], ["ws"], keepdims=0),
        helper.make_node("Add", ["ws", "b"], ["label"]),
        # per-cell logits: the same weights without the max over cells
        helper.make_node("Mul", ["sim5", "w"], ["cw"]),
        helper.make_node("ReduceSum", ["cw", "ax_last"], ["cs"], keepdims=0),
        helper.make_node("Add", ["cs", "b"], ["cell_logit"]),              # (b, 16, 4, 14795)
        helper.make_node("Softmax", ["cell_logit"], ["cell_p"], axis=-1),
        helper.make_node("MatMul", ["cell_p", "G"], ["cell_group_prob"]),  # (b, 16, 4, 7)
    ]
    graph = helper.make_graph(
        nodes, "perch_cell_head",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["b", N_T, N_F, EMB_DIM])],
        [helper.make_tensor_value_info("cell_group_prob", TensorProto.FLOAT, ["b", N_T, N_F, len(z.GROUPS) + 1]),
         helper.make_tensor_value_info("label", TensorProto.FLOAT, ["b", N_CLASSES])],
        inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    out = args.pdir / HEAD_NAME
    tmp = out.with_suffix(".partial.onnx")
    onnx.save(m, str(tmp))
    os.replace(tmp, out)
    print(f"wrote {out} ({out.stat().st_size / 2**20:.0f} MB): P {P.shape}, w {w.shape}, b {b.shape}, "
          f"G {G.shape} (class count per group: "
          + ", ".join(f"{g} {int(G[:, j].sum())}" for j, g in enumerate(z.GROUPS + ('other',))) + ")",
          flush=True)


# ---- extract ------------------------------------------------------------------------------
def ec_cells(sp: np.ndarray, ec: IC.EarthChirpScore) -> tuple[np.ndarray, np.ndarray]:
    """sp (n, 16, 4, 1536) -> per-cell insect and margin values (n, 16, 4), whose mean over the
    64 cells is the window's ec_cos / ec_margin (float64)."""
    sp = sp.astype(np.float64)
    e = sp.mean((1, 2))
    en = np.linalg.norm(e, axis=1)
    P, O = ec.P.astype(np.float64), ec.outgroup.astype(np.float64)
    best_p = P[np.argmax(e @ P.T, axis=1)]                 # argmax of the cosine = of the dot
    best_o = O[np.argmax(e @ O.T, axis=1)]
    ins = np.einsum("ntfd,nd->ntf", sp, best_p) / en[:, None, None]
    og = np.einsum("ntfd,nd->ntf", sp, best_o) / en[:, None, None]
    return ins, ins - og


def group_probs(logit: np.ndarray, groups: dict) -> np.ndarray:
    logit = logit.astype(np.float64)
    p = np.exp(logit - logit.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return np.stack([p[:, groups[g]].sum(axis=1) for g in z.GROUPS], 1)


def stage_extract(args) -> None:
    head = args.pdir / HEAD_NAME
    if not head.exists():
        raise SystemExit(f"{head} missing: run --stage build-head first (needs pip install onnx)")
    w = EW.cached_windows(args)
    ecw = pd.read_csv(args.pdir / "earthchirp_windows.csv")
    ecw = ecw[ecw["recording"].isin(set(w["recording"]))].reset_index(drop=True)
    if not (ecw["recording"].equals(w["recording"]) and np.allclose(ecw["start_s"], w["start_s"])):
        raise SystemExit("earthchirp_windows.csv does not line up with window_scores.csv")
    recs = list(pd.unique(w["recording"]))
    rows = {r: g for r, g in w.groupby("recording", sort=False)}
    wavs = EW.wav_paths(args.audio_dir)
    missing = [r for r in recs if r not in wavs]
    if missing:
        raise SystemExit(f"recordings in window_scores.csv with no wav: {missing}")
    _, groups = IC.labels_and_groups()
    sess = IC.open_session(args.model, args.device)
    hsess = IC.open_session(head, args.device)
    ec = EW.earthchirp(args)
    prob_cols = [f"prob_{g}" for g in z.GROUPS]

    final = args.pdir / f"cell_maps{EW.suffix(args)}.npy"
    tmp = final.with_suffix(".partial.npy")
    M = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(w), N_T, N_F, len(LANES)))
    check, row, t0 = [], 0, time.monotonic()
    with cf.ThreadPoolExecutor(1) as pool:          # read the next file while the GPU runs
        nxt = pool.submit(z.load_windows, wavs[recs[0]], HOP_S)
        for i, rec in enumerate(recs):
            wins = nxt.result()
            if i + 1 < len(recs):
                nxt = pool.submit(z.load_windows, wavs[recs[i + 1]], HOP_S)
            want = rows[rec]
            if len(wins) != len(want) or not np.allclose(want["start_s"], np.arange(len(wins)) * HOP_S):
                raise SystemExit(f"{rec}: {len(wins)} rebuilt windows != {len(want)} in window_scores.csv")
            if want.index[0] != row:
                raise SystemExit(f"{rec}: window_scores.csv rows are not contiguous per recording")
            d_prob = d_logit = d_ec = d_pool = 0.0
            for k in range(0, len(wins), args.batch):
                e, sp, lab = sess.run(["embedding", "spatial_embedding", "label"],
                                      {"inputs": IC.normalize_windows(wins[k:k + args.batch].copy(), 0.25)})
                sp = np.asarray(sp, np.float32)
                cg, hl = hsess.run(["cell_group_prob", "label"], {"x": sp})
                ins, mar = ec_cells(sp, ec)
                n = len(sp)
                ref = want.iloc[k:k + n]
                refe = ecw.iloc[row + k:row + k + n]
                d_prob = max(d_prob, float(np.abs(group_probs(lab, groups) - ref[prob_cols].to_numpy()).max()))
                d_logit = max(d_logit, float(np.abs(hl - lab).max()))
                d_ec = max(d_ec, float(np.abs(ins.mean((1, 2)) - refe["ec_cos"].to_numpy()).max()),
                           float(np.abs(mar.mean((1, 2)) - refe["ec_margin"].to_numpy()).max()))
                d_pool = max(d_pool, float(np.abs(sp.mean((1, 2)) - np.asarray(e).reshape(n, -1)).max()))
                M[row + k:row + k + n] = np.concatenate(
                    [cg[..., :len(z.GROUPS)], ins[..., None], mar[..., None]], -1).astype(np.float16)
            check.append({"recording": rec, "windows": len(wins), "max_prob_diff": d_prob,
                          "max_head_logit_diff": d_logit, "max_ec_diff": d_ec, "max_pool_diff": d_pool})
            row += len(wins)
            if (i + 1) % 10 == 0 or i + 1 == len(recs):
                el = time.monotonic() - t0
                c = pd.DataFrame(check)
                print(f"  {i + 1}/{len(recs)} recordings, {row}/{len(w)} windows, {row / el:.1f} win/s, "
                      f"~{(len(w) - row) / (row / el) / 60:.1f} min left; max diffs so far: prob "
                      f"{c.max_prob_diff.max():.1e}, head logit {c.max_head_logit_diff.max():.1e}, "
                      f"EC {c.max_ec_diff.max():.1e}", flush=True)
    M.flush()
    del M
    chk = pd.DataFrame(check)
    chk.to_csv(args.pdir / f"cell_maps_check{EW.suffix(args)}.csv", index=False)
    lim = {"max_prob_diff": MAX_PROB_DIFF, "max_head_logit_diff": MAX_LOGIT_DIFF, "max_ec_diff": MAX_EC_DIFF}
    bad = [f"{k} {chk[k].max():.2e} > {v}" for k, v in lim.items() if chk[k].max() > v]
    print("checks over " + f"{len(chk)} recordings: " + ", ".join(f"{k} {chk[k].max():.2e} (limit {v})"
                                                                  for k, v in lim.items())
          + f", embedding vs mean of cells {chk['max_pool_diff'].max():.1e}", flush=True)
    if bad:
        raise SystemExit("check failed: " + "; ".join(bad) + f"; maps left in {tmp}")
    os.replace(tmp, final)
    print(f"wrote {final} ({row} windows x {N_T} x {N_F} x {len(LANES)} lanes: {', '.join(LANES)})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("extract", "build-head"), default="extract")
    ap.add_argument("--model", type=Path,
                    default=ROOT / "external" / "bacpipe_model_checkpoints" / "perch_v2" / "perch_v2_no_dft.onnx")
    ap.add_argument("--lib", type=Path, default=ROOT / "external" / "earthchirp_lib")
    ap.add_argument("--audio-dir", type=Path, default=ROOT / "dataset")
    ap.add_argument("--pdir", type=Path, default=ROOT / "data" / "perch_windows")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N recordings only, to *_limit files")
    args = ap.parse_args()
    if args.stage == "build-head":
        stage_build_head(args)
    else:
        stage_extract(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
