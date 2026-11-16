"""Command line: `biodetect segment | speckle-null | cluster | figure | demo-data`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _segment(a: argparse.Namespace) -> int:
    from . import segmentation as seg

    files = seg.find_wavs(a.inputs)
    out = a.out or Path("results") / ("boxes" if a.min_cells <= 0 else f"boxes_min{a.min_cells}")
    boxes = seg.run(files, out, a.chunk_s, a.min_cells, a.workers)
    print(f"{len(boxes)} boxes from {len(files)} recordings -> {out}")
    return 0


def _speckle(a: argparse.Namespace) -> int:
    from . import speckle_null

    tab = speckle_null.run(a.box_dir, a.pieces, a.seed, a.workers)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("->", Path(a.box_dir) / "speckle_null")
    return 0


def _cluster(a: argparse.Namespace) -> int:
    from . import clustering

    info = clustering.run(a.box_dir, a.out, a.scaling, a.shuffle_seed, a.iterations, a.frame_cols,
                          a.skip_features, a.files, a.workers)
    keys = ["boxes", "clusters", "clusters_nonempty", "clusters_ge10", "largest_cluster_share", "AMI",
            "AMI_vs_run_order", "seconds_lamda"]
    print(json.dumps({k: info[k] for k in keys if k in info}, indent=1))
    print("->", info["output"])
    return 0


def _figure(a: argparse.Namespace) -> int:
    from . import figure

    view = tuple(a.view) if a.view else None
    print(figure.render(a.wav, a.start, a.chunk_s, a.top_khz, a.min_cells, view, a.out))
    return 0


def _demo_data(a: argparse.Namespace) -> int:
    from . import synthetic

    ev = synthetic.write(a.folder, a.files, a.duration_s, seed=a.seed)
    print(f"{len(ev)} synthetic events in {a.files} recordings -> {a.folder}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="biodetect", description="Unsupervised detection of animal sounds in "
                                "soundscape recordings (Guerrero et al. 2023 pipeline).")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("segment", help="noise reduction + Otsu segmentation -> bounding boxes")
    s.add_argument("inputs", nargs="+", help=".wav files or folders (searched recursively)")
    s.add_argument("--out", type=Path, default=None, help="run folder (default results/boxes[_min<N>])")
    s.add_argument("--chunk-s", type=float, default=60.0, help="piece length in seconds (paper: 1 min recordings)")
    s.add_argument("--min-cells", type=int, default=0,
                   help="drop segments smaller than N cells (NOT in the paper; 0 = paper)")
    s.add_argument("--workers", type=int, default=1)
    s.set_defaults(fn=_segment)

    s = sub.add_parser("speckle-null", help="noise-null table for choosing --min-cells without labels")
    s.add_argument("box_dir", type=Path, help="a segment run made WITHOUT --min-cells")
    s.add_argument("--pieces", type=int, default=60)
    s.add_argument("--seed", type=int, default=20261015)
    s.add_argument("--workers", type=int, default=1)
    s.set_defaults(fn=_speckle)

    s = sub.add_parser("cluster", help="27 features per box + LAMDA clustering")
    s.add_argument("box_dir", type=Path)
    s.add_argument("--out", type=Path, default=None, help="default <box_dir>/lamda")
    s.add_argument("--scaling", choices=["minmax", "rank"], default="minmax",
                   help="minmax = paper; rank = NOT in the paper (percentiles), written to <out>/rank")
    s.add_argument("--shuffle-seed", type=int, default=None, help="present the boxes in a random order")
    s.add_argument("--iterations", type=int, default=10)
    s.add_argument("--frame-cols", type=int, default=0, help="w of eq. 1; 0 = one frame per box")
    s.add_argument("--skip-features", action="store_true", help="reuse <out>/features.csv.gz")
    s.add_argument("--files", nargs="*", default=None, help="only these recordings")
    s.add_argument("--workers", type=int, default=1)
    s.set_defaults(fn=_cluster)

    s = sub.add_parser("figure", help="four-panel figure of one piece")
    s.add_argument("wav", type=Path)
    s.add_argument("--start", type=float, default=0.0, help="piece start (s)")
    s.add_argument("--chunk-s", type=float, default=60.0)
    s.add_argument("--top-khz", type=float, default=None)
    s.add_argument("--min-cells", type=int, default=0, help="draw boxes below this size dotted")
    s.add_argument("--view", type=float, nargs=2, default=None, metavar=("FROM_S", "TO_S"))
    s.add_argument("--out", type=Path, default=None)
    s.set_defaults(fn=_figure)

    s = sub.add_parser("demo-data", help="write synthetic recordings with known events")
    s.add_argument("folder", type=Path)
    s.add_argument("--files", type=int, default=3)
    s.add_argument("--duration-s", type=float, default=120.0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=_demo_data)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except (FileNotFoundError, ValueError) as e:
        print(f"biodetect: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
