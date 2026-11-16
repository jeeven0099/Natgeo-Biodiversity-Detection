"""Run the whole pipeline on synthetic recordings and render the README figures.

    python examples/synthetic_demo.py

Writes everything under results/demo/ (git-ignored) and the figures to docs/figures/.
Because the events are synthetic, their true times and bands are known, so the script
also reports how many events overlap a box: a sanity check of the implementation, not
an evaluation on real soundscapes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from biodetect import clustering, figure, segmentation, speckle_null, synthetic  # noqa: E402
from biodetect.viewer import gallery  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "results" / "demo"
FIGS = ROOT / "docs" / "figures"
MIN_CELLS = 50


def event_recall(events: pd.DataFrame, boxes: pd.DataFrame, tol_hz: float = 100.0) -> pd.Series:
    """Share of events of each kind that overlap at least one box in time and frequency."""
    def hit(e):
        b = boxes[boxes.file == e.file]
        return bool(((b.begin_s < e.end_s) & (b.end_s > e.begin_s)
                     & (b.low_hz <= e.high_hz + tol_hz) & (b.high_hz >= e.low_hz - tol_hz)).any())
    return events.assign(hit=[hit(e) for e in events.itertuples()]).groupby("kind").hit.mean()


def main() -> None:
    audio = DEMO / "audio"
    events = synthetic.write(audio, n_files=3, duration_s=120.0, seed=0)
    files = segmentation.find_wavs([audio])

    full = segmentation.run(files, DEMO / "boxes", log=lambda *_: None)
    tab = speckle_null.run(DEMO / "boxes", pieces=6)
    kept = segmentation.run(files, DEMO / f"boxes_min{MIN_CELLS}", min_cells=MIN_CELLS, log=lambda *_: None)
    paper = clustering.run(DEMO / f"boxes_min{MIN_CELLS}", log=lambda *_: None)
    rank = clustering.run(DEMO / f"boxes_min{MIN_CELLS}", scaling="rank", skip_features=True, log=lambda *_: None)

    print(f"{len(events)} synthetic events in {len(files)} recordings")
    print(f"boxes: {len(full)} (paper), {len(kept)} at min_cells {MIN_CELLS}")
    print("noise-null cutoff table:")
    print(tab[["min_cells", "real_per_piece", "noise_share_paper", "noise_share_same_thr", "real_kept_frac"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("events overlapping a box (all boxes / kept boxes):")
    print(pd.DataFrame({"all": event_recall(events, full), f"min{MIN_CELLS}": event_recall(events, kept)}).round(3))
    for name, info in [("minmax (paper)", paper), ("rank", rank)]:
        print(f"LAMDA {name}: {info['clusters']} clusters, AMI band {info['AMI']['band']:.3f}, "
              f"duration {info['AMI']['duration']:.3f}, recording {info['AMI']['file']:.3f}")

    FIGS.mkdir(parents=True, exist_ok=True)
    png = figure.render(audio / "synthetic_01.wav", 0.0, 60.0, min_cells=MIN_CELLS, view=(0.0, 20.0),
                        out=FIGS / "pipeline_synthetic.png")
    clus = Path(rank["output"]).resolve()
    s = gallery.cluster_summary(clus)
    fig = gallery.plot_tiles(clus, gallery.sample_cluster(clus, int(s.index[0]), n=8, seed=0), cols=4,
                             title=f"rank-LAMDA cluster {int(s.index[0])}: tile [0] is its representative")
    fig.savefig(FIGS / "cluster_tiles_synthetic.png", dpi=100)
    plt.close(fig)
    (DEMO / "summary.json").write_text(json.dumps({
        "events": len(events), "boxes": len(full), f"boxes_min{MIN_CELLS}": len(kept),
        "clusters_minmax": paper["clusters"], "clusters_rank": rank["clusters"]}, indent=1))
    print("figures ->", png.parent)


if __name__ == "__main__":
    main()
