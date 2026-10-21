"""End to end through the command line, on two synthetic recordings."""
import json

import numpy as np
import pandas as pd

from biodetect import features as feat
from biodetect.cli import main


def test_pipeline_end_to_end(wav_dir, tmp_path):
    boxes_dir = tmp_path / "boxes"
    assert main(["segment", str(wav_dir), "--out", str(boxes_dir)]) == 0
    files = pd.read_csv(boxes_dir / "files.csv")
    assert sorted(files.file) == ["synthetic_01", "synthetic_02"]
    boxes = pd.read_csv(boxes_dir / "boxes.csv.gz")
    pieces = pd.read_csv(boxes_dir / "per_piece.csv")
    assert len(pieces) == 2                                       # 70 s = one piece, the 10 s tail joins it
    assert pieces.segments_kept.sum() == len(boxes)

    raven = pd.read_csv(boxes_dir / "raven_tables" / "synthetic_01.Guerrero.Table.1.selections.txt", sep="\t")
    assert list(raven.columns[:7]) == ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)",
                                       "Low Freq (Hz)", "High Freq (Hz)"]
    assert len(raven) == (boxes.file == "synthetic_01").sum()

    assert main(["speckle-null", str(boxes_dir), "--pieces", "2"]) == 0
    tab = pd.read_csv(boxes_dir / "speckle_null" / "cutoff_table.csv")
    assert tab.real_kept_frac.iloc[0] == 1.0 and tab.real_kept_frac.is_monotonic_decreasing

    min_dir = tmp_path / "boxes_min50"
    assert main(["segment", str(wav_dir), "--out", str(min_dir), "--min-cells", "50"]) == 0
    kept = pd.read_csv(min_dir / "boxes.csv.gz")
    assert len(kept) == (boxes.n_cells >= 50).sum() > 0

    assert main(["cluster", str(min_dir)]) == 0
    assert main(["cluster", str(min_dir), "--scaling", "rank", "--skip-features"]) == 0
    assert main(["cluster", str(min_dir), "--scaling", "rank", "--skip-features", "--shuffle-seed", "1"]) == 0
    lam = min_dir / "lamda"
    f = pd.read_csv(lam / "features.csv.gz")
    assert len(f) == len(kept) and np.isfinite(f[feat.FEATURES].to_numpy()).all()
    assert (f.nu * f.nh >= f.n_cells).all()
    for d in (lam, lam / "rank", lam / "rank" / "shuffle1"):
        info = json.loads((d / "run_info.json").read_text())
        b = pd.read_csv(d / "boxes_clustered.csv.gz")
        assert info["boxes"] == len(b) == len(kept)
        assert b.cluster.max() < info["clusters"]
        assert len(pd.read_csv(d / "representatives.csv")) == info["clusters_nonempty"]
    assert "AMI_vs_run_order" in json.loads((lam / "rank" / "shuffle1" / "run_info.json").read_text())

    png = tmp_path / "fig.png"
    assert main(["figure", str(wav_dir / "synthetic_01.wav"), "--out", str(png), "--min-cells", "50"]) == 0
    assert png.stat().st_size > 10_000


def test_viewer_functions_without_widgets(wav_dir, tmp_path):
    from biodetect.viewer import gallery
    from biodetect.viewer.box_viewer import find_runs, plot_window

    out = tmp_path / "boxes"
    main(["segment", str(wav_dir / "synthetic_01.wav"), "--out", str(out), "--min-cells", "30"])
    main(["cluster", str(out)])
    assert list(find_runs(tmp_path)) == ["boxes"]
    plot_window(out, "synthetic_01", 0.0, 10.0)
    plot_window(out, "synthetic_01", 0.0, 10.0, view="Otsu mask")
    clus = out / "lamda"
    s = gallery.cluster_summary(clus)
    assert s.boxes.sum() == len(pd.read_csv(out / "boxes.csv.gz"))
    sample = gallery.sample_cluster(clus, int(s.index[0]), n=4)
    assert sample.gad.iloc[0] == sample.gad.max()
    gallery.plot_tiles(clus, sample)
    gallery.plot_composition(clus)
    gallery.save_name(clus, int(s.index[0]), "test name")
    assert gallery.load_names(clus) == {int(s.index[0]): "test name"}
    x, sr = gallery.hear(gallery.recording_paths(clus.resolve())["synthetic_01"], sample.iloc[0], "box only", 0.5)
    assert len(x) > 0 and sr == 48000


def test_cli_reports_errors(tmp_path, capsys):
    assert main(["segment", str(tmp_path / "missing.wav")]) == 2
    assert "error" in capsys.readouterr().err
