"""The recordings the learners fit on: the survey's and the audio added in the app.

  survey   every recording of the long-form caches (data/perch_windows/window_scores.csv) and every
           wav under dataset/. On the laptop both; on the server only the caches (no dataset/ audio).
  added    data/uploads (--uploads-dir, scan_audio.py): a recording counts once its scan is
           complete (scan_ok: SCAN_VERSION and the audio's sha256 match). Left out: deleted ones
           (deleted.csv; their sidecar is gone too, so their log rows name no known recording), ones
           not yet scanned, and ones whose name a survey recording already has. Such a name is
           ambiguous (the server has no dataset/, so its app lists the upload: rows logged under the
           name may be the upload's), so its rows are dropped too.
Log rows of any other recording are no labels (review_labels.labels_from_logs drops them).

The window rows, segments and cell maps of the added recordings go behind the caches' exactly as
review_app.Store._register puts them there (scan_audio.RowStack, the same scan arrays), so every
learner sees an added recording as the app shows it. Audio: dataset/ wavs plus the added files; a
survey recording without audio here (the server) is still known: the learners that need its audio
(species: Perch on the box; box refiner: the STFT) use their caches for it and leave out what is not
cached, with a count.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perch_viewer as pv  # noqa: E402
import scan_audio as SA  # noqa: E402

PDIR = pv.ROOT / "data" / "perch_windows"


class Sources:
    def __init__(self, uploads_dir: Path | None = SA.UPLOADS_DIR, pdir: Path = PDIR):
        self.pdir = Path(pdir)
        self.wavs = pv.wav_paths()
        w = pd.read_csv(self.pdir / "window_scores.csv", usecols=["recording"])
        self.survey = set(pd.unique(w["recording"]))
        self.uploads_dir = Path(uploads_dir) if uploads_dir is not None else None
        self.up = SA.Uploads(self.uploads_dir) if self.uploads_dir is not None else None
        self.deleted = self.up.deleted() if self.up else set()
        self.uploads: dict[str, dict] = {}            # rec -> sidecar info (+ path), in the order added
        self.left_out: dict[str, str] = {}            # rec -> why it is not used
        taken = {r.lower() for r in self.survey | set(self.wavs)}
        for rec, info in (self.up.recordings() if self.up else {}).items():
            if rec in self.deleted:
                self.left_out[rec] = "deleted"
            elif rec.lower() in taken:
                self.left_out[rec] = "a survey recording has its name (rows of the name dropped)"
            elif not self.up.scan_ok(rec, info):
                self.left_out[rec] = "not scanned (or its scan is out of date)"
            else:
                self.uploads[rec] = info

    def known(self) -> set[str]:
        clash = {r.lower() for r, why in self.left_out.items() if why.startswith("a survey recording")}
        return {r for r in self.survey | set(self.wavs) | set(self.uploads) if r.lower() not in clash}

    def audio(self) -> dict[str, str]:
        """rec -> audio file: dataset/ wavs and the added files."""
        return {**self.wavs, **{r: i["path"] for r, i in self.uploads.items()}}

    def nyquist_khz(self, rec: str) -> float | None:
        if rec in self.wavs:
            return sf.info(self.wavs[rec]).samplerate / 2000
        if rec in self.uploads:
            return float(self.uploads[rec]["samplerate"]) / 2000
        return None

    def describe(self) -> str:
        s = (f"{len(self.survey | set(self.wavs))} survey recordings ({len(self.wavs)} with audio here), "
             f"{len(self.uploads)} added recordings")
        if self.left_out:
            why = pd.Series(self.left_out).value_counts()
            s += " (left out: " + ", ".join(f"{n} {k}" for k, n in why.items()) + ")"
        return s

    def scans(self):
        """(rec, scan) of each added recording, loaded one at a time."""
        for rec in self.uploads:
            yield rec, SA.load_scan(self.up.scan_file(rec))

    # ---- the caches with the added recordings behind them ------------------------------------
    def windows(self):
        """review_taxon.Windows over the survey caches and the added recordings."""
        import review_taxon as RT
        win = RT.Windows.from_disk(self.pdir)
        if self.uploads:
            E, starts = SA.RowStack(win.E), [win.win_starts]
            for rec, scan in self.scans():
                win.win_rows[rec] = (E.append(scan["emb"]), len(scan["start_s"]))
                starts.append(scan["start_s"])
                win.seg_by_rec[rec] = SA.segments(scan, rec)
            win.E, win.win_starts = E, np.concatenate(starts)
        return win

    def segments(self) -> pd.DataFrame:
        """pv.load_segments plus the added recordings' segments (no truth_* columns: NaN there)."""
        seg = pv.load_segments(self.pdir)
        more = [SA.segments(scan, rec) for rec, scan in self.scans()]
        return pd.concat([seg, *more], ignore_index=True) if more else seg

    def cell_maps(self) -> pv.CellMaps | None:
        """pv.CellMaps with the added recordings' cell rows behind the cache's (None without the cache)."""
        if not (self.pdir / pv.CELL_FILE.name).exists():
            return None
        cells = pv.CellMaps(self.pdir)
        if self.uploads:
            n_win = sum(n for _, n in cells.rows.values())
            if len(cells.M) != n_win or n_win != len(pd.read_csv(self.pdir / "window_scores.csv", usecols=["start_s"])):
                raise SystemExit(f"{pv.CELL_FILE.name} does not cover every window: added recordings cannot go behind it")
            cells.M = SA.RowStack(cells.M)
            for rec, scan in self.scans():
                cells.rows[rec] = (cells.M.append(scan["cells"]), len(scan["start_s"]))
        return cells

    def window_starts(self) -> dict[str, np.ndarray]:
        """rec -> window starts (box_embed.BoxEmbedder.starts), the added recordings included."""
        import box_embed as BE
        out = BE.window_starts(self.pdir)
        for rec, scan in self.scans():
            out[rec] = scan["start_s"]
        return out
