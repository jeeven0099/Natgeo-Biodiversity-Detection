"""Index every labelled clip into a single manifest, without copying audio.

The seven "Labelled Data-*" directories are Google Drive download splits of one
folder; their union is taken here. Files are referenced in place, so the
manifest is cheap to rebuild and the 13 GB of audio is never duplicated.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import soundfile as sf

# Clips longer than this are treated as weak labels: the tag is known to occur
# somewhere in the file, but not in every window of it.
WEAK_LABEL_SECONDS = 10.0

# Two files in the corpus are 48-byte header-only stubs (2 frames). Anything
# this short cannot carry a vocalisation and is dropped from training.
MIN_DURATION_S = 0.1

# e.g. Collared Scops Owl_S01634_20250314_064000_sel11.wav
STEM_RE = re.compile(r"(?P<recorder>S\d+)_(?P<date>\d{8})_(?P<time>\d{6})_sel(?P<sel>\d+)", re.I)

FIELDS = [
    "clip_id", "path", "label_raw", "coarse", "fine", "rank", "keep",
    "recorder", "session_id", "date", "time", "sel", "hour",
    "duration_s", "samplerate", "channels", "subtype", "weak_label", "split_part",
    "exclude_reason",
]


def load_taxonomy(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["label_raw"]: row for row in csv.DictReader(fh)}


def iter_clips(root: Path):
    """Yield (wav_path, label_raw, split_part) for every labelled clip."""
    for split_dir in sorted(root.glob("Labelled Data-*")):
        if not split_dir.is_dir():
            continue  # skip the .zip archives
        base = split_dir / "Labelled Data"
        if not base.is_dir():
            continue
        part = split_dir.name.rsplit("-", 1)[-1]
        for wav in base.rglob("*.wav"):
            rel = wav.relative_to(base)
            if len(rel.parts) > 1:
                label_raw = rel.parts[0]
            else:
                # A handful of Cicada clips sit loose in the split root; their
                # label is the filename prefix before the recorder id.
                label_raw = wav.name.split("_S0", 1)[0]
            yield wav, label_raw, part


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--taxonomy", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = args.root
    taxonomy = load_taxonomy(args.taxonomy or root / "config" / "taxonomy.csv")
    out_path = args.out or root / "data" / "manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows, unmapped, unparsed, unreadable, seen = [], set(), [], [], {}

    for wav, label_raw, part in iter_clips(root):
        tax = taxonomy.get(label_raw)
        if tax is None:
            unmapped.add(label_raw)
            continue

        key = f"{label_raw}/{wav.name}"
        if key in seen:
            # The splits are disjoint; a collision means a broken download.
            print(f"WARNING duplicate across splits: {key}", file=sys.stderr)
            continue
        seen[key] = True

        m = STEM_RE.search(wav.stem)
        if m is None:
            unparsed.append(str(wav))
            continue

        try:
            info = sf.info(str(wav))
        except Exception as exc:  # noqa: BLE001 - report and skip corrupt files
            unreadable.append(f"{wav}: {exc}")
            continue

        # One 20-minute recording = one independent sample. Clips sharing a
        # session_id must never straddle a train/test boundary.
        session_id = f"{m['recorder']}_{m['date']}_{m['time']}"
        keep, reason = tax["keep"] == "1", ""
        if not keep:
            reason = "label_excluded"
        elif info.duration < MIN_DURATION_S:
            keep, reason = False, "too_short"
        rows.append({
            "clip_id": key,
            "path": wav.relative_to(root).as_posix(),
            "label_raw": label_raw,
            "coarse": tax["coarse"],
            "fine": tax["fine"],
            "rank": tax["rank"],
            "keep": int(keep),
            "exclude_reason": reason,
            "recorder": m["recorder"],
            "session_id": session_id,
            "date": m["date"],
            "time": m["time"],
            "sel": int(m["sel"]),
            "hour": int(m["time"][:2]),
            "duration_s": round(info.duration, 3),
            "samplerate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
            "weak_label": int(info.duration > WEAK_LABEL_SECONDS),
            "split_part": part,
        })

    rows.sort(key=lambda r: (r["coarse"], r["fine"], r["session_id"], r["sel"]))
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    excluded = Counter(r["exclude_reason"] for r in rows if r["exclude_reason"])
    print(f"wrote {len(rows)} rows -> {out_path}")
    print(f"kept {sum(r['keep'] for r in rows)}; excluded {dict(excluded)}")
    if unmapped:
        print(f"UNMAPPED labels (add to taxonomy.csv): {sorted(unmapped)}", file=sys.stderr)
    for name, items in (("unparsed filenames", unparsed), ("unreadable files", unreadable)):
        if items:
            print(f"{name}: {len(items)}", file=sys.stderr)
            for i in items[:5]:
                print(f"  {i}", file=sys.stderr)
    return 1 if (unmapped or unparsed or unreadable) else 0


if __name__ == "__main__":
    raise SystemExit(main())
