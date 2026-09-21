"""Resolve Perch's non-bird genera to higher taxonomy, once, into a committed CSV.

Perch v2 has a single flat head of 14,795 classes, so an aggregate target like
"any Orthoptera" has to be expanded into its member species by hand. Birds need
no lookup - the model ships perch_v2_ebird_classes.csv, a parallel array whose
'no_ebird_code' entries are exactly the non-birds - so only the 1,730 non-bird
genera go to GBIF.

Output config/perch_genus_taxonomy.csv is checked in. Compute nodes usually have
no outbound network, and the backbone taxonomy is stable enough that re-querying
per job would only add a failure mode.

  python scripts/build_taxonomy_map.py --assets <model assets dir>
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GBIF = "https://api.gbif.org/v1/species/match"
RANKS = ("kingdom", "phylum", "class", "order", "family")


def load_labels(assets: Path) -> tuple[list[str], list[str]]:
    labels = (assets / "labels.csv").read_text(encoding="utf-8").splitlines()[1:]
    ebird = (assets / "perch_v2_ebird_classes.csv").read_text(encoding="utf-8").splitlines()[1:]
    if len(labels) != len(ebird):
        raise SystemExit(f"asset mismatch: {len(labels)} labels vs {len(ebird)} ebird codes")
    return labels, ebird


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--assets", type=Path, required=True,
                    help="Perch SavedModel assets dir holding labels.csv")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--retry-unresolved", action="store_true",
                    help="Re-query cached genera that came back NONE or errored")
    args = ap.parse_args()

    import requests

    out = args.out or args.root / "config" / "perch_genus_taxonomy.csv"
    labels, ebird = load_labels(args.assets)

    # Only binomials without an eBird code need looking up.
    genera = sorted({l.split()[0] for l, e in zip(labels, ebird)
                     if e == "no_ebird_code" and " " in l})
    print(f"{len(labels)} classes -> {len(genera)} non-bird genera to resolve", flush=True)

    # Resume support: a rerun should cost nothing for genera already done.
    done: dict[str, dict] = {}
    if out.exists():
        done = {r["genus"]: r for r in csv.DictReader(out.open(encoding="utf-8"))}
        print(f"  {len(done)} already cached in {out}", flush=True)
    todo = [g for g in genera if g not in done]
    if args.retry_unresolved:
        stale = [g for g, r in done.items()
                 if r["match"] in ("NONE", "") or r["match"].startswith("ERROR")]
        for g in stale:
            del done[g]
        todo = sorted(set(todo) | set(stale))
        print(f"  retrying {len(stale)} unresolved", flush=True)

    session = requests.Session()

    def lookup(genus: str) -> dict:
        row = {"genus": genus, **{r: "" for r in RANKS}, "match": "", "confidence": ""}
        # Bare genus first; on a cross-kingdom homonym GBIF returns NONE rather
        # than guessing, so retry constrained to Animalia. Every class in this
        # model is an animal, so the hint adds information rather than bias.
        for params in ({"name": genus, "rank": "GENUS", "strict": "false"},
                       {"name": genus, "rank": "GENUS", "strict": "false",
                        "kingdom": "Animalia"}):
            try:
                r = session.get(GBIF, params=params, timeout=30)
                r.raise_for_status()
                j = r.json()
                if j.get("matchType", "NONE") == "NONE":
                    continue
                for k in RANKS:
                    row[k] = j.get(k, "") or ""
                row["match"] = j.get("matchType")
                row["confidence"] = str(j.get("confidence", ""))
                if "kingdom" in params:
                    row["match"] += "_ANIMALIA_HINT"
                return row
            except Exception as exc:  # noqa: BLE001 - record the failure, keep going
                row["match"] = f"ERROR: {type(exc).__name__}"
                return row
        row["match"] = "NONE"
        return row

    results = list(done.values())
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, row in enumerate(ex.map(lookup, todo), 1):
                results.append(row)
                if i % 200 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)

    results.sort(key=lambda r: r["genus"])
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["genus", *RANKS, "match", "confidence"])
        w.writeheader()
        w.writerows(results)

    bad = [r for r in results if r["match"] in ("NONE", "") or r["match"].startswith("ERROR")]
    print(f"wrote {len(results)} genera -> {out}")
    print(f"  unresolved: {len(bad)}")
    for r in bad[:20]:
        print(f"    {r['genus']}: {r['match']}", file=sys.stderr)

    from collections import Counter
    for rank in ("class", "order"):
        top = Counter(r[rank] for r in results if r[rank]).most_common(8)
        print(f"  top {rank}: " + ", ".join(f"{k}({v})" for k, v in top))

    # Flatten to one row per model output index. This is what extract_perch.py
    # actually reads: it keeps the cluster offline-safe and means the extractor
    # never has to know where the SavedModel assets live.
    tax = {r["genus"]: r for r in results}
    idx_out = out.parent / "perch_class_index.csv"
    n_bird = 0
    with idx_out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "label", "ebird", "kind", "class", "order", "family"])
        for i, (lab, eb) in enumerate(zip(labels, ebird)):
            if " " not in lab:
                w.writerow([i, lab, "", "sound_event", "", "", ""])
                continue
            if eb != "no_ebird_code":
                # An eBird code is issued only for birds, so this is exact.
                w.writerow([i, lab, eb, "species", "Aves", "", ""])
                n_bird += 1
                continue
            t = tax.get(lab.split()[0], {})
            # A few birds are missing from the eBird 2021 taxonomy; GBIF still
            # calls them Aves, so count them as birds here too.
            if t.get("class") == "Aves":
                n_bird += 1
            w.writerow([i, lab, "", "species", t.get("class", ""),
                        t.get("order", ""), t.get("family", "")])
    print(f"wrote {len(labels)} rows -> {idx_out} ({n_bird} Aves)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
