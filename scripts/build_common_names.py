"""English common names for every species the review app can show: config/common_names.csv.

    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\build_common_names.py            # eBird + GBIF
    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\build_common_names.py --offline  # eBird + cache only

Names covered: Perch's 14,597 species classes (config/perch_class_index.csv) and EarthChirp's species
prototypes (external/earthchirp_lib/global_reflib_index.parquet; genus and group prototypes have no
common name and are skipped).

Sources, in order:
  1. Birds: the eBird / Clements taxonomy v2025 shipped with bacpipe
     (external/bacpipe/bacpipe/embedding_evaluation/eBird_taxonomy_v2025-4.csv), joined on Perch's
     eBird code, else on the scientific name. One name per species (PRIMARY_COM_NAME).
  2. Everything else: GBIF's public species API. Only the scientific name is sent.
       /species/match?name=...&strict=true&kingdom=Animalia; only an EXACT match is used, and a
       synonym is followed to its accepted taxon. Then /species/{key}/vernacularNames, English only.
     The English name is picked by a fixed rule (no hand choices):
       a. a name flagged preferred (in practice the IUCN Red List's);
       b. else the first Catalogue of Life name;
       c. else the name given by the most sources (ties: first listed).
     Up to 4 other English names go to `other_english` (searchable in the app, not shown in the top 5).
GBIF answers are cached in data/common_names_gbif.jsonl (one line per name, git-ignored), so a
rerun only asks for names not yet cached.

Output columns: scientific_name (as the model spells it), common_name, other_english, source
(eBird 2025 / GBIF: <dataset>), gbif_key, for every name that has a common name.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EBIRD = ROOT / "external" / "bacpipe" / "bacpipe" / "embedding_evaluation" / "eBird_taxonomy_v2025-4.csv"
PERCH_INDEX = ROOT / "config" / "perch_class_index.csv"
EC_INDEX = ROOT / "external" / "earthchirp_lib" / "global_reflib_index.parquet"
CACHE = ROOT / "data" / "common_names_gbif.jsonl"
OUT = ROOT / "config" / "common_names.csv"
API = "https://api.gbif.org/v1"
UA = {"User-Agent": "biodiversity-review-app/1.0 (common-name lookup)"}
N_OTHER = 4


def get_json(url: str, tries: int = 4) -> dict:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)
    return {}


def pick_english(rows: list[dict]) -> tuple[str, str, list[str]]:
    """(name, source, others) from GBIF vernacularNames rows, by the rule in the module docstring."""
    eng = [r for r in rows if r.get("language") == "eng" and str(r.get("vernacularName", "")).strip()]
    if not eng:
        return "", "", []
    clean = lambda r: " ".join(str(r["vernacularName"]).split())  # noqa: E731
    pref = [r for r in eng if r.get("preferred")]
    col = [r for r in eng if str(r.get("source", "")).startswith("Catalogue of Life")]
    if pref:
        best = pref[0]
    elif col:
        best = col[0]
    else:
        counts = Counter(clean(r).lower() for r in eng)
        best = max(eng, key=lambda r: counts[clean(r).lower()])          # max keeps the first on ties
    name = clean(best)
    others, seen = [], {name.lower()}
    for r in eng:
        n = clean(r)
        if n.lower() not in seen:
            seen.add(n.lower())
            others.append(n)
    return name, str(best.get("source", "")), others[:N_OTHER]


def gbif_lookup(sci: str) -> dict:
    m = get_json(f"{API}/species/match?" + urllib.parse.urlencode(
        {"name": sci, "strict": "true", "kingdom": "Animalia"}))
    out = {"query": sci, "match": m.get("matchType", "NONE"), "key": None, "name": "", "source": "", "others": []}
    if m.get("matchType") != "EXACT" or "usageKey" not in m:
        return out
    key = m.get("acceptedUsageKey") or m["usageKey"]
    out["key"] = key
    rows, off = [], 0
    while True:
        v = get_json(f"{API}/species/{key}/vernacularNames?limit=300&offset={off}")
        rows += v.get("results", [])
        if v.get("endOfRecords", True) or not v.get("results"):
            break
        off += 300
    out["name"], out["source"], out["others"] = pick_english(rows)
    return out


def wanted() -> tuple[pd.DataFrame, list[str]]:
    """Perch species table and EarthChirp species names (lower case, as in the index)."""
    idx = pd.read_csv(PERCH_INDEX, keep_default_na=False)
    idx = idx[idx["kind"] == "species"]
    ec = pd.read_parquet(EC_INDEX)
    ec_species = sorted(set(ec.loc[ec["level"] == "species", "name"]))
    return idx, ec_species


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="no GBIF calls: eBird plus whatever is cached")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    idx, ec_species = wanted()
    eb = pd.read_csv(EBIRD, encoding="utf-8-sig", keep_default_na=False)
    eb = eb[eb["CATEGORY"] == "species"]
    by_code = dict(zip(eb["SPECIES_CODE"], eb["PRIMARY_COM_NAME"], strict=True))
    by_sci = {s.lower(): c for s, c in zip(eb["SCI_NAME"], eb["PRIMARY_COM_NAME"], strict=True)}

    rows, todo = {}, []
    for lab, code, cls in zip(idx["label"], idx["ebird"], idx["class"], strict=True):
        c = by_code.get(code) or (by_sci.get(lab.lower()) if cls == "Aves" else None)
        if c:
            rows[lab.lower()] = {"scientific_name": lab, "common_name": c, "other_english": "",
                                 "source": "eBird 2025", "gbif_key": ""}
        else:
            todo.append(lab)
    for n in ec_species:
        if n.lower() not in rows:
            todo.append(n[:1].upper() + n[1:])
    todo = list(dict.fromkeys(todo))
    print(f"{len(rows)} names from eBird; {len(todo)} for GBIF")

    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                cache[d["query"]] = d
    miss = [n for n in todo if n not in cache]
    print(f"{len(todo) - len(miss)} cached, {len(miss)} to fetch" + (" (skipped: --offline)" if args.offline else ""))
    if miss and not args.offline:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        lock, done, t0 = threading.Lock(), [0], time.time()

        def one(n):
            try:
                d = gbif_lookup(n)
            except Exception as e:                       # leave it uncached; a rerun retries it
                print(f"  failed {n}: {e}")
                return
            with lock:
                cache[n] = d
                with CACHE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(d) + "\n")
                done[0] += 1
                if done[0] % 250 == 0:
                    print(f"  {done[0]}/{len(miss)} in {time.time() - t0:.0f} s", flush=True)
        with ThreadPoolExecutor(args.threads) as ex:
            list(ex.map(one, miss))

    n_exact = n_named = 0
    for n in todo:
        d = cache.get(n)
        if not d:
            continue
        n_exact += d["match"] == "EXACT"
        if d["name"]:
            n_named += 1
            rows[n.lower()] = {"scientific_name": n, "common_name": d["name"],
                               "other_english": "; ".join(d["others"]), "source": f"GBIF: {d['source']}",
                               "gbif_key": d["key"]}
    out = pd.DataFrame(sorted(rows.values(), key=lambda r: r["scientific_name"].lower()))
    out.to_csv(OUT, index=False, encoding="utf-8", lineterminator="\n")
    print(f"GBIF: {n_exact} exact matches, {n_named} with an English name, of {len(todo)}")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} names")


if __name__ == "__main__":
    main()
