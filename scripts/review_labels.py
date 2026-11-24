"""Turn the review logs into labels for the three learners: box edges, taxon (group), species.

    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\review_labels.py
    .venv-perch\\Scripts\\python.exe -X utf8 scripts\\review_labels.py --out data\\review\\labels

Reads data/review/annotation_log.csv and data/review/audit_windows.csv (written by review_app.py;
append-only, the newest row of a box or window is its state) and config/*.csv for names. Writes,
rebuilt from scratch on every run, in --out:

  box_labels.csv      one row per box answered yes: the box the app suggested (tight_*) and the box
                      the reviewer accepted, with the change of each edge. Inside the accepted box is
                      "in"; inside the suggestion but outside the accepted box is "out" (trimmed off);
                      everything else is not a label. shown_box (from box_shown.csv) names the box
                      the reviewer started from (refined / tight / tight (control) / previous / raven
                      / drawn); a refined start anchors the accepted box, so box_refine.py scores only
                      boxes shown plain.
  taxon_labels.csv    one row per (box or time span, group, 0/1):
                        decision  yes -> the lane's group is present in the accepted box (1);
                                  no  -> it is absent from that box (0); other groups untouched;
                        named     each named species -> its group is present (1), whatever the
                                  decision (a bird named in an insect box marked no gives Aves 1 and
                                  Insecta 0);
                        audit_gap in a window closed as complete for a group, the time no box of that
                                  group covers -> absent (0), full band. The only negatives that are
                                  not a reviewer's "no".
  species_labels.csv  one row per (box, named name): the name resolved to rank / genus / family /
                      order / class / group, its Perch index or EarthChirp name, and where it stood in
                      the model's top 5. Positives only: a top-5 name the reviewer did not tick may
                      still be calling, so it is not a negative. A learner may use "named X above the
                      model's top 1 Y" as a ranking pair.
  audit_labels.csv    one row per (complete window, ticked group): its boxes and its uncovered time;
                      the base for recall.
  withheld.csv        rows that give no label, with the reason: unknown (withheld, not negative) or
                      conflict (no for a group while naming a species of that group).

Rows raven_autolabel.py wrote (reviewer "raven-auto") are labels like any other, except that model
boxes merged into one Raven box (same accepted box) give that box's taxon and species rows once.
Nothing here guesses a label: an unboxed stretch is a negative only inside a complete audit window;
Raven gaps and unreviewed boxes are never negatives. Every row carries the recording's split from
data/splits.csv ("" for the 26 recordings outside it and for audio added in the app), so learners
can hold out whole recordings.

Recordings (review_sources.py, --uploads-dir): the survey's (the long-form caches, dataset/) and the
scanned audio added in the app (data/uploads). Rows of any other recording are left out: a deleted
upload (deleted.csv), one not yet scanned, a name both an upload and a survey recording have.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import insecta_compare as IC  # noqa: E402
import raven_match as RM  # noqa: E402
import review_app as R  # noqa: E402
import review_sources as SRC  # noqa: E402

pv, z = R.pv, R.z
ROOT = R.ROOT
OUT_DIR = R.REVIEW_DIR / "labels"
SPLITS_FILE = ROOT / "data" / "splits.csv"
GROUP_OF_CLASS = dict(z.SPECIES_CLASS)            # Perch class -> group (Aves, Amphibia, Insecta, Mammalia)
EDGES = ("begin_s", "end_s", "lo_khz", "hi_khz")
MIN_GAP_S = 1e-3                                  # audit gaps shorter than this are rounding, not time

BOX_COLS = ["box_id", "recording", "split", "reviewer", "logged_at", "model", "proposal", "lane", "group",
            "kappa", "box_max_score", "threshold", "box_begin_s", "box_end_s"] + [f"tight_{e}" for e in EDGES] \
    + [f"accepted_{e}" for e in EDGES] + ["box_edited"] + [f"d_{e}" for e in EDGES] + ["audit_window"] \
    + ["log_id", "shown_box"]
DEFAULT_SHOWN = {"pipeline": "tight", "raven": "raven", "none": "drawn"}   # rows without a box_shown.csv row
TAXON_COLS = ["box_id", "recording", "split", "reviewer", "model", "lane", "group", "label", "reason",
              "begin_s", "end_s", "lo_khz", "hi_khz", "box_max_score", "threshold", "audit_window"]
SPECIES_COLS = ["box_id", "recording", "split", "reviewer", "model", "lane", "decision", "begin_s", "end_s",
                "lo_khz", "hi_khz", "name", "name_source", "canonical", "rank", "genus", "family", "order",
                "class", "group", "perch_index", "ec_name", "matched_by", "top5_rank", "model_top1",
                "top1_named", "log_id", "learned_shown"]
AUDIT_LABEL_COLS = ["audit_window", "recording", "split", "reviewer", "window_begin_s", "window_end_s", "group",
                    "n_boxes", "covered_s", "withheld_s", "gap_s"]
WITHHELD_COLS = ["box_id", "recording", "split", "reviewer", "model", "lane", "decision", "species", "reason"]


# ---- names -> taxonomy -------------------------------------------------------------------
class Names:
    """Resolves a logged name (Perch label, EarthChirp name, Raven label, common name, genus or a
    higher taxon) to its rank and lineage. Unmatched names resolve to rank "unresolved"; nothing is
    guessed. Order: Perch label, EarthChirp name, higher taxon, genus, unique common name, Raven label."""

    def __init__(self):
        idx = pd.read_csv(ROOT / "config" / "perch_class_index.csv", keep_default_na=False)
        labels, groups = IC.labels_and_groups()
        group_of = np.full(len(labels), "", dtype=object)
        for g, ix in groups.items():
            group_of[ix] = g
        self.perch = {}
        for i, r in idx.iterrows():
            key = r["label"].lower().replace("_", " ")
            genus = r["label"].split()[0] if r["kind"] == "species" and " " in r["label"] else ""
            self.perch.setdefault(key, dict(
                canonical=r["label"], rank="species" if r["kind"] == "species" else "sound_event", genus=genus,
                family=r["family"], order=r["order"], **{"class": r["class"]}, group=group_of[i],
                perch_index=int(r["index"]), ec_name=""))
        gt = pd.read_csv(ROOT / "config" / "perch_genus_taxonomy.csv", keep_default_na=False)
        self.genus = {g.lower(): dict(family=f, order=o, **{"class": c}, match=m)
                      for g, f, o, c, m in zip(gt["genus"], gt["family"], gt["order"], gt["class"], gt["match"],
                                               strict=True)}
        for r in self.perch.values():                 # genera of Perch species not in the genus table
            if r["genus"] and r["genus"].lower() not in self.genus:
                self.genus[r["genus"].lower()] = dict(family=r["family"], order=r["order"],
                                                      **{"class": r["class"]}, match="Perch species")
        # higher taxa: every class / order / family named in either table, with its lineage above it
        self.higher = {}
        for src in (idx.to_dict("records"), gt.to_dict("records")):
            for r in src:
                for rank, keep in (("family", ("order", "class")), ("order", ("class",)), ("class", ())):
                    n = r[rank]
                    if n and n.lower() not in self.higher:
                        self.higher[n.lower()] = dict(canonical=n, rank=rank, **{k: r[k] for k in keep})
        lib = ROOT / "external" / "earthchirp_lib" / "global_reflib_index.parquet"
        ec = pd.read_parquet(lib) if lib.exists() else pd.DataFrame(columns=["name", "level"])
        self.ec = {n.lower(): lv for n, lv in zip(ec["name"], ec["level"], strict=False)}
        self.common: dict[str, set[str]] = {}
        cf = ROOT / "config" / "common_names.csv"
        if cf.exists():
            cn = pd.read_csv(cf, keep_default_na=False)
            for s, c, o in zip(cn["scientific_name"], cn["common_name"], cn["other_english"], strict=False):
                for n in [c, *o.split("; ")]:
                    if n.strip():
                        self.common.setdefault(n.strip().lower(), set()).add(s)

    @staticmethod
    def _blank(name: str) -> dict:
        return dict(canonical=name, rank="unresolved", genus="", family="", order="", **{"class": ""},
                    group="", perch_index="", ec_name="", matched_by="")

    def _lineage(self, out: dict) -> dict:
        """Fills a blank lineage from the genus table (555 Perch labels have no class), then the group."""
        g = self.genus.get(out["genus"].lower()) if out["genus"] else None
        if g and not out["class"]:
            out.update(family=out["family"] or g["family"], order=out["order"] or g["order"], **{"class": g["class"]})
        if not out["group"]:
            out["group"] = GROUP_OF_CLASS.get(out["class"], "")
        if not out["group"] and out["ec_name"]:     # EarthChirp's library holds only Orthoptera and Cicadidae
            out.update(group="Insecta", **{"class": out["class"] or "Insecta"})
        return out

    def resolve(self, name: str, _depth: int = 0) -> dict:
        key = " ".join(name.strip().lower().replace("_", " ").split())
        out = self._blank(name.strip())
        if not key:
            return out
        if key in self.perch:          # also an EarthChirp name: keep it, for EarthChirp's own ranking
            return self._lineage({**out, **self.perch[key], "matched_by": "Perch label",
                                  "ec_name": key if key in self.ec else ""})
        if key in self.ec:
            lv = self.ec[key]
            out.update(ec_name=key, matched_by=f"EarthChirp {lv}", group="Insecta")
            if lv == "group" and key in self.higher:
                h = self.higher[key]
                out.update(canonical=h["canonical"], rank=h["rank"], **{k: h.get(k, "") for k in ("order", "class")})
                out[h["rank"]] = h["canonical"]
                return self._lineage(out)
            gen = key.split()[0]
            out.update(canonical=key.capitalize(), rank="species" if lv == "species" else "genus",
                       genus=gen.capitalize())
            g = self.genus.get(gen)
            if g:
                out.update(family=g["family"], order=g["order"], **{"class": g["class"]})
            return self._lineage(out)
        if key in self.higher:
            h = self.higher[key]
            out.update(canonical=h["canonical"], rank=h["rank"], matched_by="higher taxon",
                       **{k: h.get(k, "") for k in ("order", "class")})
            out[h["rank"]] = h["canonical"]
            return self._lineage(out)
        if " " not in key and key in self.genus:
            g = self.genus[key]
            out.update(canonical=key.capitalize(), rank="genus", genus=key.capitalize(), family=g["family"],
                       order=g["order"], **{"class": g["class"]}, matched_by=f"genus ({g['match']})")
            return self._lineage(out)
        sci = self.common.get(key, set())
        if len(sci) == 1 and _depth == 0:
            r = self.resolve(next(iter(sci)), 1)
            if r["rank"] != "unresolved":
                return {**r, "matched_by": f"common name -> {r['matched_by']}"}
        rg = z.LABEL_TO_GROUP.get(z.norm_label(name))
        if rg:
            out.update(rank="raven_label", group="" if rg == "Unknown" else rg, matched_by="Raven label")
            return out
        if len(sci) > 1:
            out["matched_by"] = f"ambiguous common name ({len(sci)} species)"
        return out


# ---- helpers -----------------------------------------------------------------------------
def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def latest_boxes(rows: list[dict]) -> list[dict]:
    """The newest row of every box, in first-logged order."""
    last = {}
    for r in rows:
        last[R.row_id(r)] = r          # dict keeps the first insertion position
    return list(last.values())


def latest_windows(rows: list[dict]) -> list[dict]:
    last = {}
    for r in rows:
        last[R.audit_id(r["recording"], float(r["window_begin_s"]), float(r["window_end_s"]))] = r
    return list(last.values())


def union(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def clip(iv: list[tuple[float, float]], a: float, b: float) -> list[tuple[float, float]]:
    return [(max(x, a), min(y, b)) for x, y in iv if y > a and x < b]


def gaps(a: float, b: float, cover: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out, t = [], a
    for x, y in union(clip(cover, a, b)):
        if x - t > MIN_GAP_S:
            out.append((t, x))
        t = max(t, y)
    if b - t > MIN_GAP_S:
        out.append((t, b))
    return out


def length(iv: list[tuple[float, float]]) -> float:
    return float(sum(b - a for a, b in iv))


def top5_names(s: str) -> list[str]:
    return [t.rsplit("=", 1)[0].strip() for t in s.split(" | ") if t.strip()]


def write(path: Path, cols: list[str], rows: list[dict]) -> None:
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


# ---- build -------------------------------------------------------------------------------
def build(log_rows: list[dict], audit_rows: list[dict], splits: dict[str, str], names: Names,
          top_khz: dict[str, float], shown: dict[str, str] | None = None,
          learned: dict[str, str] | None = None) -> dict[str, list[dict]]:
    """shown: log_id -> shown_box (box_shown.csv); rows without one (app version < 3) were shown
    the tightened / Raven / drawn box (DEFAULT_SHOWN). learned: log_id -> the species suggestions
    learned from reviews that the reviewer saw (species_shown.csv; "" = none, the unanchored rows)."""
    shown, learned = shown or {}, learned or {}
    box, taxon, species, withheld = [], [], [], []
    acc_of = {}                      # box id -> (row, accepted box, groups present, groups withheld)
    raven_seen = set()               # (recording, accepted box) of raven-auto rows: a merge counts once
    for r in latest_boxes(log_rows):
        bid, rec = R.row_id(r), r["recording"]
        lane_g = R.lane_group(r["lane"])
        acc = [float(r[f"accepted_{e}"]) for e in EDGES]
        sources = (r["species_source"] or "").split("; ")
        named = [(n.strip(), sources[i] if i < len(sources) else "")
                 for i, n in enumerate(r["species"].split("; ")) if n.strip()]
        res = [(n, s, names.resolve(n)) for n, s in named]
        base = dict(box_id=bid, recording=rec, split=splits.get(rec, ""), reviewer=r["reviewer"], model=r["model"],
                    lane=r["lane"])
        if r["decision"] == "unknown":
            withheld.append({**base, "decision": "unknown", "species": r["species"], "reason": "unknown"})
            acc_of[bid] = (r, acc, set(), {lane_g} | {x["group"] for _, _, x in res if x["group"]})
            continue
        if r["decision"] == "no" and any(x["group"] == lane_g for _, _, x in res):
            withheld.append({**base, "decision": "no", "species": r["species"],
                             "reason": f"conflict: no for {lane_g} but names a {lane_g} species"})
            acc_of[bid] = (r, acc, set(), {lane_g})
            continue
        ab = dict(zip(EDGES, acc, strict=False))
        once = (rec, tuple(acc))
        repeat = r["reviewer"] == RM.REVIEWER and once in raven_seen
        if r["reviewer"] == RM.REVIEWER:
            raven_seen.add(once)
        tax = {}
        if lane_g:
            tax[(lane_g, int(r["decision"] == "yes"))] = ["decision"]
        for _, _, x in res:
            if x["group"]:
                tax.setdefault((x["group"], 1), []).append("named")
        for (g, lab), why in ({} if repeat else tax).items():
            taxon.append({**base, "group": g, "label": lab, "reason": "+".join(dict.fromkeys(why)), **ab,
                          "box_max_score": r["box_max_score"], "threshold": r["threshold"]})
        present = {g for (g, lab) in tax if lab == 1}
        acc_of[bid] = (r, acc, present, set())
        if r["decision"] == "yes":
            tight = [float(r[f"tight_{e}"]) for e in EDGES]
            prop = {"Raven": "raven", "reviewer": "none"}.get(r["model"], "pipeline")
            box.append({**base, "logged_at": r["logged_at"], "proposal": prop, "group": lane_g, "kappa": r["kappa"],
                        "box_max_score": r["box_max_score"], "threshold": r["threshold"],
                        "box_begin_s": float(r["box_begin_s"]), "box_end_s": float(r["box_end_s"]),
                        **{f"tight_{e}": v for e, v in zip(EDGES, tight, strict=False)},
                        **{f"accepted_{e}": v for e, v in zip(EDGES, acc, strict=False)}, "box_edited": r["box_edited"],
                        **{f"d_{e}": round(a - t, 3) for e, a, t in zip(EDGES, acc, tight, strict=False)},
                        "log_id": r.get("log_id", ""),
                        "shown_box": shown.get(r.get("log_id", ""), DEFAULT_SHOWN[prop])})
        top = top5_names(r["top5"])
        low = [t.lower() for t in top]
        named_low = {n.lower() for n, _ in named} | {x["canonical"].lower() for _, _, x in res}
        for n, s, x in ([] if repeat else res):
            k = next((i + 1 for i, t in enumerate(low) if t in (n.lower(), x["canonical"].lower())), "")
            species.append({**base, "decision": r["decision"], **ab, "name": n, "name_source": s, **x,
                            "top5_rank": k, "model_top1": top[0] if top else "",
                            "top1_named": int(bool(low) and low[0] in named_low),
                            "log_id": r.get("log_id", ""), "learned_shown": learned.get(r.get("log_id", ""), "")})
    # audit windows closed as complete: uncovered time of each ticked group is absent
    audit = []
    windows = {}
    for w in latest_windows(audit_rows):
        if w["status"] != "complete":
            continue
        rec, a, b = w["recording"], float(w["window_begin_s"]), float(w["window_end_s"])
        wid = R.audit_id(rec, a, b)
        for g in [x for x in w["groups"].split("; ") if x]:
            windows[(rec, g, wid)] = (a, b)
            cover, held, n = [], [], 0
            for r, acc, present, wh in acc_of.values():
                if r["recording"] != rec or not (acc[0] < b and acc[1] > a):
                    continue
                if g in present:
                    cover.append((acc[0], acc[1]))
                    n += 1
                elif g in wh:
                    held.append((acc[0], acc[1]))
            gp = gaps(a, b, cover + held)
            for x, y in gp:
                taxon.append(dict(box_id=wid, recording=rec, split=splits.get(rec, ""), reviewer=w["reviewer"],
                                  model="audit", lane="", group=g, label=0, reason="audit_gap", begin_s=round(x, 3),
                                  end_s=round(y, 3), lo_khz=0.0, hi_khz=top_khz.get(rec, ""), box_max_score="",
                                  threshold=""))
            cov = union(clip(cover, a, b))
            audit.append(dict(audit_window=wid, recording=rec, split=splits.get(rec, ""), reviewer=w["reviewer"],
                              window_begin_s=a, window_end_s=b, group=g, n_boxes=n, covered_s=round(length(cov), 3),
                              withheld_s=round(length(gaps_inside(union(clip(held, a, b)), cov)), 3),
                              gap_s=round(length(gp), 3)))
    # mark labels that lie in a complete window of their group (the recall set)
    for t in taxon:
        if t["reason"] != "audit_gap":
            t["audit_window"] = in_window(windows, t["recording"], t["group"], t["begin_s"], t["end_s"])
    for x in box:
        x["audit_window"] = in_window(windows, x["recording"], x["group"], x["accepted_begin_s"], x["accepted_end_s"])
    for t in taxon:
        if t["reason"] == "audit_gap":
            t["audit_window"] = t["box_id"]
    return dict(box=box, taxon=taxon, species=species, audit=audit, withheld=withheld)


def gaps_inside(held: list[tuple[float, float]], cover: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Parts of held not inside cover."""
    out = []
    for a, b in held:
        out += gaps(a, b, cover)
    return out


def in_window(windows: dict, rec: str, group: str, a: float, b: float) -> str:
    return "; ".join(wid for (r, g, wid), (x, y) in windows.items() if r == rec and g == group and a < y and b > x)


def labels_from_logs(log_rows: list[dict], audit_rows: list[dict], shown_rows=(),
                     species_shown_rows=(), sources: SRC.Sources | None = None) -> dict[str, list[dict]]:
    """build() with the splits, the names tables and each audited recording's Nyquist;
    shown_rows: box_shown.csv rows; species_shown_rows: species_shown.csv rows.
    sources (review_sources.Sources; default: the survey + data/uploads): rows of recordings it does
    not know are left out (a deleted or unscanned upload, a name both an upload and a survey
    recording have). Added recordings have no split (""): the learners hold out whole recordings."""
    src = sources if sources is not None else SRC.Sources()
    splits = {r["session_id"]: r["split"] for r in read_csv_rows(SPLITS_FILE)}
    known = src.known()
    n_log, n_audit = len(log_rows), len(audit_rows)
    log_rows = [r for r in log_rows if r["recording"] in known]
    audit_rows = [r for r in audit_rows if r["recording"] in known]
    if len(log_rows) < n_log or len(audit_rows) < n_audit:
        print(f"left out (no known recording: a deleted or unscanned upload): {n_log - len(log_rows)} log rows, "
              f"{n_audit - len(audit_rows)} audit rows", flush=True)
    top_khz = {}
    for r in {r["recording"] for r in audit_rows}:
        k = src.nyquist_khz(r)
        if k is not None:
            top_khz[r] = k
    return build(log_rows, audit_rows, splits, Names(), top_khz, {r["log_id"]: r["shown_box"] for r in shown_rows},
                 {r["log_id"]: r["learned_shown"] for r in species_shown_rows})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=R.LOG_FILE)
    ap.add_argument("--audit", type=Path, default=R.AUDIT_FILE)
    ap.add_argument("--shown", type=Path, default=R.SHOWN_FILE)
    ap.add_argument("--species-shown", type=Path, default=R.SPECIES_SHOWN_FILE)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--uploads-dir", type=Path, default=SRC.SA.UPLOADS_DIR, help="audio added in the app")
    args = ap.parse_args()
    log_rows, audit_rows = read_csv_rows(args.log), read_csv_rows(args.audit)
    src = SRC.Sources(args.uploads_dir)
    print(src.describe())
    out = labels_from_logs(log_rows, audit_rows, read_csv_rows(args.shown), read_csv_rows(args.species_shown), src)
    args.out.mkdir(parents=True, exist_ok=True)
    for key, cols in (("box", BOX_COLS), ("taxon", TAXON_COLS), ("species", SPECIES_COLS),
                      ("audit", AUDIT_LABEL_COLS), ("withheld", WITHHELD_COLS)):
        write(args.out / f"{key}_labels.csv" if key != "withheld" else args.out / "withheld.csv", cols, out[key])
    n_box = len(latest_boxes(log_rows))
    print(f"{len(log_rows)} log rows -> {n_box} boxes; {len(audit_rows)} audit rows -> "
          f"{len(latest_windows(audit_rows))} windows")
    print(f"box labels      {len(out['box'])} (edited {sum(str(x['box_edited']) == '1' for x in out['box'])})")
    tx = pd.DataFrame(out["taxon"], columns=TAXON_COLS)
    if len(tx):
        print("taxon labels    " + ", ".join(f"{g} {int((d.label == 1).sum())}+/{int((d.label == 0).sum())}-"
                                             for g, d in tx.groupby("group")))
    else:
        print("taxon labels    0")
    sp = pd.DataFrame(out["species"], columns=SPECIES_COLS)
    print(f"species labels  {len(sp)} ({', '.join(f'{k} {v}' for k, v in sp['rank'].value_counts().items())})")
    print(f"audit labels    {len(out['audit'])} window x group")
    print(f"withheld        {len(out['withheld'])}")
    print(f"written to {R.rel(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
