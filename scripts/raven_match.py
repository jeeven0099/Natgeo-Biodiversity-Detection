"""Raven boxes matched to model boxes, and what a Raven label says; used by raven_autolabel.py (which
logs the matches) and review_app.py (whose Raven source lists the Raven boxes left unmatched).

Matching, per recording, on time only:
  1. each model box picks the Raven box of its group (a model lane counts for a Raven group when it is
     one of review_app's raven_lanes(group): the default hybrid lanes, so Insecta is EC-Insecta) that
     overlaps it with the highest time IoU; ties go to the earlier Raven box. A model box picks at most
     one Raven box (1:1).
  2. a Raven box with picks is matched when the merged span of its model boxes (first begin to last
     end) has time IoU >= IOU_CUT with it: "one" (one model box) or "merge" (several, all becoming the
     Raven box). Otherwise it is "new" (a box the model missed), and its model boxes stay as they are.
  3. a Raven box no model box picked is "new" too (no model box of its group, or every overlapping one
     went to another Raven box).
Raven boxes whose label is skipped (Unknown) take no part: they never claim a model box.
"""
from __future__ import annotations

IOU_CUT = 0.3
REVIEWER = "raven-auto"                    # the log's reviewer for every row raven_autolabel.py writes
SHOWN_ONE = "raven-auto"                   # box_shown.csv: one model box fitted to its Raven box
SHOWN_MERGE = "raven-auto merge"           # ... several model boxes merged into one Raven box
SHOWN_NEW = "raven"                        # ... a new box from a Raven box (the app's own Raven value)
NAME_SOURCE = "Raven"                      # species_source of a name taken from a Raven label

SKIP = {"unknown", "unknwon"}
# spellings merged before anything else (lower case -> the label)
TYPOS = {"aircraft": "Aircraft", "aircrfat": "Aircraft", "cicafa": "Cicada",
         "brown boobokok": "Brown Boobook",
         "pin-striped tit-babbler": "Pin-striped Tit-Babbler", "pin-striped tit-babler": "Pin-striped Tit-Babbler",
         "pin-striped tit babbler": "Pin-striped Tit-Babbler"}
# labels that name the group only (the box is a call of the group; no species): the taxon learner's
GROUP_ONLY = {"bird", "birds", "bid", "3 birds", "crow", "insect", "frog",
              "aircraft", "gunfire", "helicopter", "siren", "human voice", "rain", "thunder", "wind"}
# labels logged under another name: Cicada as its family; Collared Scops Owl as Perch's label for
# Collared Scops-Owl (the hyphen-less spelling matches no names table)
RENAME = {"cicada": "Cicadidae", "collared scops owl": "Otus lettia"}


def label_names(label: str) -> list[str] | None:
    """The names a Raven label gives the box: None = skip the box, [] = its group only, [name]."""
    s = " ".join(str(label).split())
    low = s.lower()
    if low in SKIP:
        return None
    if low.endswith("?"):                  # unsure of the species, sure of the group
        return []
    s = TYPOS.get(low, s)
    low = s.lower()
    if low in GROUP_ONLY:
        return []
    return [RENAME.get(low, s)]


def t_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = min(a1, b1) - max(a0, b0)
    span = max(a1, b1) - min(a0, b0)
    return inter / span if inter > 0 and span > 0 else 0.0


def match(model: list[dict], raven: list[dict], lanes_of, cut: float = IOU_CUT) -> list[dict]:
    """model: dicts with id, lane, begin_s, end_s; raven: dicts with key, group, label, begin_s, end_s
    (one per distinct box); lanes_of(group) -> the lanes whose boxes count for that group.
    Returns one entry per Raven box that takes part: dict(raven, kind one|merge|new, models (the model
    boxes it got, in time order; for new, those it got but could not use), iou (merged span; 0 for none),
    why (new only))."""
    use = [r for r in raven if r["group"] != "Unknown" and label_names(r["label"]) is not None]
    lanes = {r["group"]: set(lanes_of(r["group"])) for r in use}
    order = sorted(range(len(use)), key=lambda i: (use[i]["begin_s"], use[i]["end_s"], i))
    got: dict[int, list[dict]] = {i: [] for i in order}
    for m in model:
        best, bi = 0.0, None
        for i in order:
            r = use[i]
            if m["lane"] not in lanes[r["group"]]:
                continue
            v = t_iou(m["begin_s"], m["end_s"], r["begin_s"], r["end_s"])
            if v > best:                   # strict: the earlier Raven box keeps a tie
                best, bi = v, i
        if bi is not None:
            got[bi].append(m)
    out = []
    for i in order:
        r, ms = use[i], sorted(got[i], key=lambda m: (m["begin_s"], m["end_s"], m["lane"]))
        if not ms:
            overlap = any(m["lane"] in lanes[r["group"]] and m["begin_s"] < r["end_s"] and m["end_s"] > r["begin_s"]
                          for m in model)
            out.append(dict(raven=r, kind="new", models=[], iou=0.0,
                            why="its model boxes went to other Raven boxes" if overlap else "no model box"))
            continue
        v = t_iou(min(m["begin_s"] for m in ms), max(m["end_s"] for m in ms), r["begin_s"], r["end_s"])
        if v >= cut:
            out.append(dict(raven=r, kind="one" if len(ms) == 1 else "merge", models=ms, iou=v, why=""))
        else:
            out.append(dict(raven=r, kind="new", models=ms, iou=v, why=f"IoU {v:.2f} < {cut:g}"))
    return out
