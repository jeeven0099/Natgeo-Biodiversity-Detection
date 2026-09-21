"""Assign recording sessions - not clips - to train/val/test folds.

All audio comes from one recorder (S01634) over 7 days. Clips cut from the same
20-minute recording share an individual animal, a background and often a single
calling bout, so a random per-clip split leaks heavily and inflates accuracy.
The independent unit is the session, so sessions are what get split here.

Sessions are assigned greedily, rarest class first, to whichever fold is
furthest below its quota for that class - an iterative-stratification scheme
that keeps scarce classes present in every fold where the data allows.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

# A fine class needs at least this many distinct sessions to be evaluated
# honestly: fewer, and train and test would share a recording.
MIN_SESSIONS_FOR_FINE = 3
MIN_CLIPS_FOR_FINE = 20

FOLDS = (("train", 0.70), ("val", 0.15), ("test", 0.15))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--seed", type=int, default=20260920)
    args = ap.parse_args()

    root = args.root
    rows = [r for r in csv.DictReader((root / "data" / "manifest.csv").open(encoding="utf-8"))
            if r["keep"] == "1"]

    sess_fine: dict[str, Counter] = defaultdict(Counter)
    fine_sessions: dict[str, set[str]] = defaultdict(set)
    fine_clips: Counter = Counter()
    for r in rows:
        sess_fine[r["session_id"]][r["fine"]] += 1
        fine_sessions[r["fine"]].add(r["session_id"])
        fine_clips[r["fine"]] += 1

    # Decide which fine classes can carry a species head at all.
    viable, coarse_only = set(), {}
    for fine, sessions in fine_sessions.items():
        if len(sessions) >= MIN_SESSIONS_FOR_FINE and fine_clips[fine] >= MIN_CLIPS_FOR_FINE:
            viable.add(fine)
        else:
            coarse_only[fine] = (len(sessions), fine_clips[fine])

    # Assignment runs in two passes. A purely greedy deficit pass strands
    # scarce classes: their sessions come up first, and train's 70% quota wins
    # every early comparison, so a class spread over 10 sessions can still land
    # entirely in train. Pass 1 therefore reserves coverage for val and test
    # before pass 2 does the bulk fill.
    targets = {f: {fine: fine_clips[fine] * frac for fine in viable} for f, frac in FOLDS}
    got: dict[str, Counter] = {f: Counter() for f, _ in FOLDS}
    assigned: dict[str, str] = {}

    def place(session: str, fold: str) -> None:
        assigned[session] = fold
        got[fold].update({f: c for f, c in sess_fine[session].items() if f in viable})

    # Pass 1: guarantee each viable class reaches val and test, scarcest first.
    # Folds are filled smallest-quota first, since those are hardest to satisfy.
    for fine in sorted(viable, key=lambda f: (len(fine_sessions[f]), fine_clips[f])):
        if len(fine_sessions[fine]) < MIN_SESSIONS_FOR_FINE:
            continue
        for fold, frac in sorted(FOLDS, key=lambda x: x[1]):
            if fold == "train" or got[fold][fine] > 0:
                continue
            free = [s for s in fine_sessions[fine] if s not in assigned]
            # Leave at least one session for train to keep the class trainable.
            if len(free) <= 1:
                break
            want = fine_clips[fine] * frac
            # Prefer the session closest to this fold's share, so the dominant
            # session is not dumped into a 15% fold.
            place(min(free, key=lambda s: abs(sess_fine[s][fine] - want)), fold)

    # Pass 2: remaining sessions go wherever the normalised deficit is largest.
    def rarity(session: str) -> tuple:
        present = [f for f in sess_fine[session] if f in viable]
        scarcest = min((len(fine_sessions[f]) for f in present), default=10**6)
        return (scarcest, -sum(sess_fine[session].values()), session)

    for session in sorted((s for s in sess_fine if s not in assigned), key=rarity):
        best, best_score = None, None
        for fold, _ in FOLDS:
            score = sum(
                (targets[fold][f] - got[fold][f]) / max(fine_clips[f], 1)
                for f in sess_fine[session] if f in viable
            )
            # Tie-break toward the emptier fold so totals track the ratios.
            score += 1e-6 * (sum(targets[fold].values()) - sum(got[fold].values()))
            if best_score is None or score > best_score:
                best, best_score = fold, score
        place(session, best)

    out = root / "data" / "splits.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session_id", "split", "n_clips"])
        for s in sorted(assigned):
            w.writerow([s, assigned[s], sum(sess_fine[s].values())])

    n_sess = Counter(assigned.values())
    n_clip = Counter()
    for r in rows:
        n_clip[assigned[r["session_id"]]] += 1
    print(f"wrote {out}")
    print(f"sessions: {dict(n_sess)}")
    print(f"clips:    {dict(n_clip)}")
    print(f"\nspecies-head classes ({len(viable)}):")
    for f in sorted(viable, key=lambda x: -fine_clips[x]):
        per = " ".join(f"{fold}={got[fold][f]:>4d}" for fold, _ in FOLDS)
        print(f"  {f:32s} clips={fine_clips[f]:5d} sess={len(fine_sessions[f]):3d}  {per}")
    print(f"\ncoarse-head only - too few sessions or clips ({len(coarse_only)}):")
    for f, (s, c) in sorted(coarse_only.items(), key=lambda x: -x[1][1]):
        print(f"  {f:32s} clips={c:5d} sess={s:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
