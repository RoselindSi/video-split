"""Sample false_gap candidates and ask ONE question: which of three states.

`false_gap` names a POSITION -- the candidate fell outside every annotated
segment -- and not a cause. Six things produce that position and they need
opposite handling, so the class was reported as "no hand action" once in this
project and that was wrong.

THE THREE QUESTIONS THIS BATCH ANSWERS, and nothing else:

    OBSERVED_ABSENT       the view is adequate and there is no manipulation.
                          Empty table, hands away, waiting. Safe to drop.
    NOT_OBSERVABLE        hands out of frame, occlusion, camera moved. NOT safe
                          to drop -- dropping it decides there is no boundary
                          there, and nobody saw whether there was one.
    PRESENT_UNANNOTATED   there IS manipulation and the annotation does not
                          cover it. THE DANGEROUS ONE. Treating these as false
                          positives trains a system to delete real actions, and
                          they are evidence of a missing label, not a bad
                          candidate.

WHY A SAMPLE AND NOT ALL 220. The three proportions decide whether a data
validity gate is worth building and what it may do; they do not need to be
precise. If most are NOT_OBSERVABLE the gate cannot reject and only excludes
from the denominator. If most are OBSERVED_ABSENT an automatic drop is worth
calibrating. If PRESENT_UNANNOTATED is large the candidates are right and the
GT is wrong, which is a different project.

NO MODEL SCORE SELECTS ANYTHING. Sampling is by recording and position only,
so the result can be read as a property of the false_gap population rather
than of the scorer's errors. Each row carries `why_selected`.

THE VERDICT IS NOT `boundary_exists`. Asking whether there is a boundary here
puts the policy layer inside the annotation, and this batch is upstream of
that: it asks only whether the question is answerable and whether anything is
happening.

Usage:
    python -m src.auditor.boundary.false_gap_sample \
        --predictions results/boundary/error_audit/predictions.jsonl \
        --n 100 --out data/gold/false_gap_sheet.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

STATES = ("observed_absent", "not_observable", "present_unannotated",
          "uncertain")

COLUMNS = ["candidate_id", "recording_id", "pred_time", "clip",
           # the one judgement
           "state",                 # STATES
           # the evidence behind it, so a state can be checked rather than
           # trusted -- the reset regex read three of four notes backwards and
           # only its own printing caught it
           "hand_visible",          # present | absent | unknown
           "interaction_visible",   # present | absent | unknown
           "camera_stable",         # yes | no | unknown
           "what_is_happening",     # free text, one line
           "auditor", "notes"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip_dir", help="only include candidates whose clip "
                                       "already exists here")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for e in (json.loads(l) for l in open(a.predictions, encoding="utf-8")
              if l.strip()):
        for x in e.get("predicted_peaks", []):
            if x.get("status") == "false_gap":
                rows.append((e["recording_id"], x))
    print(f"{len(rows)} false_gap candidates over "
          f"{len({r for r, _ in rows})} recordings")

    if a.clip_dir:
        keep = []
        for rid, x in rows:
            p = os.path.join(a.clip_dir,
                             f"{rid}_false_gap_t{x['pred_time']:.1f}.mp4")
            if os.path.exists(p):
                keep.append((rid, x, p))
        print(f"  {len(keep)} have a clip under {a.clip_dir}")
        rows = keep
    else:
        rows = [(rid, x, "") for rid, x in rows]

    # SPREAD ACROSS RECORDINGS. Twenty candidates from one recording answer
    # the question for that kitchen and no other; the proportions are meant to
    # describe the population.
    by = defaultdict(list)
    for rid, x, p in rows:
        by[rid].append((rid, x, p))
    rng = random.Random(a.seed)
    for v in by.values():
        rng.shuffle(v)
    picked, i = [], 0
    while len(picked) < min(a.n, len(rows)):
        added = False
        for rid in sorted(by):
            if i < len(by[rid]):
                picked.append(by[rid][i])
                added = True
                if len(picked) >= a.n:
                    break
        if not added:
            break
        i += 1

    out = []
    for rid, x, p in picked:
        out.append({
            "candidate_id": f"{rid}_false_gap_t{x['pred_time']:.1f}",
            "recording_id": rid, "pred_time": x["pred_time"],
            "clip": os.path.basename(p) if p else "",
            "state": "", "hand_visible": "", "interaction_visible": "",
            "camera_stable": "", "what_is_happening": "",
            "auditor": "", "notes": "",
        })
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)

    print(f"\n{len(out)} sampled over {len({r['recording_id'] for r in out})} "
          f"recordings, at most "
          f"{max(Counter(r['recording_id'] for r in out).values())} each")
    print(f"  state is one of: {' | '.join(STATES)}")
    print(f"\n  WHAT EACH ANSWER LICENSES:")
    print(f"    observed_absent       a data validity gate MAY drop these")
    print(f"    not_observable        it may NOT -- only exclude them from the")
    print(f"                          denominator, never decide them")
    print(f"    present_unannotated   the candidate is right and the label is")
    print(f"                          missing; these are found boundaries")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
