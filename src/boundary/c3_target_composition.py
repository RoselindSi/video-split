"""What is actually in the REVIEW band, by subtype.

The project's headline problem has been "60.9% of events go to a human". That
number has been treated as a modelling failure throughout. This asks a
question nobody asked of it: how many of those events have no boundary to
decide in the first place.

The batch3 relabel turned 25 machine-labelled events out of the clean set, and
most became annotation_convention -- the GT segmentation cuts there and
nothing visible happens. Those are not hard negatives. A verifier asked to
call them is being asked to reproduce a labelling rule from pixels that do not
encode it, and no representation fixes that.

If they concentrate in the REVIEW band, part of the review load is not a model
problem at all, and the cheapest available fix is upstream of every model in
this repo: stop generating those candidates.

WHAT THIS IS NOT. A subtype is not a licence to discard an event. The
un-decidable classes are counted and named, and the decision about candidate
generation is left to a person, because "the model cannot judge it" and "it
should not have been asked" are different claims and only the second justifies
a filter. The report shows both the count and which recordings they come from,
since a class that lives in three recordings is a data-collection artefact and
one spread across forty is a convention.

Usage:
    python -m src.boundary.c3_target_composition \
        --pair_labels data/gold/pair_labels_v1.csv \
        --pair_labels data/gold/batch3_pair_labels_v1_relabel_v1.csv \
        --decisions /workspace/tr1/results/hal/c3/policy_dev_decisions.primary_transportability_frontier.csv \
        --out /workspace/tr1/results/hal/c3/target_composition.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

from src.boundary.pair_taxonomy import (
    SUBTYPES, SUBTYPE_TO_SUPERVISION, CLEAN_BINARY, load_pair_labels,
)

# classes where the pair carries no decidable visual boundary
UNDECIDABLE = ("annotation_convention", "camera_or_viewpoint_shift",
               "visibility_or_offscreen", "ambiguous")


def rid(e):
    m = re.match(r"(recording_\d+)_", e)
    return m.group(1) if m else "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair_labels", action="append", required=True)
    ap.add_argument("--decisions", help="per-event decisions CSV")
    ap.add_argument("--out")
    a = ap.parse_args()

    stored = {}
    for p in a.pair_labels:
        for e, v in load_pair_labels(p).items():
            stored[e] = v["temporal_pair_subtype"]
    print(f"{len(stored)} labelled events from {len(a.pair_labels)} file(s)")
    counts = Counter(stored.values())
    tot = sum(counts.values())
    print(f"\n  {'subtype':<32} {'n':>4}  {'share':>6}")
    for s in SUBTYPES:
        n = counts.get(s, 0)
        print(f"  {s:<32} {n:>4}  {n / tot:>5.1%}"
              + ("   <- no decidable boundary" if s in UNDECIDABLE else ""))
    und = sum(counts.get(s, 0) for s in UNDECIDABLE)
    print(f"\n  {und}/{tot} = {und / tot:.1%} of all labelled events carry no "
          f"decidable visual boundary")

    # spread across recordings: three recordings is an artefact, forty is a
    # convention, and the remedy differs
    print(f"\n  {'subtype':<32} {'recordings':>10}  most concentrated")
    for s in UNDECIDABLE:
        rs = Counter(rid(e) for e, v in stored.items() if v == s)
        if not rs:
            continue
        top = rs.most_common(3)
        print(f"  {s:<32} {len(rs):>10}  "
              + ", ".join(f"{k}:{v}" for k, v in top))

    out = {"counts": dict(counts), "undecidable_share": und / tot}

    if not a.decisions:
        print("\n  no --decisions given, so the REVIEW-band question -- the "
              "one this file exists for -- is not answered.")
    else:
        if not os.path.exists(a.decisions):
            raise SystemExit(
                f"--decisions {a.decisions} does not exist.\n"
                f"The relabelled decisions are produced by the policy refit, "
                f"so that has to run first:\n"
                f"  1. c3_local_eval  --batch3_pair_labels <relabelled> "
                f"--dump_events <scored csv>\n"
                f"  2. c3_selective_policy --events <scored csv> --select "
                f"--dump_decisions <prefix>\n"
                f"Until then, pass the OLD decisions file to see the "
                f"composition against the old partition -- valid as a "
                f"before/after stratum, stale for anything forward-looking.")
        dec = {}
        with open(a.decisions, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                dec[r["event_id"]] = r.get("decision", "")
        joined = {e: s for e, s in stored.items() if e in dec}
        print(f"\n{'=' * 70}\nBY DECISION  ({len(joined)} of {len(stored)} "
              f"events carry a decision)\n{'=' * 70}")
        if len(joined) < 0.5 * len(stored):
            print("  !! fewer than half the labelled events appear in the "
                  "decisions file. It was produced by a policy fitted on the "
                  "OLD labels and scored on the old population, so this join "
                  "is a stale partition -- fine as a fixed stratum for "
                  "before/after comparison, wrong for anything forward-looking.")
        table = defaultdict(Counter)
        for e, s in joined.items():
            table[dec[e]][s] += 1
        for d in sorted(table):
            n = sum(table[d].values())
            u = sum(table[d].get(s, 0) for s in UNDECIDABLE)
            print(f"\n  {d}  ({n} events)")
            for s, c in table[d].most_common():
                print(f"    {s:<32} {c:>4}  {c / n:>5.1%}"
                      + ("   <- undecidable" if s in UNDECIDABLE else ""))
            print(f"    -> {u}/{n} = {u / n:.1%} carry no decidable boundary")
            out[f"decision_{d}"] = {"n": n, "undecidable": u,
                                    "by_subtype": dict(table[d])}
        rev = table.get("REVIEW", Counter())
        if rev:
            n = sum(rev.values())
            u = sum(rev.get(s, 0) for s in UNDECIDABLE)
            print(f"\n{'=' * 70}")
            print(f"  {u} of the {n} REVIEW events ({u / n:.1%}) are events no "
                  f"verifier could decide, because there is no visual boundary "
                  f"to find.")
            print(f"  Removing them at CANDIDATE GENERATION would cut the "
                  f"review load by that much with no model change at all. "
                  f"Whether they\n  can be detected upstream is a separate "
                  f"question -- these subtypes were assigned by a human "
                  f"watching video, and a generator\n  has only the signal it "
                  f"already uses.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
