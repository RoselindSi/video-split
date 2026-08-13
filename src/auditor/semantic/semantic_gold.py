"""The frozen 48-event semantic gold: schema check, join, and the collinearity test.

THE TEST THAT MATTERS IS THE ONE THAT KILLED THE OLD SCHEMA. The previous
six-way status was four columns recording one judgement: `label_support` added
0.00 bits beyond `label_completeness`, and seven field combinations mapped to
six statuses with a diagonal table. Three heads fitted on those columns would
have been three views of one label, and their agreement a property of the
schema rather than evidence about the video.

The new six fields were designed to be independent axes. Designed to be is not
the same as are, and the same conditional-entropy check that exposed the old
one is run here on the new one. A field that adds under about 0.1 bits beyond
another is not a separate axis whatever its value list says, and finding that
out now costs nothing while finding it out after two heads are trained costs
the heads.

WHAT IS ALREADY VISIBLE WITHOUT ANY MODEL:

    upstream_timing_issue = yes on 41 of 48. The freeze rule says a wrong
    upstream segment can be recorded but cannot excuse a semantic error, and
    the annotator applied it -- but a field that is `yes` 85% of the time
    carries about 0.6 bits and cannot be a head. It is a co-occurrence
    finding: nearly every semantic problem in this sample sits on a segment
    whose boundaries are also wrong.

    action_presence is valid_action on 45 of 48. Two minority classes with 2
    and 1 events cannot be scored at all, and saying so here prevents an
    AUROC being computed on three events later.

    the trainable claim_support contrast is 29 YES against 6 NO, with 12
    partial and 1 uncertain excluded. Six negatives is the binding constraint
    on everything downstream, and it is a supervision problem, not an
    architecture one.

THE JOIN back to audit_188_gold_v2.jsonl is numeric, not string: audit keys
write `22/t466.8` while event ids write `_t466.8`, and two keys are half a
second off their event. Those two are matched only when the next-nearest event
in the same recording is far enough away for there to be no second candidate,
and each fuzzy match is printed by name rather than folded into a count.

Usage:
    python -m src.auditor.semantic.semantic_gold \
        --csv data/gold/semantic_ontology_gold_48.csv \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --out data/gold/semantic_ontology_gold_48.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict

EID = re.compile(r"^recording_0*(\d+)_.*_t(\d+(?:\.\d+)?)$")
KEY = re.compile(r"^(\d+)/t(\d+(?:\.\d+)?)$")

SCHEMA = {
    "claim_support": ["yes", "partial", "no", "uncertain"],
    "granularity": ["adequate", "too_coarse", "too_fine",
                    "mixed_inconsistent", "not_applicable", "uncertain"],
    "major_action_missing": ["yes", "no", "uncertain"],
    "action_presence": ["valid_action", "no_valid_action",
                        "mixed_action_and_no_action", "onset_only",
                        "terminal_only", "uncertain"],
    "segment_structure": ["adequate", "oversegmented", "undersegmented",
                          "mixed", "spurious_no_action", "uncertain"],
    "upstream_timing_issue": ["yes", "no", "uncertain"],
}


def H(values):
    n = len(values)
    c = Counter(values)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def H_cond(a, b):
    """H(a | b) -- bits still needed for `a` once `b` is known."""
    by = defaultdict(list)
    for x, y in zip(a, b):
        by[y].append(x)
    return sum(len(v) / len(a) * H(v) for v in by.values())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/gold/semantic_ontology_gold_48.csv")
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--fuzzy_join_s", type=float, default=0.6,
                    help="a key this far from an event still joins, but only "
                         "when the next-nearest event is much further, so "
                         "there is no second candidate")
    ap.add_argument("--collinear_bits", type=float, default=0.1,
                    help="a field adding fewer bits than this beyond another "
                         "is not a separate axis")
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.csv, newline="",
                                           encoding="utf-8-sig"))
            if (r.get("audit_key") or "").strip()]
    print(f"{len(rows)} audited events")

    # ------------------------------------------------------- schema check
    off = 0
    print(f"\nSCHEMA -- every value against the frozen list:")
    for f, allowed in SCHEMA.items():
        c = Counter((r.get(f) or "").strip() for r in rows)
        bad = [k for k in c if k not in allowed]
        print(f"  {f:24s} {dict(c.most_common())}")
        if bad:
            print(f"    !! OFF-SCHEMA {bad}")
            off += 1
        missing = [k for k in allowed if k not in c]
        if missing:
            print(f"     unused values: {missing}")
    print(f"  fields with off-schema values: {off}")

    # ------------------------------------------------------------- join
    gold = [json.loads(l) for l in open(a.gold, encoding="utf-8")
            if l.strip()]
    by_rec = defaultdict(list)
    for g in gold:
        m = EID.match(g["event_id"])
        if m:
            by_rec[int(m.group(1))].append((float(m.group(2)), g))
    joined, fuzzy, unmatched = 0, [], []
    for r in rows:
        m = KEY.match(r["audit_key"])
        if not m:
            unmatched.append(r["audit_key"])
            continue
        rid, t = int(m.group(1)), float(m.group(2))
        cands = sorted(((abs(x - t), x, g) for x, g in by_rec.get(rid, ())),
                       key=lambda z: z[0])
        if cands and cands[0][0] <= 0.01:
            r["event_id"] = cands[0][2]["event_id"]
            joined += 1
        elif (cands and cands[0][0] <= a.fuzzy_join_s
              and (len(cands) == 1 or cands[1][0] > 5 * cands[0][0])):
            r["event_id"] = cands[0][2]["event_id"]
            fuzzy.append((r["audit_key"], cands[0][2]["event_id"],
                          round(cands[0][0], 2),
                          round(cands[1][0], 1) if len(cands) > 1 else None))
            joined += 1
        else:
            unmatched.append(r["audit_key"])
    print(f"\nJOIN to {a.gold}: {joined}/{len(rows)}")
    for k, eid, d, nxt in fuzzy:
        print(f"  fuzzy: {k} -> {eid}  ({d}s off; next-nearest {nxt}s away)")
    if unmatched:
        print(f"  !! unmatched: {unmatched}")

    # ------------------------------------------- the collinearity check
    print(f"\nCOLLINEARITY -- the check that exposed the old six-way status:")
    cols = {f: [(r.get(f) or "").strip() for r in rows] for f in SCHEMA}
    print(f"  {'field':<24}{'H (bits)':>10}{'classes':>9}")
    for f, v in cols.items():
        print(f"  {f:<24}{H(v):>10.2f}{len(set(v)):>9}")
    print(f"\n  bits field A still needs once field B is known "
          f"(low = not a separate axis):")
    print(f"  {'':<24}" + "".join(f"{f[:9]:>11}" for f in cols))
    flat = []
    for fa, va in cols.items():
        line = f"  {fa:<24}"
        for fb, vb in cols.items():
            line += "          -" if fa == fb else f"{H_cond(va, vb):>11.2f}"
        print(line)
        for fb, vb in cols.items():
            if fa != fb and H_cond(va, vb) < a.collinear_bits:
                flat.append((fa, fb, H_cond(va, vb)))
    if flat:
        print(f"\n  !! COLLINEAR PAIRS (under {a.collinear_bits} bits):")
        for fa, fb, b in flat:
            print(f"     {fa} adds only {b:.2f} bits beyond {fb} -- one axis, "
                  f"not two")
    else:
        print(f"\n  no pair falls under {a.collinear_bits} bits. The six "
              f"fields are not restatements of\n  each other on this sample, "
              f"which is what the old schema failed.")

    # ------------------------------------------------- what is trainable
    print(f"\nTRAINABLE, before any model:")
    cs = Counter(r["claim_support"] for r in rows)
    print(f"  claim_support YES vs NO: {cs['yes']} vs {cs['no']}  "
          f"(excluding {cs['partial']} partial, {cs['uncertain']} uncertain)")
    for f in ("granularity", "segment_structure", "action_presence",
              "upstream_timing_issue", "major_action_missing"):
        c = Counter((r.get(f) or "").strip() for r in rows)
        tiny = {k: v for k, v in c.items() if v < 5}
        maj = c.most_common(1)[0]
        print(f"  {f:<24} majority {maj[0]} {maj[1]}/{len(rows)} "
              f"({100*maj[1]/len(rows):.0f}%)"
              + (f"; classes under 5: {tiny}" if tiny else ""))
    print(f"\n  A class with a handful of events cannot be scored. Naming the "
          f"sizes here is what\n  stops an AUROC being reported on three "
          f"events later.")

    if a.out:
        json.dump({"source_csv": a.csv, "n": len(rows),
                   "fuzzy_joins": fuzzy, "unmatched": unmatched,
                   "events": rows},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
