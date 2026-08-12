"""The TIMING sheet, and nothing else. Sampled without looking at any peak.

FROZEN SCOPE. This file used to emit a second sheet asking for
instance_relation on events already resolving to EARLY / LATE / DUPLICATE, so
that misaligned examples could be labelled. That pool came back at 8 events
over 6 recordings and the ceiling above it is 13, so it is not worth
annotation time and it is no longer generated. If the relation pool ever grows
materially that is a new decision to be argued then, rather than a dormant
code path still running on assumptions that have since been measured false.

WHAT THIS SHEET DECIDES. The detector's peaks land near stored segment
boundaries LESS often than randomly shifted peaks do -- 0.54x against a
circular-shift null, on a boundary set carrying no peak-based selection, so
that number is clean. Against the 125 human-corrected boundaries the same
peaks score 6.21x. Both cannot be read at face value: the audit corpus was
itself built from peak-to-GT matching, so an event is in it partly BECAUSE a
peak was nearby. The 6.21 is contaminated by that selection, and the
conclusion it appears to support -- that the stored annotation times are
systematically displaced -- is therefore supported but NOT established.

What settles it is human-verified boundary times on events chosen without any
reference to a peak. That is what these 45 events are.

SO THE SAMPLING MUST STAY PEAK-BLIND, and it is. The pool is defined by
`instance_relation in POSITIVE` and `alignment == UNDECIDABLE`, both derived
from the ontology and the timing gold; neither touches predictions.jsonl, and
this file does not import the peak reader at all. No detector score, peak
proximity or old timing class reaches the sheet, the sampler, or the ordering.
Adding any of them later would silently reintroduce the exact bias the
experiment exists to remove.

WHAT IS SHOWN: instance_relation, the candidate time, the recording, and the
segment labels around it. Existence is settled for these events by an earlier
careful pass, and re-asking it invites relitigating that pass, so it is shown
rather than re-elicited -- with an explicit disagreement field, because an
annotator who disagrees needs somewhere to say so that is NOT the timing
column. A blank time has to mean "I could not localise it" and never "I
dispute the relation", or the two become indistinguishable across 45 rows.

`cannot_localize` is a first-class answer. A transition with no single instant
is a real observation about the video, and forcing a number would manufacture
precisely the precision this experiment is trying to measure.

Usage:
    python -m src.auditor.boundary.alignment_audit_sheet \
        --migrated data/gold/pair_schema_v2_migrated.csv \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --context data/gold/audit_188_context.jsonl \
        --out data/gold/alignment_timing.csv

and once it is filled, the test it exists for:

    python -m src.auditor.boundary.alignment_from_peaks \
        --predictions .../predictions.jsonl \
        --timing_csv data/gold/alignment_timing.csv --null_shift 200
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict

from src.auditor.boundary.labels import (
    TOL, MAX_RETIME_S, cand_time, recording_of, relation, find_duplicates)

POSITIVE = ("new_action", "same_action_new_instance")

TIMING_Q = [
    ("t1_disagree_with_relation",
     "The recorded instance_relation is shown. Leave blank if you agree. "
     "If you disagree, write what you would call it -- do NOT express a "
     "disagreement by leaving the time blank."),
    ("t2_boundary_time_s",
     "The time of the boundary, in seconds. One number if it is an instant. "
     "Write `cannot_localize` if there is no single instant."),
    ("t3_interval_start_s",
     "If the transition occupies an interval, its start. Leave both interval "
     "fields blank for an instant."),
    ("t4_interval_end_s", "...and its end."),
    ("t5_candidate_verdict",
     "About the CANDIDATE time shown, not the boundary: keep / move / "
     "duplicate_of_another_candidate / no_single_time"),
    ("t6_notes", "Anything else, including why a field was unanswerable."),
]


def build(mig, gold, tol, max_retime):
    dup = find_duplicates(gold, {}, tol)
    rows = {}
    for eid in set(mig) | set(gold):
        g = gold.get(eid)
        cls, _off, _why = relation(eid, g, dup, tol, max_retime)
        rows[eid] = {
            "event_id": eid, "instance_relation": mig.get(eid, ""),
            "alignment": cls, "recording_id": recording_of(eid, g),
            "candidate_time": cand_time(eid),
            "split": (g or {}).get("split", ""), "in_gold": g is not None}
    return rows


def sample(pool, n, seed, per_recording=2):
    """Round-robin over recordings, capped, so a grouped split has groups.

    Order inside a recording is a shuffle on `seed` and nothing else. No
    score, no peak distance and no source category enters here."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in pool:
        by[r["recording_id"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    out, depth = [], 0
    while len(out) < n and depth < per_recording:
        for rid in sorted(by):
            if len(by[rid]) > depth and len(out) < n:
                out.append(by[rid][depth])
        depth += 1
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--migrated", required=True)
    ap.add_argument("--gold", action="append")
    ap.add_argument("--context", action="append")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--max_retime_s", type=float, default=MAX_RETIME_S)
    ap.add_argument("--n_timing", type=int, default=45)
    ap.add_argument("--per_recording", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--out", default="data/gold/alignment_timing.csv")
    a = ap.parse_args()
    # argparse APPENDS to an action="append" default rather than replacing it
    a.gold = a.gold or ["data/gold/audit_188_gold_v2.jsonl"]
    a.context = a.context or ["data/gold/audit_188_context.jsonl"]

    with open(a.migrated, newline="", encoding="utf-8-sig") as f:
        mig = {r["event_id"]: r["instance_relation"]
               for r in csv.DictReader(f)}
    gold, ctx = {}, {}
    for p in a.gold:
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                gold[r["event_id"]] = r
    for p in a.context:
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                ctx[r["event_id"]] = r

    rows = build(mig, gold, a.tol, a.max_retime_s)
    print(f"{len(mig)} migrated, {len(gold)} timing gold, "
          f"{len(rows)} events in the union")

    pool = [r for r in rows.values()
            if r["instance_relation"] in POSITIVE
            and r["alignment"] == "UNDECIDABLE"]
    print(f"\nTIMING pool -- ontology positive, candidate timing UNDECIDABLE:")
    print(f"  {len(pool)} events over "
          f"{len({r['recording_id'] for r in pool})} recordings")
    for k, v in Counter(r["instance_relation"] for r in pool).most_common():
        print(f"    {k:<28} {v:>4}")
    print(f"\n  the pool is defined by instance_relation and the timing gold. "
          f"No peak, score or\n  source category enters the definition, the "
          f"sampler or the row order -- that is the\n  experiment, not a "
          f"convenience.")

    if a.report_only:
        print("\n--report_only: nothing written.")
        return

    picked = sample(pool, a.n_timing, a.seed, a.per_recording)
    random.Random(a.seed).shuffle(picked)

    shown = ["event_id", "recording_id", "candidate_time_s",
             "instance_relation", "previous_segment_label",
             "containing_segment_label", "next_segment_label"]
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(shown + [q for q, _ in TIMING_Q])
        w.writerow([""] * len(shown) + [h for _, h in TIMING_Q])
        for r in picked:
            c = ctx.get(r["event_id"], {})
            w.writerow([
                r["event_id"], r["recording_id"], r["candidate_time"],
                r["instance_relation"],
                c.get("prev_segment_label")
                or c.get("nearest_previous_segment_label") or "",
                c.get("containing_segment_label") or "",
                c.get("next_segment_label")
                or c.get("nearest_next_segment_label") or ""]
                + [""] * len(TIMING_Q))

    print(f"\n{len(picked)} rows over "
          f"{len({r['recording_id'] for r in picked})} recordings -> {a.out}")
    print(f"  hidden: detector score, peak proximity, old timing class and "
          f"every corrected_* field.\n  shown: instance_relation, because "
          f"existence is settled and re-asking it would\n  relitigate an "
          f"earlier pass.")
    print(f"\n  render clips:\n    python -m "
          f"src.auditor.semantic.render_ontology_clips --sheet {a.out} \\\n"
          f"      --data ... --out_dir ... --max_span_s 30")
    print(f"  then, filled:\n    python -m "
          f"src.auditor.boundary.alignment_from_peaks \\\n"
          f"      --predictions .../predictions.jsonl --timing_csv {a.out} "
          f"--null_shift 200")


if __name__ == "__main__":
    main()
