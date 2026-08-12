"""Targeted audit to make candidate alignment trainable. Coverage, then sheets.

The count said NO-GO, and the reason matters more than the verdict. 58 ontology
positives, 13 with timing gold, 12 ALIGNED against 1 MISALIGNED. The binding
constraint is NOT that misaligned events are rare -- it is that 45 of the 58
are absent from the timing gold entirely, so nothing is known about them
either way. Those two diagnoses call for different audits, and the tolerance
sweep settles which one this is: the misaligned count is 1 at every tolerance
from 0.25s to 1.5s. A threshold cannot separate a population this small. The
pool is empty, not badly cut.

So the audit is not one sheet. Two things are missing on two disjoint
populations, and asking for both everywhere would waste most of the work:

    TIMING     events whose instance_relation is known and positive, with no
               corrected boundary time. 45 of them. Existence is settled; the
               question is only WHERE.

    RELATION   events with a corrected time that already resolves to EARLY /
               LATE / DUPLICATE, but whose instance_relation is UNKNOWN. These
               are where misaligned examples can come from at all -- the 13
               events that have both are 12:1 aligned, so filling only the
               timing sheet is very unlikely to reach 30 misaligned.

SAMPLED FOR LEARNABILITY, NOT PREVALENCE. The relation sheet deliberately
over-samples source categories enriched for mislocalisation. The resulting
class balance therefore says nothing about how often the model mislocalises in
the wild, and any model trained on it needs its operating point set on a
population that was not sampled this way. That is stated here so it is not
rediscovered from a suspiciously good number later.

WHAT EACH SHEET SHOWS AND HIDES. The timing sheet SHOWS instance_relation --
existence is settled for those events and re-asking it would invite
relitigation of a label that came from a different, careful pass. It still
carries `t1_disagree_with_relation`, because an annotator who thinks the
recorded relation is wrong needs somewhere to say so that is not the timing
field. The relation sheet HIDES the alignment class and the corrected time,
since those are derived from the judgement being collected.

CLIPS: the sheets carry event_id and recording_id, so
src.auditor.semantic.render_ontology_clips renders them as-is. Use a tighter
--max_span_s for these; a timing question wants a readable ruler, not a
two-minute span.

Usage:
    python -m src.auditor.boundary.alignment_audit_sheet \
        --migrated data/gold/pair_schema_v2_migrated.csv \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --context data/gold/audit_188_context.jsonl \
        --report_only
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict

from src.auditor.boundary.labels import (
    TOL, MAX_RETIME_S, cand_time, recording_of, nearest_corrected, relation,
    find_duplicates)

POSITIVE = ("new_action", "same_action_new_instance")
UNSET = ("UNKNOWN", "cannot_determine", "")
MISALIGNED = ("EARLY", "LATE", "DUPLICATE")

# Categories where the model produced a peak that the audit had to move or
# reject. A candidate the pipeline got exactly right is not a source of
# misaligned examples, so sampling uniformly would spend the budget on EXACT.
ENRICHED = ("early", "late", "duplicate", "false_near_edge",
            "false_mid_segment", "missed_signal_present_not_top",
            "missed_weak_signal", "false_gap")

TIMING_Q = [
    ("t1_disagree_with_relation",
     "The recorded instance_relation is shown. Leave blank if you agree. "
     "Write what you would call it if you do not -- do NOT encode a "
     "disagreement by leaving the time blank."),
    ("t2_boundary_time_s",
     "The time of the boundary, in seconds. If it is a point, one number."),
    ("t3_interval_start_s",
     "If the transition occupies an interval rather than an instant, its "
     "start. Leave both interval fields blank for a point."),
    ("t4_interval_end_s", "...and its end."),
    ("t5_candidate_verdict",
     "About the CANDIDATE time shown, not the boundary: keep / move / "
     "duplicate_of_another_candidate / no_single_time"),
    ("t6_notes", "Anything else, including why a field was unanswerable."),
]

RELATION_Q = [
    ("r1_instance_relation",
     "new_action / same_action_new_instance / same_instance / "
     "initial_action_start / terminal_action_end / cannot_determine"),
    ("r2_transition_shape",
     "point / gap / gradual / overlap / not_observable / not_applicable"),
    ("r3_why",
     "One line: what changed, or what did not."),
    ("r4_notes", "Anything else."),
]


def load(a):
    with open(a.migrated, newline="", encoding="utf-8-sig") as f:
        mig = {r["event_id"]: r["instance_relation"]
               for r in csv.DictReader(f)}
    gold = {}
    for p in a.gold:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    gold[r["event_id"]] = r
    ctx = {}
    for p in a.context:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    ctx[r["event_id"]] = r
    return mig, gold, ctx


def build(mig, gold, tol, max_retime):
    dup = find_duplicates(gold, {}, tol)
    rows = {}
    for eid in set(mig) | set(gold):
        g = gold.get(eid)
        rel = mig.get(eid, "")
        cls, off, why = relation(eid, g, dup, tol, max_retime)
        rows[eid] = {
            "event_id": eid, "instance_relation": rel, "alignment": cls,
            "offset_s": off, "recording_id": recording_of(eid, g),
            "candidate_time": cand_time(eid),
            "corrected_time": nearest_corrected(eid, g) if g else None,
            "source_category": (g or {}).get("source_category", ""),
            "split": (g or {}).get("split", ""), "in_gold": g is not None}
    return rows


def coverage(rows):
    """The 2x2 that says which audit is needed, and how much of each."""
    def cell(has_rel, has_time):
        return [r for r in rows.values()
                if (r["instance_relation"] not in UNSET) == has_rel
                and (r["alignment"] not in ("UNDECIDABLE",)) == has_time]
    print(f"\n{'=' * 74}\nWHAT IS MISSING, AND ON WHICH EVENTS\n{'=' * 74}")
    print(f"  {'':<34}{'timing usable':>16}{'timing missing':>17}")
    for has_rel, lab in ((True, "instance_relation known"),
                         (False, "instance_relation UNKNOWN")):
        a_, b_ = cell(has_rel, True), cell(has_rel, False)
        print(f"  {lab:<34}{len(a_):>16}{len(b_):>17}")
    print("\n  the trainable cell is `relation known + timing usable`, and it "
          "is only useful\n  where the relation is POSITIVE -- alignment is "
          "not a question about a non-boundary.")

    pos_no_time = [r for r in rows.values()
                   if r["instance_relation"] in POSITIVE
                   and r["alignment"] == "UNDECIDABLE"]
    mis_no_rel = [r for r in rows.values()
                  if r["alignment"] in MISALIGNED
                  and r["instance_relation"] in UNSET]
    ali_no_rel = [r for r in rows.values()
                  if r["alignment"] == "EXACT"
                  and r["instance_relation"] in UNSET]
    print(f"\n  TIMING pool   positive relation, no usable time : "
          f"{len(pos_no_time):>4} over "
          f"{len({r['recording_id'] for r in pos_no_time})} recordings")
    print(f"  RELATION pool misaligned already, relation UNKNOWN: "
          f"{len(mis_no_rel):>4} over "
          f"{len({r['recording_id'] for r in mis_no_rel})} recordings")
    print(f"                aligned already,    relation UNKNOWN: "
          f"{len(ali_no_rel):>4} over "
          f"{len({r['recording_id'] for r in ali_no_rel})} recordings")
    if mis_no_rel:
        print("\n  misaligned-pool source categories:")
        for k, v in Counter(r["source_category"]
                            for r in mis_no_rel).most_common(8):
            print(f"    {v:>4}  {k or '(none)'}")
    return pos_no_time, mis_no_rel, ali_no_rel


def sample(pool, n, seed, per_recording=2):
    """Cap per recording so a grouped split still has groups to hold out."""
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


def write_sheet(path, rows, ctx, questions, shown_extra=()):
    shown = (["event_id", "recording_id", "candidate_time_s"]
             + list(shown_extra)
             + ["previous_segment_label", "containing_segment_label",
                "next_segment_label"])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(shown + [q for q, _ in questions])
        w.writerow([""] * len(shown) + [h for _, h in questions])
        for r in rows:
            c = ctx.get(r["event_id"], {})
            w.writerow(
                [r["event_id"], r["recording_id"], r["candidate_time"]]
                + [r.get(k, "") for k in shown_extra]
                + [c.get("prev_segment_label")
                   or c.get("nearest_previous_segment_label") or "",
                   c.get("containing_segment_label") or "",
                   c.get("next_segment_label")
                   or c.get("nearest_next_segment_label") or ""]
                + [""] * len(questions))


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
    ap.add_argument("--n_relation", type=int, default=60)
    ap.add_argument("--per_recording", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--timing_out", default="data/gold/alignment_timing.csv")
    ap.add_argument("--relation_out",
                    default="data/gold/alignment_relation.csv")
    a = ap.parse_args()
    # argparse APPENDS to an action="append" default rather than
    # replacing it, so passing --gold once would silently load the
    # default file as well. Defaults are applied here instead.
    a.gold = a.gold or ["data/gold/audit_188_gold_v2.jsonl"]
    a.context = a.context or ["data/gold/audit_188_context.jsonl"]

    mig, gold, ctx = load(a)
    rows = build(mig, gold, a.tol, a.max_retime_s)
    print(f"{len(mig)} migrated, {len(gold)} timing gold, "
          f"{len(rows)} events in the union")
    pos_no_time, mis_no_rel, ali_no_rel = coverage(rows)

    if a.report_only:
        print("\n--report_only: no sheets written. Read the two pool sizes "
              "above before\nchoosing --n_timing and --n_relation; a pool "
              "smaller than the target is the\nanswer to a different "
              "question than a pool that is large but unlabelled.")
        return

    t_rows = sample(pos_no_time, a.n_timing, a.seed, a.per_recording)
    # keep the misaligned pool first, then top up with aligned so the sheet is
    # not obviously "these are the wrong ones" to the annotator
    r_pool = mis_no_rel + [r for r in ali_no_rel
                           if r["source_category"] in ENRICHED]
    r_rows = sample(r_pool, a.n_relation, a.seed, a.per_recording)
    random.Random(a.seed).shuffle(r_rows)

    write_sheet(a.timing_out, t_rows, ctx, TIMING_Q,
                shown_extra=("instance_relation",))
    write_sheet(a.relation_out, r_rows, ctx, RELATION_Q)

    print(f"\nTIMING sheet   {len(t_rows):>3} rows over "
          f"{len({r['recording_id'] for r in t_rows})} recordings "
          f"-> {a.timing_out}")
    print(f"  shows instance_relation; existence is settled for these and "
          f"the question is only WHERE")
    print(f"RELATION sheet {len(r_rows):>3} rows over "
          f"{len({r['recording_id'] for r in r_rows})} recordings "
          f"-> {a.relation_out}")
    print(f"  hides the alignment class and the corrected time -- both are "
          f"derived from what is\n  being collected. Of these, "
          f"{sum(1 for r in r_rows if r['alignment'] in MISALIGNED)} are "
          f"already misaligned and "
          f"{sum(1 for r in r_rows if r['alignment'] == 'EXACT')} aligned.")

    print(f"\n  CEILING IF EVERY ROW COMES BACK USABLE: the timing sheet can "
          f"add at most\n  {len(t_rows)} events to the 13 that exist. Whether "
          f"that reaches 30 misaligned depends\n  entirely on the rate, and "
          f"the 13 observed so far are 12:1 aligned. If the timing\n  sheet "
          f"returns the same ratio it yields about "
          f"{len(t_rows) // 13} more misaligned, not 29 --\n  which is why "
          f"the relation sheet exists and why it is sampled from categories\n"
          f"  where the model already had to be corrected.")
    print(f"\n  render clips with:\n    python -m "
          f"src.auditor.semantic.render_ontology_clips --sheet "
          f"{a.timing_out} \\\n      --data ... --out_dir ... "
          f"--max_span_s 30")


if __name__ == "__main__":
    main()
