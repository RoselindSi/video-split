"""Task-level timing gold from the semantic pool: freeze, merge, and say what it is not.

TWO SCHEMAS, ONE POOL. The 48 frozen semantic events and 17 of the 41
enrichment events now carry human TASK-LEVEL boundary times. The two sheets do
not have the same columns -- the 48 add `motion_phase_points_json`,
`canonical_task_times_json` and `timing_precision`, the 17 add
`onset_from_idle` and `no_task_boundary` -- so they are read separately and
merged with the differences kept rather than flattened.

MOTION PHASE IS NOT A BOUNDARY. Two events carry `motion_phase_only` with
their times in `motion_phase_points_json` and an EMPTY canonical list, and the
notes say why: the frozen ontology treats a continuous held sequence as one
same_instance, so a flip-cycle phase change is not a task boundary. Those
times are carried through and are never emitted as boundaries. Merging them in
would put three fake boundaries into any test that consumes this file.

`no_task_boundary` IS A LABEL, NOT A GAP. Eight events across the two sheets
say there is no task-level boundary at all. They have no times and they are
not missing data -- they are negatives, and the only file in this project that
has ever carried them.

WHAT THIS POOL IS NOT, and it matters before anything is run on it:

    NOT PEAK-BLIND. Both sheets are drawn from audit_188_gold_v2, which was
    sampled on BOUNDARY ERROR CATEGORIES -- an event is in it partly because
    of where the detector fired or failed to. The 45-event alignment gold was
    built specifically to avoid that, and these 65 cannot simply extend it.
    Any timing test run here inherits a selection the earlier one was designed
    to exclude, and the direction of that bias is not obvious: `missed_*`
    events have no peak at all while `false_*` events have one by definition.

    NOT THE SAME QUESTION as the 45. Those recorded where the boundary is for
    events whose instance_relation was already known positive. These record
    task-level timing for events selected for a SEMANTIC audit, and eight of
    them say no boundary exists.

So this file freezes and describes. It does not feed the event-matched timing
test, because doing that silently would put a peak-selected pool through a
procedure whose whole value was that its pool was not.

Usage:
    python -m src.auditor.boundary.task_timing_gold \
        --csv data/gold/task_timing_gold_48.csv \
        --csv data/gold/task_timing_gold_enrich17.csv \
        --out data/gold/task_timing_gold.json
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from collections import Counter


def cell(s):
    s = (s or "").strip()
    if not s or s in ("[]", "null", "None"):
        return []
    try:
        return json.loads(s)
    except ValueError:
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []


def yes(s):
    return str(s or "").strip().lower() in ("yes", "true", "1")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()

    events, dupes = {}, []
    for p in a.csv:
        with open(p, newline="", encoding="utf-8-sig") as f:
            rows = [r for r in csv.DictReader(f)
                    if (r.get("audit_key") or "").strip()]
        src = os.path.basename(p)
        for r in rows:
            k = r["audit_key"]
            pts = [float(x) for x in cell(r.get("human_boundary_points_json"))]
            ivs = [[float(x), float(y)] for x, y in
                   cell(r.get("human_boundary_intervals_json"))]
            phase = [float(x) for x in
                     cell(r.get("motion_phase_points_json"))]
            canon = [float(x) for x in
                     cell(r.get("canonical_task_times_json"))]
            st = (r.get("task_timing_status") or "").strip()
            e = {
                "audit_key": k, "source_sheet": src,
                "recording_id": r["recording_id"],
                "candidate_time": float(r["candidate_time"]),
                "task_timing_status": st,
                # raw, untouched
                "task_points": pts, "task_intervals": ivs,
                "motion_phase_points": phase,
                "canonical_task_times_sheet": canon,
                "timing_precision": (r.get("timing_precision") or "").strip(),
                "onset_from_idle": yes(r.get("onset_from_idle")),
                "no_task_boundary_flag": yes(r.get("no_task_boundary")),
                "timing_note": r.get("timing_note", ""),
            }
            # derived: the boundaries this event actually asserts.
            # motion_phase_points are DELIBERATELY excluded -- the annotator
            # marked them as phase changes that the frozen ontology does not
            # treat as task boundaries.
            e["boundaries"] = ([("point", x, x) for x in pts]
                               + [("interval", min(lo, hi), max(lo, hi))
                                  for lo, hi in ivs])
            e["asserts_no_boundary"] = (st == "no_task_boundary"
                                        or e["no_task_boundary_flag"])
            if k in events:
                dupes.append((k, events[k]["source_sheet"], src))
            events[k] = e
        print(f"  {src:<44} {len(rows):>3} rows")

    print(f"\n{len(events)} events over "
          f"{len({e['recording_id'] for e in events.values()})} recordings")
    if dupes:
        print(f"  !! {len(dupes)} audit_keys appear in more than one sheet; "
              f"the later file won: {dupes[:4]}")

    print(f"\n  task_timing_status: "
          f"{dict(Counter(e['task_timing_status'] for e in events.values()).most_common())}")
    nb = [e for e in events.values() if e["asserts_no_boundary"]]
    print(f"  events asserting NO task boundary: {len(nb)} "
          f"-- negatives, and the only file here that has ever carried them")
    ph = [e for e in events.values() if e["motion_phase_points"]]
    print(f"  events with motion-phase points: {len(ph)}, holding "
          f"{sum(len(e['motion_phase_points']) for e in ph)} times. NOT "
          f"emitted as\n    boundaries -- the frozen ontology treats a "
          f"continuous held sequence as one instance.")
    n_b = sum(len(e["boundaries"]) for e in events.values())
    with_b = [e for e in events.values() if e["boundaries"]]
    print(f"  {n_b} task boundaries over {len(with_b)} events "
          f"({sum(1 for e in with_b if len(e['boundaries']) > 1)} carry more "
          f"than one)")
    empty = [e for e in events.values()
             if not e["boundaries"] and not e["asserts_no_boundary"]]
    if empty:
        print(f"  !! {len(empty)} events have neither a boundary nor a "
              f"no-boundary assertion: "
              f"{[e['audit_key'] for e in empty][:5]}")

    # the sheet's own canonical list against the one derivable here
    mism = []
    for e in events.values():
        if not e["canonical_task_times_sheet"]:
            continue
        mine = sorted([x for _k, x, y in e["boundaries"] if _k == "point"]
                      + [round((x + y) / 2, 3) for _k, x, y
                         in e["boundaries"] if _k == "interval"])
        theirs = sorted(round(x, 3) for x in e["canonical_task_times_sheet"])
        if [round(x, 2) for x in mine] != [round(x, 2) for x in theirs]:
            mism.append((e["audit_key"], mine, theirs))
    print(f"\n  canonical-times check: {len(mism)} events where the sheet's "
          f"canonical list disagrees\n    with points+interval-midpoints "
          f"derived here")
    for k, mine, theirs in mism[:4]:
        print(f"    {k}: derived {mine}  sheet {theirs}")

    print(f"\n{'=' * 74}\nWHAT THIS POOL IS NOT\n{'=' * 74}")
    print("  NOT PEAK-BLIND. Both sheets come from audit_188_gold_v2, which "
          "was sampled on\n  BOUNDARY ERROR categories -- an event is in it "
          "partly because of where the detector\n  fired or failed to. The "
          "45-event alignment gold exists precisely to avoid that,\n  and "
          "these cannot simply extend it. The direction of the bias is not "
          "obvious either:\n  `missed_*` events have no peak at all while "
          "`false_*` events have one by definition.")
    print("\n  NOT THE SAME QUESTION. The 45 recorded where the boundary is "
          "for events already\n  known to be ontology positives. These record "
          "task-level timing for events picked\n  for a SEMANTIC audit, and "
          "some of them say no boundary exists at all.")

    if a.out:
        json.dump({"sources": a.csv, "n": len(events),
                   "events": list(events.values())},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
