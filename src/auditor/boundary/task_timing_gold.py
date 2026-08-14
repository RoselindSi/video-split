"""Task-level timing gold from the semantic pool: freeze, merge, and say what it is not.

ONE POOL, 69 EVENTS: the 48 frozen semantic events plus 21 enrichment ones,
both under the same schema after the standard was agreed and both sheets
re-checked against it. The re-check moved eight events, all in the same
direction: five amplitude or fold-to-wipe changes that had been recorded as
`point` became `motion_phase_only`, two multi-boundary spans collapsed to a
single instance reset, and four new rows came in as `no_task_boundary`. The
standard tightened and nothing loosened.

MOTION PHASE IS NOT A BOUNDARY. Ten events carry times in
`motion_phase_points_json` with an EMPTY canonical list, and the notes say
why: the frozen ontology treats a continuous held sequence as one instance, so
an amplitude change inside a wipe is not a task boundary. Those 21 times are
carried through and are never emitted as boundaries. Merging them in would put
21 fake boundaries into anything that consumes this file, and they are exactly
the kind of time that looks usable.

TWO KINDS OF EXISTENCE NEGATIVE, 20 in total, and they are not the same
mistake to make:

    no_action_change    12   nothing changed here
    phase_change_only    8   something visibly changed and it is not a task
                             boundary. A detector firing here is a harder
                             error to fault than one firing in continuous idle

Both are negatives and this is the only file in the project that has ever
carried either.

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
    task-level timing for events selected for a SEMANTIC audit, and 20 of them
    say no boundary exists at all.

So this file freezes and describes. It does not feed the event-matched timing
test, because doing that silently would put a peak-selected pool through a
procedure whose whole value was that its pool was not.

Usage:
    python -m src.auditor.boundary.task_timing_gold \
        --csv data/gold/task_timing_gold_48.csv \
        --csv data/gold/task_timing_gold_enrich21.csv \
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
            # TWO KINDS OF EXISTENCE NEGATIVE, and they are not the same
            # error to make. `no_task_boundary` says nothing changed here.
            # `motion_phase_only` says something visibly changed and it is not
            # a task boundary under the frozen ontology -- a detector firing
            # there is a harder mistake to fault than one firing in continuous
            # idle. Both are negatives; collapsing them would lose the
            # distinction that makes them worth having.
            e["asserts_no_boundary"] = (st in ("no_task_boundary",
                                               "motion_phase_only")
                                        or e["no_task_boundary_flag"])
            e["negative_kind"] = (
                "phase_change_only" if st == "motion_phase_only" else
                "no_action_change" if e["asserts_no_boundary"] else None)
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
    print(f"    {sum(1 for e in nb if e['negative_kind'] == 'no_action_change')}"
          f" no_action_change (nothing changed) + "
          f"{sum(1 for e in nb if e['negative_kind'] == 'phase_change_only')}"
          f" phase_change_only\n    (something visibly changed and it is not "
          f"a task boundary -- a harder negative)")
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
