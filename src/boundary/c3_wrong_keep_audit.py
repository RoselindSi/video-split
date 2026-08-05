"""Export the events a policy AUTO_KEEPs that are not sharp_visible_transition,
for human review, before any Stage 0 gate is designed.

The development policy auto-keeps 64 of 412 events, 58 of them correct. The
other 6 are what makes full-taxonomy precision 0.906 instead of the
clean-binary 0.951, and they are the reason a cascade Stage 2 was structurally
infeasible: Stage 2 only converts REVIEW into decisions, it never withdraws an
AUTO_KEEP, so those 6 were already counted before it acted. Reaching 0.95 from
58/64 by adding correct keeps alone needs 56 more with zero errors, and the
per-class cap on `ambiguous` (1 of 3 already accepted, cap 0.0) cannot be
repaired by adding anything at all.

So these 6 are the whole problem, and 6 events can be looked at individually.
What matters is whether they share an observable signature that a cheap
feature could catch -- which is exactly what a Stage 0 observability gate would
have to learn, and what decides whether such a gate is worth building.

Prints each one with every score and reliability number available, and writes a
review sheet with blank columns for the visual verdict. Deliberately does not
guess at a mechanism: the columns to fill are what a person sees in the clip.

Usage:
    python -m src.boundary.c3_wrong_keep_audit \
        --decisions /workspace/tr1/results/hal/c3/policy_dev_decisions.csv \
        --events /workspace/tr1/results/hal/c3/local_events.csv \
        --events /workspace/tr1/results/hal/c3/local_events_batch3.csv \
        --out /workspace/tr1/results/hal/c3/wrong_keeps_review.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter

CORRECT = "sharp_visible_transition"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", required=True,
                    help="c3_selective_policy --dump_decisions CSV")
    ap.add_argument("--events", action="append",
                    help="c3_local_eval --dump_events CSV(s), for columns the "
                         "decisions file does not carry (detect_longest_gap_s)")
    ap.add_argument("--decision", default="AUTO_KEEP",
                    help="AUTO_KEEP by default; AUTO_REJECT audits the other "
                         "side, where a wrong call means a real boundary was "
                         "silently dropped")
    ap.add_argument("--out")
    a = ap.parse_args()

    extra = {}
    for p in a.events or []:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                extra[r["event_id"]] = r

    rows = list(csv.DictReader(open(a.decisions, newline="", encoding="utf-8")))
    sel = [r for r in rows if r["decision"] == a.decision]
    if a.decision == "AUTO_KEEP":
        wrong = [r for r in sel
                 if (r["subtype"] and r["subtype"] != CORRECT)
                 or (not r["subtype"] and r["y"] != "1")]
    else:
        wrong = [r for r in sel
                 if (r["subtype"] == CORRECT) or (not r["subtype"] and r["y"] == "1")]

    print(f"{len(rows)} events, {len(sel)} {a.decision}, {len(wrong)} of them wrong")
    print(f"  by subtype: {dict(Counter(r['subtype'] or '(none)' for r in wrong))}")
    print(f"  by source:  {dict(Counter(r['source'] for r in wrong))}")

    score_cols = [c for c in rows[0]
                  if c not in ("event_id", "recording_id", "source", "y", "subtype",
                               "reliability", "decision", "reason")]
    print()
    for r in sorted(wrong, key=lambda x: x["subtype"]):
        e = extra.get(r["event_id"], {})
        print(f"  {r['event_id']}")
        print(f"      subtype {r['subtype'] or '(none)'}   y {r['y'] or '(none)'}   "
              f"source {r['source']}   reason {r['reason']}")
        print(f"      reliability {r['reliability']}   "
              f"longest detection gap {e.get('detect_longest_gap_s', '?')}s")
        for c in score_cols:
            print(f"      {c:<32} {r.get(c, '')}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        fields = (["event_id", "recording_id", "source", "subtype", "y", "reason",
                   "reliability", "detect_longest_gap_s"] + score_cols
                  + ["visual_verdict", "why_scored_high", "separable_by",
                     "notes"])
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in sorted(wrong, key=lambda x: x["subtype"]):
                e = extra.get(r["event_id"], {})
                w.writerow({**{k: r.get(k, "") for k in fields if k in r},
                            "detect_longest_gap_s": e.get("detect_longest_gap_s", ""),
                            "visual_verdict": "", "why_scored_high": "",
                            "separable_by": "", "notes": ""})
        print(f"\nwrote {a.out}")
        print("  Fill three columns per event, from the clip rather than from "
              "the scores:")
        print("    visual_verdict   what it actually is on screen")
        print("    why_scored_high  what the model plausibly latched onto")
        print("    separable_by     a CHEAP observable that would have caught it, "
              "or 'none' -- 'none' on most of them is the answer that says a "
              "Stage 0 gate is not learnable from these features, and is as "
              "useful as finding a pattern")


if __name__ == "__main__":
    main()
