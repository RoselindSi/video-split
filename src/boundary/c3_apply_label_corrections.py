"""Apply the label corrections two independent annotators agree on.

Only where BOTH annotators decided, agreed with each other, and BOTH differ
from the stored label. That rule is deliberately narrow and it is the whole
justification: two independent votes against a stored label is evidence about
that label, while one reviewer disagreeing is a tie between two opinions.

WHAT IS NOT CORRECTED, AND WHY IT MATTERS MORE THAN WHAT IS.

  Events where the annotators split are left exactly as they are. Their
  disagreement is uniform -- every one is annotator 1 saying same where
  annotator 2 says sharp -- which is a threshold difference a definition can
  settle and a majority vote cannot. Flipping them on a 1-1 tie broken by the
  stored label would be circular, and flipping them toward the more recent
  annotator would be arbitrary.

  Events nobody audited are left alone, and there are 277 of them. This is the
  part that must not be misread: correcting 8 labels does not fix the label
  noise, it fixes the 8 instances of it that happen to have been measured. On
  the 33 audited and decided events the second annotator differed from the
  store 10 times. If that rate holds, roughly a quarter to a third of the 313
  clean labels are wrong, and no amount of modelling recovers from that. The
  8 are a floor on the problem, not a solution to it.

The corrected files are WRITTEN BESIDE the originals, never over them. Every
number this project has produced was computed against the originals, and a
silent in-place edit would make those numbers unreproducible while looking
like an improvement.

A per-row audit trail records the old value, the new value, both annotators'
calls and their stated reasons, so a correction can be re-examined without
re-watching the clip.

Usage:
    python -m src.boundary.c3_apply_label_corrections \
        --sheet data/gold/observable_audit_your_call_36.csv \
        --sheet data/gold/observable_audit_annotator2_36.csv \
        --pair_labels data/gold/pair_labels_v1.csv \
        --pair_labels data/gold/batch3_pair_labels_v1.csv \
        --suffix _corrected_v1 --out_dir data/gold
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter

from src.boundary.pair_taxonomy import (
    SUBTYPES, SUBTYPE_TO_SUPERVISION, load_pair_labels,
)
from src.boundary.c3_annotator_agreement import load_sheet, read_csv

SHARP = "sharp_visible_transition"
SAME = "same_action_internal_motion"
CALL_TO_SUBTYPE = {"sharp": SHARP, "same": SAME}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", action="append", required=True,
                    help="repeatable; give exactly two")
    ap.add_argument("--pair_labels", action="append", required=True)
    ap.add_argument("--suffix", default="_corrected_v1")
    ap.add_argument("--out_dir", default="data/gold")
    a = ap.parse_args()
    if len(a.sheet) != 2:
        raise SystemExit("give exactly two --sheet files")

    s1, s2 = load_sheet(a.sheet[0]), load_sheet(a.sheet[1])
    shared = sorted(set(s1) & set(s2))
    print(f"{len(shared)} shared audited events")

    stored = {}
    for p in a.pair_labels:
        for eid, v in load_pair_labels(p).items():
            stored[eid] = v

    corrections, split, unanimous_ok = {}, [], 0
    for e in shared:
        c1, c2 = s1[e]["call"], s2[e]["call"]
        cur = (stored.get(e) or {}).get("temporal_pair_subtype")
        if c1 not in CALL_TO_SUBTYPE or c2 not in CALL_TO_SUBTYPE:
            continue
        if c1 != c2:
            split.append((e, c1, c2, cur))
            continue
        want = CALL_TO_SUBTYPE[c1]
        if cur == want:
            unanimous_ok += 1
        elif cur in (SHARP, SAME):
            corrections[e] = {"old": cur, "new": want,
                              "why1": s1[e].get("ans", ""),
                              "why2": s2[e].get("ans", "")}
        elif cur is not None:
            # both annotators call it clean but the store has it as gradual,
            # offscreen, ambiguous... a different kind of disagreement, and
            # promoting it into the clean set would ADD an event to every
            # downstream population rather than relabel one. Reported, never
            # applied here.
            split.append((e, c1, c2, cur))

    print(f"  {unanimous_ok} unanimous and already correct")
    print(f"  {len(corrections)} unanimous and DIFFERENT from the store "
          f"-> corrected")
    print(f"  {len(split)} not corrected (annotators split, or the store has "
          f"a non-clean subtype)")
    for e, v in sorted(corrections.items()):
        print(f"    {e:<50} {v['old'].split('_')[0]} -> {v['new'].split('_')[0]}")
    for e, c1, c2, cur in split:
        print(f"    [left alone] {e:<44} A1 {c1:<6} A2 {c2:<6} store "
              f"{cur or '(none)'}")

    if not corrections:
        print("\nnothing to correct")
        return

    os.makedirs(a.out_dir, exist_ok=True)
    trail = os.path.join(a.out_dir, f"label_corrections{a.suffix}.csv")
    with open(trail, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "old_subtype", "new_subtype",
                    "old_supervision", "new_supervision",
                    "annotator1_call", "annotator2_call",
                    "annotator1_observability", "annotator2_observability",
                    "evidence"])
        for e, v in sorted(corrections.items()):
            w.writerow([e, v["old"], v["new"],
                        SUBTYPE_TO_SUPERVISION[v["old"]],
                        SUBTYPE_TO_SUPERVISION[v["new"]],
                        s1[e]["call"], s2[e]["call"], v["why1"], v["why2"],
                        "two independent annotators agreed against the store"])

    for p in a.pair_labels:
        rows = read_csv(p)
        if not rows:
            continue
        cols = list(rows[0])
        if "temporal_pair_subtype" not in cols:
            print(f"  !! {p} has no temporal_pair_subtype column, skipped")
            continue
        n = 0
        for r in rows:
            v = corrections.get((r.get("event_id") or "").strip())
            if not v:
                continue
            r["temporal_pair_subtype"] = v["new"]
            # the supervision is rewritten too. Leaving strong_separate beside
            # same_action_internal_motion would put the row in the clean set
            # with the OLD sign -- a correction that silently does nothing,
            # which is worse than none.
            if "pair_supervision" in cols:
                r["pair_supervision"] = SUBTYPE_TO_SUPERVISION[v["new"]]
            if "notes" in cols:
                r["notes"] = ((r.get("notes") or "") +
                              " | corrected by two-annotator agreement").strip(" |")
            n += 1
        out = os.path.join(a.out_dir, os.path.basename(p).replace(
            ".csv", f"{a.suffix}.csv"))
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, cols)
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {out}  ({n} rows changed of {len(rows)})")

    missing = set(corrections) - {
        (r.get("event_id") or "").strip()
        for p in a.pair_labels for r in read_csv(p)}
    if missing:
        print(f"\n  !! {len(missing)} corrections had no row in any "
              f"--pair_labels file and were NOT written: {sorted(missing)}")

    print(f"\nwrote {trail}")
    print(f"\n{len(corrections)} of {len(stored)} stored labels changed "
          f"({len(corrections) / max(len(stored), 1):.1%}). On the 33 audited "
          f"and decided events the annotators differed from the store far more "
          f"often than that;\nwhat is corrected here is the measured part of "
          f"the label noise, not the noise. Read any improvement downstream as "
          f"a lower bound on what full relabelling would give, never as the "
          f"problem being solved.")


if __name__ == "__main__":
    main()
