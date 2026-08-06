"""Blind re-annotation sheet for a pair-label file whose provenance is in doubt.

Written for a specific finding. The 240 batch3 subtypes in
data/gold/batch3_pair_labels_v1.csv were not assigned by a person watching
video: they were converted from batch3_blind_review_complete_240.csv, whose
temporal_truth column came from Claude reading 2.5 fps contact sheets over a
+/-3 s window. 168 of them are in the current 313-event clean set -- 54% of
every number this project has reported. The other 175 labels, in
pair_labels_v1.csv, came from human video review.

THE EXCLUDED ROWS MATTER AS MUCH AS THE KEPT ONES, and are included by
default. 72 of the 240 never entered any evaluation because those same
machine-made calls sent them to soft_transition or exclude. If any of those
were wrong the clean set is not merely noisy, it is selected -- a population
shaped by the labels under review. Re-annotating only the 168 that survived
would leave that selection in place and invisible.

BLIND. The sheet carries the clip and the segment labels; the existing
subtype, the supervision, and whether the row is currently in the clean set
all live in a separate key. Someone re-annotating a label they can see is
confirming it.

ALREADY-AUDITED EVENTS ARE KEPT unless --exclude_audited is given. 20 of the
36 audited events are batch3 rows and already carry two independent calls;
answering them again costs little and gives a within-annotator consistency
check, which nothing else here provides.

The full seven-way vocabulary is asked for, not the clean binary. Whether a
row belongs in the clean set at all is exactly what is in question.

Usage:
    python -m src.boundary.c3_relabel_sheet \
        --pair_labels data/gold/batch3_pair_labels_v1.csv \
        --manifest /workspace/tr1/results/hal/batch3/batch3_manifest.jsonl \
        --blind_csv /workspace/tr1/results/hal/batch3/batch3_blind_review.csv \
        --out_dir /workspace/tr1/results/hal/c3/relabel_batch3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

from src.boundary.pair_taxonomy import SUBTYPES, CLEAN_BINARY, load_pair_labels
from src.boundary.c3_annotator_agreement import read_csv, load_sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair_labels", required=True)
    ap.add_argument("--manifest", help="jsonl with event_id/recording_id/t, for rendering")
    ap.add_argument("--blind_csv", action="append", default=[])
    ap.add_argument("--exclude_audited", action="append", default=[],
                    help="audit sheets whose events to leave out")
    ap.add_argument("--only_events",
                    help="restrict to the event_ids in this file (first column, "
                         "'#' comments and a header skipped) -- e.g. "
                         "data/gold/claude_labelled_events.txt")
    ap.add_argument("--clean_only", action="store_true",
                    help="drop rows the current labels exclude. NOT the default: "
                         "those exclusions were made by the labels under review")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    stored = load_pair_labels(a.pair_labels)
    print(f"{len(stored)} rows in {a.pair_labels}")
    print(f"  by subtype: "
          f"{dict(Counter(v['temporal_pair_subtype'] for v in stored.values()))}")
    n_clean = sum(1 for v in stored.values()
                  if v["pair_supervision"] in CLEAN_BINARY)
    print(f"  {n_clean} currently in the clean set, {len(stored) - n_clean} "
          f"excluded by these same labels")

    done = set()
    for p in a.exclude_audited:
        done |= set(load_sheet(p))
    ids = sorted(stored)
    if a.only_events:
        want = set()
        for ln in open(a.only_events, encoding="utf-8"):
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.startswith("event_id"):
                continue
            want.add(ln.split(",")[0].strip())
        missing = want - set(ids)
        ids = [e for e in ids if e in want]
        print(f"  --only_events: {len(ids)} of {len(want)} listed events found "
              f"in this label file")
        if missing:
            print(f"    {len(missing)} listed events are not in "
                  f"{a.pair_labels} -- they belong to another label file or "
                  f"never made it into one")
    if a.clean_only:
        ids = [e for e in ids if stored[e]["pair_supervision"] in CLEAN_BINARY]
        print(f"  --clean_only: {len(ids)} rows. The excluded rows are dropped "
              f"by the very labels being re-annotated, so a bias in them stays "
              f"invisible.")
    if done:
        before = len(ids)
        ids = [e for e in ids if e not in done]
        print(f"  --exclude_audited removed {before - len(ids)}")
    print(f"{len(ids)} events to re-annotate")

    pos = {}
    if a.manifest:
        for l in open(a.manifest, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                pos[r["event_id"]] = r
    ctx = {}
    for p in a.blind_csv:
        recs = ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
                if p.endswith(".jsonl") else read_csv(p))
        for r in recs:
            ctx.setdefault(r["event_id"], r)

    os.makedirs(a.out_dir, exist_ok=True)
    if pos:
        miss = [e for e in ids if e not in pos]
        if miss:
            print(f"  !! {len(miss)} events have no manifest row and cannot be "
                  f"rendered: {miss[:4]}")
        with open(os.path.join(a.out_dir, "relabel_manifest.jsonl"), "w",
                  encoding="utf-8") as f:
            for e in ids:
                if e in pos:
                    f.write(json.dumps(pos[e], ensure_ascii=False) + "\n")

    cols = ["event_id", "prev_segment_label", "next_segment_label",
            "containing_segment_label"]
    with open(os.path.join(a.out_dir, "relabel_blind_context.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for e in ids:
            c = ctx.get(e, {})
            w.writerow({k: c.get(k, "") for k in cols})

    with open(os.path.join(a.out_dir, "relabel_sheet.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "subtype(" + "|".join(SUBTYPES) + ")",
                    "confidence(1_guess|2_lean|3_sure)", "why_this_subtype",
                    "notes"])
        for e in ids:
            w.writerow([e, "", "", "", ""])

    with open(os.path.join(a.out_dir, "relabel_key.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "existing_subtype", "existing_supervision",
                    "currently_in_clean_set", "already_audited"])
        for e in ids:
            v = stored[e]
            w.writerow([e, v["temporal_pair_subtype"], v["pair_supervision"],
                        int(v["pair_supervision"] in CLEAN_BINARY),
                        int(e in done)])

    print(f"\nwrote relabel_sheet.csv / relabel_key.csv / "
          f"relabel_blind_context.csv"
          + (" / relabel_manifest.jsonl" if pos else "") + f" in {a.out_dir}")
    print("The existing subtype is in the KEY, not the sheet. Re-annotating a "
          "label you can see is confirming it, and the whole reason this file "
          "exists is that\nthe existing labels are the thing in doubt.")
    print("Definitions: docs/pair_taxonomy_definitions.md -- the ones the "
          "original labelling used, reproduced rather than improved.")


if __name__ == "__main__":
    main()
