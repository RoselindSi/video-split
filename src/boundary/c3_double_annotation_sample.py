"""Sample for a second, independent annotation of the pair taxonomy.

The first audit found a human agreeing with the taxonomy label on 21 of 33
REVIEW-band events, every call made at "sure". That is one reviewer against
one label, and it cannot say which of two very different things happened:

  the reviewer drifted from a convention that is itself stable, in which case
  0.636 is not a ceiling and the C3.2 line still has a premise

  the convention is not reproducible, in which case no representation can
  learn it, the 60.9% REVIEW rate is a property of the TARGET, and the work
  belongs in the annotation protocol

A second independent annotator separates them. That is the only question this
sample exists to answer, and it is worth 40 clips.

THREE ARMS THAT MUST NEVER BE POOLED. They answer different questions and one
of them is deliberately enriched:

  A  the disagreements    all of them, so the second annotator's verdict on
                          each is known. ENRICHED BY CONSTRUCTION -- the
                          agreement rate within this arm means nothing, it was
                          selected for disagreement. It is diagnostic only:
                          does annotator 2 side with the reviewer or the label?
  B  fresh REVIEW         events in the REVIEW band that nobody has audited.
                          THIS is the unbiased estimate of taxonomy agreement
                          where it matters, and the number that decides.
  C  non-REVIEW control   clean events the policy already decides. If
                          agreement is high here and low in B, the problem is
                          specific to the hard band; if it is low in both, the
                          taxonomy is unstable everywhere and the REVIEW rate
                          was never a modelling failure.

Pooling A into B would report an agreement rate depressed by construction, and
it would be the headline number. They are written to separate files.

THE FULL SEVEN-WAY SUBTYPE IS ASKED FOR, not the clean binary. The likeliest
mechanism behind the 12 disagreements is not that one side saw a boundary and
the other did not -- it is that the event is really gradual_phase_transition or
ambiguous and was forced into a two-way choice. An annotator restricted to
sharp/same cannot report that, and the measurement would attribute a taxonomy
problem to human error.

BLIND. No label, no first-reviewer call, no arm, no model score, and the rows
are shuffled so arm A does not arrive as a block. The key is a separate file.

Usage:
    python -m src.boundary.c3_double_annotation_sample \
        --features /workspace/tr1/results/hal/c3/hand_trajectory_features.csv \
        --first_sheet data/gold/observable_audit_your_call_36.csv \
        --first_key /workspace/tr1/results/hal/c3/observable_audit/audit_key.csv \
        --n_fresh 16 --n_control 12 \
        --blind_csv data/gold/audit_188_context.jsonl \
        --blind_csv /workspace/tr1/results/hal/batch3/batch3_blind_review.csv \
        --out_dir /workspace/tr1/results/hal/c3/double_annotation
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

import numpy as np

from src.boundary.pair_taxonomy import SUBTYPES

SHARP = "sharp_visible_transition"


def read_csv(path):
    # utf-8-sig: the returned sheet carries a BOM and every prefix lookup
    # against its first column silently misses without this.
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def col(row, prefix):
    return next((k for k in row if k.startswith(prefix)), None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--first_sheet", required=True)
    ap.add_argument("--first_key", required=True)
    ap.add_argument("--n_fresh", type=int, default=16)
    ap.add_argument("--n_control", type=int, default=12)
    ap.add_argument("--max_per_recording", type=int, default=2)
    ap.add_argument("--blind_csv", action="append", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    feat = {r["event_id"]: r for r in read_csv(a.features)}
    key = {r["event_id"]: r for r in read_csv(a.first_key)}
    sheet = read_csv(a.first_sheet)
    c_call = col(sheet[0], "your_call")
    if not c_call:
        raise SystemExit("no your_call column in --first_sheet")

    seen, disagree = set(), []
    for r in sheet:
        eid = r["event_id"]
        seen.add(eid)
        k = key.get(eid)
        if not k:
            continue
        call = (r.get(c_call) or "").strip().lower()
        truth = "sharp" if str(k.get("y", "")).strip() in ("1", "1.0") else "same"
        if call in ("sharp", "same") and call != truth:
            disagree.append(feat.get(eid) or {"event_id": eid,
                                              "recording_id": k["recording_id"]})
    print(f"first audit: {len(sheet)} rows, {len(disagree)} disagreements")
    if not disagree:
        print("  no disagreements -- arm A is empty and this sample reduces to "
              "a plain agreement estimate")

    rng = np.random.RandomState(a.seed)

    def draw(pool, n):
        pool = [r for r in pool if r["event_id"] not in seen]
        rng.shuffle(pool)
        out, per = [], Counter()
        for r in pool:
            if len(out) >= n:
                break
            if per[r["recording_id"]] >= a.max_per_recording:
                continue
            out.append(r)
            per[r["recording_id"]] += 1
        return out

    rows = list(feat.values())
    fresh = draw([r for r in rows if r.get("decision") == "REVIEW"], a.n_fresh)
    control = draw([r for r in rows if r.get("decision")
                    and r["decision"] != "REVIEW"], a.n_control)
    print(f"arm B fresh REVIEW: {len(fresh)} of {a.n_fresh} requested")
    print(f"arm C control (non-REVIEW clean): {len(control)} of {a.n_control}")
    if len(fresh) < a.n_fresh or len(control) < a.n_control:
        print("  !! an arm came up short. Arm B is the arm that decides, so a "
              "short B widens its interval -- check the per-recording cap and "
              "the decision column before accepting it.")

    picked = ([("A_disagreement", r) for r in disagree]
              + [("B_fresh_review", r) for r in fresh]
              + [("C_control", r) for r in control])
    order = rng.permutation(len(picked))
    picked = [picked[i] for i in order]
    print(f"\n{len(picked)} clips, shuffled so arm A does not arrive as a block: "
          f"{dict(Counter(k for k, _ in picked))}")

    ctx = {}
    for p in a.blind_csv:
        recs = ([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
                if p.endswith(".jsonl") else read_csv(p))
        for r in recs:
            ctx.setdefault(r["event_id"], r)
    miss = sum(1 for _, r in picked if r["event_id"] not in ctx)
    if miss:
        print(f"  !! {miss}/{len(picked)} have no segment-label context and "
              f"would render blank -- pass the file that covers them")

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "double_manifest.jsonl"), "w",
              encoding="utf-8") as f:
        for _, r in picked:
            f.write(json.dumps({"event_id": r["event_id"],
                                "recording_id": r["recording_id"],
                                "t": float(r["event_id"].rsplit("_t", 1)[1])},
                               ensure_ascii=False) + "\n")

    cols = ["event_id", "prev_segment_label", "next_segment_label",
            "containing_segment_label"]
    with open(os.path.join(a.out_dir, "double_blind_context.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for _, r in picked:
            c = ctx.get(r["event_id"], {})
            w.writerow({k: c.get(k, "") for k in cols})

    # The full seven-way subtype, and a supervision column left blank: the
    # annotator says what the video shows, not how it should be trained on.
    # Conflating the two is what produced the bucket this taxonomy replaced.
    with open(os.path.join(a.out_dir, "double_annotation_sheet.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "subtype(" + "|".join(SUBTYPES) + ")",
                    "confidence(1_guess|2_lean|3_sure)", "why_this_subtype",
                    "notes"])
        for _, r in picked:
            w.writerow([r["event_id"], "", "", "", ""])

    with open(os.path.join(a.out_dir, "double_annotation_key.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "recording_id", "arm", "existing_subtype",
                    "existing_y", "decision"])
        for arm, r in picked:
            w.writerow([r["event_id"], r["recording_id"], arm,
                        r.get("subtype", ""), r.get("y", ""),
                        r.get("decision", "")])

    print(f"\nwrote double_manifest.jsonl / double_blind_context.csv / "
          f"double_annotation_sheet.csv / double_annotation_key.csv in "
          f"{a.out_dir}")
    print("\nARM B IS THE NUMBER THAT DECIDES. Arm A is enriched for "
          "disagreement and its agreement rate is meaningless; report the "
          "three arms separately and never pool them.")
    print("The second annotator must work from the SAME subtype definitions "
          "the first labelling used (docs/pair_taxonomy_definitions.md, "
          "generated from pair_taxonomy.py) -- handing them a fresh definition "
          "would measure whether two people agree given a new rulebook, which "
          "is not what failed.")


if __name__ == "__main__":
    main()
