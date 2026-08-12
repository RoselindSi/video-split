"""A blind audit of what the six semantic statuses are actually recording.

The six statuses -- correct, partially_correct, incorrect, correct_but_coarse,
correct_but_oversegmented, uncertain -- mix at least four axes: whether a
stated claim is supported, whether something is missing, granularity, and
segment structure. `correct_but_oversegmented` is not a semantic judgement at
all; it is a boundary judgement wearing a semantic name. This sheet does not
assume the split is right. It collects free prose per event and lets the
mapping to base fields be tested rather than declared.

THE SHEET DOES NOT SHOW THE OLD STATUS. It also hides `corrected_primary_verb`
/ `corrected_object` / `corrected_secondary_verbs`, the three
`label_completeness` / `granularity` / `support` columns, and the auditor's
`notes` prose -- every one of those states or nearly states the conclusion.
The stratification key is written to a separate file so the join survives
while the annotator sees none of it.

WHAT IT DOES SHOW is the segment context, unaltered, because deciding WHICH
label is under judgement is part of what is being tested:

    105 of 188 events have NO containing segment label. The candidate sits on
    a boundary, so the semantic status refers to the previous segment, or the
    next one, or the pair -- and the schema never says which. 28 of those have
    only a next label and 20 only a previous one.

    41 of 188 have previous and next labels that are more than half the same
    words, several of them character-identical ('Fold and unfold a paper
    sheet' on both sides). For those, `incorrect` cannot mean "this label is
    wrong about the video" -- the two labels are the same label. It means the
    SEGMENTATION is wrong. 13 of the 34 `incorrect` events are in this set.

So `q2_label_referent` is a question on the sheet, not a field the loader
fills in. If annotators cannot name the referent consistently, the base fields
below are undefined for those events no matter how carefully they are worded,
because `primary_verb_supported` needs a primary verb to be about.

THE FIELDS THIS IS TESTING, transcribed from the plan so the prose can be
scored against them afterwards rather than reinterpreted:

    primary_verb_supported      yes / no / uncertain
    secondary_claims_supported  all / some / none / not_applicable / uncertain
    object_supported            yes / no / uncertain
    major_action_missing        yes / no / uncertain
    granularity                 adequate / too_coarse / uncertain
    segment_structure           single_semantic_instance /
                                multiple_semantic_phases /
                                structural_oversegmentation_suspected /
                                uncertain

The annotator fills PROSE, not these values. Mapping prose to fields is a
separate pass, and it is the pass that answers the question: if two readers
map the same prose to different fields, the schema is still too coarse; if the
prose itself cannot answer a field, the gold is missing evidence rather than
missing structure. Those are different diagnoses with different fixes.

SAMPLING is stratified on the old status and, inside each status, on whether a
containing label exists -- the referent ambiguity has to appear in every
stratum or it will look like a property of one status. `uncertain` (n=3) is
taken whole.

Usage:
    python -m src.auditor.semantic.ontology_audit_sheet \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --context data/gold/audit_188_context.jsonl \
        --n_per_status 9 \
        --out data/gold/semantic_ontology_audit.csv \
        --key_out data/gold/semantic_ontology_audit_key.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict

TIME = re.compile(r"_t(\d+(?:\.\d+)?)$")

# Everything the annotator must not see, with the reason, so that a later
# change to the sheet has to argue with this list rather than quietly widen it.
BLINDED = {
    "legacy_semantic_label_status": "the answer",
    "label_completeness": "the answer in another column",
    "label_granularity": "the answer in another column",
    "label_support": "the answer in another column",
    "semantic_relation": "derived from the answer",
    "object_relation": "derived from the answer",
    "corrected_primary_verb": "states what the auditor thought was right",
    "corrected_secondary_verbs": "states what the auditor thought was right",
    "corrected_object": "states what the auditor thought was right",
    "semantic_correction_action": "the decision that followed the answer",
    "notes": "prose that argues for the answer",
    "naming_contrastive_role": "downstream of the answer",
}

QUESTIONS = [
    ("q1_video_shows",
     "What does the video actually show in this window? Plain description "
     "of the action, the object, and anything that changes."),
    ("q2_label_referent",
     "WHICH label are you judging? containing / previous / next / both / "
     "cannot tell. If the candidate sits between two segments, say which one "
     "the question is even about."),
    ("q3_clearly_right",
     "Which parts of that label are clearly supported by the video?"),
    ("q4_clearly_wrong",
     "Which parts are clearly NOT supported? Say `none` if every stated part "
     "holds."),
    ("q5_missing",
     "Is a significant action visible that the label does not mention?"),
    ("q6_too_coarse",
     "Is the label accurate but lumping together things that deserve "
     "separate description?"),
    ("q7_phases",
     "Does the segment contain more than one phase, and if so, is splitting "
     "it a real semantic difference or the same action repeated?"),
    ("q8_notes", "Anything else, including why a question was unanswerable."),
]


def cand_time(eid):
    m = TIME.search(eid)
    return float(m.group(1)) if m else None


def referent_pattern(c):
    has = lambda k: bool(c.get(k))
    p = has("prev_segment_label") or has("nearest_previous_segment_label")
    n = has("next_segment_label") or has("nearest_next_segment_label")
    return ("CONTAIN" if has("containing_segment_label") else "-",
            "PREV" if p else "-", "NEXT" if n else "-")


def near_duplicate(c):
    """Are the two neighbour labels effectively the same sentence?

    Word-level Jaccard, deliberately crude. It is a flag for the sampler and
    for the report, not a measurement -- '8th tissue fold' against '9th tissue
    fold' should trip it, and it does, but so would a genuine pair that
    happens to share vocabulary. It is checked by eye in the audit."""
    p = c.get("prev_segment_label") or c.get("nearest_previous_segment_label")
    n = c.get("next_segment_label") or c.get("nearest_next_segment_label")
    if not p or not n:
        return False
    tok = lambda s: set(re.sub(r"[^a-z ]", "", s.lower()).split())
    a, b = tok(p), tok(n)
    return bool(a and b) and len(a & b) / len(a | b) >= 0.5


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--n_per_status", type=int, default=9)
    ap.add_argument("--status_field", default="legacy_semantic_label_status")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key_out", required=True,
                    help="stratification key, kept OUT of the sheet")
    a = ap.parse_args()

    gold = [json.loads(l) for l in open(a.gold, encoding="utf-8") if l.strip()]
    ctx = {}
    for l in open(a.context, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            ctx[r["event_id"]] = r

    missing_ctx = [g["event_id"] for g in gold if g["event_id"] not in ctx]
    if missing_ctx:
        print(f"!! {len(missing_ctx)} gold events have no context row; they "
              f"cannot be audited without segment labels and are excluded")
    gold = [g for g in gold if g["event_id"] in ctx]

    # ------------------------------------------------- the referent situation
    pats = Counter(referent_pattern(ctx[g["event_id"]]) for g in gold)
    print(f"{len(gold)} events. Which segment label exists:")
    for k, v in pats.most_common():
        print(f"  {'/'.join(k):22s} {v:3d}")
    no_contain = [g for g in gold
                  if not ctx[g["event_id"]].get("containing_segment_label")]
    dups = [g for g in gold if near_duplicate(ctx[g["event_id"]])]
    print(f"  no containing label: {len(no_contain)}  "
          f"-- referent of the status is unstated for these")
    print(f"  neighbour labels >=50% identical words: {len(dups)}  "
          f"-- `incorrect` here is a claim about segmentation, not wording")
    print(f"    of which status=incorrect: "
          f"{sum(1 for g in dups if g[a.status_field] == 'incorrect')}")

    # ------------------------------------------------------------- sampling
    rng = random.Random(a.seed)
    by_status = defaultdict(list)
    for g in gold:
        by_status[g[a.status_field]].append(g)

    picked = []
    print(f"\nsampling {a.n_per_status} per status, split on referent:")
    for st in sorted(by_status, key=lambda s: -len(by_status[s])):
        rows = by_status[st]
        want = min(a.n_per_status, len(rows))
        # inside a status, take half from each referent pattern so the
        # ambiguity is present in every stratum rather than concentrated
        with_c = [g for g in rows
                  if ctx[g["event_id"]].get("containing_segment_label")]
        without = [g for g in rows if g not in with_c]
        rng.shuffle(with_c)
        rng.shuffle(without)
        half = want // 2
        take = with_c[:half] + without[:want - half]
        if len(take) < want:  # one side was short; refill from the other
            rest = [g for g in with_c[half:] + without[want - half:]]
            take += rest[:want - len(take)]
        picked += take
        print(f"  {st:28s} {len(take):2d} of {len(rows):3d}  "
              f"(containing {sum(1 for g in take if ctx[g['event_id']].get('containing_segment_label'))}, "
              f"dup-neighbour {sum(1 for g in take if near_duplicate(ctx[g['event_id']]))})")

    rng.shuffle(picked)  # so status is not readable from row order

    # ----------------------------------------------------------------- write
    shown = ["event_id", "recording_id", "candidate_time_s",
             "previous_segment_label", "containing_segment_label",
             "next_segment_label", "clip_path", "contact_sheet_path"]
    cols = shown + [q for q, _ in QUESTIONS]
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow([""] * len(shown) + [h for _, h in QUESTIONS])
        for g in picked:
            c = ctx[g["event_id"]]
            w.writerow([
                g["event_id"], g["recording_id"], cand_time(g["event_id"]),
                c.get("prev_segment_label")
                or c.get("nearest_previous_segment_label") or "",
                c.get("containing_segment_label") or "",
                c.get("next_segment_label")
                or c.get("nearest_next_segment_label") or "",
                g.get("clip_path", ""), g.get("contact_sheet_path", "")]
                + [""] * len(QUESTIONS))

    with open(a.key_out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "old_status", "has_containing_label",
                    "neighbour_labels_near_duplicate", "label_completeness",
                    "label_granularity", "label_support",
                    "corrected_primary_verb", "corrected_object",
                    "corrected_secondary_verbs"])
        for g in picked:
            c = ctx[g["event_id"]]
            w.writerow([g["event_id"], g[a.status_field],
                        bool(c.get("containing_segment_label")),
                        near_duplicate(c), g.get("label_completeness", ""),
                        g.get("label_granularity", ""),
                        g.get("label_support", ""),
                        g.get("corrected_primary_verb", ""),
                        g.get("corrected_object", ""),
                        g.get("corrected_secondary_verbs", "")])

    print(f"\n{len(picked)} rows -> {a.out}")
    print(f"key (NOT for the annotator) -> {a.key_out}")
    print(f"blinded {len(BLINDED)} fields: {', '.join(sorted(BLINDED))}")


if __name__ == "__main__":
    main()
