"""Enrichment batch for `claim_support = no`, sized against the measured yield.

THE TARGET MAY NOT BE REACHABLE FROM THIS CORPUS, and the 48 already audited
say so before any new annotation is spent. Every one of them carries both its
old six-way status and its new claim_support, nine per status by construction,
so the conversion is measured rather than assumed:

    old status                    n    -> claim_support = no
    incorrect                     9       4   (0.44)
    uncertain                     3       1   (0.33)
    correct_but_oversegmented     9       1   (0.11)
    correct                       9       0
    correct_but_coarse            9       0
    partially_correct             9       0

`partially_correct` and `correct_but_coarse` produced ZERO negatives between
them across 18 events. That is not noise, it is the freeze rule working: those
two statuses were mostly granularity and structure complaints, and the new
schema routes them to `granularity` and `segment_structure` while
`claim_support` stays `yes`. Sampling them for negatives would spend
annotation on events whose answer is already known.

WHAT THAT IMPLIES FOR THE TARGET. 25 unaudited `incorrect` events remain. At
0.44 they yield about 11 more negatives, taking the total from 6 to roughly
17. Auditing every remaining event in the entire 188-event corpus -- all 140
of them -- projects to about 19. THIRTY CLEAN NEGATIVES IS NOT IN THIS POOL,
and the reason is the same structural one the alignment arm hit: the corpus
was sampled on BOUNDARY error categories, so it is enriched for boundary
problems and only incidentally contains semantic ones.

The yields come from nine events per status and their intervals are wide --
4/9 is compatible with anything from about 0.14 to 0.79 -- so the projection
is a planning estimate, not a measurement. Even its optimistic end does not
comfortably reach 30.

SO THIS SAMPLER DOES THE REACHABLE PART AND SAYS WHAT IT CANNOT DO. It orders
the pool by measured NO-yield, takes the whole `incorrect` remainder first,
and prints the projected total against the target so the shortfall is a number
rather than a surprise after the audit. Reaching 30 needs a different sampling
frame -- segments drawn for semantic error rather than boundary error -- and
that is a separate decision.

NO MODEL SCORE ENTERS. The pool is ordered by old status and nothing else, and
within a status the order is a seeded shuffle. Sampling on the current model's
confidence would produce a benchmark that is easy exactly where the model is
already right.

Usage:
    python -m src.auditor.semantic.enrichment_sample \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --context data/gold/audit_188_context.jsonl \
        --done data/gold/semantic_ontology_gold_48.json \
        --n 60 --out data/gold/semantic_enrichment.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict

# measured on the 48, nine per status; see the docstring for the caveat
YIELD = {"incorrect": 4 / 9, "uncertain": 1 / 3,
         "correct_but_oversegmented": 1 / 9, "correct": 0.0,
         "correct_but_coarse": 0.0, "partially_correct": 0.0}

FIELDS = [
    ("claim_support", "yes / partial / no / uncertain -- about THIS segment "
                      "as the model saw it. A wrong upstream cut is recorded "
                      "in upstream_timing_issue, it does not excuse a wrong "
                      "label."),
    ("granularity", "adequate / too_coarse / too_fine / mixed_inconsistent / "
                    "not_applicable / uncertain"),
    ("major_action_missing", "yes / no / uncertain"),
    ("action_presence", "valid_action / no_valid_action / "
                        "mixed_action_and_no_action / onset_only / "
                        "terminal_only / uncertain"),
    ("segment_structure", "adequate / oversegmented / undersegmented / mixed "
                          "/ spurious_no_action / uncertain"),
    ("upstream_timing_issue", "yes / no / uncertain"),
    ("actual_action_summary", "what the video actually shows"),
    ("existing_label_summary", "what the stored labels claim"),
    ("human_boundary_note", "boundary times, if any are worth recording"),
    ("semantic_audit_note", "anything else"),
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--done", default="data/gold/semantic_ontology_gold_48.json")
    ap.add_argument("--n", type=int, default=0,
                    help="batch size. 0 means every event in a status with a "
                         "NON-ZERO measured no-yield, which is the whole "
                         "informative pool; a larger value fills the rest "
                         "from statuses measured at zero and says so")
    ap.add_argument("--target_no", type=int, default=30)
    ap.add_argument("--target_yes", type=int, default=50)
    ap.add_argument("--status_field", default="legacy_semantic_label_status")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/gold/semantic_enrichment.csv")
    a = ap.parse_args()

    gold = {}
    for line in open(a.gold, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            gold[r["event_id"]] = r
    ctx = {}
    for line in open(a.context, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            ctx[r["event_id"]] = r
    done_blob = json.load(open(a.done, encoding="utf-8"))["events"]
    done = {e["event_id"] for e in done_blob if e.get("event_id")}
    have = Counter(e["claim_support"] for e in done_blob)
    print(f"{len(done)} already audited: "
          f"{have['yes']} yes / {have['no']} no / {have['partial']} partial "
          f"/ {have['uncertain']} uncertain")

    pool = defaultdict(list)
    for eid, g in gold.items():
        if eid in done:
            continue
        pool[g[a.status_field]].append(eid)
    order = sorted(pool, key=lambda s: -YIELD.get(s, 0.0))

    print(f"\nREMAINING POOL, ordered by MEASURED no-yield:")
    print(f"  {'old status':<28}{'left':>6}{'yield':>8}{'proj. NO':>10}")
    proj_all = 0.0
    for s in order:
        y = YIELD.get(s, 0.0)
        proj_all += len(pool[s]) * y
        print(f"  {s:<28}{len(pool[s]):>6}{y:>8.2f}{len(pool[s]) * y:>10.1f}")
    print(f"  {'':<28}{sum(len(v) for v in pool.values()):>6}{'':>8}"
          f"{proj_all:>10.1f}")
    print(f"\n  auditing the ENTIRE remaining corpus projects "
          f"{have['no'] + proj_all:.0f} negatives against a\n  target of "
          f"{a.target_no}. The pool does not contain the target, and no "
          f"sampling strategy\n  over it will. Reaching {a.target_no} needs a "
          f"frame drawn for SEMANTIC error rather\n  than boundary error -- "
          f"a separate decision, not a bigger batch.")

    informative = sum(len(pool[s]) for s in order if YIELD.get(s, 0.0) > 0)
    n = a.n or informative
    if n > informative:
        print(f"\n  !! --n {n} exceeds the {informative} events in statuses "
              f"with a non-zero measured\n     yield. The extra "
              f"{n - informative} come from statuses that produced ZERO "
              f"negatives across 18\n     audited events, so they cost "
              f"annotation and cannot move the binding constraint.\n     "
              f"They are still worth auditing if the YES side or the "
              f"granularity head needs them.")

    rng = random.Random(a.seed)
    picked, proj_no, proj_yes = [], 0.0, 0.0
    for s in order:
        ids = sorted(pool[s])
        rng.shuffle(ids)
        for eid in ids:
            if len(picked) >= n:
                break
            picked.append((eid, s))
            proj_no += YIELD.get(s, 0.0)
            proj_yes += 1 - YIELD.get(s, 0.0)
        if len(picked) >= n:
            break

    print(f"\nBATCH of {len(picked)}, taken in yield order:")
    for s, n in Counter(x[1] for x in picked).most_common():
        print(f"  {s:<28}{n:>4}")
    print(f"  projected new negatives {proj_no:.1f} -> total "
          f"{have['no'] + proj_no:.0f} / {a.target_no}")
    print(f"  projected new non-negatives {proj_yes:.1f}; yes-total will "
          f"clear {a.target_yes} comfortably")
    print(f"  the yes side was never the constraint and this batch is not "
          f"sized for it.")

    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        shown = ["audit_key", "event_id", "recording_id", "candidate_time",
                 "previous_segment_label", "containing_segment_label",
                 "next_segment_label"]
        w.writerow(shown + [k for k, _ in FIELDS])
        w.writerow([""] * len(shown) + [h for _, h in FIELDS])
        for eid, _s in picked:
            g, c = gold[eid], ctx.get(eid, {})
            rid = g["recording_id"]
            t = eid.rsplit("_t", 1)[-1]
            w.writerow([
                f"{int(rid.split('_')[-1])}/t{t}", eid, rid, t,
                c.get("prev_segment_label")
                or c.get("nearest_previous_segment_label") or "",
                c.get("containing_segment_label") or "",
                c.get("next_segment_label")
                or c.get("nearest_next_segment_label") or ""]
                + [""] * len(FIELDS))
    print(f"\nwrote {a.out}")
    print(f"  the old status is NOT in the sheet -- it is the sampling key "
          f"and showing it would\n  anchor the answer it was used to predict. "
          f"Render windows with\n  src.auditor.semantic.naming_targets, which "
          f"takes this file's event_ids.")


if __name__ == "__main__":
    main()
