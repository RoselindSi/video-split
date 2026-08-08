"""Safe to DELETE is not the complement of safe to ADMIT.

`admission_safe` found 265 unsafe candidates, and reusing that set as the
positive class for AUTO_REJECT would be a category error. It contains sharp
transitions sitting 1-3 s from their corrected boundary (a human retimes
those), gradual phases (soft supervision, not deletion), annotation-convention
splits (an annotation-side decision), and offscreen or ambiguous moments
(nothing is known about them at all). Deleting any of those loses a real
boundary or a real supervision signal.

So the product semantics are written down first and the gold is built from
them, not from what is convenient to compute:

    AUTO_REJECT = this candidate can be deleted outright. No human needs to
    retime it, it is not worth keeping as a soft transition, and no valid
    boundary is lost by removing it.

HARD_REJECT_SAFE therefore requires ALL of:

    subtype           same_action_internal_motion or camera_or_viewpoint_shift
                      -- the only two classes whose definition is "nothing
                      happened here that a boundary should mark"
    temporal_truth              spurious
    candidate_boundary_validity invalid
    no_valid_boundary           True
    boundary_time_unresolved    False
    gt_boundary_relation        not multiple_valid  (a human decides those)
    salvage                     no corrected boundary time within SALVAGE_S of
                                the candidate, and no corrected times recorded
                                at all -- if a boundary can be recovered by
                                moving the timestamp, the candidate is
                                salvageable and deleting it destroys work

EVERYTHING ELSE IS A NEGATIVE, including the undecidable. There is no
"excluded" category here, deliberately. In the admission experiment an event
with no timing record could honestly be set aside; here, a teacher that
deletes an offscreen candidate has destroyed something whether or not our gold
knows what it was, so it must be counted as a false reject. An exclusion would
be a hole in the number exactly where the product risk lives.

THE FUNNEL IS THE POINT OF THIS FILE. Before any API call it answers whether
the branch has a ceiling worth paying for: if only ~18 of 188 audited events
are safe to delete, then a perfect teacher buys under 10% review reduction and
the experiment is not worth running at all. Each exclusion is counted
separately so the ceiling can be read with and without the salvage rule.

Usage:
    python -m src.auditor.reject_safe \
        --pair_labels data/gold/pair_labels_v1.csv \
        --pair_labels data/gold/batch3_pair_labels_v1_relabel_v1.csv \
        --exclude_review /workspace/tr1/results/hal/c3/teacher_observe_only.json \
        --decisions .../policy_decisions_v4.primary_transportability_frontier.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

DELETABLE_SUBTYPES = ("same_action_internal_motion", "camera_or_viewpoint_shift")
SALVAGE_S = 2.0


def load_gold(paths):
    g = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    g[r["event_id"]] = r
    return g


def cand_time(eid):
    try:
        return float(eid.rsplit("_t", 1)[1])
    except (IndexError, ValueError):
        return None


def reject_safe(eid, subtype, g, salvage_s=SALVAGE_S):
    """(bool, blocking_reason). The reason is the FIRST rule that excluded it,
    in a fixed order, so the funnel counts partition the population instead of
    double-counting events that fail several rules."""
    if subtype is None:
        return False, "no subtype label"
    if subtype not in DELETABLE_SUBTYPES:
        return False, f"subtype {subtype}"
    if g is None:
        return False, "no audit record"
    if g.get("temporal_truth") != "spurious":
        return False, f"temporal_truth {g.get('temporal_truth')}"
    if g.get("candidate_boundary_validity") != "invalid":
        return False, f"validity {g.get('candidate_boundary_validity')}"
    if not g.get("no_valid_boundary"):
        return False, "a valid boundary was recorded here"
    if g.get("boundary_time_unresolved"):
        return False, "boundary_time_unresolved"
    if g.get("gt_boundary_relation") == "multiple_valid":
        return False, "multiple_valid"
    times = g.get("corrected_boundary_times_json") or []
    if times:
        t = cand_time(eid)
        near = (t is not None
                and min(abs(float(x) - t) for x in times) <= salvage_s)
        return False, ("salvageable: a corrected boundary is within "
                       f"{salvage_s}s" if near else
                       "a corrected boundary was recorded elsewhere")
    return True, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", action="append",
                    default=["data/gold/audit_188_gold_v2.jsonl"])
    ap.add_argument("--pair_labels", action="append", required=True)
    ap.add_argument("--exclude_review", action="append", default=[],
                    help="teacher result files whose events are development "
                         "data and must not be reused as a held-out test")
    ap.add_argument("--decisions", help="frozen student policy decisions, to "
                                        "see whether a reject proposal set "
                                        "exists at all")
    ap.add_argument("--salvage_s", type=float, default=SALVAGE_S)
    ap.add_argument("--out")
    a = ap.parse_args()

    from src.boundary.pair_taxonomy import load_pair_labels
    labels = {}
    for p in a.pair_labels:
        for e, v in load_pair_labels(p).items():
            labels[e] = v["temporal_pair_subtype"]
    gold = load_gold(a.gold)
    dev = set()
    for p in a.exclude_review:
        if os.path.exists(p):
            for r in json.load(open(p, encoding="utf-8")).get("results", []):
                dev.add(r["event_id"])
    print(f"{len(gold)} audited events, {len(labels)} subtype labels, "
          f"{len(dev)} teacher-development events to exclude")

    rows = []
    for eid, sub in labels.items():
        ok, why = reject_safe(eid, sub, gold.get(eid), a.salvage_s)
        rows.append({"event_id": eid, "subtype": sub, "reject_safe": ok,
                     "reason": why, "dev": eid in dev,
                     "audited": eid in gold})

    aud = [r for r in rows if r["audited"]]
    pos = [r for r in aud if r["reject_safe"]]
    print(f"\n{'=' * 78}\nHARD_REJECT_SAFE OVER THE {len(aud)} AUDITED EVENTS"
          f"\n{'=' * 78}")
    print(f"  safe to delete   {len(pos)}")
    print(f"  everything else  {len(aud) - len(pos)}   "
          f"(negatives, including the undecidable -- deleting one of those is "
          f"a false reject)")

    print(f"\n  where the other {len(aud) - len(pos)} are lost, first blocking "
          f"rule only:")
    for why, n in Counter(r["reason"] for r in aud
                          if not r["reject_safe"]).most_common():
        print(f"    {n:>4}  {why}")

    print(f"\n  the two deletable subtypes, funnel:")
    for s in DELETABLE_SUBTYPES:
        g = [r for r in aud if r["subtype"] == s]
        k = sum(1 for r in g if r["reject_safe"])
        print(f"    {s:<32} {k:>3} safe of {len(g):>3} audited")
        for why, n in Counter(r["reason"] for r in g
                              if not r["reject_safe"]).most_common(6):
            print(f"        {n:>3}  {why}")

    held = [r for r in aud if not r["dev"]]
    hpos = [r for r in held if r["reject_safe"]]
    print(f"\n{'=' * 78}\nAFTER REMOVING THE TEACHER-DEVELOPMENT EVENTS"
          f"\n{'=' * 78}")
    print(f"  {len(held)} held-out audited events, {len(hpos)} reject-safe, "
          f"{len(held) - len(hpos)} negatives")
    lost = [r for r in aud if r["dev"] and r["reject_safe"]]
    print(f"  reject-safe events burned as development data: {len(lost)}")

    # ------------------------------------------------------------- ceiling
    print(f"\n{'=' * 78}\nPRODUCT CEILING, BEFORE ANY API CALL\n{'=' * 78}")
    print(f"  A teacher that certified EVERY reject-safe event and never "
          f"touched anything else would\n  remove {len(hpos)} of {len(held)} "
          f"held-out candidates = {len(hpos) / max(1, len(held)):.1%} review "
          f"reduction, at precision 1.000.")
    print(f"  That is the CEILING, not a forecast: it assumes perfect recall "
          f"and perfect precision at once.\n  The pre-registered floor is n "
          f">= 20 with precision >= 0.95, so the branch is worth paying for "
          f"only if\n  {len(hpos)} is comfortably above 20 -- a ceiling near "
          f"the floor leaves no room for the teacher to be imperfect.")
    if len(hpos) < 20:
        print(f"\n  !! {len(hpos)} < 20. Even a flawless teacher cannot reach "
              f"the pre-registered minimum on this set.\n     Nothing is "
              f"gained by calling the API before this number changes.")

    if a.decisions and os.path.exists(a.decisions):
        dec, sc = {}, {}
        with open(a.decisions, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                dec[r["event_id"]] = r.get("decision")
                sc[r["event_id"]] = r.get("score") or r.get("fused_score")
        print(f"\n{'=' * 78}\nTHE STUDENT'S OWN REJECT PROPOSALS\n{'=' * 78}")
        print(f"  decisions in {os.path.basename(a.decisions)}: "
              f"{dict(Counter(dec.values()))}")
        rej = [e for e, d in dec.items() if d == "AUTO_REJECT"]
        if not rej:
            print("  !! the frozen policy proposes NO rejects -- reject_below "
                  "is -1.0, which was added as an action-space option and\n"
                  "     never turned on. There is no proposal tail to certify, "
                  "so the cascade has no first stage. Either a reject\n"
                  "     threshold is chosen (and that is a new selection, with "
                  "all the nested-vs-pooled care that implies), or the\n"
                  "     teacher runs on the full REVIEW band and the cost "
                  "saving of the cascade does not apply.")
        else:
            byr = defaultdict(lambda: [0, 0])
            for e in rej:
                r = next((x for x in rows if x["event_id"] == e), None)
                if r is None:
                    continue
                byr[r["subtype"]][0] += 1
                byr[r["subtype"]][1] += int(r["reject_safe"])
            n_ok = sum(v[1] for v in byr.values())
            print(f"  {len(rej)} proposed rejects, {n_ok} of them reject-safe "
                  f"= precision {n_ok / max(1, len(rej)):.3f} with no teacher")
            print(f"  {'subtype':<34} {'proposed':>9} {'reject-safe':>12}")
            for s, v in sorted(byr.items(), key=lambda x: -x[1][0]):
                print(f"  {s:<34} {v[0]:>9} {v[1]:>12}")
            print("  This is the baseline the teacher has to beat. If it is "
                  "already at the deployment bar the teacher adds cost only.")

    if a.out:
        json.dump({"salvage_s": a.salvage_s,
                   "deletable_subtypes": list(DELETABLE_SUBTYPES),
                   "n_audited": len(aud), "n_reject_safe": len(pos),
                   "n_held_out": len(held), "n_held_out_reject_safe": len(hpos),
                   "events": rows},
                  open(a.out, "w", encoding="utf-8"), indent=2, default=str)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
