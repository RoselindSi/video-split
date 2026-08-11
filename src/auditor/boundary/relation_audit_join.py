"""Does the stored POINT class mean what the target says it means?

The relation audit asked one question on 54 events whose instance_relation was
UNKNOWN, 33 of them carrying a stored POINT. Joining the answers back gives the
number this whole line of work was for: of the events the model is trained to
call boundaries, how many are a new action, how many are the same action
starting again, and how many are one continuous instance.

A stored POINT that turns out to be `same_instance` is not a hard example. It
is a wrong target, and no representation reaches a target that contains its own
negation.

THE POLICY IS APPLIED, NOT REINVENTED. configs/auditor/instance_relation_policy
_v1.yaml already says what each relation earns: new_action is eligible,
same_instance is not, same_action_new_instance is conditional on the transition
showing a gap or a point, and both one-sided cases are undecided and route to
REVIEW. This file reads that file. Deciding eligibility here would put the same
rule in two places and they would drift.

WHERE THE SHAPE IS UNKNOWN, A CONDITIONAL RELATION STAYS UNRESOLVED rather
than defaulting either way. These events were sampled BECAUSE their relation
was unknown, and many carry no shape either; resolving them by assumption would
manufacture the contamination estimate instead of measuring it.

54 EVENTS AND ONE ANNOTATOR. Every proportion below has a wide interval and one
person's reading behind it. The second sheet is what turns this into an
agreement measurement; until then these are counts, not rates.

Usage:
    python -m src.auditor.boundary.relation_audit_join \
        --sheet data/gold/relation_audit_annotator1_filled.csv \
        --key .../relation_audit/relation_audit_key.csv \
        --migrated data/gold/pair_schema_v2_migrated.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict

POINT, NONE = "POINT_TRANSITION", "NO_TRANSITION"


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def col(row, prefix):
    return (next((v for k, v in row.items()
                  if k and k.startswith(prefix)), "") or "").strip()


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    import math
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--key", help="relation_audit_key.csv, for stored "
                                  "morphology and shape")
    ap.add_argument("--migrated", help="pair_schema_v2_migrated.csv, used for "
                                       "transition_shape when --key is absent")
    ap.add_argument("--labels", help="boundary_v1_labels json, used for the "
                                     "stored morphology when --key is absent")
    ap.add_argument("--policy",
                    default="configs/auditor/instance_relation_policy_v1.yaml")
    ap.add_argument("--out")
    a = ap.parse_args()

    import yaml
    pol = yaml.safe_load(open(a.policy, encoding="utf-8"))["instance_relation"]

    calls = {}
    for r in read_csv(a.sheet):
        e, c = r.get("event_id"), col(r, "your_call")
        if e and c:
            calls[e] = {"relation": c, "confidence": col(r, "confidence"),
                        "why": col(r, "why_one_line")}
    key = {r["event_id"]: r for r in read_csv(a.key)} if a.key and os.path.exists(a.key) else {}
    mig = ({r["event_id"]: r for r in read_csv(a.migrated)}
           if a.migrated and os.path.exists(a.migrated) else {})
    lab = {}
    if a.labels and os.path.exists(a.labels):
        import json as _j
        lab = {e["event_id"]: e
               for e in _j.load(open(a.labels, encoding="utf-8"))["events"]}
    print(f"{len(calls)} filled rows; key for {len(key)}; migrated table for "
          f"{len(mig)}; labels for {len(lab)}")
    if not key and not lab:
        print("  !! no --key and no --labels, so the stored morphology is "
              "unavailable and the headline table cannot be produced.")
    unknown_rel = [c["relation"] for c in calls.values()
                   if c["relation"] not in pol]
    if unknown_rel:
        print(f"  !! relations not in the policy file: "
              f"{dict(Counter(unknown_rel))}. Add them to the schema and the "
              f"policy before reading anything below.")

    def stored(e):
        if e in key and key[e].get("stored_morphology"):
            return key[e]["stored_morphology"]
        if e in lab:
            return lab[e].get("morphology") or "MASKED"
        return "?"

    def shape(e):
        if e in key and key[e].get("transition_shape"):
            return key[e]["transition_shape"]
        return (mig.get(e, {}).get("transition_shape") or "UNKNOWN")

    def shape_observed(e):
        """Was the shape SEEN by someone, or inherited from the old subtype?

        This matters for the conditional rule. A same_action_new_instance
        qualifies on shape in {gap, point}, and most of them carry `point` --
        but that `point` came from `sharp_visible_transition` in the very
        label being questioned, while the annotator's own prose describes an
        idle gap. The decision is the same either way, since both qualify, and
        reading the cell as an observation would still be wrong."""
        src_ = (mig.get(e, {}) or {}).get("shape_source", "")
        return bool(src_) and not src_.startswith("legacy:")

    def src(e):
        if "_batch3_gt_boundary_" in e:
            return "batch3 gt_boundary"
        if "_batch3_raw_change_peak_" in e:
            return "batch3 raw_change_peak"
        return "dev"

    rels = sorted({c["relation"] for c in calls.values()})

    print(f"\n{'=' * 88}\nRELATION x STORED MORPHOLOGY\n{'=' * 88}")
    classes = sorted({stored(e) for e in calls})
    print(f"  {'relation':<26}" + "".join(f"{c[:16]:>18}" for c in classes)
          + f"{'total':>8}")
    for r in rels:
        g = [e for e in calls if calls[e]["relation"] == r]
        cc = Counter(stored(e) for e in g)
        print(f"  {r:<26}" + "".join(f"{cc.get(c, 0):>18}" for c in classes)
              + f"{len(g):>8}")
    print(f"  {'total':<26}"
          + "".join(f"{sum(1 for e in calls if stored(e) == c):>18}"
                    for c in classes) + f"{len(calls):>8}")

    pts = [e for e in calls if stored(e) == POINT]
    if pts:
        print(f"\n{'=' * 88}\nTHE HEADLINE: what the stored POINT class is "
              f"made of\n{'=' * 88}")
        cc = Counter(calls[e]["relation"] for e in pts)
        print(f"  {len(pts)} events carry a stored POINT_TRANSITION")
        for r, n in cc.most_common():
            lo, hi = wilson(n, len(pts))
            print(f"    {r:<28} {n:>3}  {n / len(pts):>6.1%}  "
                  f"[{lo:.1%}, {hi:.1%}]")
        wrong = sum(cc.get(k, 0) for k in ("same_instance",))
        if wrong:
            lo, hi = wilson(wrong, len(pts))
            print(f"\n  {wrong} of {len(pts)} stored POINT events are "
                  f"`same_instance` -- one continuous instance labelled a "
                  f"boundary.")
            print(f"  {wrong / len(pts):.1%} [{lo:.1%}, {hi:.1%}]. That is not "
                  f"a hard example, it is a wrong target, and it puts a floor "
                  f"under\n  what any model trained on this class can reach.")
        else:
            print(f"\n  No stored POINT came back `same_instance` on this "
                  f"sample. The class survives its own audit here, which\n  is "
                  f"a real result and rests on {len(pts)} events and one "
                  f"annotator.")

    print(f"\n{'=' * 88}\nWHAT THE POLICY WOULD DECIDE\n{'=' * 88}")
    dec = {}
    for e, c in calls.items():
        p = pol.get(c["relation"], {})
        el = p.get("boundary_eligible")
        # a relation carrying an explicit route is routed, whatever its
        # eligibility says. `cannot_determine` is boundary_eligible: false AND
        # route: REVIEW, and the policy's own doc says why -- "not a negative;
        # the evidence is missing, not the boundary". Reading only the
        # eligibility flag counted those 8 events as decided negatives.
        if p.get("route") == "REVIEW":
            d = ("REVIEW (evidence missing)" if el is False
                 else "REVIEW (policy undefined)")
        elif el is True:
            d = "boundary"
        elif el is False:
            d = "not a boundary"
        elif el == "conditional":
            sh = shape(e)
            need = ((p.get("requires") or {}).get("transition_shape")
                    or ["gap", "point"])
            d = ("boundary" if sh in need else
                 "UNRESOLVED (shape unknown)" if sh == "UNKNOWN" else
                 "not a boundary")
        else:
            d = "REVIEW (policy undefined)"
        dec[e] = d
    print(f"  {'decision':<32} {'n':>4}   against the stored label")
    for d, n in Counter(dec.values()).most_common():
        g = [e for e in calls if dec[e] == d]
        print(f"  {d:<32} {n:>4}   {dict(Counter(stored(e) for e in g))}")

    flip = [e for e in calls if stored(e) == POINT
            and dec[e] in ("not a boundary",)]
    if flip:
        print(f"\n  {len(flip)} stored POINT events the policy would NOT call "
              f"a boundary:")
        for e in flip:
            print(f"    {e[-46:]:<47} {calls[e]['relation']:<26} "
                  f"conf {calls[e]['confidence']}")
            print(f"      {calls[e]['why'][:100]}")
    unres = [e for e in calls if dec[e].startswith("UNRESOLVED")]
    if unres:
        print(f"\n  {len(unres)} conditional relations cannot be resolved "
              f"because the transition shape is unknown. These events were\n"
              f"  sampled for having an unknown relation and many have no "
              f"shape either; assuming one would manufacture the number.")

    print(f"\n{'=' * 88}\nRELATION x TRANSITION SHAPE, and x SOURCE"
          f"\n{'=' * 88}")
    shapes = sorted({shape(e) for e in calls})
    print(f"  cells read `n (o)` where o is how many of them were OBSERVED by "
          f"a pass rather than inherited from the old subtype")
    print(f"  {'relation':<26}" + "".join(f"{s[:14]:>16}" for s in shapes))
    for r in rels:
        g = [e for e in calls if calls[e]["relation"] == r]
        cc = Counter(shape(e) for e in g)
        ob = Counter(shape(e) for e in g if shape_observed(e))
        print(f"  {r:<26}"
              + "".join(f"{f'{cc.get(s, 0)} ({ob.get(s, 0)})':>16}"
                        for s in shapes))
    n_leg = sum(1 for e in calls if shape(e) != "UNKNOWN"
                and not shape_observed(e))
    if n_leg:
        print(f"\n  {n_leg} of the shapes above were INHERITED from the "
              f"seven-way subtype, not observed. That includes most of the\n"
              f"  same_action_new_instance rows, whose `point` came from "
              f"`sharp_visible_transition` in the label under question while\n"
              f"  the annotator's own prose describes an idle gap. Both values "
              f"satisfy the conditional rule so no decision moves, but the\n"
              f"  cell is not evidence that anyone saw a compact switch.")
    print()
    srcs = sorted({src(e) for e in calls})
    print(f"  {'relation':<26}" + "".join(f"{s[:20]:>24}" for s in srcs))
    for r in rels:
        g = [e for e in calls if calls[e]["relation"] == r]
        cc = Counter(src(e) for e in g)
        print(f"  {r:<26}" + "".join(f"{cc.get(s, 0):>24}" for s in srcs))

    print(f"\n  54 events and one annotator. Every proportion above has a wide "
          f"interval behind it, and the second sheet is what\n  turns these "
          f"counts into an agreement measurement.")

    if a.out:
        import json
        json.dump({"n": len(calls),
                   "stored_point": {"n": len(pts),
                                    "by_relation": dict(Counter(
                                        calls[e]["relation"] for e in pts))},
                   "policy_decisions": dict(Counter(dec.values())),
                   "events": [{"event_id": e, **calls[e],
                               "stored": stored(e), "shape": shape(e),
                               "policy": dec[e]} for e in sorted(calls)]},
                  open(a.out, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
