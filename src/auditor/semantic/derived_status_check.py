"""Is the proposed base-field -> status derivation a function? Enumerated.

The plan is to record six base fields and derive the old statuses from them by
fixed rule, the same shape as the boundary two-field reformulation. That only
works if the rules partition the product space: every combination must reach
exactly one status. Rules written in prose usually do not, and the failure is
invisible until annotation is finished and two rules fire on the same event.

So this enumerates all 3 x 5 x 3 x 3 x 3 x 4 = 1620 combinations, applies each
rule INDEPENDENTLY, and counts how many rules fire. It answers three questions
before a single event is annotated:

    how many combinations fire NO rule      -- events with no derived status
    how many fire MORE THAN ONE             -- events whose status depends on
                                               the order the rules are read in
    which pairs of rules collide, and on what

A precedence order is not a fix for a collision, it is a DECISION about which
axis dominates, and it should be made deliberately. The script prints the
collisions so the decision can be made against them; it does not invent one.

The rules are transcribed from the plan as literally as prose allows. Where
the prose is ambiguous the reading is stated in a comment, because a
convenient reading would make the check pass by construction.

Usage:
    python -m src.auditor.semantic.derived_status_check
    python -m src.auditor.semantic.derived_status_check --precedence \
        incorrect,partially_correct,correct_but_coarse,correct
"""
from __future__ import annotations

import argparse
import itertools
from collections import Counter, defaultdict

FIELDS = {
    "primary_verb_supported": ["yes", "no", "uncertain"],
    "secondary_claims_supported": ["all", "some", "none", "not_applicable",
                                   "uncertain"],
    "object_supported": ["yes", "no", "uncertain"],
    "major_action_missing": ["yes", "no", "uncertain"],
    "granularity": ["adequate", "too_coarse", "uncertain"],
    "segment_structure": ["single_semantic_instance",
                          "multiple_semantic_phases",
                          "structural_oversegmentation_suspected",
                          "uncertain"],
}


def unsupported_stated_claim(r):
    """"no unsupported stated claim" -- a stated claim that the video refuses.

    Reading: primary or object unsupported, or SOME/NONE of the secondaries
    supported. `not_applicable` means there were no secondary claims to fail,
    so it does not count as an unsupported claim; `uncertain` is not a refusal
    either, and is handled by the uncertainty rule instead."""
    return (r["primary_verb_supported"] == "no"
            or r["object_supported"] == "no"
            or r["secondary_claims_supported"] in ("some", "none"))


def some_claims_supported(r):
    return (r["primary_verb_supported"] == "yes"
            or r["object_supported"] == "yes"
            or r["secondary_claims_supported"] in ("all", "some"))


RULES = {
    "correct": lambda r: (
        r["primary_verb_supported"] == "yes"
        and r["object_supported"] == "yes"
        and not unsupported_stated_claim(r)
        and r["major_action_missing"] == "no"
        and r["granularity"] == "adequate"),

    # "some stated claims supported + some unsupported OR major action missing"
    "partially_correct": lambda r: (
        some_claims_supported(r)
        and (unsupported_stated_claim(r)
             or r["major_action_missing"] == "yes")),

    # "core primary/object claim unsupported"
    "incorrect": lambda r: (r["primary_verb_supported"] == "no"
                            or r["object_supported"] == "no"),

    # "claims supported + granularity = too_coarse"
    "correct_but_coarse": lambda r: (
        r["primary_verb_supported"] == "yes"
        and r["object_supported"] == "yes"
        and not unsupported_stated_claim(r)
        and r["granularity"] == "too_coarse"),
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precedence", default="",
                    help="comma-separated status order to test as a fix. "
                         "Without it, the raw collisions are reported")
    a = ap.parse_args()

    keys = list(FIELDS)
    combos = [dict(zip(keys, v))
              for v in itertools.product(*(FIELDS[k] for k in keys))]
    print(f"{len(combos)} combinations of {len(keys)} fields\n")

    n_fired = Counter()
    collisions = Counter()
    uncovered = []
    for r in combos:
        fired = tuple(s for s, fn in RULES.items() if fn(r))
        n_fired[len(fired)] += 1
        if not fired:
            uncovered.append(r)
        elif len(fired) > 1:
            collisions[fired] += 1

    print("rules firing per combination:")
    for k in sorted(n_fired):
        tag = ("  <-- no derived status" if k == 0 else
               "  <-- ambiguous" if k > 1 else "")
        print(f"  {k} rule(s): {n_fired[k]:5d}  "
              f"({100*n_fired[k]/len(combos):5.1f}%){tag}")

    print(f"\ncolliding rule sets ({sum(collisions.values())} combinations):")
    for k, v in collisions.most_common():
        print(f"  {' + '.join(k):55s} {v:5d}")

    if collisions:
        print("\nwhat a collision looks like:")
        for k in list(collisions)[:3]:
            ex = next(r for r in combos
                      if tuple(s for s, fn in RULES.items() if fn(r)) == k)
            print(f"  {' + '.join(k)}")
            for f in keys:
                print(f"      {f:28s} {ex[f]}")

    print(f"\ncombinations reaching NO status: {len(uncovered)}")
    if uncovered:
        # the uncovered set is usually one or two coherent regions, so
        # summarise by field value rather than listing 3000 rows
        for f in keys:
            c = Counter(r[f] for r in uncovered)
            share = {k: f"{100*v/len(uncovered):.0f}%" for k, v in c.items()}
            print(f"  {f:28s} {share}")
        print("\n  an uncovered example:")
        for f in keys:
            print(f"      {f:28s} {uncovered[0][f]}")

    if a.precedence:
        order = [s.strip() for s in a.precedence.split(",")]
        missing = [s for s in RULES if s not in order]
        if missing:
            print(f"\n!! precedence omits {missing}; they can never fire")
        still = 0
        assigned = Counter()
        for r in combos:
            hit = next((s for s in order if s in RULES and RULES[s](r)), None)
            if hit is None:
                still += 1
            assigned[hit] += 1
        print(f"\nwith precedence {' > '.join(order)}:")
        for k, v in assigned.most_common():
            print(f"  {str(k):28s} {v:5d}")
        print(f"  still unassigned: {still}")
        print("  NOTE: precedence removes the ambiguity by choosing an axis "
              "to dominate.\n  It does not make the combination well defined "
              "-- it makes the choice explicit,\n  and the choice has to be "
              "checked against real events, not against this table.")


if __name__ == "__main__":
    main()
