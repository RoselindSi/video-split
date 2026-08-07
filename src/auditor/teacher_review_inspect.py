"""Read what the reviewer actually said, grouped by the mistakes that matter.

A pass/fail count says the pilot failed. It does not say whether the model
looked at the wrong thing, described something that was not there, or saw
correctly and weighed it wrongly -- and those need different fixes. This
prints the reasoning, ordered so the expensive errors come first.

  MISSED           false keeps the reviewer approved. These are the events
                   that would enter the dataset uncorrected, so they are
                   printed in full, with the subtype a human assigned.
  OVER-CHALLENGED  true keeps the reviewer challenged. The cost side: a
                   reviewer that buys safety by challenging everything has
                   only moved work back to a person.
  CAUGHT / KEPT    the correct decisions, in one line each.

INVENTED EVIDENCE IS AUDITED SEPARATELY. A reviewer can reach the right label
from a description of something that did not happen, and on a different clip
that reasoning fails. So any claim of release, recontact, object switch or
target switch on an event a human called same-action or camera-motion is
listed with the sentence that made it, whatever the final decision was.

Usage:
    python -m src.auditor.teacher_review_inspect \
        --review /workspace/tr1/results/hal/c3/teacher_review.json
    python -m src.auditor.teacher_review_inspect --review ... --event recording_000030_exact_t12.2
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

SHARP = "sharp_visible_transition"
CLAIM_KEYS = ("contact_transition", "object_switch_observed",
              "target_switch_observed", "discrete_state_change")


def dec(r):
    return (r.get("challenge") or {}).get("final_review")


def blind(r, k, default="?"):
    return (r.get("blind") or {}).get(k, default)


def show(r, full=True):
    b, c = r.get("blind") or {}, r.get("challenge") or {}
    print(f"\n  {r['event_id']}")
    print(f"    human subtype      {r['subtype']}")
    print(f"    student score      {r.get('score') or '-'}")
    print(f"    blind decision     {b.get('blind_decision', '?')}"
          f"   ({b.get('negative_reason') or 'no negative reason'})")
    print(f"    final review       {dec(r)}"
          f"   admit={c.get('approve_for_direct_admission')}")
    if not full:
        return
    print(f"    what it says it saw:")
    for k in ("hand_visibility", "active_object_visibility",
              "same_object_before_after", "contact_transition",
              "object_switch_observed", "target_switch_observed",
              "discrete_state_change", "camera_motion_dominant",
              "transition_type", "evidence_sufficient"):
        if k in b:
            print(f"      {k:<28} {b[k]}")
    for line in (b.get("evidence") or []):
        print(f"      + {line}")
    for line in (c.get("strongest_counterevidence") or []):
        print(f"      - {line}")
    if r.get("blind_unparsed"):
        print(f"      !! blind reply did not parse: {r['blind_unparsed'][:160]}")
    if r.get("challenge_unparsed"):
        print(f"      !! challenge reply did not parse: "
              f"{r['challenge_unparsed'][:160]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", required=True)
    ap.add_argument("--event", help="print one event and nothing else")
    ap.add_argument("--subtype", help="restrict to one human subtype")
    a = ap.parse_args()

    with open(a.review, encoding="utf-8") as f:
        blob = json.load(f)
    res = blob["results"]
    if a.event:
        for r in res:
            if r["event_id"] == a.event:
                show(r)
                return
        raise SystemExit(f"{a.event} is not in this file")
    if a.subtype:
        res = [r for r in res if r["subtype"] == a.subtype]
        print(f"restricted to subtype {a.subtype}: {len(res)} events")

    missed = [r for r in res if r["arm"] == "false_keep" and dec(r) == "approve"]
    over = [r for r in res if r["arm"] == "true_keep" and dec(r) != "approve"]
    caught = [r for r in res if r["arm"] == "false_keep" and dec(r) != "approve"]
    kept = [r for r in res if r["arm"] == "true_keep" and dec(r) == "approve"]

    print(f"\n{'#' * 74}\n# MISSED -- wrong admissions the reviewer approved "
          f"({len(missed)})\n{'#' * 74}")
    print(f"  by subtype: {dict(Counter(r['subtype'] for r in missed))}")
    for r in missed:
        show(r)

    print(f"\n{'#' * 74}\n# OVER-CHALLENGED -- real boundaries it refused "
          f"({len(over)})\n{'#' * 74}")
    for r in over:
        show(r)

    print(f"\n{'#' * 74}\n# CAUGHT ({len(caught)}) and KEPT ({len(kept)}), one "
          f"line each\n{'#' * 74}")
    for tag, group in (("caught", caught), ("kept", kept)):
        for r in group:
            print(f"  {tag:<7} {r['event_id'][:46]:<46} "
                  f"{r['subtype'][:26]:<26} {blind(r, 'transition_type')}")

    # a right answer reached from something that did not happen fails
    # differently from a wrong answer, and only this shows which
    print(f"\n{'#' * 74}\n# EVIDENCE THAT CANNOT BE TRUE\n{'#' * 74}")
    suspect = ("same_action_internal_motion", "camera_or_viewpoint_shift",
               "visibility_or_offscreen")
    n = 0
    for r in res:
        if r["subtype"] not in suspect:
            continue
        b = r.get("blind") or {}
        claims = []
        if b.get("contact_transition") in ("release", "recontact",
                                           "release_and_recontact"):
            claims.append(f"contact_transition={b['contact_transition']}")
        for k in ("object_switch_observed", "target_switch_observed",
                  "discrete_state_change"):
            if b.get(k):
                claims.append(k)
        if not claims:
            continue
        n += 1
        print(f"\n  {r['event_id']}  (human: {r['subtype']})")
        print(f"    claims: {', '.join(claims)}   final: {dec(r)}")
        for line in (b.get("evidence") or []):
            print(f"      + {line}")
    if not n:
        print("  none -- every structured claim is consistent with the human "
              "subtype")
    else:
        print(f"\n  {n} event(s) assert an interaction change on a clip a human "
              f"called same-action, camera motion or unobservable. Where the "
              f"final answer is\n  right anyway, it is right for a reason that "
              f"was not in the video, and that reasoning will not repeat.")

    print(f"\n{'#' * 74}\n# WHERE THE TWO PASSES DISAGREE\n{'#' * 74}")
    flip = [r for r in res
            if blind(r, "blind_decision") == "approve_visible_sharp"
            and dec(r) != "approve"]
    stick = [r for r in res
             if blind(r, "blind_decision") != "approve_visible_sharp"
             and dec(r) == "approve"]
    print(f"  blind approved then challenged: {len(flip)}  "
          f"({sum(1 for r in flip if r['arm'] == 'false_keep')} of them really "
          f"were wrong admissions)")
    print(f"  blind rejected then approved:   {len(stick)}  "
          f"({sum(1 for r in stick if r['arm'] == 'true_keep')} of them really "
          f"were correct)")
    print("  The first row is the challenge pass doing its job; the second is "
          "it undoing a correct blind rejection.")

    print(f"\n{'#' * 74}\n# BY SUBTYPE\n{'#' * 74}")
    print(f"  {'subtype':<32} {'n':>3} {'approved':>9} {'challenged':>11}")
    for s in sorted({r["subtype"] for r in res}):
        g = [r for r in res if r["subtype"] == s]
        ap_ = sum(1 for r in g if dec(r) == "approve")
        print(f"  {s:<32} {len(g):>3} {ap_:>9} {len(g) - ap_:>11}"
              + ("   <- should be approved" if s == SHARP else
                 "   <- should be challenged"))


if __name__ == "__main__":
    main()
