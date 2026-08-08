"""Test the reject certificate on observations already collected. No API calls.

The proposed rule reads:

    no_interaction_change = same_active_object_before_after == yes
                            and contact_change == none
                            and object_switch == no
                            and target_switch == no
                            and discrete_object_state_change == no
    certified = evidence_sufficient and visibility sufficient
                and no_interaction_change

Every field it needs was already recorded by the observe-only run, so whether
this rule would delete a real boundary is answerable now, on events that have
been through both the teacher and the human auditor, before a single call is
made for Phase A.

WHAT THIS CAN AND CANNOT SAY. The events in the teacher files were drawn from
the AUTO_KEEP pool -- high student scores. Phase A would draw from the LOW
tail, where the sharp events may look different, so a false-reject count here
is NOT a forecast of the Phase A rate. What it can establish is whether the
certificate is safe BY CONSTRUCTION: a rule that deletes a real boundary on
any population is not a rule whose go/no-go can be "false reject = 0", and
that verdict transfers even though the rate does not.

CAMERA MOTION IS NOT A REJECT REASON ON ITS OWN, and this file follows the
proposal in only recording it. A moving camera does not mean the hand-object
interaction was unchanged; it means the evidence is harder to read, which is a
reason to send a human, not to delete.

Usage:
    python -m src.auditor.reject_certificate \
        --review /workspace/tr1/results/hal/c3/teacher_observe_only.json \
        --reject_safe /workspace/tr1/results/hal/c3/reject_safe.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

CONTINUITY = {"same_active_object_before_after": "yes",
              "contact_change": "none",
              "object_switch": "no",
              "target_switch": "no",
              "discrete_object_state_change": "no"}


def rev(r):
    return (r.get("review") if ("review" in r or "eligible" in r)
            else r.get("blind")) or {}


def certify(b):
    """(certified, the fields that stopped it). The blockers are returned
    because a rule saved by one field on most events is one field away from
    deleting them, and that margin is invisible in a pass/fail count."""
    if not b:
        return False, ["no parseable response"]
    fail = []
    if not b.get("evidence_sufficient"):
        fail.append("evidence_sufficient=false")
    for k in ("hand_visibility", "active_object_visibility"):
        if b.get(k) == "insufficient":
            fail.append(f"{k}=insufficient")
    for k, want in CONTINUITY.items():
        if b.get(k) != want:
            fail.append(f"{k}={b.get(k)} (needs {want})")
    return (not fail), fail


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", action="append", required=True)
    ap.add_argument("--reject_safe", required=True)
    ap.add_argument("--gold", action="append",
                    default=["data/gold/audit_188_gold_v2.jsonl"])
    a = ap.parse_args()

    safe = {r["event_id"]: r
            for r in json.load(open(a.reject_safe, encoding="utf-8"))["events"]}
    gold = {}
    for p in a.gold:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    gold[r["event_id"]] = r

    res = []
    for p in a.review:
        blob = json.load(open(p, encoding="utf-8"))
        for r in blob["results"]:
            r["_file"] = os.path.basename(p)
            res.append(r)
    print(f"{len(res)} reviewed events from {len(a.review)} file(s)")

    cert, blocked = [], []
    for r in res:
        ok, why = certify(rev(r))
        (cert if ok else blocked).append((r, why))

    print(f"\n{'=' * 78}\nTHE CERTIFICATE ON ALREADY-COLLECTED OBSERVATIONS"
          f"\n{'=' * 78}")
    print(f"  certified safe to delete   {len(cert)}")
    print(f"  refused                    {len(blocked)}")

    bad = [(r, why) for r, why in cert
           if not (safe.get(r["event_id"], {}).get("reject_safe"))]
    good = [(r, why) for r, why in cert if r not in [x for x, _ in bad]]
    print(f"\n  of the {len(cert)} certified: {len(cert) - len(bad)} really "
          f"are safe to delete, {len(bad)} are NOT")

    if bad:
        print(f"\n  THE {len(bad)} FALSE REJECTS. Each is a boundary this rule "
              f"would have deleted unreviewed:")
        for r, _ in bad:
            b = rev(r)
            g = gold.get(r["event_id"], {})
            s = safe.get(r["event_id"], {})
            print(f"\n    {r['event_id']}")
            print(f"      human subtype {r.get('subtype')}   truth "
                  f"{g.get('temporal_truth')}   validity "
                  f"{g.get('candidate_boundary_validity')}   corrected "
                  f"{g.get('primary_corrected_boundary_time')}")
            print(f"      not reject-safe because: {s.get('reason')}")
            print(f"      the teacher saw: {b.get('before_interaction')}")
            print(f"                   ->  {b.get('after_interaction')}")
            print(f"      camera_motion_dominant={b.get('camera_motion_dominant')}"
                  f"  hand_visibility={b.get('hand_visibility')}")
        print(f"\n  false rejects by subtype: "
              f"{dict(Counter(r.get('subtype') for r, _ in bad))}")
    else:
        print("\n  No false reject on this sample. That is necessary and not "
              "sufficient: see the margin below, since a rule held back by a\n"
              "  single field on most events is one field away from deleting "
              "them.")

    # how close the refusals came. A certificate that survives because one
    # field happened to fire is not a safe certificate, it is a lucky one.
    margins = Counter(len(why) for _, why in blocked)
    print(f"\n{'=' * 78}\nHOW CLOSE THE REFUSALS CAME\n{'=' * 78}")
    print(f"  {'blockers':>9} {'events':>7}   (1 means a single field stood "
          f"between this event and deletion)")
    for k in sorted(margins):
        print(f"  {k:>9} {margins[k]:>7}")
    one = [(r, why) for r, why in blocked if len(why) == 1]
    real = [(r, why) for r, why in one
            if not safe.get(r["event_id"], {}).get("reject_safe")]
    if real:
        print(f"\n  {len(real)} real boundaries were held back by ONE field:")
        for r, why in real:
            print(f"    {r['event_id'][-46:]:<47} saved by {why[0]}")
        print("  Those are the events that decide whether this rule is robust "
              "or lucky. A schema change that softens any of\n  those fields "
              "deletes them.")

    print(f"\n  which fields do the refusing, over all "
          f"{len(blocked)} refusals:")
    for k, n in Counter(w.split("=")[0] for _, why in blocked
                        for w in why).most_common():
        print(f"    {n:>4}  {k}")


if __name__ == "__main__":
    main()
