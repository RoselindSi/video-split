"""Read what the reviewer actually said, in the four layers that fail
differently.

A pass/fail count says the pilot failed. It cannot say whether the model
misread the video, read it correctly and reasoned about time wrongly, or did
both and the rule then let it through. Those need different fixes, so the
output separates them:

  1 OBSERVATION   what it says it saw -- hands, objects, contact, switches.
                  Wrong here is a perception failure.
  2 LOCALISATION  when the change starts and ends, whether it is concentrated,
                  whether a unique transition point falls inside the
                  tolerance. Wrong here with layer 1 right is a temporal
                  reasoning failure, which is what v1 showed.
  3 EVIDENCE      both sides in full, not just the strongest. v1's tell was an
                  h0 line that was correct and then ignored, and a summary
                  that keeps only the winning side hides exactly that.
  4 ELIGIBILITY   every rule check, PASS and FAIL alike, recomputed from the
                  config. Printing only the failures cannot distinguish "the
                  model was wrong" from "our rule is too strict".

For v2 runs with repeats, the draws are diffed FIELD BY FIELD. Knowing an
event flipped says nothing; knowing that object_switch held steady while
change_concentrated_near_candidate moved says the perception is stable and the
adjudication is not, which is a different bottleneck and a different fix.

v1 files use blind/challenge; v2 uses review/eligible/draws. The format is
detected rather than assumed.

Usage:
    python -m src.auditor.teacher_review_inspect --review ...v2.json \
        --subtype gradual_phase_transition --verbose
    python -m src.auditor.teacher_review_inspect --review ...v2.json \
        --gold false_keep --route provisional_admission --show_eligibility
    python -m src.auditor.teacher_review_inspect --review ...v2.json \
        --flipped_only --show_draws
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

SHARP = "sharp_visible_transition"

OBS = ["before_interaction", "after_interaction", "hand_visibility",
       "active_object_visibility", "same_active_object_before_after",
       "contact_change", "object_switch", "target_switch",
       "discrete_object_state_change", "camera_motion_dominant"]
LOC = ["pre_state_stable", "post_state_stable",
       "change_begins_before_candidate", "change_continues_after_candidate",
       "change_concentrated_near_candidate", "unique_transition_point_visible",
       "transition_within_tolerance"]
# the localisation fields a gradual event should mark, in the order that reads
# left to right as "does this change sit at the candidate"
GRID = ["change_begins_before_candidate", "change_continues_after_candidate",
        "change_concentrated_near_candidate", "unique_transition_point_visible",
        "transition_within_tolerance"]


def is_v2(r):
    return "review" in r or "eligible" in r


def rev(r):
    return (r.get("review") if is_v2(r) else r.get("blind")) or {}


def verdict(r):
    if is_v2(r):
        return r.get("route", "?")
    return (r.get("challenge") or {}).get("final_review", "?")


def admitted(r):
    return (r.get("route") == "provisional_admission" if is_v2(r)
            else verdict(r) == "approve")


def checks(b, rule):
    """Every rule item with its outcome, not only the ones that failed.
    Printing failures alone cannot separate a wrong model from a strict rule."""
    if not rule:
        return []
    out = [("evidence_sufficient", bool(b.get("evidence_sufficient")),
            b.get("evidence_sufficient")),
           (f"decision == {rule['require_decision']}",
            b.get("decision") == rule["require_decision"], b.get("decision"))]
    for k in rule["require_yes"]:
        out.append((f"{k} == yes", b.get(k) == "yes", b.get(k)))
    for k in rule["forbid_no"]:
        out.append((f"{k} != no", b.get(k) != "no", b.get(k)))
    for k in rule["forbid_yes"]:
        out.append((f"{k} != yes", b.get(k) != "yes", b.get(k)))
    for k in rule.get("forbid_insufficient", []):
        out.append((f"{k} != insufficient", b.get(k) != "insufficient", b.get(k)))
    return out


def show(r, rule, args):
    b = rev(r)
    print(f"\n{'-' * 74}\n  {r['event_id']}")
    print(f"    human subtype {r['subtype']}   arm {r.get('arm')}   "
          f"-> {verdict(r)}")
    if not b:
        print(f"    !! nothing parsed: {(r.get('unparsed') or r.get('blind_unparsed') or '')[:200]}")
        return
    if args.verbose:
        print("    OBSERVATION")
        for k in OBS:
            if k in b:
                print(f"      {k:<34} {b[k]}")
        print("    TEMPORAL LOCALISATION")
        for k in LOC:
            if k in b:
                print(f"      {k:<34} {b[k]}")
    if args.show_evidence or args.verbose:
        for tag, key in (("H1 EVIDENCE", "h1_evidence"),
                         ("H0 EVIDENCE", "h0_evidence"),
                         ("EVIDENCE", "evidence")):
            if b.get(key):
                print(f"    {tag}")
                for line in b[key]:
                    print(f"      {'+' if 'H0' not in tag else '-'} {line}")
        for k in ("strongest_h1_evidence", "strongest_h0_evidence"):
            if b.get(k):
                print(f"    {k:<26} {b[k]}")
        print(f"    ADJUDICATION  decision={b.get('decision')}"
              f"  negative_reason={b.get('negative_reason_if_h0')}")
    if (args.show_eligibility or args.verbose) and rule:
        cs = checks(b, rule)
        print("    ELIGIBILITY")
        for name, ok, val in cs:
            print(f"      {'PASS' if ok else 'FAIL'}  {name:<42} ({val})")
        bad = [n for n, ok, _ in cs if not ok]
        print(f"      RESULT: {'ELIGIBLE' if not bad else 'HUMAN REVIEW'}"
              + (f"   blockers: {', '.join(bad)}" if bad else ""))
    if (args.show_safety or args.verbose) and r.get("safety"):
        s = r["safety"]
        print(f"    SAFETY PASS  blocker={s.get('blocker_found')}  "
              f"final={s.get('final')}")
        if s.get("specific_contradiction"):
            print(f"      cited: {s['specific_contradiction']}")
    if args.show_draws and r.get("draws"):
        ds = r["draws"]
        print(f"    DRAWS ({len(ds)})")
        keys = [k for k in OBS + LOC
                if len({json.dumps(( d.get('review') or {}).get(k))
                        for d in ds}) > 1]
        if not keys:
            print("      every observed field identical across draws; only the "
                  "decision moved" if len({d['decision'] for d in ds}) > 1
                  else "      identical across draws")
        for k in keys:
            vals = [str((d.get("review") or {}).get(k)) for d in ds]
            print(f"      {k:<34} {' | '.join(vals)}")
        print(f"      {'decision':<34} "
              f"{' | '.join(str(d['decision']) for d in ds)}")
        print(f"      {'eligible':<34} "
              f"{' | '.join(str(d['eligible']) for d in ds)}")
        if any("review" not in d for d in ds):
            print("      !! this file predates per-draw observations, so only "
                  "the verdicts can be diffed. Re-run to compare fields.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", required=True)
    ap.add_argument("--config", help="eligibility rule; defaults to the config "
                                     "path recorded in the review file")
    ap.add_argument("--event")
    ap.add_argument("--subtype")
    ap.add_argument("--route")
    ap.add_argument("--gold", choices=["false_keep", "true_keep"])
    ap.add_argument("--flipped_only", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--show_evidence", action="store_true")
    ap.add_argument("--show_eligibility", action="store_true")
    ap.add_argument("--show_safety", action="store_true")
    ap.add_argument("--show_draws", action="store_true")
    a = ap.parse_args()

    blob = json.load(open(a.review, encoding="utf-8"))
    res = blob["results"]
    v2 = any(is_v2(r) for r in res)
    rule = None
    cfgp = a.config or blob.get("config")
    if v2 and cfgp and os.path.exists(cfgp):
        rule = json.load(open(cfgp, encoding="utf-8")).get("eligibility")
    elif v2:
        print("  !! no eligibility config found, so the rule cannot be "
              "recomputed; --show_eligibility will be empty")
    print(f"{os.path.basename(a.review)}: {len(res)} events, "
          f"format {'v2' if v2 else 'v1'}")

    if a.event:
        for r in res:
            if r["event_id"] == a.event:
                a.verbose = True
                show(r, rule, a)
                return
        raise SystemExit(f"{a.event} not in this file")

    sel = res
    if a.subtype:
        sel = [r for r in sel if r["subtype"] == a.subtype]
    if a.route:
        sel = [r for r in sel if r.get("route") == a.route]
    if a.gold:
        sel = [r for r in sel if r.get("arm") == a.gold]
    if a.flipped_only:
        sel = [r for r in sel if not r.get("stable", True)]
    print(f"  {len(sel)} after filters")

    for r in sel:
        show(r, rule, a)

    # the grid that decides the next change: for each event, whether the model
    # marked the change as sitting at the candidate at all
    if v2 and sel:
        print(f"\n{'=' * 100}\nLOCALISATION GRID\n{'=' * 100}")
        hdr = ["begins_before", "continues_after", "concentrated",
               "unique_point", "in_tol"]
        print(f"  {'event':<34} {'gold':<11} "
              + " ".join(f"{h:<16}" for h in hdr) + f"{'dec':<6} elig")
        for r in sel:
            b = rev(r)
            cells = " ".join(f"{str(b.get(k)):<16}" for k in GRID)
            print(f"  {r['event_id'][-33:]:<34} {r.get('arm', ''):<11} {cells}"
                  f"{str(b.get('decision')):<6} {r.get('eligible')}")
        print("\n  A gradual event should read yes / yes / no / no / no. Rows "
              "that read otherwise while the human called them gradual are the "
              "model failing to\n  localise, not the rule failing to block -- "
              "and the fix for those is not another threshold.")

    print(f"\n{'=' * 74}\nSUMMARY\n{'=' * 74}")
    print(f"  {'subtype':<32} {'n':>3} {'admitted':>9} {'to human':>9}")
    for s in sorted({r["subtype"] for r in sel}):
        g = [r for r in sel if r["subtype"] == s]
        na = sum(1 for r in g if admitted(r))
        print(f"  {s:<32} {len(g):>3} {na:>9} {len(g) - na:>9}"
              + ("   <- should be admitted" if s == SHARP else ""))
    if v2:
        fl = [r for r in sel if not r.get("stable", True)]
        print(f"\n  unstable across draws: {len(fl)}/{len(sel)}")


if __name__ == "__main__":
    main()
