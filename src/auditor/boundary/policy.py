"""Ontology + model output -> an audit action. Deterministic, no parameters.

Reads configs/auditor/boundary_ontology_v1.yaml and applies it. The separation
matters because the three previous failures were all one layer reaching into
another: a perception head penalised for not reproducing an annotation
convention, an operating point selected on the same events it was scored on,
and a deterministic rule that never once overrode the model it was supposed to
check.

REVIEW IS THE DEFAULT AND EVERY OTHER ACTION MUST BE EARNED. A rule that does
not fire produces REVIEW, never an action, and the reason list records which
condition carried it there so a decision can be traced without rerunning the
model.

A GATE WHOSE INPUT IS MISSING BLOCKS. The observability and nuisance rules
name fields the untrained heads do not emit, and the first version read them
with `.get(k)` and skipped the check when the answer was None -- a safety gate
that passes when it cannot be evaluated. Absent input now routes to REVIEW with
that stated as the reason, which is why a model whose abstain heads carry no
gradient automates nothing at all. That is the honest state, not a regression.

THE SUBTYPE IS A LABEL AND DEPLOYMENT DOES NOT HAVE IT. `subtype_overrides` is
how annotation_convention and ambiguous events avoid an automatic decision, and
in evaluation the subtype is read from the gold. Passing --deployment withholds
it, which is the only configuration whose numbers describe what would happen on
new video. The gap between the two columns is exactly what a learned abstain
head has to close.

ACTIONS THE ONTOLOGY DISABLES ARE NOT COMPUTED SILENTLY. AUTO_REJECT and
SUGGEST_RETIME are both `enabled: false` in v1 -- AUTO_REJECT because no
evidence has earned it and SUGGEST_RETIME because 6 EARLY and 4 LATE over 8
recordings cannot support one. When a disabled action's conditions are met the
event still routes to REVIEW and the reason says so, so the coverage that
action WOULD have had stays measurable without it being taken.
"""
from __future__ import annotations

import json
import os

MORPHOLOGY = ["POINT_TRANSITION", "INTERVAL_TRANSITION", "NO_TRANSITION",
              "UNOBSERVABLE"]
RELATION = ["EXACT", "EARLY", "LATE", "DUPLICATE", "NO_VALID"]


def load_ontology(path="configs/auditor/boundary_ontology_v1.yaml"):
    try:
        import yaml
    except ImportError:
        raise SystemExit("pyyaml is needed to read the ontology; the rules are "
                         "config rather than code on purpose, so this is not "
                         "an optional dependency")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _listify(v):
    return v if isinstance(v, list) else [v]


def decide(pred, onto, subtype=None):
    """pred: the model's per-event output dict with class probabilities.
    Returns (action, reasons, blocked_action_or_None)."""
    reasons = []
    m = max(MORPHOLOGY, key=lambda k: pred["morphology"][k])
    p_m = pred["morphology"][m]
    r = max(RELATION, key=lambda k: pred["relation"][k])
    p_r = pred["relation"][r]
    reasons.append(f"morphology={m}({p_m:.2f})")
    reasons.append(f"relation={r}({p_r:.2f})")

    mo = onto["morphology"].get(m, {})
    if mo.get("automatic_decision") is False:
        return "REVIEW", reasons + [f"{m} is never decided automatically"], None
    ov = (onto.get("subtype_overrides") or {}).get(subtype or "", {})
    if ov.get("automatic_decision") is False:
        return "REVIEW", reasons + [
            f"subtype {subtype} is never decided automatically"], None

    req = (onto.get("observability") or {}).get("require_for_automation", {})
    for k, allowed in req.items():
        v = pred.get(k)
        if isinstance(v, dict):
            v = max(v, key=v.get)
        if v is None:
            return "REVIEW", reasons + [
                f"{k} is required for automation and the model does not emit "
                f"it"], None
        if v not in _listify(allowed):
            return "REVIEW", reasons + [f"{k}={v} blocks automation"], None
    nu = (onto.get("nuisance") or {}).get("camera_dominant", {})
    if nu.get("blocks_automatic_decision"):
        cd = pred.get("camera_dominant")
        if cd is None:
            return "REVIEW", reasons + [
                "camera_dominant is required for automation and the model "
                "does not emit it"], None
        if cd >= 0.5:
            return "REVIEW", reasons + ["camera-dominant blocks automation"], None

    acts = onto["actions"]
    for name in ("AUTO_KEEP", "SUGGEST_RETIME", "AUTO_REJECT"):
        spec = acts.get(name) or {}
        if m not in _listify(spec.get("morphology", [])):
            continue
        if r not in _listify(spec.get("relation", [])):
            continue
        if p_m < spec.get("min_confidence", 1.1):
            reasons.append(f"{name}: morphology confidence {p_m:.2f} below "
                           f"{spec['min_confidence']}")
            continue
        if p_r < spec.get("min_confidence", 1.1):
            reasons.append(f"{name}: relation confidence {p_r:.2f} below "
                           f"{spec['min_confidence']}")
            continue
        lim = spec.get("max_abs_offset_sec")
        if lim is not None and abs(pred.get("offset", 0.0)) > lim:
            reasons.append(f"{name}: |offset| "
                           f"{abs(pred.get('offset', 0.0)):.2f}s exceeds {lim}")
            continue
        if subtype in (spec.get("forbid_subtypes") or []):
            reasons.append(f"{name}: subtype {subtype} is forbidden")
            continue
        if not spec.get("enabled", True):
            # the conditions held; the action is switched off. Recorded rather
            # than dropped so the coverage it would have had stays measurable
            return "REVIEW", reasons + [
                f"{name} conditions met but the action is disabled in the "
                f"ontology"], name
        return name, reasons, None
    return "REVIEW", reasons + ["no action's conditions were met"], None


def decide_all(preds, onto, subtypes=None):
    out = []
    for p in preds:
        st = (subtypes or {}).get(p["event_id"])
        a, why, blocked = decide(p, onto, st)
        out.append({**p, "audit_action": a, "policy_reason": why,
                    "blocked_action": blocked})
    return out


def morphology_only(pred, onto, spec_name="AUTO_KEEP"):
    """Would this event clear the LEARNED conditions alone -- morphology,
    relation and confidence -- with every gate the model cannot currently
    evaluate set aside?

    This is the size of the hole. Those events are held back today by the
    gold subtype and by gates that block for want of an input; both of those
    disappear on new video, and what remains is whatever an abstain head
    learns to catch."""
    spec = (onto["actions"].get(spec_name) or {})
    m = max(MORPHOLOGY, key=lambda k: pred["morphology"][k])
    r = max(RELATION, key=lambda k: pred["relation"][k])
    if m not in _listify(spec.get("morphology", [])):
        return False
    if r not in _listify(spec.get("relation", [])):
        return False
    c = spec.get("min_confidence", 1.1)
    if pred["morphology"][m] < c or pred["relation"][r] < c:
        return False
    lim = spec.get("max_abs_offset_sec")
    if lim is not None and abs(pred.get("offset") or 0.0) > lim:
        return False
    return True


def main():
    import argparse
    from collections import Counter
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True,
                    help="output of src.auditor.boundary.train --out")
    ap.add_argument("--ontology",
                    default="configs/auditor/boundary_ontology_v1.yaml")
    ap.add_argument("--deployment", action="store_true",
                    help="withhold the gold subtype, which is what new video "
                         "looks like")
    ap.add_argument("--out")
    a = ap.parse_args()

    onto = load_ontology(a.ontology)
    blob = json.load(open(a.predictions, encoding="utf-8"))
    preds = blob["events"]
    subtypes = {p["event_id"]: p.get("subtype") for p in preds}
    rows = decide_all(preds, onto, None if a.deployment else subtypes)

    print(f"{len(rows)} events, ontology v{onto.get('version')} from "
          f"{os.path.basename(a.ontology)}"
          + ("   DEPLOYMENT: the gold subtype is withheld" if a.deployment
             else "   EVALUATION: the gold subtype is available"))
    print(f"\n  {'action':<18} {'n':>5}")
    for k, n in Counter(r["audit_action"] for r in rows).most_common():
        print(f"  {k:<18} {n:>5}")
    bl = Counter(r["blocked_action"] for r in rows if r["blocked_action"])
    if bl:
        print(f"\n  would have fired if the ontology enabled them:")
        for k, n in bl.most_common():
            print(f"    {k:<18} {n:>5}")

    # ------------------------------------------------------- the gap
    masked = [r for r in rows if not r.get("morphology_true")]
    print(f"\n{'=' * 78}\nWHAT HOLDS EACH EVENT BACK\n{'=' * 78}")
    why = Counter(r["policy_reason"][-1].split(" is required")[0]
                  .split("=")[0].split(" conditions")[0]
                  for r in rows)
    for k, n in why.most_common(8):
        print(f"  {n:>5}  {k}")

    print(f"\n{'=' * 78}\nTHE SIZE OF THE HOLE\n{'=' * 78}")
    hole = [r for r in rows if morphology_only(r, onto)]
    hm = [r for r in hole if not r.get("morphology_true")]
    print(f"  {len(hole)} of {len(rows)} events clear the LEARNED AUTO_KEEP "
          f"conditions on their own -- morphology, relation and\n  confidence, "
          f"with every gate the model cannot currently evaluate set aside.")
    print(f"  {len(hm)} of those have NO morphology target at all: "
          f"{dict(Counter(r['subtype'] for r in hm))}")
    print(f"  Those are the events an abstain head has to catch. Today they "
          f"are held back by the gold subtype and by gates that\n  block for "
          f"want of an input, and neither survives contact with new video.")
    if masked:
        conf = [max(r["morphology"].values()) for r in masked]
        import statistics
        print(f"\n  all {len(masked)} masked events: morphology confidence "
              f"median {statistics.median(conf):.2f}, "
              f"{sum(1 for c in conf if c > 0.95)} above 0.95")
    if a.out:
        json.dump({"ontology": os.path.abspath(a.ontology), "events": rows},
                  open(a.out, "w", encoding="utf-8"), indent=2, default=str)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
