"""Boundary evidence + semantic evidence -> one action. Deterministic, no parameters.

Reads `configs/auditor/joint_policy_v1.yaml` and applies it. The two auditors
run independently -- the boundary one label-blind, the semantic one label-aware
-- and this is the only place their outputs meet.

THE SEMANTIC SIDE IS NOT A BOUNDARY TRUTH SOURCE. It contributes continuity
evidence. If `label_L` and `label_R` are both `wipe counter` but a release,
reset and restart is observed between them, the answer is still
`same_action_new_instance -> BOUNDARY`. Identical labels are not an argument
against a boundary, and 56% of the disputed events in the double-audit sit on
exactly that configuration.

TRI-STATE, NOT BOOLEAN, and this is the point the whole file turns on:

    OBSERVED_PRESENT   looked for, found
    OBSERVED_ABSENT    looked for, view adequate, not there
    NOT_OBSERVABLE     not assessable

`no reset was seen` and `it was seen that there was no reset` are different
claims. Only OBSERVED_ABSENT satisfies a condition requiring absence. Collapsing
the two is the same defect as the observability gate that read its input with
`.get(k)` and skipped the check when the answer was None -- a safety gate that
passes when it cannot be evaluated.

AUTO_REJECT_CANDIDATE IS NOT AUTO_REJECT. It names a region of evidence space
in which a reject would be defensible, and it is measured rather than executed.
`enabled: false` in the config, and the driver may not turn it on before an
independent gold has produced a false-reject rate with an interval. Every
operating point in this project that was authorised before that step was later
withdrawn.

Usage:
    python -m src.auditor.joint_policy --self_test
    python -m src.auditor.joint_policy --events events.jsonl --out decided.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

CONFIG = "configs/auditor/joint_policy_v1.yaml"

PRESENT, ABSENT, UNKNOWN = "OBSERVED_PRESENT", "OBSERVED_ABSENT", "NOT_OBSERVABLE"
TRI = (PRESENT, ABSENT, UNKNOWN)


BOUNDARY_ONTOLOGY = "configs/auditor/boundary_ontology_v1.yaml"


def load_config(path=CONFIG, ontology=BOUNDARY_ONTOLOGY):
    """The joint policy, plus the observability levels it must not restate.

    A missing ontology is a hard error rather than a fallback to a hardcoded
    list: a joint policy that silently invents its own visibility levels would
    disagree with the boundary policy about when automation is allowed, and
    both would still print a decision."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("pyyaml is required to read the frozen policy; the "
                         "policy is not duplicated in code on purpose.")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not os.path.exists(ontology):
        raise SystemExit(f"{ontology} not found. The observability levels are "
                         f"defined there and this file may not restate them.")
    with open(ontology, encoding="utf-8") as f:
        ont = yaml.safe_load(f)
    req = (ont.get("observability") or {}).get("require_for_automation")
    if not req:
        raise SystemExit(f"{ontology} has no observability."
                         f"require_for_automation; without it nothing here "
                         f"knows when a view is adequate.")
    cfg["_observability_required"] = {k: set(v) for k, v in req.items()}
    return cfg


def _tri(v):
    """Anything not an explicit tri-state value is NOT_OBSERVABLE.

    A missing field, a None, a stray boolean -- all of them mean "this was not
    assessed", never "this was assessed as absent". A `False` arriving here
    from a head that emits booleans would otherwise read as OBSERVED_ABSENT
    and satisfy the auto-reject conjunction on no evidence at all."""
    return v if v in TRI else UNKNOWN


def semantic_compatible_with_continuity(sem, thr):
    """BOTH cross terms must support. One direction is not symmetry.

    `label_R` can be generic enough to describe the left segment while
    `label_L` is specific and fails on the right; taking either one alone
    would call that pair continuous."""
    lr = sem.get("support_L_labelR")
    rl = sem.get("support_R_labelL")
    if lr is None or rl is None:
        return False, "a cross-support term is missing"
    if lr < thr or rl < thr:
        return False, (f"cross support below {thr}: "
                       f"L|label_R={lr:.2f}, R|label_L={rl:.2f}")
    return True, f"cross support {lr:.2f} / {rl:.2f}"


def decide(bnd, sem, cfg):
    """One event -> {action, precedence_block, reasons, blocked_by, evidence_used}."""
    thr = cfg["thresholds"]
    reasons, blocked = [], []
    used = {}

    reset = _tri(bnd.get("release_reset_restart"))
    used["release_reset_restart"] = reset
    morph = bnd.get("morphology")
    used["morphology"] = morph
    cont = bnd.get("interaction_continuity")
    used["interaction_continuity"] = cont

    # ---- precedence 1: a release forces a boundary ------------------------
    # Evaluated first and never revisited. Semantic continuity cannot reach it.
    if reset == PRESENT:
        return {
            "action": "KEEP_BOUNDARY",
            "precedence_block": "release_reset_restart_forces_boundary",
            "instance_relation": "same_action_new_instance",
            "reasons": ["release/reset/restart observed between the segments; "
                        "two repetitions of one action are two instances"],
            "blocked_by": [],
            "evidence_used": used,
            "followup": [],
        }

    # ---- hard rule: direction change alone is not a boundary --------------
    only = set(bnd.get("only_evidence_is") or [])
    if only == {"motion_direction_change"}:
        blocked.append("direction_change_is_not_boundary")
        reasons.append("the only boundary evidence is a change of motion "
                       "direction; wiping left then right is one instance")

    # ---- precedence 2: the auto-reject CANDIDATE conjunction --------------
    ok, why = True, []

    strong = (morph == "NO_TRANSITION"
              and float(bnd.get("morphology_confidence") or 0.0)
              >= thr["morphology_strong_min"])
    if not strong:
        ok = False
        why.append(f"morphology is not strong NO_TRANSITION "
                   f"({morph} @ {bnd.get('morphology_confidence')})")

    if cont != "continuous":
        ok = False
        why.append(f"interaction continuity is {cont!r}, not 'continuous'")

    # THE TRI-STATE CHECK. Absence must be observed, not merely unreported.
    if reset != ABSENT:
        ok = False
        why.append(f"release/reset/restart is {reset}, and only "
                   f"OBSERVED_ABSENT licenses a reject")
        if reset == UNKNOWN:
            blocked.append("unseen_reset_is_not_absent_reset")

    obs = bnd.get("observability") or {}
    # THE LEVELS COME FROM boundary_ontology_v1, NOT FROM HERE. They are
    # defined once under `observability.require_for_automation`; writing
    # ["clear", "partial"] again in this file would be a second source of
    # truth that drifts the first time someone adds a level.
    req = cfg["_observability_required"]
    inadequate = [k for k, allowed in req.items() if obs.get(k) not in allowed]
    if inadequate:
        ok = False
        why.append("observability inadequate: "
                   + ", ".join(f"{k}={obs.get(k)!r}" for k in inadequate))
        blocked.append("unobservable_is_not_no_boundary")

    if bnd.get("nearby_point_transition_sec") is not None and \
            float(bnd["nearby_point_transition_sec"]) < \
            thr["no_nearby_point_transition_sec"]:
        ok = False
        why.append(f"a POINT_TRANSITION sits "
                   f"{bnd['nearby_point_transition_sec']}s away, inside the "
                   f"{thr['no_nearby_point_transition_sec']}s guard")

    compat, cwhy = semantic_compatible_with_continuity(sem, thr["cross_support_min"])
    used["cross_support"] = cwhy
    if not compat:
        ok = False
        why.append(f"semantic evidence not compatible with continuity: {cwhy}")

    # Identical labels are evidence of nothing on their own.
    if (sem.get("label_L") is not None
            and sem.get("label_L") == sem.get("label_R")
            and not compat):
        blocked.append("same_label_is_not_same_instance")
        reasons.append("label_L == label_R, which is not evidence of one "
                       "instance; the cross-support terms decide")

    if ok and not blocked:
        act = cfg["actions"]["AUTO_REJECT_CANDIDATE"]
        return {
            "action": "AUTO_REJECT_CANDIDATE",
            "precedence_block": "auto_reject_candidate",
            "instance_relation": "same_instance",
            "reasons": ["all five conditions met: "
                        + "; ".join(["strong NO_TRANSITION",
                                     "interaction continuous",
                                     "reset OBSERVED_ABSENT",
                                     "observability adequate",
                                     "semantic compatible with continuity"])],
            "blocked_by": [],
            "evidence_used": used,
            "enabled": bool(act.get("enabled")),
            "followup": list(act.get("followup") or []),
        }

    return {
        "action": "REVIEW",
        "precedence_block": "default",
        "instance_relation": None,
        "reasons": reasons + why,
        "blocked_by": blocked,
        "evidence_used": used,
        "followup": ["human"],
    }


# --------------------------------------------------------------------------
# SELF TEST. Each of the four hard rules gets a case that must make it fire.
# A rule that never fires is indistinguishable from a rule that is not wired
# in, and this project has shipped one of those: an observability gate that
# read its input with .get() and skipped itself when the answer was missing.
# --------------------------------------------------------------------------
CLEAR = {"hand_visibility": "clear", "interaction_visibility": "clear"}

CASES = [
    ("reset observed -> KEEP even with identical labels",
     {"morphology": "NO_TRANSITION", "morphology_confidence": 0.99,
      "interaction_continuity": "continuous",
      "release_reset_restart": PRESENT, "observability": CLEAR},
     {"label_L": "wipe counter", "label_R": "wipe counter",
      "support_L_labelR": 0.95, "support_R_labelL": 0.95},
     "KEEP_BOUNDARY", None),

    ("all five met -> AUTO_REJECT_CANDIDATE",
     {"morphology": "NO_TRANSITION", "morphology_confidence": 0.99,
      "interaction_continuity": "continuous",
      "release_reset_restart": ABSENT, "observability": CLEAR},
     {"label_L": "wipe counter", "label_R": "wipe counter",
      "support_L_labelR": 0.9, "support_R_labelL": 0.9},
     "AUTO_REJECT_CANDIDATE", None),

    ("reset NOT_OBSERVABLE -> REVIEW, not reject",
     {"morphology": "NO_TRANSITION", "morphology_confidence": 0.99,
      "interaction_continuity": "continuous",
      "release_reset_restart": UNKNOWN, "observability": CLEAR},
     {"label_L": "wipe counter", "label_R": "wipe counter",
      "support_L_labelR": 0.9, "support_R_labelL": 0.9},
     "REVIEW", "unseen_reset_is_not_absent_reset"),

    ("reset field missing entirely -> same as NOT_OBSERVABLE",
     {"morphology": "NO_TRANSITION", "morphology_confidence": 0.99,
      "interaction_continuity": "continuous", "observability": CLEAR},
     {"label_L": "a", "label_R": "a",
      "support_L_labelR": 0.9, "support_R_labelL": 0.9},
     "REVIEW", "unseen_reset_is_not_absent_reset"),

    ("reset arrives as False -> NOT treated as absent",
     {"morphology": "NO_TRANSITION", "morphology_confidence": 0.99,
      "interaction_continuity": "continuous",
      "release_reset_restart": False, "observability": CLEAR},
     {"label_L": "a", "label_R": "a",
      "support_L_labelR": 0.9, "support_R_labelL": 0.9},
     "REVIEW", "unseen_reset_is_not_absent_reset"),

    ("occluded -> REVIEW, not no-boundary",
     {"morphology": "NO_TRANSITION", "morphology_confidence": 0.99,
      "interaction_continuity": "continuous",
      "release_reset_restart": ABSENT,
      "observability": {"hand_visibility": "occluded",
                        "interaction_visibility": "clear"}},
     {"label_L": "a", "label_R": "a",
      "support_L_labelR": 0.9, "support_R_labelL": 0.9},
     "REVIEW", "unobservable_is_not_no_boundary"),

    ("identical labels but cross support fails -> REVIEW",
     {"morphology": "NO_TRANSITION", "morphology_confidence": 0.99,
      "interaction_continuity": "continuous",
      "release_reset_restart": ABSENT, "observability": CLEAR},
     {"label_L": "wipe counter", "label_R": "wipe counter",
      "support_L_labelR": 0.2, "support_R_labelL": 0.9},
     "REVIEW", "same_label_is_not_same_instance"),

    ("only one cross term supports -> not compatible",
     {"morphology": "NO_TRANSITION", "morphology_confidence": 0.99,
      "interaction_continuity": "continuous",
      "release_reset_restart": ABSENT, "observability": CLEAR},
     {"label_L": "rinse mug", "label_R": "wipe counter",
      "support_L_labelR": 0.9, "support_R_labelL": 0.1},
     "REVIEW", None),

    ("direction change is the only evidence -> REVIEW, never KEEP",
     {"morphology": "POINT_TRANSITION", "morphology_confidence": 0.99,
      "interaction_continuity": "continuous",
      "release_reset_restart": ABSENT, "observability": CLEAR,
      "only_evidence_is": ["motion_direction_change"]},
     {"label_L": "wipe counter", "label_R": "wipe counter",
      "support_L_labelR": 0.9, "support_R_labelL": 0.9},
     "REVIEW", "direction_change_is_not_boundary"),

    ("a nearby POINT_TRANSITION blocks the reject",
     {"morphology": "NO_TRANSITION", "morphology_confidence": 0.99,
      "interaction_continuity": "continuous",
      "release_reset_restart": ABSENT, "observability": CLEAR,
      "nearby_point_transition_sec": 1.2},
     {"label_L": "a", "label_R": "a",
      "support_L_labelR": 0.9, "support_R_labelL": 0.9},
     "REVIEW", None),
]


def self_test(cfg):
    bad = 0
    fired = Counter()
    for name, bnd, sem, want, want_block in CASES:
        got = decide(bnd, sem, cfg)
        ok = got["action"] == want
        if want_block:
            ok = ok and want_block in got["blocked_by"]
        for b in got["blocked_by"]:
            fired[b] += 1
        mark = "ok  " if ok else "FAIL"
        if not ok:
            bad += 1
        print(f"  {mark} {name}")
        print(f"       -> {got['action']}  blocked_by={got['blocked_by']}")
        if not ok:
            print(f"       expected {want}"
                  + (f" + {want_block}" if want_block else ""))
            for r in got["reasons"]:
                print(f"         · {r}")

    print(f"\n  hard rules that fired at least once:")
    for r in cfg["hard_rules"]:
        n = fired.get(r, 0)
        print(f"    {'ok  ' if n else 'NEVER FIRED'} {r}  ({n})")
    never = [r for r in cfg["hard_rules"] if not fired.get(r)]
    if never:
        bad += len(never)
        print(f"\n  !! {len(never)} hard rule(s) never fired. A rule that "
              f"never fires cannot be\n     distinguished from a rule that "
              f"is not wired in.")

    print(f"\n  AUTO_REJECT_CANDIDATE enabled: "
          f"{cfg['actions']['AUTO_REJECT_CANDIDATE']['enabled']}  "
          f"(must stay false until "
          f"{cfg['actions']['AUTO_REJECT_CANDIDATE']['requires_before_enable']})")
    if cfg["actions"]["AUTO_REJECT_CANDIDATE"]["enabled"]:
        bad += 1
        print("  !! it is enabled in the config and nothing has calibrated it.")
    return bad


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--events",
                    help="jsonl with {boundary: {...}, semantic: {...}} per "
                         "line")
    ap.add_argument("--out")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.config):
        raise SystemExit(f"{a.config} not found; the policy lives in the "
                         f"config, not in this file")
    cfg = load_config(a.config)
    print(f"joint policy v{cfg['version']}  frozen={cfg.get('frozen')}  "
          f"semantic role: {cfg['semantic_auditor']['role']}")
    print(f"  observability levels read from {BOUNDARY_ONTOLOGY}: "
          f"{ {k: sorted(v) for k, v in cfg['_observability_required'].items()} }")

    if a.self_test or not a.events:
        print("\nself test:")
        bad = self_test(cfg)
        raise SystemExit(1 if bad else 0)

    rows, counts = [], Counter()
    for line in open(a.events, encoding="utf-8"):
        if not line.strip():
            continue
        e = json.loads(line)
        d = decide(e.get("boundary") or {}, e.get("semantic") or {}, cfg)
        counts[d["action"]] += 1
        for b in d["blocked_by"]:
            counts[f"blocked:{b}"] += 1
        rows.append(dict(e, decision=d))

    print(f"\n{len(rows)} events")
    for k, v in counts.most_common():
        print(f"  {k:<44}{v:>5}  {v / max(len(rows), 1):.1%}")
    print(f"\n  AUTO_REJECT_CANDIDATE is measured, not executed: "
          f"enabled={cfg['actions']['AUTO_REJECT_CANDIDATE']['enabled']}. "
          f"The coverage\n  above is what a reject WOULD cover, which is the "
          f"number risk calibration needs.")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
