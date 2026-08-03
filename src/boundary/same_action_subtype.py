"""Heuristic motion-subtype tagger for the clean-145 dev set's 37
same_action_internal_motion negatives, built to answer one question before
committing to C2 (slow/motif latent) vs C3 (hand-object local crop):

  Does C1's predictive-surprise failure concentrate in direction-reversal /
  periodic-repetition events (-> supports C2: the predictor is tracking a
  low-level motion PHASE, not a high-level action identity) or in
  camera/viewpoint-heavy events (-> supports C3: global pooled features are
  being polluted by egomotion, not by the interaction itself)?

Data source: `gold.notes`, the free-text audit rationale already written for
every negative when the pair taxonomy was built (data/gold/audit_188_gold_v2.jsonl).
These notes were written by a human/VLM audit BEFORE this diagnostic
question existed, so they are not circular with respect to it -- they
describe what is visually happening, not why C1 might succeed or fail.

Subtypes (priority order matters -- a note can mention several things; the
FIRST rule that matches wins, ordered most-specific-first so a generic
"repetitive" mention doesn't swallow a more specific "flip" or "regrasp"
case):

  object_pose_flip      the object itself is rotated/flipped to a new face
                         (e.g. slipper upper -> insole side)
  direction_reversal     gross-motor direction/orientation change (wiping
                         direction, back-and-forth, orientation change)
  regrasp_reposition     ONLY the grip/finger/hand configuration changes;
                         notes explicitly say no idle interval / object
                         switch / restart / goal change (the plastic-wrap
                         sealing family is almost entirely this)
  periodic_repetition    explicit circular/cyclic/periodic motion, repeated
                         peaks from the same repeating gesture
  spatial_phase_shift    same continuous action moves to a different spatial
                         region of the SAME object (kettle lid -> body, bowl
                         rim position A -> B)
  camera_or_scene_noise  note attributes the false signal to something other
                         than the person's motion (rare in this set, kept for
                         completeness -- if empty, that itself is informative:
                         it means camera-egomotion pollution is not what the
                         ORIGINAL auditors saw as the driver here)
  other                  no rule matched; read the note by hand

Usage:
    python -m src.boundary.same_action_subtype \
        --gold data/gold/audit_188_gold_v2.jsonl \
        --context data/gold/audit_188_context.jsonl \
        --pair_labels data/gold/pair_labels_v1.csv \
        --out data/gold/same_action_subtype_v1.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re

SUBTYPES = ["object_pose_flip", "direction_reversal", "regrasp_reposition",
            "periodic_repetition", "spatial_phase_shift", "camera_or_scene_noise",
            "other"]

_RULES = [
    ("object_pose_flip", re.compile(
        r"\bflip(s|ped|ping)?\b|upper side|insole side|footbed side|"
        r"rotat(e|ed|ing) the (object|item)", re.I)),
    ("direction_reversal", re.compile(
        r"direction change|wiping direction|back-and-forth|back and forth|"
        r"side to side|orientation change|reverses?|bottle orientation", re.I)),
    ("periodic_repetition", re.compile(
        r"circular motion|circular wiping|periodic|repeated (small )?peaks|"
        r"cyclic|repetitive-motion sensitivity|multiple peaks", re.I)),
    ("spatial_phase_shift", re.compile(
        r"spatial phase|different (part|region|position) of the (same|bowl)|"
        r"moves? toward the (same|upper)|lid and top cap toward|"
        r"shift to a different part", re.I)),
    ("regrasp_reposition", re.compile(
        # The dominant template in this dev set's plastic-wrap-sealing family:
        # "only the <grip/finger/position> change(s); there is no idle
        # interval, [disengagement,] object switch, restart, or change in
        # [the overall] action goal" -- match the STRUCTURE (no idle interval
        # + object switch/action goal, i.e. the auditor's explicit ruling
        # that nothing but local hand configuration changed), not specific
        # body-part nouns, since the noun varies (grasp/grip/finger/hand
        # position/pulling position/pressing location...) but this
        # no-idle-interval ruling is consistent.
        r"no idle interval.{0,80}(object switch|action goal)|"
        r"\bregrasp\b|repositioning of the fingers|"
        r"repetitive-motion response within the ongoing action", re.I)),
    ("camera_or_scene_noise", re.compile(
        r"camera (motion|movement|shift)|viewpoint change|head (motion|turn)",
        re.I)),
]


def classify(note: str) -> tuple[str, str]:
    """Returns (subtype, matched_snippet). First matching rule wins."""
    for subtype, pat in _RULES:
        m = pat.search(note or "")
        if m:
            lo, hi = max(0, m.start() - 20), min(len(note), m.end() + 20)
            return subtype, note[lo:hi].strip()
    return "other", (note or "")[:60]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/gold/audit_188_gold_v2.jsonl")
    ap.add_argument("--context", default="data/gold/audit_188_context.jsonl")
    ap.add_argument("--pair_labels", default="data/gold/pair_labels_v1.csv")
    ap.add_argument("--out", default="data/gold/same_action_subtype_v1.csv")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, ".")
    from src.auditor import gold_schema as S
    from src.boundary import pair_taxonomy as T

    gold = S.load_gold(a.gold)
    labels = T.load_pair_labels(a.pair_labels)

    events = []
    for g in gold:
        role = g.get("boundary_contrastive_role")
        if role not in ("positive", "motion_hard_negative"):
            continue
        events.append({"event_id": g["event_id"],
                       "recording_id": g.get("recording_id"),
                       "y": 1 if role == "positive" else 0, "gold": g})
    kept = T.apply_to_events(events, labels, verbose=False)
    neg = [e for e in kept if e["y"] == 0]
    print(f"clean-145: {len(kept)} total; tagging {len(neg)} "
          f"same_action_internal_motion negatives")

    rows = []
    for e in neg:
        note = e["gold"].get("notes", "") or ""
        sub, snip = classify(note)
        rows.append({"event_id": e["event_id"], "recording_id": e["recording_id"],
                     "subtype": sub, "evidence": snip, "notes": note})

    from collections import Counter
    dist = Counter(r["subtype"] for r in rows)
    print("subtype distribution:", dict(dist))
    if dist.get("other", 0):
        print(f"  !! {dist['other']} events matched no rule -- read these by hand "
              f"before trusting the cross-tab, they may need a new rule or a "
              f"manual tag")

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["event_id", "recording_id", "subtype",
                                          "evidence", "notes"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
