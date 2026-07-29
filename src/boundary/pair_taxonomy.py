"""Temporal pair taxonomy: what KIND of left/right relationship an audited
event actually is, and therefore how it may enter a loss.

Why this replaces the old binary role. The contrastive adapter's nested CV
gave train AUROC 0.758 / test AUROC 0.499 (chance). Train being mediocre --
not high -- rules out plain cross-recording overfitting: the model could not
fit even its own training pairs. Manual video review of the audited events
found why: `motion_hard_negative` was a bucket holding at least five
different phenomena, all forced to satisfy the same "pull left/right
together" constraint:

  - genuinely the same action, just internal motion (pressing/smoothing
    plastic wrap around a bowl rim)          -> pulling together is CORRECT
  - a gradual phase transition with no instant switch (locating the wrap
    edge, then starting to pull it)          -> pulling together is WRONG,
                                                but so is pushing apart
  - the camera/viewpoint moved, hands left frame, no new object interaction
                                             -> a nuisance, not a semantic
                                                relation at all
  - the action left the frame / is unobservable (crumpling plastic, then
    hands and object exit; the bin and the "discard" are never visible)
                                             -> no evidence either way
  - an annotation-convention split rather than a visible change

Forcing one geometry on all of these is self-contradictory supervision, and
a contrastive loss is far more sensitive to it than a plain BCE head -- which
is consistent with the adapter being WORSE than the v1 logistic baseline
(0.576 vs 0.702) rather than merely no better.

So every event gets two new fields:

  temporal_pair_subtype : what the video shows (observation)
  pair_supervision      : how it may be used in training (policy)

kept separate so that changing training policy later does not require
re-watching video. SUBTYPE_TO_SUPERVISION is the default mapping; a human
may override `pair_supervision` per row.
"""
from __future__ import annotations

import csv
import json

# --- what the video shows ---------------------------------------------------
SUBTYPES = [
    "sharp_visible_transition",     # clear, fast change of interaction/state
    "same_action_internal_motion",  # one ongoing action, motion only
    "gradual_phase_transition",     # real change, but no instantaneous switch
    "camera_or_viewpoint_shift",    # dominant change is global/camera motion
    "visibility_or_offscreen",      # relevant moment not observable in frame
    "annotation_convention",        # split exists by labelling rule, not vision
    "ambiguous",                    # cannot be resolved from the clip
]

# --- how it may enter a loss ------------------------------------------------
SUPERVISIONS = [
    "strong_separate",      # left/right must be far apart  (y=1 in clean set)
    "strong_align",         # left/right must be close      (y=0 in clean set)
    "soft_transition",      # real but gradual -- no hard push/pull yet
    "nuisance_invariance",  # representation should IGNORE this difference
    "exclude",              # not usable as supervision at all
]

SUBTYPE_TO_SUPERVISION = {
    "sharp_visible_transition": "strong_separate",
    "same_action_internal_motion": "strong_align",
    "gradual_phase_transition": "soft_transition",
    # camera shift starts as `exclude`, not `nuisance_invariance`: using it for
    # invariance training needs a camera-motion signal we do not compute yet,
    # and mislabelling a real transition that merely coincides with head motion
    # would teach the model to ignore true boundaries.
    "camera_or_viewpoint_shift": "exclude",
    "visibility_or_offscreen": "exclude",
    "annotation_convention": "exclude",
    "ambiguous": "exclude",
}

# Only these two participate in the first clean binary experiment.
CLEAN_BINARY = {"strong_separate": 1, "strong_align": 0}


# Subtypes assigned by direct human video review (recorded here so the
# judgement is versioned with the code rather than living only in chat).
# These pre-fill the relabelling sheet; everything else starts blank.
REVIEWED_SUBTYPES = {
    "recording_000406_false_mid_segment_t595.5": (
        "same_action_internal_motion",
        "continuous pressing/smoothing of plastic wrap along the bowl rim; "
        "intent and interaction target unchanged"),
    "recording_000406_exact_t490.3": (
        "gradual_phase_transition",
        "pulling wrap gradually becomes laying it over the bowl; no clear "
        "instantaneous switch point"),
    "recording_000406_missed_weak_signal_t556.6": (
        "gradual_phase_transition",
        "holding roll, locating the film edge, gradually starting to pull; "
        "continuous process"),
    "recording_000419_missed_weak_signal_t688.9": (
        "visibility_or_offscreen",
        "continues crumpling plastic, then hands and object leave frame; bin "
        "and the discard action are never visible"),
    "recording_000438_missed_signal_present_not_top_t181.0": (
        "camera_or_viewpoint_shift",
        "hand leaves frame while the whole view/scene shifts; no visible new "
        "object interaction"),
    "recording_000419_false_near_edge_t309.5": (
        "sharp_visible_transition",
        "previous action ends, hand reaches for and picks up the wrap roll; "
        "interaction target changes"),
    "recording_000438_missed_signal_present_not_top_t182.0": (
        "sharp_visible_transition",
        "wrap roll enters frame and starts being picked up and manipulated"),
}


def load_pair_labels(path):
    """Read a relabelled sheet (CSV or JSONL). Returns
    {event_id: {"temporal_pair_subtype", "pair_supervision", ...}}.
    `pair_supervision` falls back to SUBTYPE_TO_SUPERVISION when blank, so a
    human only has to fill the subtype unless they want to override policy."""
    rows = []
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    else:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        eid = (r.get("event_id") or "").strip()
        if not eid:
            continue
        sub = (r.get("temporal_pair_subtype") or "").strip() or None
        sup = (r.get("pair_supervision") or "").strip() or None
        if sub and sub not in SUBTYPES:
            raise ValueError(f"{eid}: unknown temporal_pair_subtype {sub!r} "
                             f"(allowed: {SUBTYPES})")
        if sup and sup not in SUPERVISIONS:
            raise ValueError(f"{eid}: unknown pair_supervision {sup!r} "
                             f"(allowed: {SUPERVISIONS})")
        if sup is None and sub is not None:
            sup = SUBTYPE_TO_SUPERVISION[sub]
        out[eid] = {"temporal_pair_subtype": sub, "pair_supervision": sup,
                    "notes": r.get("notes", "")}
    return out


def apply_to_events(events, pair_labels, keep=("strong_separate", "strong_align"),
                    verbose=True):
    """Filter `events` to those whose pair_supervision is in `keep`, and reset
    each event's y from the supervision (NOT from the old binary role).
    Unlabelled events are dropped with a count, never silently kept under
    their old label -- that would reintroduce the contradictory supervision
    this taxonomy exists to remove."""
    out, n_unlabelled, dropped = [], 0, {}
    for e in events:
        lab = pair_labels.get(e["event_id"])
        if lab is None or lab.get("pair_supervision") is None:
            n_unlabelled += 1
            continue
        sup = lab["pair_supervision"]
        if sup not in keep:
            dropped[sup] = dropped.get(sup, 0) + 1
            continue
        ev = dict(e)
        ev["pair_supervision"] = sup
        ev["temporal_pair_subtype"] = lab.get("temporal_pair_subtype")
        ev["y"] = CLEAN_BINARY[sup] if sup in CLEAN_BINARY else e["y"]
        out.append(ev)
    if verbose:
        print(f"pair taxonomy: kept {len(out)} events ({keep}); "
              f"dropped by supervision {dropped}; unlabelled {n_unlabelled}")
        if n_unlabelled:
            print(f"  !! {n_unlabelled} events have no subtype yet -- relabel them "
                  f"before treating this subset as complete")
    return out
