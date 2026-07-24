"""Deterministic rule layer: turn ATOMIC visual observations from the VLM
into the full Gold v2 judgment fields, instead of asking the VLM to directly
free-guess fields that are not actually visual questions.

Why this exists: the first two full-72 runs asked the VLM to emit
`gt_boundary_relation` / `model_boundary_behavior` directly, and it was the
worst-scoring field both times (8B: correct on the correct_detection/
spurious_motion_response rows but near-random on the missed/weak-response
distinction; 32B on a 15-event subset: 3/15 correct, worse than 8B's
already-poor showing). The reason is structural, not a model-capacity
problem: a VLM watching an 8-second clip cannot know whether a candidate
peak was suppressed by NMS vs never cleared the decode threshold -- that is
decode-mechanics information, not something visible in the video. The
boundary audit's own `source_category` (which automatic-metric bucket this
event came from -- e.g. 'missed_weak_signal', 'false_mid_segment', 'late')
already encodes exactly the structural fact needed, so this derives the
field instead of asking the VLM to guess it.

Two derivations:

  derive_boundary_fields(temporal_truth, source_category, pred_time, pred_score)
      -> (gt_boundary_relation, model_boundary_behavior)
      Needs no VLM guess beyond `temporal_truth` itself (a genuine visual
      question the VLM is well suited for).

  derive_semantic_fields(observed_primary_verb, observed_secondary_verbs,
                         observed_object, additional_action_visible, label_text)
      -> label_support / label_completeness / label_granularity /
         semantic_relation / object_relation
      Compares what Pass A/B actually SAW (never anchored on the label) to
      the ORIGINAL annotation text, using the frozen naming ontology's own
      verb/object normalization (src/analysis/build_ontology.py) instead of
      asking the VLM to self-report a parent/child/compatible relation. This
      is a best-effort v1 rule, not a claim of parity with human judgment --
      cases it cannot resolve return 'unresolved'/'unknown' rather than
      guessing, which is the honest signal for "route to human review."
"""
from __future__ import annotations

from src.analysis.build_ontology import (
    norm_verb, extract_verbs, extract_object, OBJECT_NORM,
    STRICT_INVERSE, CONTEXTUAL_INVERSE, GENERIC_VERBS,
)

# --- boundary fields ---------------------------------------------------

# source_category -> whether THIS event's own timestamp is anchored on a real
# GT boundary (missed_*/late/exact/early -- gt_time in context is the real
# annotated point) or on a model prediction with no GT match at this exact
# point (duplicate/false_* -- gt_time in context is only the NEAREST
# reference GT, not a real annotation at this point).
_GT_ANCHORED_CATEGORIES = {
    "missed_signal_present_not_top", "missed_weak_signal",
    "late", "exact", "early",
}
_WEAK_SIGNAL_FLOOR = 0.15  # below this, the nearest local score is noise-level


def derive_boundary_fields(temporal_truth, source_category, pred_time, pred_score):
    """-> (gt_boundary_relation, model_boundary_behavior)."""
    if temporal_truth == "unresolved":
        return "unresolved", "not_evaluable"
    if temporal_truth == "ambiguous":
        return "multiple_valid", "not_evaluable"

    gt_anchored = source_category in _GT_ANCHORED_CATEGORIES
    has_matched_pred = pred_time is not None

    if gt_anchored:
        if temporal_truth == "valid":
            if has_matched_pred:
                behav = "duplicate" if source_category == "duplicate" else "correct_detection"
            else:
                score = pred_score or 0.0
                behav = "weak_response" if score >= _WEAK_SIGNAL_FLOOR else "missed"
            return "correctly_annotated", behav
        else:  # spurious
            behav = "spurious_motion_response" if has_matched_pred else "correct_rejection"
            return "spurious_gt", behav
    else:  # FP-anchored: no real GT annotated at THIS point
        if temporal_truth == "valid":
            behav = "duplicate" if source_category == "duplicate" else "correct_detection"
            return "missing_from_gt", behav
        else:
            # correctly absent GT at a spurious model candidate
            return "correctly_annotated", "spurious_motion_response"


# --- semantic fields -----------------------------------------------------

def _canon_verb(word):
    w = (word or "").strip().lower()
    if not w:
        return None
    for t in w.split():
        c = norm_verb(t)
        if c:
            return c
    return None


def _canon_object(word):
    w = (word or "").strip().lower()
    return OBJECT_NORM.get(w)


def _verb_pair_relation(observed_verb, label_verb):
    """same / incompatible / unknown for ONE observed vs ONE label verb, using
    the frozen ontology's canonical forms + inverse-action tables."""
    ov_raw, lb_raw = (observed_verb or "").strip().lower(), (label_verb or "").strip().lower()
    if not ov_raw or not lb_raw:
        return "unknown"
    ov_c, lb_c = _canon_verb(observed_verb), _canon_verb(label_verb)
    if ov_raw == lb_raw or (ov_c and lb_c and ov_c == lb_c):
        return "same"
    if ov_c and lb_c:
        if lb_c in STRICT_INVERSE.get(ov_c, []) or ov_c in STRICT_INVERSE.get(lb_c, []):
            return "incompatible"
        if lb_c in CONTEXTUAL_INVERSE.get(ov_c, []) or ov_c in CONTEXTUAL_INVERSE.get(lb_c, []):
            return "incompatible"
        return "unknown"
    # at least one side didn't resolve to a canonical ontology verb -- fall
    # back to substring containment as a weak same/paraphrase signal
    if ov_raw in lb_raw or lb_raw in ov_raw:
        return "same"
    return "unknown"


def derive_semantic_fields(observed_primary_verb, observed_secondary_verbs,
                           observed_object, additional_action_visible, label_text):
    """label_text: the ORIGINAL annotation string for the segment containing
    this moment (e.g. 'Rinse mug under running water'). May be None (no
    label context available) -> everything unresolved/not_applicable."""
    if not label_text:
        return {
            "label_support": "uncertain", "label_completeness": "unresolved",
            "label_granularity": "not_applicable", "semantic_relation": "unknown",
            "object_relation": "unknown",
        }

    label_verbs = extract_verbs(label_text)
    label_obj, _tool, _container, _unresolved = extract_object(label_text)

    verb_rel = "unknown"
    for lv in (label_verbs or [None]):
        r = _verb_pair_relation(observed_primary_verb, lv)
        if r == "same":
            verb_rel = "same"
            break
        if r == "incompatible" and verb_rel == "unknown":
            verb_rel = "incompatible"

    ov_obj_c = _canon_object(observed_object)
    if label_obj and ov_obj_c:
        obj_rel = "same" if ov_obj_c == label_obj else "wrong_object"
    else:
        obj_rel = "unknown"

    # a label whose verb is one of the ontology's known-generic verbs
    # (manipulate/present/display/...) is a real (correct) but coarser
    # description whenever the observed verb is itself more specific --
    # this is 'too_coarse', not 'incorrect'.
    label_is_generic = any(lv in GENERIC_VERBS for lv in (label_verbs or []))
    observed_is_generic = _canon_verb(observed_primary_verb) in GENERIC_VERBS

    if verb_rel == "incompatible" or obj_rel == "wrong_object":
        support = "contradicted"
        completeness = "incorrect" if verb_rel == "incompatible" else "wrong_object"
        granularity = "not_applicable"
    elif verb_rel == "same" and obj_rel in ("same", "unknown"):
        support = "supported"
        completeness = "missing_secondary" if additional_action_visible in ("yes", True) else "complete"
        granularity = "too_coarse" if (label_is_generic and not observed_is_generic) else "appropriate"
    elif label_is_generic and not observed_is_generic and obj_rel in ("same", "unknown"):
        # generic label verb (manipulate/adjust/move/...) doesn't strictly
        # match but also doesn't CONTRADICT the specific observed verb, and
        # the object lines up -- a vague-but-not-wrong parent description.
        support = "supported"
        completeness = "missing_secondary" if additional_action_visible in ("yes", True) else "complete"
        granularity = "too_coarse"
    else:
        # verb didn't resolve either way -- don't guess a support verdict
        support = "uncertain"
        completeness = "unresolved"
        granularity = "unresolved"

    return {
        "label_support": support,
        "label_completeness": completeness,
        "label_granularity": granularity,
        "semantic_relation": verb_rel,
        "object_relation": obj_rel,
    }
