"""Gold v2 schema for the visual-auditor dev set (72 hand-labeled boundary
events).

This is the single source of truth for (a) which fields a visual auditor is
expected to produce, (b) the closed vocabulary allowed for each categorical
field, and (c) how to load the frozen gold labels + per-event annotation
context. Both the inference driver (`run_visual_auditor.py`) and the scorer
(`eval_auditor.py`) import from here so they can never drift apart.

The schema is deliberately ORTHOGONAL: temporal truth (what actually happened
in the video), semantic-label quality (is the annotation right), and policy
(what to do about it) are separate heads. A wrong GT boundary and a wrong GT
*label* are different failures and must be gradeable independently -- see the
`Schema_v2` sheet of audit_72_gold_v2_machine_readable.xlsx.

The gold set is a DEV/CALIBRATION set, not a training set: it exists to answer
"can a strong video model reproduce a human's structured judgment", not to be
fit. It is frozen -- do not relabel it to chase auditor agreement.
"""
from __future__ import annotations

import json
import os

# --- closed vocabularies (from Schema_v2) -----------------------------------
# Every categorical field an auditor emits must land in one of these. A value
# outside the set is a schema violation (counted separately in eval, never
# silently coerced).

TEMPORAL_TRUTH = ["valid", "spurious", "ambiguous", "unresolved"]

GT_BOUNDARY_RELATION = [
    "correctly_annotated", "missing_from_gt", "gt_offset",
    "spurious_gt", "multiple_valid", "unresolved",
]

MODEL_BOUNDARY_BEHAVIOR = [
    "correct_detection", "correct_rejection", "weak_response", "missed",
    "mislocalized", "duplicate", "spurious_motion_response",
    "decoder_missed", "not_evaluable",
]

CANDIDATE_BOUNDARY_VALIDITY = ["valid", "invalid", "ambiguous", "unresolved"]

LABEL_SUPPORT = ["supported", "contradicted", "uncertain"]

LABEL_COMPLETENESS = [
    "complete", "missing_secondary", "partially_correct",
    "wrong_object", "incorrect", "unresolved",
]

LABEL_GRANULARITY = [
    "appropriate", "too_coarse", "too_fine", "mixed",
    "not_applicable", "unresolved",
]

SEMANTIC_RELATION = [
    "same", "synonym", "parent", "child",
    "compatible", "incompatible", "unknown",
]

OBJECT_RELATION = ["same", "wrong_instance", "wrong_object", "unspecified", "unknown"]

BOUNDARY_CONTRASTIVE_ROLE = ["positive", "motion_hard_negative", "exclude"]

NAMING_CONTRASTIVE_ROLE = ["strong_positive", "soft_positive", "hard_negative", "exclude"]

TEMPORAL_CORRECTION_ACTION = [
    "keep", "add_boundary", "remove_boundary", "shift_boundary",
    "review_convention", "exclude",
]

SEMANTIC_CORRECTION_ACTION = [
    "keep", "expand_or_soften", "collapse_granularity",
    "repair_partial_label", "replace_label", "exclude_or_review",
]

REVIEW_CONFIDENCE = ["high", "medium", "low"]

# field -> allowed vocabulary, for the fields an auditor is scored on.
ENUM_FIELDS = {
    "temporal_truth": TEMPORAL_TRUTH,
    "gt_boundary_relation": GT_BOUNDARY_RELATION,
    "model_boundary_behavior": MODEL_BOUNDARY_BEHAVIOR,
    "candidate_boundary_validity": CANDIDATE_BOUNDARY_VALIDITY,
    "label_support": LABEL_SUPPORT,
    "label_completeness": LABEL_COMPLETENESS,
    "label_granularity": LABEL_GRANULARITY,
    "semantic_relation": SEMANTIC_RELATION,
    "object_relation": OBJECT_RELATION,
    "boundary_contrastive_role": BOUNDARY_CONTRASTIVE_ROLE,
    "naming_contrastive_role": NAMING_CONTRASTIVE_ROLE,
    "temporal_correction_action": TEMPORAL_CORRECTION_ACTION,
    "semantic_correction_action": SEMANTIC_CORRECTION_ACTION,
    "review_confidence": REVIEW_CONFIDENCE,
}

BOOL_FIELDS = ["no_valid_boundary", "boundary_time_unresolved",
               "corrected_target_known", "auto_proposal_eligible"]

# free-text / structured correction targets (graded loosely, not by exact enum)
TARGET_FIELDS = ["corrected_primary_verb", "corrected_secondary_verbs", "corrected_object"]
TIME_FIELDS = ["primary_corrected_boundary_time"]

# The subset that constitutes the auditor's actual predictions. Everything
# else in the gold row (paths, source_row_index, legacy_*) is bookkeeping.
AUDITOR_OUTPUT_FIELDS = (
    list(ENUM_FIELDS) + BOOL_FIELDS + TARGET_FIELDS + TIME_FIELDS
)


def is_valid_value(field: str, value) -> bool:
    """True iff `value` is in-vocabulary for `field` (enum/bool fields only)."""
    if field in ENUM_FIELDS:
        return value in ENUM_FIELDS[field]
    if field in BOOL_FIELDS:
        return isinstance(value, bool)
    return True  # target/time fields are free-form


# --- loading ----------------------------------------------------------------

def load_gold(path: str) -> list[dict]:
    """Load the frozen gold JSONL (one row per event)."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"no rows in gold file {path}")
    return rows


def load_context(path: str) -> dict[str, dict]:
    """Load per-event annotation context (the ORIGINAL segment labels the
    auditor is asked to verify), keyed by event_id."""
    ctx = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            ctx[d["event_id"]] = d
    return ctx


def default_gold_paths(repo_root: str | None = None) -> tuple[str, str]:
    """(gold_jsonl, context_jsonl) at their committed locations."""
    if repo_root is None:
        # src/auditor/gold_schema.py -> repo root is two levels up
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return (
        os.path.join(repo_root, "data", "gold", "audit_72_gold_v2.jsonl"),
        os.path.join(repo_root, "data", "gold", "audit_72_context.jsonl"),
    )
