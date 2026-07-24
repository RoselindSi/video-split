"""Three-pass prompt construction for the visual auditor.

The whole point of the three-pass design is to stop the auditor from
*anchoring* on the annotation it is supposed to be checking. If you show a
strong VLM the clip, the GT label, the evaluator's error category, and ask
"is this label right?", it tends to rationalize the label it was given. So:

  Pass A  BLIND. Only the raw clip. No GT label, no model prediction, no
          score curve, no evaluator category. The model describes what it
          actually sees and -- critically -- separates "the picture is moving"
          from "the action changed". This is the un-anchored visual truth.

  Pass B  SEMANTIC. Pass A's description + the ORIGINAL annotation label(s).
          Now verify the label: is it supported, complete, the right
          granularity, and what would the corrected target be. (Recommended
          model: a reasoning/"think" model, since this is a comparison task.)

  Pass C  TEMPORAL. Pass A's description + GT time / model-predicted time /
          adjacent labels. Judge whether an action-level boundary truly exists
          here, how the GT relates to it, and how the current model behaved.

Each pass returns STRICT JSON with a fixed key set. We parse the JSON out of
the reply (models sometimes wrap it in prose / code fences) in
`parse_json_reply`. Field values are normalized/validated downstream against
`gold_schema` -- prompts ask for the closed vocabulary but never trust it.
"""
from __future__ import annotations

import json
import re

from . import gold_schema as S


# --- output contracts (documented per pass, embedded into the prompt) -------

PASS_A_KEYS = {
    "before_action": "short verb phrase for the action just BEFORE the middle of the clip",
    "after_action": "short verb phrase for the action just AFTER the middle of the clip",
    "object_before": "main object being manipulated before",
    "object_after": "main object being manipulated after",
    "state_change": "one sentence: what physically changes (object/hand/contact state)",
    "semantic_action_changed": "yes | no | unclear -- did the high-level ACTION INTENT change (not just motion)",
    "motion_change_without_semantic_change": "yes | no | unclear -- is there strong visual motion but the SAME ongoing action (e.g. repetitive wiping, direction reversal, regrasp)",
    "candidate_boundary_time": "absolute seconds of the single most likely action boundary in this clip, or null",
    "candidate_boundary_interval": '{"start": s, "end": s} absolute-second window in which a boundary plausibly lies, or null',
    "observed_secondary_actions": "list of short additional/co-occurring actions visible (may be empty)",
    "visual_evidence": "clear | partial | insufficient -- how well the clip shows the relevant moment",
}

PASS_B_KEYS = {
    "label_support": " | ".join(S.LABEL_SUPPORT),
    "label_completeness": " | ".join(S.LABEL_COMPLETENESS),
    "label_granularity": " | ".join(S.LABEL_GRANULARITY),
    "semantic_relation": " | ".join(S.SEMANTIC_RELATION) + " (original vs corrected action)",
    "object_relation": " | ".join(S.OBJECT_RELATION) + " (original vs corrected object)",
    "corrected_primary_verb": "the correct primary verb given the video (may equal the label's verb)",
    "corrected_secondary_verbs": "list of secondary verbs the video supports (may be empty)",
    "corrected_object": "the correct object noun",
    "rationale": "one sentence grounded in the visual evidence, not in the label wording",
}

PASS_C_KEYS = {
    "temporal_truth": " | ".join(S.TEMPORAL_TRUTH) + " -- does a real action-level boundary exist near here, independent of GT/model",
    "gt_boundary_relation": " | ".join(S.GT_BOUNDARY_RELATION),
    "model_boundary_behavior": " | ".join(S.MODEL_BOUNDARY_BEHAVIOR),
    "candidate_boundary_validity": " | ".join(S.CANDIDATE_BOUNDARY_VALIDITY),
    "corrected_boundary_time": "absolute seconds of the true boundary, or null if none/unresolved",
    "corrected_boundary_interval": '{"start": s, "end": s} or null',
    "rationale": "one sentence",
}


def _contract_block(keys: dict[str, str]) -> str:
    lines = ["Return ONLY a JSON object with EXACTLY these keys:"]
    for k, desc in keys.items():
        lines.append(f'  "{k}": <{desc}>')
    lines.append("No prose before or after. No markdown fences. Use null for unknown, not empty string.")
    return "\n".join(lines)


# --- Pass A: blind visual description ---------------------------------------

PASS_A_SYSTEM = (
    "You are a meticulous egocentric-video action analyst. You are shown a "
    "short first-person (head-mounted camera) clip of a hand task. You have NO "
    "access to any annotation, label, model prediction, or 'correct answer' -- "
    "describe only what you can actually see.\n\n"
    "Your single most important job is to distinguish a SEMANTIC ACTION CHANGE "
    "(the person switches to a different action/goal -- e.g. stops scrubbing a "
    "mug and starts rinsing it; picks up a new object) from mere MOTION CHANGE "
    "within one ongoing action (repetitive wiping, a direction reversal, "
    "regrasping, tool repositioning). Strong visual motion does NOT imply an "
    "action boundary."
)


def build_pass_a(clip_start: float, clip_end: float) -> str:
    return (
        f"This clip spans roughly {clip_start:.1f}s to {clip_end:.1f}s of the "
        f"recording (absolute times). Watch the whole clip, then judge whether "
        f"the ACTION changes and, if so, when.\n\n"
        + _contract_block(PASS_A_KEYS)
    )


# --- Pass B: semantic label verification ------------------------------------

PASS_B_SYSTEM = (
    "You verify whether a human-written action label matches what a video "
    "actually shows. You are given (1) a blind visual description produced "
    "without seeing the label, and (2) the original annotation label(s). Judge "
    "the LABEL against the VISUAL EVIDENCE -- do not assume the label is "
    "correct, and do not rewrite it into something the video does not show. It "
    "is common for a label to be a correct-but-coarser (parent) description of "
    "a finer action; that is 'too_coarse', not 'incorrect'."
)


def build_pass_b(pass_a: dict, containing_label, prev_label, next_label) -> str:
    ctx = {
        "blind_visual_description": pass_a,
        "original_labels": {
            "segment_containing_this_moment": containing_label,
            "previous_segment_label": prev_label,
            "next_segment_label": next_label,
        },
    }
    return (
        "Evidence:\n" + json.dumps(ctx, ensure_ascii=False, indent=2) + "\n\n"
        "Decide how well the original label captures the action(s) and object "
        "in the video, and what the corrected target should be.\n\n"
        + _contract_block(PASS_B_KEYS)
    )


# --- Pass C: temporal / boundary verification -------------------------------

PASS_C_SYSTEM = (
    "You decide whether a real ACTION-LEVEL boundary exists at a point in an "
    "egocentric video, and how the annotation and a detection model relate to "
    "it. 'temporal_truth' is about the VIDEO ONLY -- whether a genuine action "
    "transition occurs -- independent of whether the GT marked it or the model "
    "fired. A boundary can be real but unannotated (missing_from_gt), or "
    "annotated where no real transition occurs (spurious_gt). Repetitive or "
    "reversing motion inside one action is NOT a boundary."
)


def build_pass_c(pass_a: dict, gt_time, pred_time, pred_score,
                 prev_label, next_label, has_score_plot: bool) -> str:
    ctx = {
        "blind_visual_description": pass_a,
        "annotated_gt_boundary_time": gt_time,
        "model_predicted_time": pred_time,
        "model_peak_score": pred_score,
        "label_before_this_point": prev_label,
        "label_after_this_point": next_label,
    }
    plot = ("A probability-vs-time plot is also attached (GT dashed, prediction "
            "marked). Use it only as a hint about the model, never as ground "
            "truth for whether a boundary exists.\n" if has_score_plot else "")
    return (
        "Evidence:\n" + json.dumps(ctx, ensure_ascii=False, indent=2) + "\n\n"
        + plot +
        "First decide temporal_truth from the video, THEN describe how the GT "
        "and the model relate to that truth.\n\n"
        + _contract_block(PASS_C_KEYS)
    )


# --- reply parsing ----------------------------------------------------------

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_reply(text: str) -> dict:
    """Extract the first JSON object from a model reply, tolerating code
    fences and surrounding prose. Returns {} on failure (callers treat an
    unparseable pass as 'unresolved')."""
    if not text:
        return {}
    t = text.strip()
    # strip ```json ... ``` fences if present
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t.strip())
    try:
        return json.loads(t)
    except Exception:
        pass
    m = _JSON_OBJ.search(t)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}
