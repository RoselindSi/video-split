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


# Pass B and Pass C are deliberately ATOMIC: they ask ONLY questions
# answerable from the video itself. Fields like label_support/
# semantic_relation/gt_boundary_relation/model_boundary_behavior used to be
# asked directly and were the worst-performing fields in two full-72 runs --
# not because the model wasn't smart enough, but because they aren't visual
# questions (semantic_relation needs the frozen naming ontology's own
# verb-comparison logic; gt_boundary_relation/model_boundary_behavior need
# decode-mechanics info -- was a candidate peak suppressed by NMS vs never
# above threshold -- that isn't visible in an 8-second clip at all). Those
# are now DERIVED deterministically in derive_fields.py from these atomic
# observations + the existing structured context. See that module's
# docstring for the full rationale.

PASS_B_KEYS = {
    "reasoning": "2-3 sentences, written BEFORE any field below: state the blind description's verb+object, state the label's verb+object, then explicitly say where they agree or disagree. Do not restate the label as if it were the conclusion.",
    "observed_primary_verb": "the single main verb for what the video shows here (your own observation -- do not just copy the label's verb)",
    "observed_secondary_verbs": "list of additional verbs/actions the video shows co-occurring or immediately following (may be empty)",
    "observed_object": "the main object noun being manipulated, as you observe it",
    "additional_action_visible": "yes | no -- is there a second, distinct action visible beyond the single primary verb above (this drives missing_secondary detection -- do not answer based on the label, only on what you see)",
    "conflicts_with_blind_description": "yes | no -- does the ORIGINAL LABEL disagree with your own observation above on verb, object, or granularity? Decide this after observing, not before.",
}

PASS_C_KEYS = {
    "reasoning": "2-3 sentences, written BEFORE any field below: state what Pass A already concluded (semantic_action_changed / motion_change_without_semantic_change), then state whether the GT/prediction context here changes that conclusion and why. Do not jump straight to 'valid' because a GT time was given.",
    "agrees_with_pass_a_motion_judgment": "yes | no | pass_a_uncertain -- does your temporal_truth conclusion below align with what the blind Pass A description already implied (semantic_action_changed / motion_change_without_semantic_change)? Decide this FIRST. If you override Pass A, your rationale must name the NEW evidence that justifies it.",
    "temporal_truth": " | ".join(S.TEMPORAL_TRUTH) + " -- does a real action-level boundary exist near here, independent of GT/model. This is the ONLY judgment call in this pass; do not also decide how the model or annotation behaved -- that is derived separately from this answer.",
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
    "You report what a video ACTUALLY SHOWS, for later comparison against a "
    "human-written label by separate, deterministic code -- you do not judge "
    "the label yourself. You are given (1) a blind visual description "
    "produced without seeing the label, and (2) the original annotation "
    "label(s), shown only as extra context for your OWN observation (e.g. to "
    "know which object/segment is meant), not as something to confirm or "
    "deny.\n\n"
    "KNOWN FAILURE MODE, correct for it: earlier runs of this exact pipeline "
    "asked the model to directly judge 'does this label match', and it showed "
    "a strong bias toward rubber-stamping the label as correct even when the "
    "blind description clearly disagreed. Report your OWN independent "
    "observation of verb/object/secondary-actions first; only the "
    "conflicts_with_blind_description field compares to the label, and it "
    "must reflect what you actually observed, not what would make the label "
    "'probably close enough'.\n\n"
    "WORKED EXAMPLE of the reasoning style expected (own observation "
    "disagrees with a topically-close label): blind description says the "
    "clip shows picking up a folded tissue and beginning to unfold/open it; "
    "the label says 'fold tissue into a compact rectangle'. reasoning: "
    "'Blind description: unfolding a tissue. Label: folding a tissue. These "
    "are OPPOSITE actions on the same object, not a paraphrase.' -> "
    "observed_primary_verb='unfold', observed_object='tissue', "
    "conflicts_with_blind_description=yes. Report 'unfold', not 'fold', even "
    "though both actions involve a tissue."
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
    "You decide ONE thing: whether a real ACTION-LEVEL boundary exists at a "
    "point in an egocentric video ('temporal_truth'), about the VIDEO ONLY -- "
    "whether a genuine action transition occurs -- independent of whether the "
    "GT marked it or the model fired. A boundary can be real but unannotated, "
    "or annotated where no real transition occurs. Repetitive or reversing "
    "motion inside one action is NOT a boundary. Do NOT try to also decide "
    "how the annotation or the detection model behaved (whether a candidate "
    "peak was suppressed, below threshold, etc.) -- that is decode-mechanics "
    "information not visible in a video clip, and is derived separately from "
    "your temporal_truth answer by code, not by you.\n\n"
    "KNOWN FAILURE MODE, correct for it: earlier runs of this exact pipeline "
    "showed a strong bias toward calling everything 'valid' (or hedging to "
    "'ambiguous') even when the blind Pass A description already said the "
    "motion was ordinary within-action motion (motion_change_without_"
    "semantic_change=yes, semantic_action_changed=no). Pass A's judgment was "
    "made WITHOUT seeing the GT time or the model's prediction, so it is "
    "strong independent evidence -- do not let seeing an annotated GT time "
    "talk you into 'valid' by default. If Pass A already concluded this is "
    "just motion inside one action, your default should be temporal_truth= "
    "'spurious' unless you find CONCRETE ADDITIONAL evidence, not already "
    "visible to Pass A, that a genuine action changed.\n\n"
    "WORKED EXAMPLE of the reasoning style expected (temporal_truth is "
    "'spurious' despite a GT time being given): Pass A already said "
    "motion_change_without_semantic_change=yes for two remote-control flips "
    "separated by a brief static hold; the GT time falls inside that static "
    "hold, between the two real flips. reasoning: 'Pass A found no semantic "
    "change here -- the GT sits in a stable no-action gap between two real "
    "flips elsewhere. A GT timestamp existing is not evidence a boundary is "
    "real.' -> agrees_with_pass_a_motion_judgment=yes, temporal_truth="
    "spurious. Do not default to 'valid' just because an annotated GT time "
    "was provided in the evidence above."
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
