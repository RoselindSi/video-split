"""Visual-auditor inference driver (three-pass, per-event).

For each of the 72 frozen gold events, runs an independent video-vision model
through three passes and fuses the result into ONE record whose fields match
the Gold v2 schema, so `eval_auditor.py` can score it against the human label
field-by-field. This answers the question the mentor put first: can a strong
video model reproduce a human's structured audit -- BEFORE we trust it to
mine training pairs or train any second-stage model.

  Pass A (blind)     : clip only, no label/GT/category -> visual truth, and the
                       motion-change-vs-semantic-change distinction.
  Pass B (semantic)  : Pass A + the ORIGINAL annotation label -> label quality
                       + corrected target.
  Pass C (temporal)  : Pass A + GT time / model time / adjacent labels -> is a
                       real boundary here, and how do GT and the model relate.

Confidence is NOT the model's own verbal confidence. It is a consistency score
across `--repeats` runs (fps-jittered) plus a blind-vs-conditioned agreement
check (does Pass C's temporal_truth agree with Pass A's action-changed call).
Only high-consistency, resolved cases get auto_proposal_eligible=True.

Usage
-----
Smoke test (no GPU / no video, proves the plumbing + scorer end to end):
    python -m src.auditor.run_visual_auditor --backend mock \
        --out /tmp/auditor_pred.jsonl && \
    python -m src.auditor.eval_auditor --pred /tmp/auditor_pred.jsonl

Real run (server, videos + weights present). Blind pass on an Instruct model,
reasoning passes on a Think model (per the auditor design):
    python -m src.auditor.run_visual_auditor --backend qwen \
        --model_id_a  Qwen/Qwen3.5-VL-27B-Instruct \
        --model_id_bc Qwen/Qwen3.5-VL-27B-Think \
        --media_dir /workspace/tr1/results/boundary/error_audit/media \
        --repeats 3 --score_plot \
        --out /workspace/tr1/results/auditor/auditor_pred.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter

from . import gold_schema as S
from . import prompts as P
from . import vision_backends as VB
from . import derive_fields as D


# --- small aggregation helpers ---------------------------------------------

def _mode(values):
    """Majority value + agreement fraction; ignores None/empty."""
    vals = [v for v in values if v not in (None, "", [])]
    if not vals:
        return None, 0.0
    c = Counter(map(_hashable, vals))
    top, n = c.most_common(1)[0]
    return _unhashable(top), n / len(vals)


def _hashable(v):
    return json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v


def _unhashable(v):
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def _median_time(values):
    nums = []
    for v in values:
        try:
            if v is not None:
                nums.append(float(v))
        except (TypeError, ValueError):
            pass
    return round(statistics.median(nums), 2) if nums else None


def _norm_enum(field, value):
    """Coerce a raw model value to the closed vocabulary; None if out-of-vocab
    (tracked as a schema miss in eval, never silently mapped)."""
    if value is None:
        return None
    v = str(value).strip().lower().replace(" ", "_")
    return v if v in S.ENUM_FIELDS.get(field, []) else None


# --- per-pass runners (with repeats) ---------------------------------------

def run_pass(backend, system, user, *, video, images, base_fps, repeats,
             temperature, keys):
    """Run a pass `repeats` times with fps jitter; return list of parsed dicts."""
    outs = []
    fps_variants = [base_fps, base_fps * 1.5, base_fps * 0.67, base_fps * 2.0]
    for i in range(repeats):
        fps = fps_variants[i % len(fps_variants)]
        temp = temperature if i > 0 else 0.0  # first run greedy, rest sampled
        raw = backend.generate(system, user, video=video, images=images,
                               fps=fps, temperature=temp, mock_keys=keys)
        outs.append(P.parse_json_reply(raw))
    return outs


def run_pass_multi(backends, system, user, *, video, images, base_fps, repeats,
                   temperature, keys):
    """Like run_pass, but across one or more backends -- e.g. a small AND a
    large Instruct checkpoint, since running the SAME model on itself
    (`repeats`) only measures whether it is internally consistent, not
    whether it is right: a systematic bias reproduces identically across
    fps-jittered repeats and reads as high confidence. Two DIFFERENT models
    agreeing is a much stronger signal. Returns (merged_runs_for_fusion,
    [per_backend_runs...]) -- the merged list feeds the existing _mode()
    consensus unchanged; the per-backend lists let the caller additionally
    check cross-model agreement."""
    per_backend = []
    merged = []
    for backend in backends:
        runs = run_pass(backend, system, user, video=video, images=images,
                        base_fps=base_fps, repeats=repeats,
                        temperature=temperature, keys=keys)
        per_backend.append(runs)
        merged.extend(runs)
    return merged, per_backend


def cross_model_agreement_values(per_backend_value_lists):
    """1.0 if all backends' dominant value agrees, 0.0 if any disagree, None
    if there's only one backend's worth of values (not applicable). Takes
    raw value lists (not run dicts) so it works for atomic VLM fields AND
    for per-repeat DERIVED fields (e.g. label_support) alike."""
    if len(per_backend_value_lists) < 2:
        return None
    dominants = [_mode(vals)[0] for vals in per_backend_value_lists]
    dominants = [d for d in dominants if d is not None]
    if len(dominants) < 2:
        return None
    return 1.0 if len(set(dominants)) == 1 else 0.0


def cross_model_agreement(per_backend_runs, field):
    """1.0 if all backends' dominant answer for `field` agrees, 0.0 if any
    disagree, None if there's only one backend (not applicable)."""
    return cross_model_agreement_values([[r.get(field) for r in runs] for runs in per_backend_runs])


def derive_semantic_per_repeat(runs, label_text):
    """Apply derive_semantic_fields to each repeat's ATOMIC observation
    (never the raw VLM output directly -- there is no VLM-predicted
    label_support/etc. any more, see derive_fields.py)."""
    return [D.derive_semantic_fields(
                r.get("observed_primary_verb"), r.get("observed_secondary_verbs"),
                r.get("observed_object"), r.get("additional_action_visible"), label_text)
            for r in runs]


def derive_boundary_per_repeat(runs, source_category, pred_time, pred_score):
    """Apply derive_boundary_fields to each repeat's temporal_truth answer."""
    out = []
    for r in runs:
        truth = _norm_enum("temporal_truth", r.get("temporal_truth")) or "unresolved"
        rel, behav = D.derive_boundary_fields(truth, source_category, pred_time, pred_score)
        out.append({"gt_boundary_relation": rel, "model_boundary_behavior": behav})
    return out


def consensus_pass_a(runs):
    changed, _ = _mode([r.get("semantic_action_changed") for r in runs])
    motion, _ = _mode([r.get("motion_change_without_semantic_change") for r in runs])
    return {
        "before_action": _mode([r.get("before_action") for r in runs])[0],
        "after_action": _mode([r.get("after_action") for r in runs])[0],
        "object_before": _mode([r.get("object_before") for r in runs])[0],
        "object_after": _mode([r.get("object_after") for r in runs])[0],
        "state_change": (runs[0] or {}).get("state_change"),
        "semantic_action_changed": changed,
        "motion_change_without_semantic_change": motion,
        "candidate_boundary_time": _median_time([r.get("candidate_boundary_time") for r in runs]),
        "observed_secondary_actions": (runs[0] or {}).get("observed_secondary_actions") or [],
        "visual_evidence": _mode([r.get("visual_evidence") for r in runs])[0],
    }


# --- deterministic fusion into the Gold v2 fields --------------------------

_TRUTH_TO_VALIDITY = {"valid": "valid", "spurious": "invalid",
                      "ambiguous": "ambiguous", "unresolved": "unresolved"}
_REL_TO_TEMPORAL_ACTION = {
    "correctly_annotated": "keep", "missing_from_gt": "add_boundary",
    "spurious_gt": "remove_boundary", "gt_offset": "shift_boundary",
    "multiple_valid": "review_convention", "unresolved": "exclude",
}


def _semantic_action(support, completeness, granularity):
    if support == "contradicted" or completeness in ("incorrect", "wrong_object"):
        return "replace_label"
    if completeness == "missing_secondary":
        return "expand_or_soften"
    if granularity == "too_coarse":
        return "collapse_granularity"
    if completeness == "partially_correct":
        return "repair_partial_label"
    if support == "uncertain" or completeness == "unresolved":
        return "exclude_or_review"
    return "keep"


def _naming_role(support, completeness, granularity):
    if support == "uncertain" or completeness == "unresolved":
        return "exclude"
    if support == "contradicted" or completeness in ("incorrect", "wrong_object"):
        return "hard_negative"
    if completeness in ("missing_secondary", "partially_correct") or granularity in ("too_coarse", "too_fine", "mixed"):
        return "soft_positive"
    return "strong_positive"


def _boundary_role(truth, motion_change):
    if truth == "valid":
        return "positive"
    if truth == "spurious":
        return "motion_hard_negative" if motion_change == "yes" else "exclude"
    return "exclude"  # ambiguous / unresolved


def fuse(pa, pb, pb_derived, pc, pc_derived):
    """pb/pc hold the VLM's ATOMIC observations (consensus across repeats);
    pb_derived/pc_derived hold the deterministically-derived judgment fields
    (consensus across per-repeat derivations) -- see derive_fields.py. Fusing
    from derived-consensus, rather than asking the VLM for these fields
    directly, is the fix for the worst-performing fields in the full-72 runs
    (gt_boundary_relation/model_boundary_behavior needed decode-mechanics
    info no video shows; label_support/semantic_relation showed a strong
    anchoring bias when asked directly of the model)."""
    truth = _norm_enum("temporal_truth", pc.get("temporal_truth")) or "unresolved"
    rel = _norm_enum("gt_boundary_relation", pc_derived.get("gt_boundary_relation")) or "unresolved"
    behav = _norm_enum("model_boundary_behavior", pc_derived.get("model_boundary_behavior")) or "not_evaluable"
    support = _norm_enum("label_support", pb_derived.get("label_support")) or "uncertain"
    comp = _norm_enum("label_completeness", pb_derived.get("label_completeness")) or "unresolved"
    gran = _norm_enum("label_granularity", pb_derived.get("label_granularity")) or "unresolved"
    sem_rel = _norm_enum("semantic_relation", pb_derived.get("semantic_relation")) or "unknown"
    obj_rel = _norm_enum("object_relation", pb_derived.get("object_relation")) or "unknown"
    motion = pa.get("motion_change_without_semantic_change")
    ctime = _median_time([pc.get("corrected_boundary_time"), pa.get("candidate_boundary_time")]) \
        if truth in ("valid", "ambiguous") else _median_time([pc.get("corrected_boundary_time")])

    return {
        "temporal_truth": truth,
        "gt_boundary_relation": rel,
        "model_boundary_behavior": behav,
        "candidate_boundary_validity": _TRUTH_TO_VALIDITY[truth],
        "label_support": support,
        "label_completeness": comp,
        "label_granularity": gran,
        "semantic_relation": sem_rel,
        "object_relation": obj_rel,
        "corrected_primary_verb": pb.get("observed_primary_verb"),
        "corrected_secondary_verbs": pb.get("observed_secondary_verbs") or [],
        "corrected_object": pb.get("observed_object"),
        "primary_corrected_boundary_time": ctime,
        "no_valid_boundary": bool(truth in ("spurious", "unresolved") and ctime is None),
        "boundary_time_unresolved": bool(truth == "unresolved"),
        "boundary_contrastive_role": _boundary_role(truth, motion),
        "naming_contrastive_role": _naming_role(support, comp, gran),
        "temporal_correction_action": _REL_TO_TEMPORAL_ACTION[rel],
        "semantic_correction_action": _semantic_action(support, comp, gran),
    }


def calibrated_confidence(pa_runs, pb_derived_runs, pc_runs, fused,
                          cross_model_temporal=None, cross_model_label=None):
    """Consistency-based confidence (not the model's stated confidence).

    pb_derived_runs: per-repeat DERIVED label_support (see
    derive_semantic_per_repeat) -- there is no VLM-predicted label_support
    any more, so repeat-agreement is measured on the derived judgment.

    cross_model_temporal / cross_model_label: from cross_model_agreement(),
    None unless a second B/C backend was actually used. Same-model repeats
    (temporal_agree/semantic_agree below) measure internal consistency only
    -- a systematic bias reproduces identically across fps-jittered repeats
    of ONE model and reads as high confidence despite being wrong (observed
    on the server: hard-slice(2) was 0/17 in the 'high confidence' bucket).
    Cross-model agreement is real diversity and is weighted in when present.
    """
    comps = {}
    # agreement across repeats on the two anchor fields
    comps["temporal_agree"] = _mode([r.get("temporal_truth") for r in pc_runs])[1]
    comps["semantic_agree"] = _mode([d.get("label_support") for d in pb_derived_runs])[1]
    # blind (Pass A) vs conditioned (Pass C) agreement
    changed = _mode([r.get("semantic_action_changed") for r in pa_runs])[0]
    truth = fused["temporal_truth"]
    if changed in ("yes", "no") and truth in ("valid", "spurious"):
        expect_valid = changed == "yes"
        comps["blind_conditioned_agree"] = 1.0 if (expect_valid == (truth == "valid")) else 0.0
    else:
        comps["blind_conditioned_agree"] = 0.5
    if cross_model_temporal is not None:
        comps["cross_model_temporal_agree"] = cross_model_temporal
    if cross_model_label is not None:
        comps["cross_model_label_agree"] = cross_model_label
    overall = sum(comps.values()) / len(comps)
    bin_ = "high" if overall >= 0.8 else ("medium" if overall >= 0.5 else "low")
    eligible = (overall >= 0.8 and truth not in ("ambiguous", "unresolved")
                and fused["label_support"] != "uncertain")
    # With a second model available, never call something high-confidence if
    # the two models actually disagreed on the anchor fields -- this is the
    # exact case the whole cross-model addition exists to catch.
    if cross_model_temporal == 0.0 or cross_model_label == 0.0:
        bin_ = "low" if bin_ == "high" else bin_
        eligible = False
    return overall, bin_, eligible, comps


# --- main -------------------------------------------------------------------

def audit_event(backend_a, backend_bc, gold_row, ctx, args, backend_bc2=None):
    center = ctx.get("pred_time") or ctx.get("gt_time") or 0.0
    half = args.clip_window / 2.0
    clip_start, clip_end = max(0.0, center - half), center + half

    video = None
    if backend_a.name != "mock":
        clip = gold_row.get("clip_path")
        if args.media_dir and clip:
            clip = os.path.join(args.media_dir, os.path.basename(clip))
        video = clip
    images = ()
    if args.score_plot and backend_a.name != "mock":
        sp = gold_row.get("score_plot_path")
        if args.media_dir and sp:
            sp = os.path.join(args.media_dir, os.path.basename(sp))
        if sp and os.path.exists(sp):
            images = (sp,)

    # Pass A -- blind
    pa_runs = run_pass(backend_a, P.PASS_A_SYSTEM, P.build_pass_a(clip_start, clip_end),
                       video=video, images=(), base_fps=args.fps, repeats=args.repeats,
                       temperature=args.temperature, keys=list(P.PASS_A_KEYS))
    pa = consensus_pass_a(pa_runs)

    backends_bc = [backend_bc] + ([backend_bc2] if backend_bc2 is not None else [])

    label_text = ctx.get("containing_segment_label")
    source_category = gold_row.get("source_category")

    # Pass B -- ATOMIC visual observation only (reasoning model[s]). Judgment
    # fields (label_support etc.) are DERIVED below, never asked of the VLM.
    pb_user = P.build_pass_b(pa, ctx.get("containing_segment_label"),
                             ctx.get("prev_segment_label"), ctx.get("next_segment_label"))
    pb_runs, pb_per_backend = run_pass_multi(
        backends_bc, P.PASS_B_SYSTEM, pb_user, video=video, images=(),
        base_fps=args.fps, repeats=args.repeats,
        temperature=args.temperature, keys=list(P.PASS_B_KEYS))
    pb = {k: _mode([r.get(k) for r in pb_runs])[0] for k in P.PASS_B_KEYS}
    pb_derived_runs = derive_semantic_per_repeat(pb_runs, label_text)
    pb_derived = {k: _mode([d.get(k) for d in pb_derived_runs])[0] for k in
                 ("label_support", "label_completeness", "label_granularity",
                  "semantic_relation", "object_relation")}

    # Pass C -- ATOMIC temporal_truth only (reasoning model[s]). gt_boundary_
    # relation/model_boundary_behavior are DERIVED below from temporal_truth +
    # source_category, never asked of the VLM (they need decode-mechanics
    # info -- was a peak suppressed by NMS vs below threshold -- that isn't
    # visible in a clip at all).
    pc_user = P.build_pass_c(pa, ctx.get("gt_time"), ctx.get("pred_time"),
                             ctx.get("pred_score"), ctx.get("prev_segment_label"),
                             ctx.get("next_segment_label"), bool(images))
    pc_runs, pc_per_backend = run_pass_multi(
        backends_bc, P.PASS_C_SYSTEM, pc_user, video=video, images=images,
        base_fps=args.fps, repeats=args.repeats,
        temperature=args.temperature, keys=list(P.PASS_C_KEYS))
    pc = {k: _mode([r.get(k) for r in pc_runs])[0] for k in P.PASS_C_KEYS}
    pc_derived_runs = derive_boundary_per_repeat(
        pc_runs, source_category, ctx.get("pred_time"), ctx.get("pred_score"))
    pc_derived = {k: _mode([d.get(k) for d in pc_derived_runs])[0] for k in
                 ("gt_boundary_relation", "model_boundary_behavior")}

    fused = fuse(pa, pb, pb_derived, pc, pc_derived)
    cm_temporal = cross_model_agreement(pc_per_backend, "temporal_truth")
    pb_derived_per_backend = [derive_semantic_per_repeat(runs, label_text) for runs in pb_per_backend]
    cm_label = cross_model_agreement_values(
        [[d.get("label_support") for d in derived] for derived in pb_derived_per_backend])
    overall, bin_, eligible, comps = calibrated_confidence(
        pa_runs, pb_derived_runs, pc_runs, fused,
        cross_model_temporal=cm_temporal, cross_model_label=cm_label)
    fused["review_confidence"] = bin_
    fused["auto_proposal_eligible"] = eligible

    return {
        "event_id": gold_row["event_id"],
        "recording_id": gold_row.get("recording_id"),
        **fused,
        "_confidence": {"overall": round(overall, 3), "components": comps},
        "_pass_a": pa,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["mock", "qwen"], default="mock")
    ap.add_argument("--model_id_a", help="Pass A (blind) model; Instruct recommended")
    ap.add_argument("--model_id_bc", help="Pass B/C (reasoning) model; Think recommended")
    ap.add_argument("--model_id_bc2", help="OPTIONAL second Pass B/C model for cross-model "
                    "agreement (e.g. a larger checkpoint alongside model_id_bc). Same-model "
                    "--repeats only measures self-consistency, which is blind to a systematic "
                    "bias (it reproduces identically every repeat); a second, differently-"
                    "sized/trained model disagreeing is a real diversity signal and downgrades "
                    "confidence/auto_proposal_eligible when the two disagree on the anchor "
                    "fields. Roughly doubles Pass B/C runtime.")
    ap.add_argument("--model_id", help="single model id used for all passes (overridden by _a/_bc)")
    ap.add_argument("--gold", help="gold jsonl (default: committed data/gold/...)")
    ap.add_argument("--context", help="context jsonl (default: committed data/gold/...)")
    ap.add_argument("--media_dir", help="directory holding the clip/plot media (basename-joined)")
    ap.add_argument("--score_plot", action="store_true", help="attach the score-plot image to Pass C")
    ap.add_argument("--repeats", type=int, default=1, help="runs per pass for the consistency signal (>=3 recommended for real runs)")
    ap.add_argument("--fps", type=float, default=2.0, help="base sampling fps for video")
    ap.add_argument("--clip_window", type=float, default=8.0, help="approx clip span (s) used only to give the model an absolute-time frame of reference")
    ap.add_argument("--temperature", type=float, default=0.5, help="sampling temperature for repeats after the first")
    ap.add_argument("--limit", type=int, default=0, help="only first N events (debug)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gold_path, ctx_path = S.default_gold_paths()
    gold_path = a.gold or gold_path
    ctx_path = a.context or ctx_path
    gold = S.load_gold(gold_path)
    ctx = S.load_context(ctx_path)
    if a.limit:
        gold = gold[:a.limit]

    backend_bc2 = None
    if a.backend == "mock":
        backend_a = backend_bc = VB.build_backend("mock")
        if a.model_id_bc2:
            backend_bc2 = VB.build_backend("mock")
    else:
        id_a = a.model_id_a or a.model_id
        id_bc = a.model_id_bc or a.model_id or id_a
        backend_a = VB.build_backend("qwen", id_a)
        backend_bc = backend_a if id_bc == id_a else VB.build_backend("qwen", id_bc)
        if a.model_id_bc2:
            backend_bc2 = VB.build_backend("qwen", a.model_id_bc2)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for row in gold:
            eid = row["event_id"]
            rec = audit_event(backend_a, backend_bc, row, ctx.get(eid, {}), a, backend_bc2=backend_bc2)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            print(f"[{n}/{len(gold)}] {eid} -> truth={rec['temporal_truth']} "
                  f"support={rec['label_support']} conf={rec['_confidence']['overall']} "
                  f"auto={rec['auto_proposal_eligible']}", file=sys.stderr)

    try:
        from src.eval.run_manifest import write_manifest
        write_manifest(a.out, input_paths=[gold_path, ctx_path],
                       extra={"backend": a.backend, "model_id_a": a.model_id_a,
                              "model_id_bc": a.model_id_bc, "model_id_bc2": a.model_id_bc2,
                              "repeats": a.repeats, "n_events": n})
    except Exception as e:
        print(f"[manifest] skipped ({e})", file=sys.stderr)
    print(f"wrote {n} auditor records -> {a.out}")


if __name__ == "__main__":
    main()
