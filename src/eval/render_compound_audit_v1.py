"""compound_audit_v1.0 -- frozen qualitative audit schema for ALL 25 compound
N9 items (not just failures, so success cases are available for comparison
against failure cases on the same visual factors). Auto-fills every field
the pipeline can compute (identity, GT/prediction, derived set-error type);
leaves every human-judgment field empty for two-pass annotation (Pass 1
blind to prediction, Pass 2 with prediction visible, Pass 3 review of
uncertain/disagreement cases) as specified in the schema.

set_error_type derivation (deterministic, matches the frozen definitions):
  exact_success               : predicted == true secondary set
  complete_omission           : true secondary non-empty, predicted empty
  overselection_true_retained : all true secondary retained, plus extras
  partial_multisecondary_miss : missing >=1 true, no extras, prediction non-empty
  substitution_or_mixed       : missing >=1 true AND has >=1 extra

Usage (server, no GPU needed):
    python -m src.eval.render_compound_audit_v1 \
        --n7_jsonl /workspace/tr1/results/naming/n7_scored.jsonl \
        --n9_jsonl /workspace/tr1/results/naming/n9_full_contrastive.jsonl \
        --out_dir /workspace/tr1/results/naming/compound_audit_v1 \
        --run_id N9_contrastive
"""
import argparse, csv, json, os, random, subprocess
from datetime import datetime, timezone

from decord import VideoReader
from PIL import Image, ImageDraw

from src.eval.eval_naming_n9b_final_comparison import grouped_cv_decode

HEADER = [
    "schema_version", "run_id", "model_name", "git_commit", "recording_id", "segment_id",
    "item_id", "item_type", "image_path", "clip_path", "fold_id", "group_id",
    "gt_canonical_name", "primary_action", "true_secondary", "predicted_secondary",
    "candidate_verbs", "secondary_scores", "threshold", "predicted_canonical_name",
    "n_true_secondary", "n_pred_secondary", "exact_success", "tp_count", "fp_count",
    "fn_count", "contains_any_true_secondary", "contains_all_true_secondary",
    "has_extra_secondary", "set_error_type",
    "secondary_duration_short", "secondary_duration_bin", "secondary_temporal_location",
    "secondary_repeated", "secondary_overlap_with_primary", "secondary_transition_only",
    "secondary_visibility", "secondary_evidence_frame_count", "motion_required_to_identify",
    "state_change_only", "occlusion_issue", "small_object_or_region", "camera_or_motion_blur",
    "contact_sheet_sampling_miss", "boundary_truncation", "adjacent_action_contamination",
    "segment_too_long", "segment_too_short", "compound_temporally_separable",
    "ontology_mismatch", "ontology_mismatch_type", "gt_label_visually_supported",
    "gt_label_over_specific", "gt_label_under_specific", "gt_primary_secondary_order_clear",
    "secondary_is_state_or_goal", "annotation_disagreement_risk",
    "same_action_family", "same_family_relation", "candidate_synonym_overlap",
    "candidate_granularity_mismatch", "object_affordance_prior_strong", "primary_visually_dominant",
    "failure_factors", "primary_failure_factor", "failure_type_free_text",
    "prediction_behavior", "model_error_explainability", "notes",
    "annotator_id", "annotation_pass1_complete", "annotation_pass2_complete",
    "annotator_confidence", "needs_second_review", "second_annotator_id",
    "review_disagreement", "adjudication_notes", "review_status", "audit_timestamp",
]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def set_error_type(true_secondary, predicted_secondary):
    missing = true_secondary - predicted_secondary
    extra = predicted_secondary - true_secondary
    if predicted_secondary == true_secondary:
        return "exact_success"
    if not predicted_secondary:
        return "complete_omission"
    if not missing and extra:
        return "overselection_true_retained"
    if not extra and missing:
        return "partial_multisecondary_miss"
    return "substitution_or_mixed"


def make_sheet(vr, start, end, frame_indices, gt_name, thumb_w=160):
    vfps = vr.get_avg_fps()
    n = len(vr)
    imgs = []
    for i in frame_indices:
        im = Image.fromarray(vr[min(i, n - 1)].asnumpy())
        w, h = im.size
        im = im.resize((thumb_w, int(h * thumb_w / w)))
        imgs.append(im)
    H = max(im.height for im in imgs)
    sheet = Image.new("RGB", (thumb_w * len(imgs), H + 30), "white")
    d = ImageDraw.Draw(sheet)
    for i, im in enumerate(imgs):
        sheet.paste(im, (i * thumb_w, 30))
    d.text((4, 4), f"GT: {gt_name}", fill="darkgreen")
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n7_jsonl", required=True)
    ap.add_argument("--n9_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_id", default="N9_contrastive")
    ap.add_argument("--model_name", default="Qwen3-VL-8B-Instruct")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    n7 = {(r["recording_id"], r["segment_idx"]): r for r in (json.loads(l) for l in open(a.n7_jsonl))}
    n9 = {(r["recording_id"], r["segment_idx"]): r for r in (json.loads(l) for l in open(a.n9_jsonl))}
    keys = sorted(set(n7) & set(n9))

    contrastive_recs = [{"recording_id": k[0], "segment_idx": k[1], "primary_letter": n9[k]["primary_letter"],
                         "gt_letters": n9[k]["gt_letters"], "con_scores": n9[k]["contrastive_scores"]}
                        for k in keys]
    oof_pred, taus = grouped_cv_decode(contrastive_recs, "con_scores", a.n_folds, a.seed)
    rng = random.Random(a.seed)
    recording_ids = sorted({r["recording_id"] for r in contrastive_recs})
    rng.shuffle(recording_ids)
    folds = [recording_ids[i::a.n_folds] for i in range(a.n_folds)]
    fold_of = {rid: fi for fi, fold in enumerate(folds) for rid in fold}

    os.makedirs(a.out_dir, exist_ok=True)
    commit = git_commit()
    now = datetime.now(timezone.utc).isoformat()
    vr_cache = {}
    rows = []

    for k in keys:
        r7, r9 = n7[k], n9[k]
        secondary_gt = set(r9["gt_letters"]) - {r9["primary_letter"]}
        if not secondary_gt:
            continue  # compound only
        rid, sid = k
        letters = "ABCDEF"[:len(r7["options"])]
        primary_verb = r7["options"][letters.index(r9["primary_letter"])]
        pred = oof_pred[k]
        candidate_verbs = [v for l, v in zip(letters, r7["options"]) if l != r9["primary_letter"]]
        true_secondary_verbs = [r7["options"][letters.index(l)] for l in sorted(secondary_gt)]
        pred_secondary_verbs = [r7["options"][letters.index(l)] for l in sorted(pred)]
        secondary_scores = {r7["options"][letters.index(l)]: round(s, 2)
                            for l, s in r9["contrastive_scores"].items()}

        vp = r7["video"]
        if vp not in vr_cache:
            vr_cache[vp] = VideoReader(vp, num_threads=1)
        vr = vr_cache[vp]
        sheet = make_sheet(vr, r7["start"], r7["end"], r7["frame_indices"], r7["gt_name"])
        image_path = os.path.join(a.out_dir, f"{rid}_seg{sid}.png")
        sheet.save(image_path)

        missing = secondary_gt - pred; extra = pred - secondary_gt
        predicted_canonical_name = f"{primary_verb}" + (
            " and " + " and ".join(pred_secondary_verbs) if pred_secondary_verbs else "")

        row = {
            "schema_version": "compound_audit_v1.0", "run_id": a.run_id, "model_name": a.model_name,
            "git_commit": commit, "recording_id": rid, "segment_id": sid, "item_id": f"{rid}_seg{sid}",
            "item_type": "compound", "image_path": image_path, "clip_path": "",
            "fold_id": fold_of[rid], "group_id": rid,
            "gt_canonical_name": r7["gt_name"], "primary_action": primary_verb,
            "true_secondary": json.dumps(true_secondary_verbs), "predicted_secondary": json.dumps(pred_secondary_verbs),
            "candidate_verbs": json.dumps(candidate_verbs), "secondary_scores": json.dumps(secondary_scores),
            "threshold": round(taus[fold_of[rid]], 2), "predicted_canonical_name": predicted_canonical_name,
            "n_true_secondary": len(secondary_gt), "n_pred_secondary": len(pred),
            "exact_success": pred == secondary_gt, "tp_count": len(pred & secondary_gt),
            "fp_count": len(extra), "fn_count": len(missing),
            "contains_any_true_secondary": bool(pred & secondary_gt),
            "contains_all_true_secondary": missing == set(),
            "has_extra_secondary": bool(extra),
            "set_error_type": set_error_type(secondary_gt, pred),
        }
        for col in HEADER:
            if col not in row:
                row[col] = ""
        rows.append(row)
        print(f"{rid} seg{sid}: {row['set_error_type']:28s} true={true_secondary_verbs} "
              f"pred={pred_secondary_verbs} -> {image_path}")

    csv_path = os.path.join(a.out_dir, "compound_audit_v1.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    n_exact = sum(r["exact_success"] for r in rows)
    print(f"\nwrote {len(rows)} compound audit rows -> {csv_path}")
    print(f"set_error_type distribution:")
    from collections import Counter
    for et, c in Counter(r["set_error_type"] for r in rows).most_common():
        print(f"  {et:28s} {c}")
    print(f"\nPass 1 (blind to prediction) fields to fill first: secondary_duration_bin, "
          f"secondary_temporal_location, secondary_overlap_with_primary, secondary_transition_only, "
          f"secondary_visibility, motion_required_to_identify, state_change_only, occlusion_issue, "
          f"contact_sheet_sampling_miss, boundary_truncation, adjacent_action_contamination, "
          f"compound_temporally_separable, ontology_mismatch, ontology_mismatch_type, "
          f"gt_label_visually_supported, gt_label_over_specific, secondary_is_state_or_goal, "
          f"same_action_family, same_family_relation, candidate_synonym_overlap, "
          f"object_affordance_prior_strong, primary_visually_dominant")
    print(f"Pass 2 (prediction visible) fields: failure_factors, primary_failure_factor, "
          f"failure_type_free_text, prediction_behavior, model_error_explainability, notes")
    print(f"Pass 3: review rows where any Pass1/2 field = uncertain, or "
          f"ontology_mismatch=yes, or gt_label_visually_supported=no")


if __name__ == "__main__":
    main()
