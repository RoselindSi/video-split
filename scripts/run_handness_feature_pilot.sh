#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/workspace/handness_v2/v2_reviewed/code}
ROOT=${ROOT:-/workspace/handness_v2}
FOLD=${FOLD:-$ROOT/bodypart_hardneg_v2/cv_combined/fold_hold_r0824}
OUT=${OUT:-$ROOT/bodypart_hardneg_v2/pixel_feature_pilot_r0824}
PYTHON=${PYTHON:-python}
HELDOUT=R0824_120917
FEATURES=(
  --features "$ROOT/qwen_features_0.pt"
  --features "$ROOT/qwen_features_1.pt"
  --features "$ROOT/qwen_features_2.pt"
  --features "$ROOT/qwen_features_3.pt"
)

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=3 "$PYTHON" -m src.semhand.pixel_feature_student train \
  --manifest "$FOLD/manifest.csv" \
  "${FEATURES[@]}" \
  --out "$OUT/student_seed0.pt" \
  --hard-example-weight 4 \
  --feature-weight 1 \
  --exclude-val-rec "$HELDOUT" \
  --seed 0 \
  >"$OUT/train.log" 2>&1
CUDA_VISIBLE_DEVICES=3 "$PYTHON" -m src.semhand.pixel_feature_student score \
  --manifest "$FOLD/manifest.csv" \
  --checkpoint "$OUT/student_seed0.pt" \
  --split val \
  --predictions "$OUT/val.csv" \
  >"$OUT/val_metrics.json"
CUDA_VISIBLE_DEVICES=3 "$PYTHON" -m src.semhand.pixel_feature_student score \
  --manifest "$FOLD/manifest.csv" \
  --checkpoint "$OUT/student_seed0.pt" \
  --split test \
  --predictions "$OUT/test.csv" \
  >"$OUT/test_metrics.json"
"$PYTHON" scripts/select_handness_threshold.py \
  --validation "$OUT/val.csv" \
  --test "$OUT/test.csv" \
  --minimum-recall .995 \
  --heldout-rec "$HELDOUT" \
  --out "$OUT/threshold.json" \
  >"$OUT/threshold.log"
"$PYTHON" scripts/evaluate_handness_hardneg_cv.py \
  --predictions "$OUT/val.csv" \
  --candidates "$ROOT/bodypart_hardneg_v1/candidates.csv" \
  --candidates "$ROOT/bodypart_hardneg_v2/candidates.csv" \
  --review "$ROOT/bodypart_hardneg_v1/human_track_review.csv" \
  --review "$ROOT/bodypart_hardneg_v2/human_track_review.csv" \
  --heldout-rec "$HELDOUT" \
  --threshold-report "$OUT/threshold.json" \
  --out "$OUT/hardneg_tracks.json" \
  >"$OUT/hardneg_tracks.log"

echo "completed: $OUT"
