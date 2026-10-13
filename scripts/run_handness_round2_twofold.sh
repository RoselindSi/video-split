#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/workspace/handness_v2/v2_reviewed/code}
ROOT=${ROOT:-/workspace/handness_v2}
OUT=${OUT:-$ROOT/bodypart_hardneg_v2/cv_combined}
PYTHON=${PYTHON:-python}

BASE=$ROOT/v2_reviewed/manifest.csv
ROUND1=$ROOT/bodypart_hardneg_v1
ROUND2=$ROOT/bodypart_hardneg_v2
FEATURES=(
  --features "$ROOT/qwen_features_0.pt"
  --features "$ROOT/qwen_features_1.pt"
  --features "$ROOT/qwen_features_2.pt"
  --features "$ROOT/qwen_features_3.pt"
)
CHECKPOINTS=()

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

prepare_fold() {
  local name=$1
  local heldout=$2
  local training_negative=$3
  local fold=$OUT/$name
  mkdir -p "$fold/w16"

  "$PYTHON" scripts/apply_handness_review.py \
    --manifest "$BASE" \
    --review "$ROUND1/human_item_review.csv" \
    --out "$fold/round1.csv"
  "$PYTHON" scripts/apply_handness_review.py \
    --manifest "$fold/round1.csv" \
    --review "$ROUND2/human_item_review.csv" \
    --out "$fold/manifest.csv" \
    --split-override "$heldout:val" \
    --split-override "$training_negative:train" \
    --split-override R0826_105135:train \
    --split-override R26_0824_113206:train \
    --split-override R26_0827_131646:train \
    --split-override R26_0827_143242:val
}

train_fold() {
  local name=$1
  local heldout=$2
  local fold=$OUT/$name
  local work=$fold/w16
  local pids=()

  if [[ ! -f "$work/head_seed0.pt" || ! -f "$work/head_seed1.pt" || \
        ! -f "$work/head_seed2.pt" ]]; then
    for seed in 0 1 2; do
      CUDA_VISIBLE_DEVICES=$seed "$PYTHON" -m src.semhand.qwen_feature_head train \
        --manifest "$fold/manifest.csv" \
        "${FEATURES[@]}" \
        --out "$work/head_seed${seed}.pt" \
        --hard-example-weight 16 \
        --exclude-val-rec "$heldout" \
        --seed "$seed" \
        >"$work/seed${seed}.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
  else
    echo "reusing checkpoints: $work"
  fi

  CHECKPOINTS=()
  for seed in 0 1 2; do
    CHECKPOINTS+=(--checkpoint "$work/head_seed${seed}.pt")
  done
  "$PYTHON" -m src.semhand.qwen_feature_head score \
    --manifest "$fold/manifest.csv" \
    "${FEATURES[@]}" \
    "${CHECKPOINTS[@]}" \
    --split val \
    --predictions "$work/val.csv" \
    >"$work/val_metrics.json"
  "$PYTHON" -m src.semhand.qwen_feature_head score \
    --manifest "$fold/manifest.csv" \
    "${FEATURES[@]}" \
    "${CHECKPOINTS[@]}" \
    --split test \
    --predictions "$work/test.csv" \
    >"$work/test_metrics.json"
  "$PYTHON" scripts/select_handness_threshold.py \
    --validation "$work/val.csv" \
    --test "$work/test.csv" \
    --minimum-recall .995 \
    --heldout-rec "$heldout" \
    --out "$work/threshold.json" \
    >"$work/threshold.log"
  "$PYTHON" scripts/evaluate_handness_hardneg_cv.py \
    --predictions "$work/val.csv" \
    --candidates "$ROUND1/candidates.csv" \
    --candidates "$ROUND2/candidates.csv" \
    --review "$ROUND1/human_track_review.csv" \
    --review "$ROUND2/human_track_review.csv" \
    --heldout-rec "$heldout" \
    --threshold-report "$work/threshold.json" \
    --out "$work/hardneg_tracks.json" \
    >"$work/hardneg_tracks.log"
}

prepare_fold fold_hold_r0824 R0824_120917 R0828_105127
prepare_fold fold_hold_r0828 R0828_105127 R0824_120917
train_fold fold_hold_r0824 R0824_120917
train_fold fold_hold_r0828 R0828_105127

echo "completed: $OUT"
