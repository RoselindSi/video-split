#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/workspace/handness_v2/v2_reviewed/code}
ROOT=${ROOT:-/workspace/handness_v2}
SOURCE=${SOURCE:-$ROOT/bodypart_hardneg_v2/cv_combined}
OUT=${OUT:-$ROOT/bodypart_hardneg_v2/pixel_cv}
PYTHON=${PYTHON:-python}

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

run_fold() {
  local name=$1
  local heldout=$2
  local manifest=$SOURCE/$name/manifest.csv
  local work=$OUT/$name
  local pids=()
  local checkpoints=()
  mkdir -p "$work"

  if [[ ! -f "$work/student_seed0.pt" || \
        ! -f "$work/student_seed1.pt" || \
        ! -f "$work/student_seed2.pt" ]]; then
    for seed in 0 1 2; do
      CUDA_VISIBLE_DEVICES=$seed "$PYTHON" -m src.semhand.pixel_student train \
        --manifest "$manifest" \
        --out "$work/student_seed${seed}.pt" \
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

  for seed in 0 1 2; do
    checkpoints+=(--checkpoint "$work/student_seed${seed}.pt")
  done
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m src.semhand.pixel_student score \
    --manifest "$manifest" \
    "${checkpoints[@]}" \
    --split val \
    --predictions "$work/val.csv" \
    >"$work/val_metrics.json"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m src.semhand.pixel_student score \
    --manifest "$manifest" \
    "${checkpoints[@]}" \
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
    --candidates "$ROOT/bodypart_hardneg_v1/candidates.csv" \
    --candidates "$ROOT/bodypart_hardneg_v2/candidates.csv" \
    --review "$ROOT/bodypart_hardneg_v1/human_track_review.csv" \
    --review "$ROOT/bodypart_hardneg_v2/human_track_review.csv" \
    --heldout-rec "$heldout" \
    --threshold-report "$work/threshold.json" \
    --out "$work/hardneg_tracks.json" \
    >"$work/hardneg_tracks.log"
}

run_fold fold_hold_r0824 R0824_120917
run_fold fold_hold_r0828 R0828_105127

echo "completed: $OUT"
