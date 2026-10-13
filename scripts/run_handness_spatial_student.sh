#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/workspace/handness_v2/v2_reviewed/code}
ROOT=${ROOT:-/workspace/handness_v2}
MANIFEST=${MANIFEST:-$ROOT/spatial_expand_v1/combined_spatial_manifest.csv}
OUT=${OUT:-$ROOT/spatial_expand_v1/spatial_student_v1}
PYTHON=${PYTHON:-python}

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

pids=()
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$seed "$PYTHON" -m src.semhand.spatial_student train \
    --manifest "$MANIFEST" \
    --out "$OUT/student_seed${seed}.pt" \
    --seed "$seed" \
    >"$OUT/seed${seed}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

checkpoints=()
for seed in 0 1 2; do
  checkpoints+=(--checkpoint "$OUT/student_seed${seed}.pt")
done
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m src.semhand.spatial_student score \
  --manifest "$MANIFEST" "${checkpoints[@]}" --split val \
  --predictions "$OUT/val.csv" >"$OUT/val_metrics.json"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m src.semhand.spatial_student score \
  --manifest "$MANIFEST" "${checkpoints[@]}" --split test \
  --predictions "$OUT/test.csv" >"$OUT/test_metrics.json"
"$PYTHON" scripts/select_handness_threshold.py \
  --validation "$OUT/val.csv" --test "$OUT/test.csv" \
  --minimum-recall .995 --out "$OUT/threshold.json" \
  >"$OUT/threshold.log"

echo "completed: $OUT"
