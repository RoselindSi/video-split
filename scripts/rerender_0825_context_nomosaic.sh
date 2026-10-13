#!/bin/bash
# Score temporal owner context once, then render a raw-pixel review video.
set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LD_LIBRARY_PATH=/workspace/glvnd/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

ROOT=/workspace/tr1/vs
REC=R26_0825_113200
BAG=/shared/datasets/.incoming/video_1520p_raw/qingyu005_202608202101/raw_databag/95aeb3bbaa2a49fdbdfe333c9e91b5fd/databag-26_0825_113200
ARM=/workspace/ten_ship_capfix
VIEWS=/workspace/owner_gate_0825_context_views
SCORES=/workspace/owner_gate_0825_context_scores.jsonl
GATES=/workspace/owner_gate_0825_context_gates
OUT=/workspace/ten_ship_context_nomosaic
CK="/workspace/distil/student/S_wide_g6_seed*.pt"

mkdir -p "$OUT/logs" "$GATES"
cd "$ROOT"

/workspace/venv_rig/bin/python3 -m src.semhand.owner_gate \
  --prep --arm "$ARM" --jobs /workspace/cam3_jobs10.txt \
  --rec "$REC" --views "$VIEWS" --source_fps 30 --context_seconds 2 \
  > "$OUT/logs/${REC}.prep.log" 2>&1

CUDA_VISIBLE_DEVICES=0 /workspace/lvs/.venv/bin/python \
  -m src.semhand.owner_gate --views "$VIEWS" --out "$SCORES" \
  --gate_dir "$GATES" --batch 5 --min_positive_frac 0.80 \
  > "$OUT/logs/${REC}.qwen.log" 2>&1

CUDA_VISIBLE_DEVICES=1 /workspace/venv_rig/bin/python3 \
  -m src.semhand.blur_check --databag "$BAG" \
  --clf_student "$CK" --self_reconfirm 1 --continue_conf 0.10 \
  --new_hand_grace 2 --camera cam3 --start 7224 --n 400 --stride 1 --fps 6 \
  --decisions "/workspace/ten_decisions/${REC}.decisions.csv" \
  --owner_gate "$GATES/${REC}.owner_gate.csv" \
  --face_verdicts "/workspace/ten_verdicts/${REC}.faceverdict.csv" \
  --no_mosaic \
  --out "$OUT/${REC}.mp4" --csv "$OUT/${REC}.csv" \
  --face_csv "$OUT/${REC}.faces.csv" \
  > "$OUT/logs/${REC}.render.log" 2>&1

echo "[$(date +%H:%M:%S)] CONTEXT_NOMOSAIC_DONE"
