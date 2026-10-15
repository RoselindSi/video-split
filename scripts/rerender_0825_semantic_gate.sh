#!/bin/bash
# Wait for the Qwen owner gate, then render the corrected clip once.
set -uo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LD_LIBRARY_PATH=/workspace/glvnd/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

ROOT=/workspace/tr1/vs
REC=R26_0825_113200
GATE=/workspace/owner_gate_0825_gates/${REC}.owner_gate.csv
OUT=/workspace/ten_ship_semantic
BAG=/shared/datasets/.incoming/video_1520p_raw/qingyu005_202608202101/raw_databag/95aeb3bbaa2a49fdbdfe333c9e91b5fd/databag-26_0825_113200
CK="/workspace/distil/student/S_wide_g6_seed*.pt"

mkdir -p "$OUT/logs"
while pgrep -f "^/workspace/lvs/.venv/bin/python -m src.semhand.owner_gate.*owner_gate_0825_views" >/dev/null; do
  sleep 15
done
if [ ! -s "$GATE" ]; then
  echo "Qwen exited without writing $GATE"
  exit 1
fi

cd "$ROOT" || exit 1
echo "[$(date +%H:%M:%S)] semantic render start"
CUDA_VISIBLE_DEVICES=1 /workspace/venv_rig/bin/python3 \
  -m src.semhand.blur_check --databag "$BAG" \
  --clf_student "$CK" --self_reconfirm 1 --continue_conf 0.10 \
  --new_hand_grace 2 --camera cam3 --start 7224 --n 400 --stride 1 --fps 6 \
  --decisions "/workspace/ten_decisions/${REC}.decisions.csv" \
  --owner_gate "$GATE" \
  --face_verdicts "/workspace/ten_verdicts/${REC}.faceverdict.csv" \
  --out "$OUT/${REC}.mp4" --csv "$OUT/${REC}.csv" \
  --face_csv "$OUT/${REC}.faces.csv" \
  > "$OUT/logs/${REC}.log" 2>&1
status=$?
if [ "$status" -ne 0 ]; then
  echo "[$(date +%H:%M:%S)] semantic render FAILED ($status)"
  exit "$status"
fi
echo "[$(date +%H:%M:%S)] SEMANTIC_RENDER_DONE"
