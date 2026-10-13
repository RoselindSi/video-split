#!/bin/bash
# Re-render the ten-video final batch with the post-decision owner-cap fix.
set -uo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LD_LIBRARY_PATH=/workspace/glvnd/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

cd /workspace/tr1/vs || exit 1

J=/workspace/cam3_jobs10.txt
S=/workspace/ten_ship_capfix
CK="/workspace/distil/student/S_wide_g6_seed*.pt"
GPUS=(0 1 2 3 4 5 6)

mkdir -p "$S/logs"
mapfile -t JOBS < "$J"

worker() {
  local gpu=$1
  local slot=$2
  local idx rec bag start n

  for ((idx=slot; idx<${#JOBS[@]}; idx+=${#GPUS[@]})); do
    IFS="|" read -r rec bag start n <<< "${JOBS[$idx]}"
    [ -z "$rec" ] && continue
    if [ -s "$S/${rec}.csv" ] && [ -s "$S/${rec}.mp4" ]; then
      echo "[$(date +%H:%M:%S)] GPU $gpu skip complete $rec"
      continue
    fi

    echo "[$(date +%H:%M:%S)] GPU $gpu start $rec"
    if ! CUDA_VISIBLE_DEVICES=$gpu /workspace/venv_rig/bin/python3 \
      -m src.semhand.blur_check --databag "$bag" \
      --clf_student "$CK" --self_reconfirm 1 --continue_conf 0.10 \
      --new_hand_grace 2 --camera cam3 --start "$start" --n "$n" \
      --stride 1 --fps 6 \
      --decisions "/workspace/ten_decisions/${rec}.decisions.csv" \
      --face_verdicts "/workspace/ten_verdicts/${rec}.faceverdict.csv" \
      --out "$S/${rec}.mp4" --csv "$S/${rec}.csv" \
      --face_csv "$S/${rec}.faces.csv" \
      > "$S/logs/${rec}.log" 2>&1 < /dev/null; then
      echo "[$(date +%H:%M:%S)] GPU $gpu FAILED $rec"
      return 1
    fi
    echo "[$(date +%H:%M:%S)] GPU $gpu done $rec"
  done
}

echo "[$(date +%H:%M:%S)] capfix render: ${#JOBS[@]} videos on ${#GPUS[@]} GPUs"
pids=()
for slot in "${!GPUS[@]}"; do
  worker "${GPUS[$slot]}" "$slot" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

if [ "$status" -ne 0 ]; then
  echo "[$(date +%H:%M:%S)] TEN_SHIP_CAPFIX_FAILED"
  exit "$status"
fi

echo "[$(date +%H:%M:%S)] TEN_SHIP_CAPFIX_DONE"
