#!/bin/bash
# Run the current temporal owner pipeline over ten crowded raw cam3 segments.
set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LD_LIBRARY_PATH=/workspace/glvnd/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

ROOT=/workspace/tr1/vs
RUN_NAME=${RUN_NAME:-crowd10}
J=${JOBS_FILE:-/workspace/crowd10_jobs.txt}
CK="/workspace/distil/student/S_wide_g6_seed*.pt"
BASE=/workspace/${RUN_NAME}_base
CACHE=/workspace/${RUN_NAME}_track_cache
HVIEWS=/workspace/${RUN_NAME}_hand_views
HSCORES=/workspace/${RUN_NAME}_hand_scores.jsonl
DECISIONS=/workspace/${RUN_NAME}_decisions
TRACK_MAPS=/workspace/${RUN_NAME}_track_maps
CLEAN=/workspace/${RUN_NAME}_clean
OVIEWS=/workspace/${RUN_NAME}_owner_views
OSCORES=/workspace/${RUN_NAME}_owner_scores.jsonl
GATES=/workspace/${RUN_NAME}_owner_gates
FINAL=/workspace/${RUN_NAME}_context_fused_nomosaic
GPUS=(0 1 2 3 4 5 6)

cd "$ROOT"
mkdir -p "$BASE/logs" "$CACHE" "$CLEAN/logs" "$FINAL/logs" \
  "$DECISIONS" "$TRACK_MAPS" "$GATES"
mapfile -t JOBS < "$J"

echo "[$(date +%H:%M:%S)] run=$RUN_NAME jobs=$J count=${#JOBS[@]}"

collect_arm() {
  local out=$1
  local cache_out=$2
  shift 2
  local -a extra=("$@")
  local -a pids=()
  local slot gpu idx rec bag start n
  for slot in "${!GPUS[@]}"; do
    gpu=${GPUS[$slot]}
    (
      for ((idx=slot; idx<${#JOBS[@]}; idx+=${#GPUS[@]})); do
        IFS="|" read -r rec bag start n <<< "${JOBS[$idx]}"
        if [ -s "$out/$rec.csv" ] \
          && [ -s "$cache_out/$rec.tracks.jsonl" ]; then
          echo "[$(date +%H:%M:%S)] skip $rec"
          continue
        fi
        echo "[$(date +%H:%M:%S)] GPU $gpu start $rec"
        CUDA_VISIBLE_DEVICES=$gpu /workspace/venv_rig/bin/python3 \
          -m src.semhand.blur_check --databag "$bag" --clf_student "$CK" \
          --self_reconfirm 1 --continue_conf 0.10 --new_hand_grace 2 \
          --camera cam3 --start "$start" --n "$n" --stride 1 --fps 6 \
          --track_cache_out "$cache_out/$rec.tracks.jsonl" \
          --metadata_only --detector_batch_size 8 \
          "${extra[@]}" --csv "$out/$rec.csv" \
          > "$out/logs/$rec.log" 2>&1 < /dev/null
        echo "[$(date +%H:%M:%S)] GPU $gpu done $rec"
      done
    ) &
    pids+=("$!")
  done
  for idx in "${pids[@]}"; do wait "$idx"; done
}

echo "[$(date +%H:%M:%S)] stage 1/6: collect metadata and track cache"
collect_arm "$BASE" "$CACHE"

echo "[$(date +%H:%M:%S)] stage 2/6: fuse fragmented raw track ids"
PYTHONPATH="$ROOT" /workspace/venv_rig/bin/python3 -m src.rig.track_fusion \
  --arm "$BASE" --jobs "$J" --out "$TRACK_MAPS"

echo "[$(date +%H:%M:%S)] stage 3/6: representative hand/not-hand semantics"
PYTHONPATH="$ROOT" /workspace/venv_rig/bin/python3 -m src.semhand.three_probe \
  --arm "$BASE" --jobs "$J" --min_hands 1 --views "$HVIEWS" \
  --out "$HSCORES" --track_maps "$TRACK_MAPS" \
  --frames_per_track 11 --min_px 150 --prep
hpids=()
for shard in 0 1 2; do
  gpu=$(( shard == 0 ? 0 : (shard == 1 ? 3 : 6) ))
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$ROOT" /workspace/lvs/.venv/bin/python \
    -m src.semhand.three_probe --arm "$BASE" --jobs "$J" --min_hands 1 \
    --views "$HVIEWS" --out "$HSCORES" --track_maps "$TRACK_MAPS" \
    --frames_per_track 11 --min_px 150 \
    --shard "$shard" --nshard 3 \
    > "$BASE/logs/hand_qwen_$shard.log" 2>&1 &
  hpids+=("$!")
done
for pid in "${hpids[@]}"; do wait "$pid"; done
cat "$HSCORES".[0-2] > "$HSCORES"

echo "[$(date +%H:%M:%S)] stage 4/6: materialize hand verdict tables"
PYTHONPATH="$ROOT" /workspace/venv_rig/bin/python3 -m src.rig.post_pass \
  --arm "$BASE" --jobs "$J" --scores "$HSCORES" \
  --track_maps "$TRACK_MAPS" --out "$DECISIONS" \
  --materialized_arm "$CLEAN" --max_owner 2
while IFS="|" read -r rec _; do
  EXTRA=(--decisions "$DECISIONS/$rec.decisions.csv" --no_mosaic)
  # Rendering is batched below, so this loop only validates table presence.
  test -f "$DECISIONS/$rec.decisions.csv"
done < "$J"

echo "[$(date +%H:%M:%S)] stage 5/6: temporal owner gate"
PYTHONPATH="$ROOT" /workspace/venv_rig/bin/python3 -m src.semhand.owner_gate \
  --prep --arm "$CLEAN" --jobs "$J" --views "$OVIEWS" \
  --source_fps 30 --context_seconds 2 --base_strong 0.90 \
  > "$FINAL/logs/owner_prep.log" 2>&1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT" /workspace/lvs/.venv/bin/python \
  -m src.semhand.owner_gate --views "$OVIEWS" --out "$OSCORES" \
  --gate_dir "$GATES" --track_maps "$TRACK_MAPS" \
  --batch 5 --min_positive_frac 0.80 \
  --base_rescue 0.95 --base_strong 0.90 --min_base_strong_frac 0.80 \
  > "$FINAL/logs/owner_qwen.log" 2>&1

# A recording with no lower-ranked owner candidates legitimately produces no
# gate rows.  Write an explicit empty table so the final render can distinguish
# that case from a missing or mistyped gate path.
GATE_HEADER="rec,tid,canonical_tid,verdict,reason,context_state,base_state,p_median,p_min,p_max,positive_frac,base_p_median,base_strong_frac,n"
while IFS="|" read -r rec _; do
  if [ ! -e "$GATES/$rec.owner_gate.csv" ]; then
    printf '%s\n' "$GATE_HEADER" > "$GATES/$rec.owner_gate.csv"
  fi
done < "$J"

echo "[$(date +%H:%M:%S)] stage 6/6: final no-mosaic cache replay"
fpids=()
for slot in "${!GPUS[@]}"; do
  gpu=${GPUS[$slot]}
  (
    for ((idx=slot; idx<${#JOBS[@]}; idx+=${#GPUS[@]})); do
      IFS="|" read -r rec bag start n <<< "${JOBS[$idx]}"
      CUDA_VISIBLE_DEVICES=$gpu /workspace/venv_rig/bin/python3 \
        -m src.semhand.blur_check --databag "$bag" --clf_student "$CK" \
        --self_reconfirm 1 --continue_conf 0.10 --new_hand_grace 2 \
        --camera cam3 --start "$start" --n "$n" --stride 1 --fps 6 \
        --track_cache_in "$CACHE/$rec.tracks.jsonl" \
        --decisions "$DECISIONS/$rec.decisions.csv" \
        --owner_gate "$GATES/$rec.owner_gate.csv" --no_mosaic \
        --out "$FINAL/$rec.mp4" --csv "$FINAL/$rec.csv" \
        --face_csv "$FINAL/$rec.faces.csv" > "$FINAL/logs/$rec.log" 2>&1
    done
  ) &
  fpids+=("$!")
done
for pid in "${fpids[@]}"; do wait "$pid"; done

echo "[$(date +%H:%M:%S)] ${RUN_NAME}_CURRENT_PIPELINE_DONE"
