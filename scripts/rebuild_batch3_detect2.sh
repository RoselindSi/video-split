#!/bin/bash
# 重建 batch3 这 14 条录像的 detect2（join 之前的 tid），供手部线的主指标使用。
#
# 为什么需要：14 条里有 8 条的 detect2 随 scratchpad 清理一起没了，
# 服务器上也没有副本（既无仓库也无产物）。没有它就只能停在「候选数」，
# 算不出主指标 —— 实际被接入的非目标框比例、受污染的目标轨迹比例。
#
# detect2 **本来就带 tid**，而且是 join 之前的，正是跑 forced_pairs 需要的。
# 所以这里只是重跑 stage 1，不需要改任何代码。
#
#   用法:  bash scripts/rebuild_batch3_detect2.sh
#   输出:  $OUT/<录像>.csv 和 $CACHE/<录像>.tracks.jsonl
set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LD_LIBRARY_PATH=/workspace/glvnd/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

ROOT=${ROOT:-/workspace/tr1/vs}
JOBS=${JOBS:-$ROOT/data/jobs/batch3_14.txt}
CK=${CK:-/workspace/distil/student/S_wide_g6_seed*.pt}

# /scratch 已满(8.0T 零剩余)、/data/storage 在这台机器上不存在。
# 按 HARD RULE「大文件只放 SAN」，落点放 /shared。跑前自己 df -h 看一眼。
OUT=${OUT:-/shared/semhand/batch3_detect2}
CACHE=${CACHE:-/shared/semhand/batch3_track_cache}
GPUS=(${GPUS:-0 1 2 3 4 5 6})

cd "$ROOT"
mkdir -p "$OUT/logs" "$CACHE"
mapfile -t JOBLIST < "$JOBS"
echo "[$(date +%H:%M:%S)] 共 ${#JOBLIST[@]} 条, 输出 $OUT"
df -h /shared | tail -1

pids=()
for slot in "${!GPUS[@]}"; do
  gpu=${GPUS[$slot]}
  (
    for ((idx=slot; idx<${#JOBLIST[@]}; idx+=${#GPUS[@]})); do
      IFS="|" read -r rec bag start n <<< "${JOBLIST[$idx]}"
      if [ -s "$OUT/$rec.csv" ] && [ -s "$CACHE/$rec.tracks.jsonl" ]; then
        echo "[$(date +%H:%M:%S)] skip $rec"; continue
      fi
      echo "[$(date +%H:%M:%S)] GPU $gpu start $rec"
      # --no_faces: 流水线数周不打码，人脸检测器的输出没有任何下游读取。
      #   ac61366 实测同一张卡上 stage 1 从 732s 降到 481s，且与带脸的那一臂
      #   在全部 598 条共享行的每个非 face 列上完全一致 —— 不花判定代价。
      # --stride 1: 源是 30fps。注意 --fps 只影响渲染，这里 --metadata_only
      #   不渲染，所以不受 fps-stride 5 倍换算那个坑影响。
      # --track_cache_in 不给: 这次就是要重新检测，没有缓存可复用。
      CUDA_VISIBLE_DEVICES=$gpu /workspace/venv_rig/bin/python3 \
        -m src.semhand.blur_check --databag "$bag" --clf_student "$CK" \
        --self_reconfirm 1 --continue_conf 0.10 --new_hand_grace 2 \
        --camera cam3 --start "$start" --n "$n" --stride 1 \
        --no_faces \
        --track_cache_out "$CACHE/$rec.tracks.jsonl" \
        --metadata_only --detector_batch_size 8 \
        --csv "$OUT/$rec.csv" \
        > "$OUT/logs/$rec.log" 2>&1 < /dev/null
      echo "[$(date +%H:%M:%S)] GPU $gpu done $rec"
    done
  ) &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p"; done

# 等产物、不等进程 —— HARD RULE: 远端等待绝不能用 pgrep -f，
# ssh 包装进程的命令行里带着整段脚本文本，会让 while pgrep 永远为真。
echo
echo "[$(date +%H:%M:%S)] 完成情况:"
done_n=0
for line in "${JOBLIST[@]}"; do
  rec="${line%%|*}"
  if [ -s "$OUT/$rec.csv" ]; then
    rows=$(( $(wc -l < "$OUT/$rec.csv") - 1 ))
    frames=$(tail -n +2 "$OUT/$rec.csv" | cut -d, -f5 | sort -u | wc -l)
    printf "  OK   %-24s %5d 行 / %3d 帧\n" "$rec" "$rows" "$frames"
    done_n=$((done_n+1))
  else
    printf "  缺   %-24s  见 %s\n" "$rec" "$OUT/logs/$rec.log"
  fi
done
echo "[$(date +%H:%M:%S)] $done_n / ${#JOBLIST[@]} 条产出"
echo
echo "校验: 帧数应为 400（窗口 400-799）。拉回本地后跑"
echo "  python3 src/semhand/verify_face_column.py --det-dir <目录> --out /tmp/x.json"
echo "  —— 它会打印每条的框数分布，和 gold 的 others 数对账。"
