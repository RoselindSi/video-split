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

# 落点按 HARD RULE「大文件只放 SAN」放 /storage（与 /workspace 是同一目录）。
# 不能放 /shared：容器里它是只读挂载，mkdir 直接报 Read-only file system。
# /scratch 已满(8.0T 剩 540K)，/data/storage 在容器里不存在。
OUT=${OUT:-/storage/roselind_semhand/batch3_detect2}
CACHE=${CACHE:-/storage/roselind_semhand/batch3_track_cache}
# GPU 是和别的租户共享的，跑前先 nvidia-smi 挑显存空的卡再用 GPUS="4 5 6" 覆盖。
GPUS=(${GPUS:-0 1 2 3 4 5 6})

cd "$ROOT"
mkdir -p "$OUT/logs" "$CACHE"
mapfile -t JOBLIST < "$JOBS"
echo "[$(date +%H:%M:%S)] 共 ${#JOBLIST[@]} 条, 输出 $OUT, GPU ${GPUS[*]}"
df -h "$OUT" | tail -1

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
        > "$OUT/logs/$rec.log" 2>&1 < /dev/null \
        && echo "[$(date +%H:%M:%S)] GPU $gpu done $rec" \
        || echo "[$(date +%H:%M:%S)] GPU $gpu FAILED $rec (rc=$?), 见 $OUT/logs/$rec.log"
      # 失败不能让 set -e 退出这个子 shell：那样这张卡上排在后面的录像全被跳过。
    done
  ) &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || true; done

# 等产物、不等进程 —— HARD RULE: 远端等待绝不能用 pgrep -f，
# ssh 包装进程的命令行里带着整段脚本文本，会让 while pgrep 永远为真。
#
# 帧数从 track cache 的 end 行取（每个处理过的帧写一行 frame，结束写
# {"type":"end","frame_count":N}），不从 csv 数：csv 是每个框一行，没检出手的
# 帧不出行。现存 6 条的 csv 唯一帧数是 395/399/399/325/400/400，按 csv 判
# 「应为 400」会把正常产物报成缺帧。csv 只校验帧号都落在窗口内。
echo
echo "[$(date +%H:%M:%S)] 完成情况:"
done_n=0
bad_n=0
for line in "${JOBLIST[@]}"; do
  IFS="|" read -r rec bag start n <<< "$line"
  if [ -s "$OUT/$rec.csv" ] && [ -s "$CACHE/$rec.tracks.jsonl" ]; then
    rows=$(( $(wc -l < "$OUT/$rec.csv") - 1 ))
    frames=$(tail -n +2 "$OUT/$rec.csv" | cut -d, -f5 | sort -u | wc -l)
    read -r lo hi < <(tail -n +2 "$OUT/$rec.csv" | awk -F, \
      'NR==1{lo=$5;hi=$5} {if($5<lo)lo=$5; if($5>hi)hi=$5} END{print lo+0, hi+0}')
    fc=$(tail -n 1 "$CACHE/$rec.tracks.jsonl" | grep -o '"frame_count": *[0-9]*' | grep -o '[0-9]*$' || true)
    flag=OK
    if [ "${fc:-x}" != "$n" ] || [ "${lo:-0}" -lt "$start" ] || [ "${hi:-0}" -gt $((start + n - 1)) ]; then
      flag=BAD; bad_n=$((bad_n+1))
    fi
    printf "  %-4s %-24s 处理 %3s/%s 帧 | csv %5d 行, %3d 帧有框, 帧号 %s-%s\n" \
      "$flag" "$rec" "${fc:-?}" "$n" "$rows" "$frames" "$lo" "$hi"
    done_n=$((done_n+1))
  else
    printf "  缺   %-24s  见 %s\n" "$rec" "$OUT/logs/$rec.log"
  fi
done
echo "[$(date +%H:%M:%S)] $done_n / ${#JOBLIST[@]} 条产出, 其中 $bad_n 条校验不过"
du -sh "$OUT" "$CACHE"
echo
echo "拉回本地后跑"
echo "  python3 src/semhand/verify_face_column.py --det-dir <目录> --out /tmp/x.json"
echo "  —— 它会打印每条的框数分布，和 gold 的 others 数对账。"
