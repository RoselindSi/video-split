#!/bin/bash
# Ten recordings nothing has touched, through the whole thing end to end.
#
# The six cam3 recordings have been looked at from every side for a week, so
# everything measured on them is measured where the decisions were also made.
# These ten share no recording with the evaluation set, with either negative
# harvest, or with the held-out pool -- they are the 561st onward under the
# same shuffle. What they answer is whether any of it travels.
#
# Five stages, chained, because each needs the one before it:
#   1  the pipeline, recording the face boxes it covers
#   2  hand-ness over the own boxes           -> the track decisions
#   3  face-ness over the face groups         -> the size-cap verdicts
#   4  the two passes turned into tables
#   5  the render that reads them
set -u
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH=/workspace/glvnd/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
cd /workspace/tr1/vs
source /workspace/venv_rig/bin/activate
J=/workspace/cam3_jobs10.txt
CK="/workspace/distil/student/S_wide_g6_seed*.pt"

# Wait for the six-recording rescore to free the card it is on.
until grep -qa VERDICTS3_DONE /workspace/face2.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] stage 1: pipeline"

R=/workspace/ten_base; mkdir -p $R/logs
gpus=(4 5 1); gi=0; pids=()
while IFS="|" read -r rec bag start n; do
  [ -z "$rec" ] && continue
  [ -s "$R/${rec}.csv" ] && continue
  g=${gpus[$gi]}; gi=$(( (gi + 1) % 3 ))
  CUDA_VISIBLE_DEVICES=$g python3 -m src.semhand.blur_check --databag "$bag" \
    --clf_student "$CK" --self_reconfirm 1 --continue_conf 0.10 \
    --new_hand_grace 2 --camera cam3 --start "$start" --n "$n" --stride 1 --fps 6 \
    --out $R/${rec}.mp4 --csv $R/${rec}.csv --face_csv $R/${rec}.faces.csv \
    > $R/logs/${rec}.log 2>&1 < /dev/null &
  pids+=($!)
  if [ ${#pids[@]} -ge 3 ]; then for p in "${pids[@]}"; do wait $p; done; pids=(); fi
done < $J
for p in "${pids[@]}"; do wait $p; done
echo "[$(date +%H:%M:%S)] stage 2: hand-ness"

PYTHONPATH=/workspace/tr1/vs python3 /workspace/threeprobe.py --arm $R --jobs $J \
  --min_hands 1 --views /workspace/ten_hviews --out /workspace/ten_hand.jsonl --prep
for i in 0 1 2; do
  g=$((i==0 ? 6 : (i==1 ? 0 : 3)))
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=/workspace/tr1/vs \
    /workspace/lvs/.venv/bin/python /workspace/threeprobe.py --arm $R --jobs $J \
    --views /workspace/ten_hviews --out /workspace/ten_hand.jsonl \
    --shard $i --nshard 3 > /workspace/ten_hand_$i.log 2>&1 &
done
wait
cat /workspace/ten_hand.jsonl.[0-9] > /workspace/ten_hand.jsonl
echo "[$(date +%H:%M:%S)] stage 3: face-ness"

PYTHONPATH=/workspace/tr1/vs python3 -m src.semhand.faceness --faces $R --jobs $J \
  --views /workspace/ten_fviews --out /workspace/ten_face.jsonl --prep
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/workspace/tr1/vs \
  /workspace/lvs/.venv/bin/python -m src.semhand.faceness --faces $R --jobs $J \
  --views /workspace/ten_fviews --out /workspace/ten_face.jsonl
echo "[$(date +%H:%M:%S)] stage 4: tables"

PYTHONPATH=/workspace/tr1/vs python3 -m src.rig.post_pass --arm $R --jobs $J \
  --scores /workspace/ten_hand.jsonl --out /workspace/ten_decisions
PYTHONPATH=/workspace/tr1/vs python3 -m src.semhand.face_verdicts --faces $R --jobs $J \
  --scores /workspace/ten_face.jsonl --out /workspace/ten_verdicts
echo "[$(date +%H:%M:%S)] stage 5: render"

S=/workspace/ten_ship; mkdir -p $S/logs
gi=0; pids=()
while IFS="|" read -r rec bag start n; do
  [ -z "$rec" ] && continue
  [ -s "$S/${rec}.csv" ] && continue
  g=${gpus[$gi]}; gi=$(( (gi + 1) % 3 ))
  CUDA_VISIBLE_DEVICES=$g python3 -m src.semhand.blur_check --databag "$bag" \
    --clf_student "$CK" --self_reconfirm 1 --continue_conf 0.10 \
    --new_hand_grace 2 --camera cam3 --start "$start" --n "$n" --stride 1 --fps 6 \
    --decisions /workspace/ten_decisions/${rec}.decisions.csv \
    --face_verdicts /workspace/ten_verdicts/${rec}.faceverdict.csv \
    --out $S/${rec}.mp4 --csv $S/${rec}.csv --face_csv $S/${rec}.faces.csv \
    > $S/logs/${rec}.log 2>&1 < /dev/null &
  pids+=($!)
  if [ ${#pids[@]} -ge 3 ]; then for p in "${pids[@]}"; do wait $p; done; pids=(); fi
done < $J
for p in "${pids[@]}"; do wait $p; done
echo TEN_DONE
