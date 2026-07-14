#!/usr/bin/env bash
# DDP (8-GPU data-parallel) training TEMPLATE for the seg curriculum.
# Use this when moving to Qwen3-VL / larger data, NOT for the Time-R1 curriculum
# (keep those single-GPU so Stage 1/2/3 stay comparable).
#
# Why DDP: GRPO's cost is dominated by rollout generation; data-parallel across
# 8 cards is ~near-linear speedup (single-GPU ~1.8h -> ~15min). 7B+LoRA fits one
# card, so this is data parallel (DDP), NOT tensor parallel.
#
# Run FROM the repo root, inside the matching venv:
#   source /workspace/tr1/env_qwen3.sh     # (or env_train.sh for a DDP smoke test)
#   bash /workspace/tr1/vs/configs/train_seg_ddp_template.sh

export WANDB_PROJECT=video-split-seg
export EXP_NAME=seg_stage1_ddp          # <-- rename per stage
export PYTHONPATH=".:$PYTHONPATH"
export CUDA_HOME=${CUDA_HOME:-/workspace/tr1/cudabin}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN                   # surface NCCL issues without spamming
export DEBUG_MODE="true"
export LOG_PATH="./logs/$EXP_NAME/$EXP_NAME.txt"

OUTDIR=./logs/$EXP_NAME
BASE_MODEL="/workspace/tr1/ckpts/Qwen3-VL-8B-Instruct"   # <-- set per run
TRAIN_DATA="/workspace/tr1/data_handtask/train_multiseg_train.json"

# ---- effective-batch math (READ THIS before running) ----------------------
# single-GPU: nproc 1 * per_device 1 * grad_accum 2 = 2 videos/opt-step
#             56 videos * 3 epochs / 2 = 84 opt-steps
# 8-GPU DDP : nproc 8 * per_device 1 * grad_accum 1 = 8 videos/opt-step
#             56 videos / 8 = 7 opt-steps/epoch
#   -> to keep ~80 opt-steps, need ~12 epochs (set below).
#   -> grad_accum dropped to 1 so we don't blow the effective batch to 16.
# If you change GPU count or data size, recompute epochs to keep opt-steps sane.
# --------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12399" \
    main.py \
    --output_dir $OUTDIR \
    --model_name_or_path $BASE_MODEL \
    --train_data_path $TRAIN_DATA \
    --dataset_name seg \
    --learning_rate 1e-4 \
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_prompt_length 8192 \
    --max_completion_length 512 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --ddp_timeout 3600 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --fix_vit true \
    --slide_window false \
    --num_train_epochs 12 \
    --run_name $EXP_NAME \
    --report_to tensorboard \
    --reward_funcs iou_seg format_seg \
    --temperature 1.0 \
    --prompt_type seg \
    --is_curriculum_learning false \
    --logging_dir $OUTDIR \
    --save_steps 20 \
    --save_only_model true

# --- what changed vs the single-GPU stage script, and why ---
# CUDA_VISIBLE_DEVICES : 0 -> 0..7        (expose all 8 cards)
# torchrun nproc       : 1 -> 8           (8 data-parallel processes)
# gradient_accumulation: 2 -> 1           (keep effective batch reasonable at 8x)
# num_train_epochs     : 3 -> 12          (fewer opt-steps/epoch at 8x -> more epochs)
# ddp_timeout          : added 3600s      (long many-segment videos make ranks wait;
#                                          avoid NCCL collective timeout)
# NCCL_DEBUG           : WARN             (visibility into multi-proc comm issues)
#
# FIRST-RUN CHECKLIST (the DDP-specific gotchas to watch):
#   - all 8 ranks start + reach step 1 (no NCCL init hang)
#   - only rank0 writes checkpoints/logs (expected)
#   - if a rank stalls on a long video -> raise ddp_timeout / check shm size
#   - reward in logs is averaged across ranks (slightly different from single-GPU)
#
# SMOKE TEST (cheap, run between Stage 2 and 3 to de-risk DDP early):
#   change nproc_per_node=2 and CUDA_VISIBLE_DEVICES=0,1, num_train_epochs=1,
#   point BASE_MODEL at Time-R1-7B; just confirm 2-rank NCCL + save + log work.
