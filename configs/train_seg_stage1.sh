#!/usr/bin/env bash
# GRPO Stage 1 (approach B): teach correct boundaries + valid multi-segment format.
# Curriculum stage 1 uses only iou_seg + format_seg; name_seg and seq come later.
#
# Base model = Time-R1-7B: it already localizes distinct sub-tasks well zero-shot
# (IoU 0.5-0.88), so GRPO starts from a strong grounding prior instead of scratch.
#
# Run FROM the time-r1 repo root (needs PYTHONPATH="." and src/ on path):
#   cd /workspace/tr1/time-r1
#   bash /workspace/tr1/vs/configs/train_seg_stage1.sh

export WANDB_PROJECT=video-split-seg
export EXP_NAME=seg_stage1_iou_format
export PYTHONPATH=".:$PYTHONPATH"
# fake nvcc so deepspeed's import-time CUDA check passes (we don't use deepspeed ops)
export CUDA_HOME=${CUDA_HOME:-/workspace/tr1/cudabin}
export DEBUG_MODE="true"
export LOG_PATH="./logs/$EXP_NAME/$EXP_NAME.txt"

OUTDIR=./logs/$EXP_NAME
BASE_MODEL="/workspace/tr1/ckpts/Time-R1-7B"
TRAIN_DATA="/workspace/tr1/data_handtask/train_multiseg_train.json"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12399" \
    main.py \
    --output_dir $OUTDIR \
    --model_name_or_path $BASE_MODEL \
    --train_data_path $TRAIN_DATA \
    --dataset_name seg \
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --max_prompt_length 8192 \
    --max_completion_length 512 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --fix_vit true \
    --slide_window false \
    --num_train_epochs 3 \
    --run_name $EXP_NAME \
    --report_to tensorboard \
    --reward_funcs iou_seg format_seg \
    --temperature 1.0 \
    --prompt_type seg \
    --is_curriculum_learning false \
    --logging_dir $OUTDIR \
    --save_steps 50 \
    --save_only_model true

# --- what changed vs Time-R1's scripts/posttrain/train_rl.sh, and why ---
# model_name_or_path : Qwen2.5-VL-3B -> Time-R1-7B   (start from a grounding prior)
# train_data_path    : Charades 2k5 -> our seg train split
# LoRA (use_peft)    : full FT -> LoRA r=16          (fits 1x96GB, no deepspeed/nvcc,
#                                                     less overfit on 56 videos)
# --deepspeed        : removed                        (zero3_offload needs CUDA toolkit
#                                                     to JIT-compile cpu_adam; avoided)
# max_completion_length: 200 -> 512                  (multi-segment output is longer)
# reward_funcs       : iou_v2 format -> iou_seg format_seg  (our multi-seg rewards)
# prompt_type        : v1 -> seg                     (whole-video segmentation prompt)
# attn_implementation: flash_attention_2 -> sdpa     (flash-attn has no Blackwell build yet)
# num_train_epochs   : 5 -> 3                         (small data, avoid overfit)
# num_generations    : kept 8 (design target 16; raise if group reward variance collapses)
