#!/usr/bin/env bash
# GRPO Stage 2 (approach B): add naming on top of stage-1 boundaries.
# reward = iou_seg + name_seg + format_seg
#
# Prereq: stage-1 finished and its LoRA merged into a full checkpoint:
#   python $ROOT/vs/src/tools/merge_lora.py \
#     --base $ROOT/ckpts/Time-R1-7B \
#     --adapter $ROOT/time-r1/logs/seg_stage1_iou_format \
#     --out $ROOT/ckpts/seg_stage1_merged
#
# Run FROM the time-r1 repo root inside venv_train:
#   source /workspace/tr1/env_train.sh
#   bash /workspace/tr1/vs/configs/train_seg_stage2.sh

export WANDB_PROJECT=video-split-seg
export EXP_NAME=seg_stage2_add_name
export PYTHONPATH=".:$PYTHONPATH"
export CUDA_HOME=${CUDA_HOME:-/workspace/tr1/cudabin}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DEBUG_MODE="true"
export LOG_PATH="./logs/$EXP_NAME/$EXP_NAME.txt"

OUTDIR=./logs/$EXP_NAME
BASE_MODEL="/workspace/tr1/ckpts/seg_stage1_merged"
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
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_prompt_length 8192 \
    --max_completion_length 512 \
    --num_generations 4 \
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
    --reward_funcs iou_seg name_seg format_seg \
    --temperature 1.0 \
    --prompt_type seg \
    --is_curriculum_learning false \
    --logging_dir $OUTDIR \
    --save_steps 20 \
    --save_only_model true

# vs stage1: BASE_MODEL = stage-1 merged ckpt; reward adds name_seg
# (embedding similarity + genericity penalty; sentence-transformers is cached
# as a module-level singleton so it loads once, not per step)
