export WANDB_PROJECT=video-split-seg
export EXP_NAME=qwen3_s1b_ddp
export PYTHONPATH=".:$PYTHONPATH"
export CUDA_HOME=${CUDA_HOME:-/workspace/tr1/cudabin}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export DEBUG_MODE="true"
export LOG_PATH="./logs/$EXP_NAME/$EXP_NAME.txt"
OUTDIR=./logs/$EXP_NAME
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12399 main.py \
    --output_dir $OUTDIR \
    --model_name_or_path /workspace/tr1/ckpts/Qwen3-VL-8B-Instruct \
    --train_data_path /workspace/tr1/data_handtask/train_multiseg_train.json \
    --dataset_name seg --learning_rate 1e-4 \
    --use_peft true --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --num_generations 4 \
    --max_completion_length 512 \
    --ddp_timeout 3600 --ddp_find_unused_parameters true \
    --logging_steps 1 --bf16 true --dtype bfloat16 --data_seed 42 \
    --gradient_checkpointing true --attn_implementation sdpa \
    --fix_vit true --slide_window false --num_train_epochs 6 \
    --run_name $EXP_NAME --report_to tensorboard \
    --reward_funcs iou_seg name_seg seq format_seg \
    --temperature 1.0 --prompt_type seg --is_curriculum_learning false \
    --logging_dir $OUTDIR --save_steps 20 --save_only_model true
