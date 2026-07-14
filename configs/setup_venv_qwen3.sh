#!/usr/bin/env bash
# Build a fresh venv for the Qwen3-VL-8B-Instruct migration.
#
# Qwen3-VL needs transformers >= 4.57 and vllm >= 0.11 (verified on the HF model
# card + Qwen vLLM docs). Both training (HF generate) and inference (vllm) work
# on this one stack, so the two-venv split we needed for Time-R1 (4.51 vs 4.53)
# is NO LONGER necessary — venv_qwen3 does both.
#
# Run:  source /workspace/tr1/env.sh   # only to get uv on PATH
#       bash /workspace/tr1/vs/configs/setup_venv_qwen3.sh
# Then: source /workspace/tr1/env_qwen3.sh   (before training or inference)

set -u
export ROOT=/workspace/tr1
export UV_PYTHON_INSTALL_DIR=$ROOT/pythons
export UV_CACHE_DIR=$ROOT/uv_cache
export CUDA_HOME=${CUDA_HOME:-$ROOT/cudabin}   # fake nvcc for deepspeed import

command -v uv >/dev/null || { echo "!! uv not found (source env.sh first)"; exit 1; }

echo "==== create venv_qwen3 (Python 3.10.12) ===="
[ -d "$ROOT/venv_qwen3" ] || uv venv "$ROOT/venv_qwen3" --python 3.10.12
source "$ROOT/venv_qwen3/bin/activate"

echo "==== torch cu128 (Blackwell sm_120) ===="
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "==== vllm >=0.11 + transformers >=4.57 + training deps ===="
uv pip install \
    "vllm>=0.11.0" \
    "transformers>=4.57.0" \
    trl peft accelerate datasets deepspeed rouge_score tensorboard \
    decord qwen-vl-utils sentence-transformers

echo "==== write env_qwen3.sh ===="
cat > "$ROOT/env_qwen3.sh" <<'EOF'
unset HISTFILE
export ROOT=/workspace/tr1
export HF_HOME=$ROOT/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export PIP_CACHE_DIR=$ROOT/pip_cache
export UV_CACHE_DIR=$ROOT/uv_cache
export UV_PYTHON_INSTALL_DIR=$ROOT/pythons
export PATH=$ROOT/bin:$PATH
export CUDA_HOME=$ROOT/cudabin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -d "$ROOT/venv_qwen3" ] && source $ROOT/venv_qwen3/bin/activate
EOF

echo ""
echo "==== VERSION CHECK (the recurring gotcha — verify before trusting) ===="
python -c "import torch,transformers,vllm,trl,peft; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('transformers', transformers.__version__, '(need >=4.57)'); \
print('vllm', vllm.__version__, '(need >=0.11)'); \
print('trl', trl.__version__)"
echo ""
echo "If torch is NOT +cu128, or transformers <4.57, or vllm <0.11 -> stop and fix"
echo "before doing anything else (a dep may have pulled the wrong torch)."

# ------------------------------------------------------------------ MIGRATION NOTES
# After the env is good, the CODE changes for Qwen3-VL (do at migration time):
#   1. trainer + merge_lora.py: Qwen2_5_VLForConditionalGeneration ->
#      Qwen3VLForConditionalGeneration
#   2. DELETE the second_per_grid_ts patch (Qwen2.5-VL specific; Qwen3-VL doesn't
#      use it) — it's guarded by an `if ... in prompt_inputs` so it's harmless if
#      left, but cleaner to remove.
#   3. lora_target_modules: UNCHANGED (Qwen3 keeps q/k/v/o/gate/up/down_proj).
#   4. reward / prompt / converter / eval: UNCHANGED (decoupled from Time-R1).
#   5. trl may be a newer version -> a couple of GRPO arg names might differ;
#      first training launch will surface them.
#   6. base = Qwen3-VL-8B-Instruct (download to $ROOT/ckpts/Qwen3-VL-8B-Instruct);
#      first step: zero-shot eval_multiseg to set a NEW baseline vs Time-R1's.
