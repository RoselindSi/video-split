#!/usr/bin/env bash
# Build a SEPARATE venv for TRAINING, pinned to Time-R1's tested transformers
# (4.51.1). Training uses HF generate (no vllm), so this venv has no vllm and
# avoids the transformers-4.53 Qwen2.5-VL generation incompatibilities.
#
# The inference venv ($ROOT/venv, transformers 4.53.2 + vllm 0.9.2) is untouched.
#
# Run:  bash /workspace/tr1/vs/configs/setup_venv_train.sh
# Then: source /workspace/tr1/env_train.sh   (before any training)

set -u
export ROOT=/workspace/tr1
export UV_PYTHON_INSTALL_DIR=$ROOT/pythons
export UV_CACHE_DIR=$ROOT/uv_cache

command -v uv >/dev/null || { echo "!! uv not found"; exit 1; }

echo "==== create venv_train (Python 3.10.12) ===="
[ -d "$ROOT/venv_train" ] || uv venv "$ROOT/venv_train" --python 3.10.12
source "$ROOT/venv_train/bin/activate"

echo "==== torch 2.7.0 + cu128 (Blackwell) ===="
uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

echo "==== training deps, transformers pinned to Time-R1's 4.51.1 ===="
uv pip install \
    "transformers==4.51.1" \
    "trl==0.17.0" \
    peft accelerate datasets deepspeed rouge_score tensorboard \
    decord qwen-vl-utils "numba==0.61.2" sentence-transformers

echo "==== write env_train.sh (source this before training) ===="
cat > "$ROOT/env_train.sh" <<'EOF'
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
[ -d "$ROOT/venv_train" ] && source $ROOT/venv_train/bin/activate
EOF

echo "==== smoke check ===="
python -c "import torch,transformers,trl,peft,datasets; \
print('torch',torch.__version__,'tf',transformers.__version__,'trl',trl.__version__,'cuda',torch.cuda.is_available())"

echo ""
echo "Done. Train with:  source $ROOT/env_train.sh  then run the training script."
