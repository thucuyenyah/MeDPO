#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
CACHE_ROOT="$PROJECT_ROOT/.cache"
ENV_NAME="${FD_CONDA_ENV:-MeDPO_env}"
HOME_ENV_PREFIX="/home/thinng/miniconda3/envs/${ENV_NAME}"
WEKA_ENV_PREFIX="$PROJECT_ROOT/conda_envs/${ENV_NAME}"
KSHIELD_SITE="$PROJECT_ROOT/.cache/kshield_site_min"
TORCH_CUDA_SITE="$PROJECT_ROOT/.cache/torch_cuda_py311"

mkdir -p \
  "$CACHE_ROOT/conda/pkgs" \
  "$CACHE_ROOT/pip" \
  "$CACHE_ROOT/huggingface/datasets" \
  "$CACHE_ROOT/wandb" \
  "$CACHE_ROOT/xdg"

if [ -f /home/thinng/miniconda3/etc/profile.d/conda.sh ]; then
  source /home/thinng/miniconda3/etc/profile.d/conda.sh
else
  module load Anaconda3
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

if [ -d "$HOME_ENV_PREFIX" ]; then
  conda activate "$HOME_ENV_PREFIX"
elif [ -d "$WEKA_ENV_PREFIX" ]; then
  conda activate "$WEKA_ENV_PREFIX"
else
  conda activate "$ENV_NAME"
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export WANDB_CACHE_DIR="$CACHE_ROOT/wandb"
if [ -z "${WANDB_API_KEY:-}" ]; then
  export WANDB_MODE=disabled
fi
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export CONDA_PKGS_DIRS="$CACHE_ROOT/conda/pkgs"
export HF_HUB_DISABLE_XET=1
if [ "$ENV_NAME" = "kshield_env" ] && [ -d "$KSHIELD_SITE" ]; then
  export PYTHONPATH="$KSHIELD_SITE:${PYTHONPATH:-}"
fi
if [ "$ENV_NAME" = "MeDPO_env" ] && [ -d "$TORCH_CUDA_SITE" ]; then
  export PYTHONPATH="$TORCH_CUDA_SITE:${PYTHONPATH:-}"
fi

echo "Python: $(which python)"
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
