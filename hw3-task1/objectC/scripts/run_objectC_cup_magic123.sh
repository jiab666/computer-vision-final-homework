#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=/root/autodl-tmp/envs/threestudio
WORK_DIR=/root/autodl-tmp/work/threestudio

export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export CPATH="$ENV_DIR/targets/x86_64-linux/include:$ENV_DIR/lib/python3.10/site-packages/nvidia/cu13/include:${CPATH:-}"
export TORCH_EXTENSIONS_DIR=/root/autodl-tmp/cache/home-cache/torch_extensions
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "$WORK_DIR"
exec "$ENV_DIR/bin/python" launch.py \
  --config configs/objectC-cup-magic123-coarse.yaml \
  --train --gpu 0 \
  "$@"
