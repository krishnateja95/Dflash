#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-$HOME/virtual_envs/Dflash/bin/activate}"
if [[ -f "${VENV_PATH}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}"
fi

python3 "${SCRIPT_DIR}/4_launch_vllm.py" Qwen/Qwen3-8B \
  --hidden-states-path /tmp \
  --target-layer-ids 1 9 17 25 34 36 \
  -- \
  --max-model-len 16384 \
  -dp 4 \
  --max_num_batched_tokens 4096 \
  --max_num_seqs 4096 \
  --gpu-memory-utilization 0.9





