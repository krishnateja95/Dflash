#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/virtual_envs/Dflash/bin/activate}"
if [[ -f "${VENV_PATH}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}"
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH:-./output/checkpoints/checkpoint_best}"
vllm serve "${CHECKPOINT_PATH}"