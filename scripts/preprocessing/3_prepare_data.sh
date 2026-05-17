#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-$HOME/virtual_envs/Dflash/bin/activate}"
if [[ -f "${VENV_PATH}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}"
fi

python "${SCRIPT_DIR}/3_prepare_data.py" \
  --max-samples 5000 \
  --model Qwen/Qwen3-8B \
  --data "${SCRIPT_DIR}/../data_generation/ultrachat_regeneration.jsonl" "${SCRIPT_DIR}/../data_generation/magpie_regeneration.jsonl" \
  --output "${SCRIPT_DIR}/../../prepared_data" \
  --assistant-pattern "<\|im_start\|>assistant\s*([\s\S]*?)<\|im_end\|>"
