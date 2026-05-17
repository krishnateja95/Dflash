#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-$HOME/virtual_envs/Dflash/bin/activate}"
if [[ -f "${VENV_PATH}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}"
fi

python "${SCRIPT_DIR}/2_script.py" \
  --limit 5000 \
  --concurrency 512 \
  --max-tokens 8196 \
  --dataset magpie \
  --outfile "${SCRIPT_DIR}/magpie_regeneration.jsonl"

python "${SCRIPT_DIR}/2_script.py" \
  --limit 5000 \
  --concurrency 2048 \
  --max-tokens 8196 \
  --dataset ultrachat \
  --outfile "${SCRIPT_DIR}/ultrachat_regeneration.jsonl"
