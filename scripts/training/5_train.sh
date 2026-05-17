#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-$HOME/virtual_envs/Dflash/bin/activate}"
if [[ -f "${VENV_PATH}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}"
fi

export TRACKIO_DIR="${TRACKIO_DIR:-${SCRIPT_DIR}/../../Qwen3-8B}"
mkdir -p "$TRACKIO_DIR"

torchrun \
  --standalone \
  --nproc_per_node=4 "${SCRIPT_DIR}/5_train.py" \
  --verifier-name-or-path Qwen/Qwen3-8B \
  --data-path "${SCRIPT_DIR}/../../prepared_data" \
  --on-missing generate \
  --on-generate delete \
  --scheduler-type cosine \
  --draft-vocab-size 32000 \
  --max-anchors 3072 \
  --target-layer-ids 1 9 17 25 34 \
  --speculator-type dflash \
  --num-layers 5 \
  --logger trackio \
  --run-name demo_qwen3_9b \
  --lr 0.0006 \
  --epochs 20 \
  --num-workers 16
