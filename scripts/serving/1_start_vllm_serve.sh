#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/virtual_envs/Dflash/bin/activate}"
if [[ -f "${VENV_PATH}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}"
fi

vllm serve Qwen/Qwen3-8B \
  --port 8000 \
  --tensor-parallel-size 1 \
  -dp 4 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --async-scheduling \
  --max_num_batched_tokens 4096 \
  --max_num_seqs 4096
