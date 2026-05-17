#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/virtual_envs/Dflash/bin/activate}"
if [[ -f "${VENV_PATH}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}"
fi

# Start serving in a separate terminal before running this script:
# vllm serve qwen3_8b_dflash

# Acceptance-rate benchmark.
# source 6_run_perf_benchmark.sh \
#   --target "http://localhost:8000" \
#   --output-dir acceptance_rate_results

# Performance benchmark.
# source 6_run_perf_benchmark.sh \
#   --target "http://localhost:8000" \
#   --subsets "HumanEval" \
#   --max-requests 80 \
#   --output-dir perf_results

# # Compare and plot performance.
# python plot_perf_compare.py \
#   --source "No Spec=perf_results_nodflash/perf_results.csv" \
#   --source "DFlash=perf_results/perf_results.csv" \
#   --metric itl \
#   --output-dir ./plots




# vllm serve qwen3_8b_dflash 

# # Acceptance rate command
# source run_perf_benchmark.sh \
#     --target "http://localhost:8000" \
#     --subsets "HumanEval" \
#     --max-requests 80 \
#     --acceptance-only \
#     --output-dir acceptance_rate_results

# source run_perf_benchmark.sh \
#     --target "http://localhost:8000" \
#     --subsets "HumanEval" \
#     --max-requests 80 \
#     --output-dir perf_results

