#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-$HOME/virtual_envs/Dflash/bin/activate}"
if [[ -f "${VENV_PATH}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}"
fi

DATASET_DIR="${DATASET_DIR:-./dataset/speculator_benchmarks}"

set -euo pipefail

benchmark_files=(
  HumanEval.jsonl
  math_reasoning.jsonl
  qa.jsonl
  question.jsonl
  rag.jsonl
  summarization.jsonl
  tool_call.jsonl
  translation.jsonl
  writing.jsonl
)

target_models=(
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3-8B-FP8"
)

draft_models=(
  "krishnateja95/Qwen3-8B-Dflash"
  "krishnateja95/Qwen3-8B-FP8-Dflash"
)

# For acceptance rate only (using the eval-guidellm framework)
# Runs both:
# 1) Baseline (no speculative decoding): each target model alone
# 2) Speculative decoding: all target/draft combinations
for target_model in "${target_models[@]}"; do
  target_name="${target_model#*/}"

  # Baseline run (no -s / no speculative decoding)
  for file in "${benchmark_files[@]}"; do
    file_name="${file%.jsonl}"
    output_dir="eval_results/${target_name}__baseline_${file_name}/"
    bash "${SCRIPT_DIR}/run_evaluation.sh" \
      -b "${target_model}" \
      -d "${DATASET_DIR}/${file}" \
      -o "${output_dir}"
  done

  # Speculative decoding runs (all target/draft combinations)
  for draft_model in "${draft_models[@]}"; do
    draft_name="${draft_model#*/}"
    for file in "${benchmark_files[@]}"; do
      file_name="${file%.jsonl}"
      output_dir="eval_results/${target_name}__${draft_name}_${file_name}/"
      bash "${SCRIPT_DIR}/run_evaluation.sh" \
        -b "${target_model}" \
        -s "${draft_model}" \
        -d "${DATASET_DIR}/${file}" \
        -o "${output_dir}"
    done
  done
done



