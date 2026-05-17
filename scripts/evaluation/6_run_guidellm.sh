
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-$HOME/virtual_envs/Dflash/bin/activate}"
if [[ -f "${VENV_PATH}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}"
fi

SPECULATOR_MODEL_PATH="${SPECULATOR_MODEL_PATH:-./output/checkpoints/checkpoint_best}"
DATASET_DIR="${DATASET_DIR:-./dataset/speculator_benchmarks}"

set -euo pipefail

# For acceptance rate only (using the eval-guidellm framework)
for file in \
  HumanEval.jsonl \
  math_reasoning.jsonl \
  qa.jsonl \
  question.jsonl \
  rag.jsonl \
  summarization.jsonl \
  tool_call.jsonl \
  translation.jsonl \
  writing.jsonl
do
  bash "${SCRIPT_DIR}/run_evaluation.sh" \
    -b "Qwen/Qwen3-8B" \
    -s "${SPECULATOR_MODEL_PATH}" \
    -d "${DATASET_DIR}/${file}"
done



