#!/usr/bin/env bash
# End-to-end guidellm + vLLM benchmarking for Qwen3-8B.
# Runs two configurations:
#   1) Baseline (no speculative decoding)        -> ./qwen3_8b/
#   2) Dflash speculative decoding (k=16)         -> ./qwen3_8b_dflash/
# Benchmark: nine task types from RedHatAI/speculator_benchmarks
#   (HumanEval, math_reasoning, qa, question, rag, summarization,
#    tool_call, translation, writing).
VENV_PATH="${VENV_PATH:-$HOME/virtual_envs/Dflash/bin/activate}"
if [[ -f "${VENV_PATH}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}"
fi
# VLLM_USE_PRECOMPILED=1
# uv pip install git+https://github.com/vllm-project/vllm.git

set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES is not set. Run canhazgpu first." >&2
  exit 1
fi

IFS=',' read -ra AVAILABLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
TOTAL_GPUS=${#AVAILABLE_GPUS[@]}

# NOTE: Qwen3-8B (8.2B params, dense) in BF16 is ~16GB of weights and fits
# comfortably on a single GPU (24GB+) at TP=1 with plenty of headroom for the
# KV cache and the Dflash draft model. One worker per GPU.
TENSOR_PARALLEL=1
NUM_WORKERS=$(( TOTAL_GPUS / TENSOR_PARALLEL ))
BASE_PORT=8009

if (( TOTAL_GPUS % TENSOR_PARALLEL != 0 )); then
  echo "ERROR: TOTAL_GPUS (${TOTAL_GPUS}) is not divisible by TENSOR_PARALLEL (${TENSOR_PARALLEL})." >&2
  exit 1
fi

DATASETS=(
  "HumanEval.jsonl"
  "math_reasoning.jsonl"
  "qa.jsonl"
  "question.jsonl"
  "rag.jsonl"
  "summarization.jsonl"
  "tool_call.jsonl"
  "translation.jsonl"
  "writing.jsonl"
)

WAIT_READY_TIMEOUT=6000
BENCH_SECONDS=90

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
QUEUE_FILE=$(mktemp)
QUEUE_LOCK=$(mktemp)

# Output folder names (per user request)
BASELINE_DIR="qwen3_8b"
DFLASH_DIR="qwen3_8b_dflash"

# ---------------------------------------------------------------------------
# Qwen3 tool-calling + reasoning flags.
#
# Reference invocation (older vLLM):
#   vllm serve Qwen/Qwen3-8B --enable-reasoning --reasoning-parser deepseek_r1
#
# Modern vLLM (>=0.8.x) has dropped --enable-reasoning. Thinking mode is now
# controlled via the chat-template kwarg; the reasoning parser handles the
# <think>…</think> blocks that Qwen3 emits.
#
# Qwen3 uses Hermes-style tool calls, so --tool-call-parser=hermes is the
# correct parser. We keep tool-call support since the dataset is
# BFCL_v4_parallel (tool_call.jsonl).
# ---------------------------------------------------------------------------
QWEN3_TOOL_ARGS=(
  --enable-auto-tool-choice
  --tool-call-parser hermes
  --reasoning-parser deepseek_r1
  --default-chat-template-kwargs '{"enable_thinking":true}'
  --max-model-len 32768
  --trust-remote-code
)

# Flatten to a single string for the queue (bash arrays don't survive pipes cleanly).
QWEN3_TOOL_ARGS_STR="${QWEN3_TOOL_ARGS[*]}"

total_groups=0

# Queue format (pipe-separated):
#   output_dir | target | technique | spec_model | method | k | extra_vllm_args | extra_spec_json
#
# For baseline (no spec decode): spec_model="", method="", k=0.

enqueue_baseline() {
  # output_dir | target | extra_vllm_args
  local output_dir="$1" target="$2" extra_args="${3:-}"
  echo "${output_dir}|${target}|baseline|||0|${extra_args}|" >> "${QUEUE_FILE}"
  total_groups=$(( total_groups + 1 ))
}

enqueue_dflash() {
  # output_dir | target | spec | k | extra_vllm_args
  local output_dir="$1" target="$2" spec="$3" k="$4" extra_args="${5:-}"
  echo "${output_dir}|${target}|dflash|${spec}|dflash|${k}|${extra_args}|" >> "${QUEUE_FILE}"
  total_groups=$(( total_groups + 1 ))
}



# ============================================================================
# Qwen3-8B dense (BF16, single 24GB+ GPU)
# 8.2B total params, 36 layers, GQA (32 Q heads / 8 KV heads),
# native 32k ctx (131k via YaRN). Reasoning mode on.
# ============================================================================

# 1) Baseline: target only, no speculative decoding -> ./qwen3_8b/
enqueue_baseline "${BASELINE_DIR}" \
                 "Qwen/Qwen3-8B" \
                 "${QWEN3_TOOL_ARGS_STR}"

# 2) Dflash speculative decoding (k = 16) -> ./qwen3_8b_dflash/
enqueue_dflash "${DFLASH_DIR}" \
               "Qwen/Qwen3-8B" \
               "krishnateja95/Qwen3-8B-Dflash" \
               16 \
               "${QWEN3_TOOL_ARGS_STR}"

total_datasets=${#DATASETS[@]}
total_jobs=$(( total_groups * total_datasets ))

echo "============================================================"
echo " Qwen3-8B benchmark: baseline + Dflash speculative decoding"
echo " ${total_groups} server groups"
echo " ${total_datasets} datasets per group = ${total_jobs} total benchmark runs"
echo " Running ${NUM_WORKERS} parallel workers (${TENSOR_PARALLEL} GPUs each, TP=${TENSOR_PARALLEL})"
echo " Output folders: ./${BASELINE_DIR}/ and ./${DFLASH_DIR}/"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  detected=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || true)
  echo "Detected ${detected} GPUs."
  if [[ -z "$detected" ]] || (( detected == 0 )); then
    echo "ERROR: No GPUs detected." >&2
    exit 1
  fi
fi

gpu_worker() {
  local worker_id="$1"
  local port=$(( BASE_PORT + worker_id ))
  local gpu_start=$(( worker_id * TENSOR_PARALLEL ))
  local gpu_end=$(( gpu_start + TENSOR_PARALLEL - 1 ))

  local gpu_list=""
  for g in $(seq "${gpu_start}" "${gpu_end}"); do
    local phys_gpu="${AVAILABLE_GPUS[$g]}"
    if [[ -n "${gpu_list}" ]]; then gpu_list+=",${phys_gpu}"; else gpu_list="${phys_gpu}"; fi
  done

  local worker_fail=0

  while true; do
    local job_line=""
    {
      flock -x 200
      job_line=$(head -n 1 "${QUEUE_FILE}" 2>/dev/null || true)
      if [[ -n "${job_line}" ]]; then
        sed -i '1d' "${QUEUE_FILE}"
      fi
    } 200>"${QUEUE_LOCK}"

    if [[ -z "${job_line}" ]]; then
      break
    fi

    IFS='|' read -r output_dir target_model technique spec_decode_model method k extra_vllm_args extra_spec_json <<< "${job_line}"

    local target_short
    target_short=$(basename "${target_model}")
    local spec_short=""
    if [[ -n "${spec_decode_model}" ]]; then
      spec_short=$(basename "${spec_decode_model}")
    fi

    local outdir="${BASE_DIR}/${output_dir}/output"
    local logdir="${BASE_DIR}/${output_dir}/vllm_logs"
    mkdir -p "$outdir" "$logdir"

    # Skip if all datasets for this (technique, k) are already done
    local all_done=true
    for dataset in "${DATASETS[@]}"; do
      local dname="${dataset%.jsonl}"
      if [[ ! -s "${logdir}/vllm_${dname}_k${k}.log" ]]; then
        all_done=false
        break
      fi
    done
    if [[ "${all_done}" == "true" ]]; then
      echo "[Worker ${worker_id} GPUs ${gpu_list}] Skipping ${output_dir}/${technique} k=${k} (already complete)"
      continue
    fi

    if [[ "${technique}" == "baseline" ]]; then
      echo "[Worker ${worker_id} GPUs ${gpu_list}] Starting server: target=${target_short} technique=baseline (no spec decode)"
    else
      echo "[Worker ${worker_id} GPUs ${gpu_list}] Starting server: target=${target_short} technique=${technique} spec=${spec_short} k=${k}"
    fi

    local server_log="${logdir}/server_k${k}.log"
    local server_pid=""

    # Build --speculative-config only when we actually have a draft model
    local spec_flag=()
    if [[ -n "${spec_decode_model}" && -n "${method}" && "${k}" -gt 0 ]]; then
      local spec_config="{\"model\":\"${spec_decode_model}\",\"num_speculative_tokens\":${k},\"method\":\"${method}\"${extra_spec_json}}"
      spec_flag=(--speculative-config "${spec_config}")
    fi

    # Launch vLLM server
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${gpu_list}" \
    vllm serve "${target_model}" \
      --host 0.0.0.0 \
      --port "${port}" \
      --tensor-parallel-size "${TENSOR_PARALLEL}" \
      --no-enable-prefix-caching \
      --max-num-seqs 64 \
      --enforce-eager \
      --gpu-memory-utilization 0.92 \
      "${spec_flag[@]}" \
      ${extra_vllm_args} \
      > "${server_log}" 2>&1 &
    server_pid=$!

    local ready=false
    echo -n "[Worker ${worker_id}] Waiting for server on port ${port} (${output_dir}/${technique} k=${k})…"
    for _ in $(seq 1 "${WAIT_READY_TIMEOUT}"); do
      if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
        echo " ready."
        ready=true
        break
      fi
      if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo " server process died."
        break
      fi
      sleep 1
    done

    if [[ "${ready}" != "true" ]]; then
      echo "[Worker ${worker_id}] ERROR: Server failed to start for ${output_dir}/${technique} k=${k}"
      tail -n 30 "${server_log}" 2>/dev/null || true
      worker_fail=1
      kill "${server_pid}" 2>/dev/null || true
      wait "${server_pid}" 2>/dev/null || true
      sleep 2
      continue
    fi

    for dataset in "${DATASETS[@]}"; do
      local dataset_name="${dataset%.jsonl}"
      echo "[Worker ${worker_id}]   Benchmarking ${dataset_name} | ${output_dir}/${technique} k=${k}"

      local log_pos_before
      log_pos_before=$(wc -c < "${server_log}")

      local out="${outdir}/${target_short}_${dataset_name}_k${k}.json"

      # guidellm invocation
      if ! GUIDELLM__MAX_CONCURRENCY=128 \
           GUIDELLM__PREFERRED_ROUTE="chat_completions" \
           guidellm benchmark \
             --target "http://localhost:${port}/v1" \
             --data "RedHatAI/speculator_benchmarks" \
             --profile throughput \
             --rate 10 \
             --output-path "${out}" \
             --data-args "{\"data_files\": \"${dataset}\"}" \
             --data-column-mapper '{"text_column":"prompt"}' \
             > "${outdir}/guidellm_${dataset_name}_k${k}.log" 2>&1; then
        echo "[Worker ${worker_id}]   WARNING: guidellm failed for ${dataset_name} | ${output_dir}/${technique} k=${k}"
        worker_fail=1
      fi

      local log_pos_after
      log_pos_after=$(wc -c < "${server_log}")
      tail -c +"$(( log_pos_before + 1 ))" "${server_log}" | head -c "$(( log_pos_after - log_pos_before ))" > "${logdir}/vllm_${dataset_name}_k${k}.log"

      echo "[Worker ${worker_id}]   Done: ${dataset_name} | ${output_dir}/${technique} k=${k}"
    done

    echo "[Worker ${worker_id}] Stopping server for ${output_dir}/${technique} k=${k}"
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    rm -f "${server_log}"
    sleep 2
  done

  return ${worker_fail}
}

################################################################################
# Launch workers
################################################################################
WORKER_PIDS=()

cleanup() {
  echo "Cleaning up all workers…"
  for pid in "${WORKER_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${WORKER_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  rm -f "${QUEUE_FILE}" "${QUEUE_LOCK}"
}
trap cleanup EXIT

for worker_id in $(seq 0 $(( NUM_WORKERS - 1 ))); do
  gpu_worker "${worker_id}" &
  WORKER_PIDS+=($!)
done

echo "Launched ${NUM_WORKERS} workers (${TENSOR_PARALLEL} GPUs each). Waiting for completion…"

fail=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

rm -f "${QUEUE_FILE}" "${QUEUE_LOCK}"

echo ""
echo "============================================================"
if [[ $fail -eq 0 ]]; then
  echo "All benchmarks completed successfully."
  exit 0
else
  echo "One or more benchmarks had failures (see above)."
  exit 1
fi