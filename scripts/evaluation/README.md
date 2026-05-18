# Evaluation

End-to-end GuideLLM evaluation and acceptance-rate parsing.

- `run_evaluation.sh`: starts vLLM, runs GuideLLM, parses acceptance rates.
- `6_run_guidellm.sh`: batch wrapper over the benchmark dataset files.
- `scripts/vllm_serve.sh`: robust vLLM launcher with health checks.
- `scripts/run_guidellm.sh`: GuideLLM runner for local/HF/built-in datasets.
- `scripts/vllm_stop.sh`: graceful shutdown helper.
- `scripts/parse_logs.py`: parses speculative decoding acceptance from logs.
- `configs/qwen3_dflash.env`: example config for config-driven runs.

## Usage

Single dataset:

```bash
bash scripts/evaluation/run_evaluation.sh \
  -b "Qwen/Qwen3-8B" \
  -s "./output/checkpoints/checkpoint_best" \
  -d "./dataset/speculator_benchmarks/math_reasoning.jsonl"
```

Batch run across common dataset files:

```bash
export DATASET_DIR="./dataset/speculator_benchmarks"
bash scripts/evaluation/6_run_guidellm.sh
```

`6_run_guidellm.sh` currently runs:

- Baseline for each target model in its `target_models` list
- Speculative runs for each draft model in its `draft_models` list

Config-driven run:

```bash
bash scripts/evaluation/run_evaluation.sh \
  -c scripts/evaluation/configs/qwen3_dflash.env
```
