# Dflash Training and Evaluation Scripts

This repository is an implementation of Dflash draft model training which contains:

- data regeneration with vLLM
- DFlash data preprocessing
- speculator training
- serving trained checkpoints
- GuideLLM benchmarking and acceptance analysis

Large artifacts (datasets, checkpoints, logs, outputs) are intentionally **not**
committed. Scripts are organized by workflow stage under `scripts/`.

Models: [Qwen3-8B-Dflash](https://huggingface.co/krishnateja95/Qwen3-8B-Dflash) [Qwen3-8B-FP8-Dflash](https://huggingface.co/krishnateja95/Qwen3-8B-FP8-Dflash)

## What Is Included

- All relevant orchestration `.sh` scripts.
- All corresponding workflow `.py` scripts.
- Folder-level READMEs with runnable commands.
- Portable shell wrappers (no machine-specific absolute paths required).

## What Is Excluded

- Checkpoints and model weights.
- Large regenerated JSONL files.
- Benchmark/evaluation logs and outputs.
- Prepared Arrow datasets and token cache artifacts.

## Repository Layout

```text
scripts/
  serving/         # Start baseline or trained model vLLM servers
  data_generation/ # Regenerate conversations from source datasets
  preprocessing/   # Build training-ready Arrow dataset + token statistics
  training/        # Hidden-state extraction server + speculator training
  benchmark/       # Throughput/latency benchmark drivers
  evaluation/      # End-to-end eval controller + helper scripts
  monitoring/      # Trackio convenience tooling
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For training/preprocessing scripts, install the `speculators` package in the
same environment (local editable install or your internal source).

Optional env setup:

```bash
cp .env.example .env
# edit values if needed
source .env
```

## Typical End-to-End Flow

1. `scripts/serving/1_start_vllm_serve.sh`
2. `scripts/data_generation/2_script.sh`
3. `scripts/preprocessing/3_prepare_data.sh`
4. `scripts/training/4_launch_vllm.sh`
5. `scripts/training/5_train.sh`
6. `scripts/serving/6_vllm_serve_dflash.sh`
7. `scripts/evaluation/run_evaluation.sh` or `scripts/benchmark/guidellm_Qwen3_8B.sh`

## Evaluation Config Example

```bash
bash scripts/evaluation/run_evaluation.sh \
  -c scripts/evaluation/configs/qwen3_dflash.env
```

## Notes

- Paths are parameterized with environment variables where possible.
- Script names intentionally preserve the original numbering to retain workflow
  familiarity.
- See section READMEs for concrete command examples.