# Benchmark

Benchmark scripts for throughput and acceptance behavior.

- `guidellm_Qwen3_8B.sh`: parallel multi-GPU sweep (baseline vs DFlash).
- `6_run_perf_benchmark.sh`: reusable eval controller entrypoint.
- `6_run_perf_bench.sh`: convenience notes/examples for benchmark commands.

## Usage

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/benchmark/guidellm_Qwen3_8B.sh
```

Outputs are created in:

- `qwen3_8b/`
- `qwen3_8b_dflash/`

Prerequisites:

- `guidellm` and `vllm` installed in active environment.
- `CUDA_VISIBLE_DEVICES` set to the GPUs you want to use.
- Access to benchmark dataset (`RedHatAI/speculator_benchmarks` in script defaults).
