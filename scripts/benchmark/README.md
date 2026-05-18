# Benchmark

Benchmark scripts for throughput and acceptance behavior.

- `guidellm_Qwen3_8B.sh`: parallel multi-GPU sweep across base/draft model combinations.
- `6_run_perf_benchmark.sh`: reusable eval controller entrypoint.
- `6_run_perf_bench.sh`: convenience notes/examples for benchmark commands.
- `plot_benchmark_rps_vs_itl.py`: generate per-benchmark RPS vs ITL PDF plots.

## Usage

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/benchmark/guidellm_Qwen3_8B.sh
```

Outputs are created in:

- `scripts/benchmark/eval_results/` (method-wise benchmark outputs)
- `scripts/benchmark/eval_results/benchmark_pdfs_rps_vs_itl/` (generated plots)

Prerequisites:

- `guidellm` and `vllm` installed in active environment.
- `CUDA_VISIBLE_DEVICES` set to the GPUs you want to use.
- Access to benchmark dataset (`RedHatAI/speculator_benchmarks` in script defaults).

Generate plots from benchmark JSON outputs:

```bash
python3 scripts/benchmark/plot_benchmark_rps_vs_itl.py \
  --eval-results-dir scripts/benchmark/eval_results \
  --output-dir scripts/benchmark/eval_results/benchmark_pdfs_rps_vs_itl
```
