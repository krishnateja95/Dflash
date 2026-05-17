# Serving

Scripts to start vLLM servers.

- `1_start_vllm_serve.sh`: serves base `Qwen/Qwen3-8B` with tuned scheduler flags.
- `6_vllm_serve_dflash.sh`: serves a trained DFlash checkpoint.

## Usage

```bash
export VENV_PATH="$HOME/virtual_envs/Dflash/bin/activate"
bash scripts/serving/1_start_vllm_serve.sh
```

```bash
export CHECKPOINT_PATH="./output/checkpoints/checkpoint_best"
bash scripts/serving/6_vllm_serve_dflash.sh
```
