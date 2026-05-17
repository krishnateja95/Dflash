# Training

Train DFlash speculator models using hidden-state extraction.

- `4_launch_vllm.py` / `4_launch_vllm.sh`: launch vLLM hidden-state extraction mode.
- `5_train.py` / `5_train.sh`: run distributed training with `torchrun`.

## Usage

In terminal 1 (hidden-state producer):

```bash
bash scripts/training/4_launch_vllm.sh
```

In terminal 2 (training):

```bash
export TRACKIO_DIR="./Qwen3-8B"
bash scripts/training/5_train.sh
```

Training checkpoints are written under `output/checkpoints/` by default.
