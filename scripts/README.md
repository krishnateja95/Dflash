# Scripts

This directory groups all workflow scripts by purpose.

- `serving/`: start vLLM for baseline or trained checkpoints.
- `data_generation/`: regenerate assistant responses using base models.
- `preprocessing/`: sanitize and convert JSONL into train-ready Arrow data.
- `training/`: hidden-state extraction launchers and speculator training.
- `benchmark/`: throughput/latency benchmark orchestrators.
- `evaluation/`: end-to-end GuideLLM evaluation pipeline.
- `monitoring/`: Trackio inspection and dashboard helper.

Each subfolder has its own README with concrete usage examples.

Tip: copy `.env.example` at repo root to `.env` and `source .env` before
running these scripts to avoid repeatedly exporting common variables.
