# Preprocessing

Prepare regenerated JSONL files into train-ready Arrow datasets and token
frequency files.

- `3_prepare_data.py`: sanitizes metadata and builds preprocessed dataset.
- `3_prepare_data.sh`: preset invocation for the regenerated Qwen3 data.

## Usage

```bash
bash scripts/preprocessing/3_prepare_data.sh
```

Default output directory:

- `prepared_data/`
