# Data Generation

Regenerate assistant outputs from source datasets through an OpenAI-compatible
vLLM endpoint.

- `2_script.py`: async high-throughput generator with retries/resume support.
- `2_script.sh`: preset runs for Magpie and UltraChat.

## Usage

Start a serving endpoint first (`scripts/serving/1_start_vllm_serve.sh`), then:

```bash
bash scripts/data_generation/2_script.sh
```

Outputs:

- `scripts/data_generation/magpie_regeneration.jsonl`
- `scripts/data_generation/ultrachat_regeneration.jsonl`
