## Benchmark Model Variants

- `Qwen3-8B`
  - Quantization: None
  - Speculative Decoding (SD): No
  - Notes: Baseline

- `Qwen3-8B-FP8`
  - Quantization: FP8
  - Speculative Decoding (SD): No
  - Notes: Quantized baseline

- `Qwen3-8B-FP8-Dflash-FP8`
  - Quantization: FP8
  - Speculative Decoding (SD): Yes (Dflash for `Qwen3-8B-FP8`)

- `Qwen3-8B-FP8-Dflash`
  - Quantization: FP8
  - Speculative Decoding (SD): Yes (Dflash for `Qwen3-8B`)

- `Qwen3-8B-Dflash`
  - Quantization: None
  - Speculative Decoding (SD): Yes (Dflash for `Qwen3-8B`)

- `Qwen3-8B-Dflash-FP8`
  - Quantization: None
  - Speculative Decoding (SD): Yes (Dflash for `Qwen3-8B-FP8`)
