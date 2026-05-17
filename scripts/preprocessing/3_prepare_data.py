#!/usr/bin/env python3
"""
Prepare data for speculator training

This script processes an input dataset and:
1. Applies chat template + tokenizes each sample
2. Produces a loss/assistant mask for each sample
3. Records token frequency statistics

The output of this script is:
1. Processed dataset ready for online training or offline datagen in output_dir
2. Token frequency statistics file at token_freq_path

Preprocessing will be skipped if the dataset already exists at the output directory.
Token frequencies are saved in the output directory by default.

Usage:
    python prepare_data.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --data sharegpt_a.jsonl sharegpt_b.jsonl \
        --output ./training_data \
        --max-samples 5000
"""

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import List

from speculators.data_generation.logging_utils import PipelineLogger
from speculators.data_generation.preprocessing import (
    load_and_preprocess_dataset,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = PipelineLogger(__name__)


def _normalize_usage(usage: object) -> dict[str, int | None]:
    """Normalize metadata.usage into a fixed nullable schema."""
    usage_dict = usage if isinstance(usage, dict) else {}
    return {
        "prompt_tokens": usage_dict.get("prompt_tokens"),
        "total_tokens": usage_dict.get("total_tokens"),
        "completion_tokens": usage_dict.get("completion_tokens"),
        "prompt_tokens_details": usage_dict.get("prompt_tokens_details"),
    }


def _normalize_metadata(metadata: object) -> dict[str, object]:
    """Normalize metadata into a stable schema expected by HF datasets."""
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    return {
        "idx": metadata_dict.get("idx"),
        "finish_reason": metadata_dict.get("finish_reason"),
        "latency_s": metadata_dict.get("latency_s"),
        "usage": _normalize_usage(metadata_dict.get("usage")),
        "endpoint": metadata_dict.get("endpoint"),
    }


def _is_problematic_record(record: dict[str, object]) -> bool:
    """Return True for rows that should be dropped before HF loading."""
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return True

    # Failed requests are serialized with metadata.error and incomplete fields.
    error = metadata.get("error")
    if isinstance(error, str) and error.strip():
        return True

    return False


def sanitize_jsonl_dataset(input_path: Path, sanitized_dir: Path) -> Path:
    """Rewrite JSONL, dropping problematic rows and normalizing metadata schema."""
    sanitized_dir.mkdir(parents=True, exist_ok=True)
    sanitized_path = sanitized_dir / f"{input_path.stem}.filtered{input_path.suffix}"
    if (
        sanitized_path.exists()
        and sanitized_path.stat().st_mtime >= input_path.stat().st_mtime
    ):
        log.info(f"Reusing sanitized dataset: {sanitized_path}")
        return sanitized_path

    total_rows = 0
    modified_rows = 0
    dropped_rows = 0

    with input_path.open("r", encoding="utf-8") as src, sanitized_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue

            total_rows += 1
            record = json.loads(line)
            if _is_problematic_record(record):
                dropped_rows += 1
                continue

            original_metadata = record.get("metadata")
            normalized_metadata = _normalize_metadata(original_metadata)

            if original_metadata != normalized_metadata:
                modified_rows += 1

            record["metadata"] = normalized_metadata
            dst.write(json.dumps(record, ensure_ascii=True))
            dst.write("\n")

    if dropped_rows > 0:
        log.warning(
            f"Dropped {dropped_rows}/{total_rows} problematic rows in {input_path}"
        )
    if modified_rows > 0:
        log.warning(
            f"Sanitized metadata for {modified_rows}/{total_rows} rows in {input_path} "
            "due to inconsistent metadata schema"
        )
    if dropped_rows == 0 and modified_rows == 0:
        log.info(f"Dataset schema already consistent: {input_path}")

    return sanitized_path


def sanitize_local_data_paths(data_paths: list[str], output_dir: Path) -> list[str]:
    """Normalize local JSONL files before preprocessing."""
    sanitized_dir = output_dir / "_sanitized_inputs"
    sanitized_paths: list[str] = []

    for raw_path in data_paths:
        data_path = Path(raw_path)
        if data_path.exists() and data_path.suffix == ".jsonl":
            sanitized_paths.append(
                str(sanitize_jsonl_dataset(data_path, sanitized_dir))
            )
        else:
            sanitized_paths.append(raw_path)

    return sanitized_paths


def normalize_data_paths(raw_data_args: list[list[str]]) -> List[str]:
    """Flatten argparse --data values into a simple list of paths."""
    return [path for group in raw_data_args for path in group]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for speculator training")

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path for target model",
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        action="append",
        nargs="+",
        required=True,
        help=(
            "One or more paths to training data. "
            "You can pass multiple files after one --data."
        ),
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=8192,
        help="Maximum sequence length for preprocessing and model (default: 8192)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: None, process all)",
    )
    parser.add_argument(
        "--token-freq-path",
        type=str,
        default=None,
        help=(
            "Path to save token frequency distribution"
            "(default: args.output / 'token_freq.pt')"
        ),
    )
    parser.add_argument(
        "--assistant-pattern",
        type=str,
        default=None,
        help=(
            "Custom regex pattern for matching assistant responses. "
            "If not provided, auto-detected from chat template."
        ),
    )
    parser.add_argument(
        "--turn-dropout",
        action="store_true",
        help=(
            "Enable turn dropout: randomly keeps first N consecutive turns "
            "per conversation for data augmentation."
        ),
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Directory to save output dataset (default: ./output)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Forcibly rerun `prepare_data.py`.Deletes existing content in output dir"
        ),
    )

    # Processing arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (must match preprocessing seed, default: 0)",
    )
    parser.add_argument(
        "--num-preprocessing-workers",
        type=int,
        default=8,
        help="Number of CPU processes for dataset preprocessing (default: 8)",
    )
    parser.add_argument(
        "--minimum-valid-tokens",
        type=int,
        default=None,
        help=(
            "Drop samples whose loss mask contains fewer than this many "
            "trainable tokens."
        ),
    )
    args = parser.parse_args()
    args.data = normalize_data_paths(args.data)
    return args


def main():
    args = parse_args()

    log.section("Preparing data")
    log.config(
        {
            "Target Model": args.model,
            "Dataset": args.data,
            "Output Dir": args.output,
        }
    )

    output = Path(args.output)
    if output.exists():
        if not args.overwrite and glob.glob(str(output / "*.arrow")):
            log.warning(
                "Dataset files already exists in output directory, skipping "
                "preprocessing. To existing overwrite files use --overwrite."
            )
            sys.exit(0)
    else:
        output.mkdir(parents=True)

    token_freq_path = (
        output / "token_freq.pt"
        if args.token_freq_path is None
        else Path(args.token_freq_path)
    )
    data_paths = sanitize_local_data_paths(args.data, output)

    dataset, _ = load_and_preprocess_dataset(
        target_model_path=args.model,
        train_data_paths=data_paths,
        seq_length=args.seq_length,
        build_dataset_num_proc=args.num_preprocessing_workers,
        seed=args.seed,
        max_samples=args.max_samples,
        token_freq_path=token_freq_path,
        assistant_pattern=args.assistant_pattern,
        turn_dropout=args.turn_dropout,
        minimum_valid_tokens=args.minimum_valid_tokens,
    )

    log.info("Done preparing data")
    log.section(f"Writing dataset to {args.output}")
    dataset.save_to_disk(args.output)


if __name__ == "__main__":
    main()