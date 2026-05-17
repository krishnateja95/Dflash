#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from typing import Any

import aiohttp
from datasets import load_dataset

# Use orjson if available for ~3-5x faster JSON encode/decode.
try:
    import orjson

    def _json_loads(s):
        return orjson.loads(s)

    def _json_dumps_line(obj) -> str:
        return orjson.dumps(obj).decode("utf-8")

except ImportError:
    orjson = None

    def _json_loads(s):
        return json.loads(s)

    def _json_dumps_line(obj) -> str:
        return json.dumps(obj, ensure_ascii=False)


DATASET_CONFIGS = {
    "magpie": {
        "id": "Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered",
        "prompt_field": "instruction",
        "default_split": "train",
    },
    "ultrachat": {
        "id": "HuggingFaceH4/ultrachat_200k",
        "prompt_field": "prompt",
        "default_split": "train_sft",
    },
    "gsm8k": {
        "id": "openai/gsm8k",
        "prompt_field": "question",
        "default_split": "train",
        "subset": "main",
    },
}

logger = logging.getLogger(__name__)


class TransientHTTPError(Exception):
    """5xx response or other retriable HTTP-level failure."""


def parse_args():
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Regenerate responses from a dataset via vLLM Chat API."
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="vLLM OpenAI-compatible Chat Completions endpoint",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name exposed by vLLM (auto-detected if not specified)",
    )
    parser.add_argument(
        "--dataset",
        default="ultrachat",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to process",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split (defaults to dataset-specific split)",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help=(
            "Dataset subset/config name "
            "(auto-detected from dataset config if not specified)"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N rows")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=256,
        help=(
            "Max concurrent in-flight requests. With dp-size N, set this to "
            "roughly N * (per-replica max-num-seqs / 2) as a starting point."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="max_tokens for generation",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of attempts per request before giving up",
    )
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=1.0,
        help="Base delay (s) for exponential backoff between retries",
    )
    parser.add_argument(
        "--sock-read-timeout",
        type=float,
        default=600.0,
        help="Per-request socket read timeout in seconds (0 disables)",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=100,
        help="Flush output file every N records (0 to flush per record)",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="Print progress every N seconds (0 to disable)",
    )
    parser.add_argument(
        "--outfile",
        default=None,
        help="Output JSONL path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already in outfile (by uuid or idx)",
    )
    parser.add_argument(
        "--language-filter",
        default=None,
        help="Only process rows where language==this (e.g., EN)",
    )
    parser.add_argument(
        "--log-file",
        default="2_script.log",
        help="Path to runtime log file",
    )
    return parser.parse_args()


def configure_logging(log_file: str) -> None:
    """Configure logging to both stdout and file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use in filenames."""
    name = re.sub(r'[/\\:*?"<>|]', "_", name)
    name = name.replace(" ", "_")
    return name.strip("._")


def load_seen(path: str):
    """Load previously processed record IDs from output file."""
    seen = set()
    if not os.path.isfile(path):
        return seen

    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = obj.get("uuid") or obj.get("idx")
            # Also check inside metadata for the new output format
            if key is None and isinstance(obj.get("metadata"), dict):
                key = obj["metadata"].get("idx")
            if key is None:
                key = obj.get("id")
            if key is not None:
                seen.add(str(key))
    return seen


async def detect_model(endpoint: str) -> str:
    """Automatically detect the model name from the vLLM server."""
    models_endpoint = endpoint.replace("/v1/chat/completions", "/v1/models")

    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(models_endpoint) as response,
        ):
            data = await response.json()
            models = data.get("data", [])
            if models:
                model_name = models[0]["id"]
                logger.info("Auto-detected model: %s", model_name)
                return model_name
            raise ValueError("No models found at endpoint")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Failed to auto-detect model from {models_endpoint}: {e}\n"
            f"Please specify model with --model argument"
        ) from e


async def post_with_retry(
    session: aiohttp.ClientSession,
    endpoint: str,
    payload: dict,
    max_retries: int,
    base_delay: float,
) -> dict:
    """POST with exponential-backoff retry on transient failures."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            async with session.post(endpoint, json=payload) as response:
                if 500 <= response.status < 600:
                    body = await response.text()
                    raise TransientHTTPError(
                        f"HTTP {response.status}: {body[:200]}"
                    )
                response.raise_for_status()
                return await response.json(loads=_json_loads)
        except (
            aiohttp.ClientConnectionError,
            aiohttp.ServerDisconnectedError,
            asyncio.TimeoutError,
            TransientHTTPError,
        ) as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                await asyncio.sleep(delay)
                continue
            raise
    # Shouldn't reach here, but guard anyway
    assert last_error is not None
    raise last_error


class ProgressTracker:
    """Tracks counts and prints periodic progress messages."""

    def __init__(self):
        self.success = 0
        self.errors = 0
        self.start_time = time.time()
        self._last_total = 0
        self._last_time = self.start_time

    def record_success(self):
        self.success += 1

    def record_error(self):
        self.errors += 1

    async def reporter(self, interval: float):
        if interval <= 0:
            return
        try:
            while True:
                await asyncio.sleep(interval)
                now = time.time()
                total = self.success + self.errors
                elapsed = now - self.start_time
                window = now - self._last_time
                window_rate = (
                    (total - self._last_total) / window if window > 0 else 0.0
                )
                avg_rate = total / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "[progress] processed=%s success=%s errors=%s "
                    "rate=%.1f/s (avg %.1f/s) elapsed=%.0fs",
                    total,
                    self.success,
                    self.errors,
                    window_rate,
                    avg_rate,
                    elapsed,
                )
                self._last_total = total
                self._last_time = now
        except asyncio.CancelledError:
            return


async def worker(
    session: aiohttp.ClientSession,
    queue: "asyncio.Queue[dict[str, Any] | None]",
    args,
    out_fh,
    endpoint: str,
    progress: ProgressTracker,
    write_state: dict,
):
    """Worker that pulls items from queue and sends them to the vLLM endpoint."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        idx = item["idx"]
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "max_tokens": args.max_tokens,
        }

        start_time = time.time()
        try:
            data = await post_with_retry(
                session,
                endpoint,
                payload,
                max_retries=args.max_retries,
                base_delay=args.retry_base_delay,
            )

            choice = data["choices"][0]
            message = choice["message"]
            generated_text = message["content"]
            reasoning_content = message.get("reasoning_content")
            if reasoning_content is None:
                reasoning_content = message.get("reasoning")
            finish_reason = choice.get("finish_reason")
            latency = time.time() - start_time

            metadata = {
                "idx": idx,
                "finish_reason": finish_reason,
                "latency_s": round(latency, 3),
                "usage": data.get("usage"),
                "endpoint": endpoint,
            }
            if reasoning_content is not None:
                metadata["reasoning_content"] = reasoning_content

            output = {
                "id": item.get("uuid") or f"sample_{idx}",
                "conversations": [
                    {"from": "human", "value": item["prompt"]},
                    {"from": "gpt", "value": generated_text},
                ],
                "metadata": metadata,
            }
            out_fh.write(_json_dumps_line(output) + "\n")
            progress.record_success()
            logger.info(
                "[output] id=%s idx=%s finish_reason=%s latency_s=%.3f text=%s",
                output["id"],
                idx,
                finish_reason,
                latency,
                generated_text,
            )
        except Exception as e:  # noqa: BLE001
            error_output = {
                "id": item.get("uuid") or f"sample_{idx}",
                "conversations": [{"from": "human", "value": item["prompt"]}],
                "metadata": {
                    "idx": idx,
                    "error": repr(e),
                    "endpoint": endpoint,
                },
            }
            out_fh.write(_json_dumps_line(error_output) + "\n")
            progress.record_error()
            logger.exception("[output-error] id=%s idx=%s", error_output["id"], idx)
        finally:
            # Batched flush: avoids one fsync per record.
            write_state["written"] += 1
            if (
                args.flush_every <= 0
                or write_state["written"] % args.flush_every == 0
            ):
                out_fh.flush()
            queue.task_done()


async def main():
    """Main async function to process dataset through vLLM endpoint."""
    args = parse_args()
    configure_logging(args.log_file)

    endpoint = args.endpoint
    logger.info("Using endpoint: %s", endpoint)
    if orjson is None:
        logger.info(
            "(orjson not installed; using stdlib json. `pip install orjson` for a speedup.)"
        )

    if args.model is None:
        args.model = await detect_model(endpoint)
    logger.info("Using model: %s", args.model)

    dataset_config = DATASET_CONFIGS[args.dataset]
    dataset_id = dataset_config["id"]
    prompt_field = dataset_config["prompt_field"]

    split = args.split if args.split is not None else dataset_config["default_split"]
    subset = args.subset if args.subset is not None else dataset_config.get("subset")

    if args.outfile is None:
        model_name = args.model.split("/")[-1] if "/" in args.model else args.model
        model_name = sanitize_filename(model_name)
        args.outfile = f"{args.dataset}_{model_name}.jsonl"

    logger.info("Using dataset: %s", dataset_id)
    logger.info("Split: %s", split)
    logger.info("Prompt field: %s", prompt_field)
    logger.info("Output file: %s", args.outfile)
    logger.info("Concurrency: %s", args.concurrency)
    logger.info("Max tokens: %s", args.max_tokens)
    logger.info("Max retries: %s", args.max_retries)
    logger.info("Log file: %s", args.log_file)

    seen_ids = load_seen(args.outfile) if args.resume else set()
    if args.resume and seen_ids:
        logger.info(
            "Resume: %s records already in output, skipping.",
            len(seen_ids),
        )

    dataset = load_dataset(dataset_id, name=subset, split=split, streaming=True)

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)

    sock_read = args.sock_read_timeout if args.sock_read_timeout > 0 else None
    timeout = aiohttp.ClientTimeout(
        total=None, sock_connect=90, sock_read=sock_read
    )
    connector = aiohttp.TCPConnector(
        limit=0,  # 0 = unlimited; we cap concurrency ourselves via worker count.
        force_close=False,
        enable_cleanup_closed=True,
    )
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    progress = ProgressTracker()
    write_state = {"written": 0}

    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, headers=headers
    ) as session:
        # noqa: ASYNC230 - sync open is fine here; writes are short and infrequent
        with open(args.outfile, "a", encoding="utf-8") as output_file:  # noqa: ASYNC230
            workers = [
                asyncio.create_task(
                    worker(
                        session,
                        queue,
                        args,
                        output_file,
                        endpoint,
                        progress,
                        write_state,
                    )
                )
                for _ in range(args.concurrency)
            ]
            reporter_task = asyncio.create_task(
                progress.reporter(args.progress_interval)
            )

            processed_count = 0
            skipped_count = 0
            try:
                for index, row in enumerate(dataset):
                    if args.limit is not None and processed_count >= args.limit:
                        break

                    if (
                        args.language_filter
                        and row.get("language") != args.language_filter
                    ):
                        continue

                    prompt = row.get(prompt_field)
                    if not prompt:
                        continue

                    uuid = row.get("uuid")
                    key = str(uuid or index)
                    if key in seen_ids:
                        skipped_count += 1
                        continue

                    await queue.put(
                        {
                            "idx": index,
                            "uuid": uuid,
                            "prompt": prompt,
                        }
                    )
                    processed_count += 1

                # Signal workers to stop
                for _ in range(len(workers)):
                    await queue.put(None)
                await asyncio.gather(*workers)
            finally:
                reporter_task.cancel()
                try:
                    await reporter_task
                except asyncio.CancelledError:
                    pass
                output_file.flush()

            elapsed = time.time() - progress.start_time
            logger.info("========================================")
            logger.info(
                "Done. Submitted: %s, skipped (resume): %s",
                processed_count,
                skipped_count,
            )
            logger.info(
                "success=%s errors=%s elapsed=%.1fs avg_rate=%.1f/s",
                progress.success,
                progress.errors,
                elapsed,
                (progress.success + progress.errors) / elapsed,
            )
            logger.info("========================================")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)