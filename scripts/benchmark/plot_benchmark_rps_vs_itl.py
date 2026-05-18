#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


METHOD_LABELS = {
    "qwen_qwen3-8b__baseline": "Qwen3-8B",
    "qwen_qwen3-8b-fp8__baseline": "Qwen3-8B-FP8",
    "qwen_qwen3-8b-fp8__krishnateja95_qwen3-8b-dflash": "Qwen3-8B-FP8-Dflash",
    "qwen_qwen3-8b__krishnateja95_qwen3-8b-dflash": "Qwen3-8B-Dflash",
    "qwen_qwen3-8b-fp8__krishnateja95_qwen3-8b-fp8-dflash": "Qwen3-8B-FP8-Dflash-FP8",
    "qwen_qwen3-8b__krishnateja95_qwen3-8b-fp8-dflash": "Qwen3-8B-Dflash-FP8",
}


BENCHMARK_REGEX = re.compile(r"^.+?_(.+)_k\d+\.json$")


def extract_benchmark_name(file_name: str) -> Optional[str]:
    match = BENCHMARK_REGEX.match(file_name)
    if not match:
        return None
    return match.group(1)


def load_point(json_path: Path) -> Optional[Tuple[float, float]]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    try:
        benchmark = data["benchmarks"][0]
        metrics = benchmark["metrics"]
        rps_median = metrics["requests_per_second"]["total"]["median"]
        itl_median = metrics["inter_token_latency_ms"]["total"]["median"]
        return float(rps_median), float(itl_median)
    except Exception:
        return None


def collect_points(eval_results_dir: Path) -> Dict[str, List[Tuple[str, str, float, float]]]:
    points_by_benchmark: Dict[str, List[Tuple[str, str, float, float]]] = {}

    for json_path in sorted(eval_results_dir.rglob("*.json")):
        rel_parts = json_path.relative_to(eval_results_dir).parts
        if len(rel_parts) < 3:
            continue

        method_dir = rel_parts[0]
        benchmark_name = extract_benchmark_name(json_path.name)
        if benchmark_name is None:
            continue

        point = load_point(json_path)
        if point is None:
            continue

        rps_median, itl_median = point
        label = METHOD_LABELS.get(method_dir, method_dir)
        points_by_benchmark.setdefault(benchmark_name, []).append(
            (method_dir, label, rps_median, itl_median)
        )

    return points_by_benchmark


def plot_benchmark(
    benchmark_name: str,
    rows: List[Tuple[str, str, float, float]],
    output_pdf: Path,
) -> None:
    rows_sorted = sorted(rows, key=lambda x: x[2])

    fig, ax = plt.subplots(figsize=(11, 7))

    for _, label, rps_median, itl_median in rows_sorted:
        ax.scatter(rps_median, itl_median, s=90, alpha=0.9, label=label)
        ax.annotate(
            label,
            xy=(rps_median, itl_median),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
        )

    ax.set_title(f"{benchmark_name} - Median ITL (ms) vs Requests Per Second", fontsize=18, weight="bold")
    ax.set_xlabel("Requests Per Second (median)", fontsize=14)
    ax.set_ylabel("Median ITL (ms)", fontsize=14)
    ax.grid(True, alpha=0.25)

    # De-duplicate legend labels while preserving order.
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        unique_handles.append(h)
        unique_labels.append(l)
    ax.legend(unique_handles, unique_labels, loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(output_pdf, format="pdf")
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate one benchmark PDF plot with six methods overlaid."
    )
    parser.add_argument(
        "--eval-results-dir",
        default=str(script_dir / "eval_results"),
        help="Root directory that contains all method result folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "eval_results" / "benchmark_pdfs_rps_vs_itl"),
        help="Directory where per-benchmark PDF files will be written.",
    )
    args = parser.parse_args()

    eval_results_dir = Path(args.eval_results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points_by_benchmark = collect_points(eval_results_dir)
    if not points_by_benchmark:
        raise SystemExit("No matching benchmark points found.")

    written = []
    for benchmark_name in sorted(points_by_benchmark):
        output_pdf = output_dir / f"{benchmark_name}_rps_vs_itl.pdf"
        plot_benchmark(benchmark_name, points_by_benchmark[benchmark_name], output_pdf)
        written.append(output_pdf)

    print(f"Wrote {len(written)} PDFs to: {output_dir}")
    for pdf in written:
        print(pdf)


if __name__ == "__main__":
    main()
