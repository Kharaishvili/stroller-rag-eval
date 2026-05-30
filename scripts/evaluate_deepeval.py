#!/usr/bin/env python
"""Run DeepEval evaluation over the configured RAG pipeline"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

from stroller_rag_eval.config import get_config
from stroller_rag_eval.evaluation.dataset import load_eval_examples
from stroller_rag_eval.evaluation.deepeval_eval import (
    DEFAULT_METRIC_NAMES,
    SUPPORTED_METRIC_NAMES,
    evaluate_records_with_deepeval,
)
from stroller_rag_eval.evaluation.runner import run_rag_over_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline with DeepEval."
    )
    parser.add_argument("--dataset", type=Path, help="Evaluation CSV path.")
    parser.add_argument(
        "--output",
        type=Path,
        help="JSONL path for normalized RAG inputs sent to DeepEval.",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        help="CSV path for DeepEval metric results.",
    )
    parser.add_argument("--top-k", type=int, help="Number of chunks to retrieve.")
    parser.add_argument("--limit", type=int, help="Limit evaluation rows for smoke tests.")
    parser.add_argument("--model", help="Evaluator model used by DeepEval.")
    parser.add_argument("--threshold", type=float, help="DeepEval metric threshold.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=SUPPORTED_METRIC_NAMES,
        default=list(DEFAULT_METRIC_NAMES),
        help=(
            "DeepEval metrics to run. Defaults to correctness against ground_truth. "
            "Faithfulness and contextual_relevancy are opt-in diagnostics."
        ),
    )
    parser.add_argument(
        "--async-mode",
        action="store_true",
        help="Run DeepEval metrics asynchronously. Default is serial for stability.",
    )
    parser.add_argument(
        "--include-reason",
        action="store_true",
        help="Ask DeepEval to generate metric reasons. Slower and more token-heavy.",
    )
    parser.add_argument(
        "--show-indicator",
        action="store_true",
        help="Show DeepEval's progress indicator.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum concurrent DeepEval test cases when --async-mode is enabled.",
    )
    parser.add_argument(
        "--truths-extraction-limit",
        type=int,
        default=5,
        help="Maximum truths extracted by the faithfulness metric.",
    )
    parser.add_argument(
        "--provider-timeout",
        type=float,
        default=60,
        help="OpenAI provider timeout in seconds for DeepEval judge calls.",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=180,
        help="DeepEval per-test-case timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()

    if args.top_k:
        config = replace(config, top_k=args.top_k)

    dataset_path = args.dataset or config.eval_dataset_path
    output_path = (
        args.output or config.eval_reports_dir / "deepeval" / "normalized_records.jsonl"
    )
    results_output_path = (
        args.results_output or config.eval_reports_dir / "deepeval" / "results.csv"
    )
    evaluator_model = args.model or config.evaluator_model
    threshold = (
        args.threshold if args.threshold is not None else config.deepeval_threshold
    )

    examples = load_eval_examples(dataset_path, limit=args.limit)
    records = run_rag_over_examples(examples, config)
    result = evaluate_records_with_deepeval(
        records,
        model=evaluator_model,
        threshold=threshold,
        output_path=output_path,
        async_mode=args.async_mode,
        include_reason=args.include_reason,
        show_indicator=args.show_indicator,
        max_concurrent=args.max_concurrent,
        truths_extraction_limit=args.truths_extraction_limit,
        metric_names=tuple(args.metrics),
        provider_timeout=args.provider_timeout,
        task_timeout=args.task_timeout,
    )
    result_rows = _deepeval_result_rows(result)
    _write_result_rows(result_rows, results_output_path)
    metric_summary = _metric_summary(result_rows)

    print("DeepEval evaluation complete")
    print(f"Examples evaluated: {len(records)}")
    print(f"Metrics: {', '.join(args.metrics)}")
    print(f"Normalized records: {output_path}")
    print(f"DeepEval results: {results_output_path}")
    for metric_name, summary in metric_summary.items():
        print(
            f"{metric_name}: avg={summary['average_score']:.4f}, "
            f"pass_rate={summary['pass_rate']:.1%}"
        )


def _deepeval_result_rows(result: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for test_result in getattr(result, "test_results", []):
        for metric_data in getattr(test_result, "metrics_data", []):
            rows.append(
                {
                    "test_name": getattr(test_result, "name", ""),
                    "metric": getattr(metric_data, "name", ""),
                    "score": getattr(metric_data, "score", None),
                    "threshold": getattr(metric_data, "threshold", None),
                    "success": getattr(metric_data, "success", None),
                    "error": getattr(metric_data, "error", None),
                    "evaluation_model": getattr(metric_data, "evaluation_model", ""),
                    "evaluation_cost": getattr(metric_data, "evaluation_cost", None),
                }
            )
    return rows


def _write_result_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "test_name",
        "metric",
        "score",
        "threshold",
        "success",
        "error",
        "evaluation_model",
        "evaluation_cost",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        metric_rows.setdefault(str(row["metric"]), []).append(row)

    summary = {}
    for metric_name, rows_for_metric in metric_rows.items():
        scores = [
            float(row["score"])
            for row in rows_for_metric
            if row["score"] is not None
        ]
        passes = [row["success"] is True for row in rows_for_metric]
        summary[metric_name] = {
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "pass_rate": sum(passes) / len(passes) if passes else 0.0,
        }
    return summary


if __name__ == "__main__":
    main()
