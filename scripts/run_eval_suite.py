#!/usr/bin/env python
"""Run the full RAG evaluation suite over one shared set of records"""

from __future__ import annotations

import argparse
import csv
import json
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
from stroller_rag_eval.evaluation.deterministic_eval import (
    evaluate_records_deterministically,
)
from stroller_rag_eval.evaluation.ragas_eval import evaluate_records_with_ragas
from stroller_rag_eval.evaluation.runner import run_rag_over_examples, write_records_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic, RAGAS, and DeepEval checks together."
    )
    parser.add_argument("--dataset", type=Path, help="Evaluation CSV path.")
    parser.add_argument("--report-dir", type=Path, help="Suite report directory.")
    parser.add_argument("--top-k", type=int, help="Number of chunks to retrieve.")
    parser.add_argument("--limit", type=int, help="Limit rows for smoke tests.")
    parser.add_argument(
        "--skip-deterministic",
        action="store_true",
        help="Skip deterministic source/refusal checks.",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS judge metrics.",
    )
    parser.add_argument(
        "--skip-deepeval",
        action="store_true",
        help="Skip DeepEval judge metrics.",
    )
    parser.add_argument(
        "--include-refusals-in-judges",
        action="store_true",
        help="Also send must_refuse=true rows to RAGAS and DeepEval.",
    )
    parser.add_argument(
        "--ragas-progress",
        action="store_true",
        help="Show RAGAS progress output.",
    )
    parser.add_argument(
        "--ragas-max-workers",
        type=int,
        default=2,
        help="Maximum concurrent RAGAS jobs.",
    )
    parser.add_argument(
        "--ragas-batch-size",
        type=int,
        default=10,
        help="Number of RAGAS jobs to submit per batch.",
    )
    parser.add_argument(
        "--ragas-max-retries",
        type=int,
        default=10,
        help="Maximum RAGAS retry attempts for transient provider errors.",
    )
    parser.add_argument(
        "--ragas-max-wait",
        type=int,
        default=60,
        help="Maximum RAGAS retry wait in seconds.",
    )
    parser.add_argument(
        "--deepeval-metrics",
        nargs="+",
        choices=SUPPORTED_METRIC_NAMES,
        default=list(DEFAULT_METRIC_NAMES),
        help="DeepEval metrics to run.",
    )
    parser.add_argument("--deepeval-model", help="Evaluator model used by DeepEval.")
    parser.add_argument(
        "--deepeval-threshold",
        type=float,
        help="DeepEval metric threshold.",
    )
    parser.add_argument(
        "--deepeval-provider-timeout",
        type=float,
        default=60,
        help="OpenAI provider timeout in seconds for DeepEval judge calls.",
    )
    parser.add_argument(
        "--deepeval-task-timeout",
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
    report_dir = args.report_dir or config.eval_reports_dir / "eval_suite"
    report_dir.mkdir(parents=True, exist_ok=True)

    records_path = report_dir / "normalized_records.jsonl"
    answerable_records_path = report_dir / "answerable_records.jsonl"
    refusal_records_path = report_dir / "refusal_records.jsonl"
    summary_path = report_dir / "summary.json"

    examples = load_eval_examples(dataset_path, limit=args.limit)
    records = run_rag_over_examples(examples, config)
    answerable_records = [record for record in records if not record.must_refuse]
    refusal_records = [record for record in records if record.must_refuse]
    judge_records = records if args.include_refusals_in_judges else answerable_records

    write_records_jsonl(records, records_path)
    write_records_jsonl(answerable_records, answerable_records_path)
    write_records_jsonl(refusal_records, refusal_records_path)

    summary: dict[str, Any] = {
        "dataset": str(dataset_path),
        "examples_evaluated": len(records),
        "answerable_examples": len(answerable_records),
        "refusal_examples": len(refusal_records),
        "judge_examples": len(judge_records),
        "judge_scope": (
            "all_rows" if args.include_refusals_in_judges else "answerable_only"
        ),
        "top_k": config.top_k,
        "records": str(records_path),
        "answerable_records": str(answerable_records_path),
        "refusal_records": str(refusal_records_path),
        "reports": {},
        "metrics": {},
    }

    print("Eval suite started")
    print(f"Examples evaluated: {len(records)}")
    print(f"Answerable examples: {len(answerable_records)}")
    print(f"Refusal examples: {len(refusal_records)}")
    print(f"Judge examples: {len(judge_records)}")
    print(f"Normalized records: {records_path}")

    if not args.skip_deterministic:
        deterministic_path = report_dir / "deterministic_results.csv"
        deterministic_results = evaluate_records_deterministically(
            records,
            output_path=deterministic_path,
        )
        deterministic_summary = {
            "expected_source_retrieval_pass_rate": float(
                deterministic_results["retrieved_expected_source"].mean()
            ),
            "source_filter_purity_pass_rate": float(
                deterministic_results["retrieved_only_expected_sources"].mean()
            ),
            "refusal_behavior_pass_rate": float(
                deterministic_results["refusal_behavior_ok"].mean()
            ),
        }
        summary["reports"]["deterministic"] = str(deterministic_path)
        summary["metrics"]["deterministic"] = deterministic_summary

    if not args.skip_ragas:
        ragas_path = report_dir / "ragas_results.csv"
        if judge_records:
            ragas_result = evaluate_records_with_ragas(
                judge_records,
                evaluator_model=config.evaluator_model,
                embedding_model=config.embedding_model,
                output_path=ragas_path,
                show_progress=args.ragas_progress,
                max_workers=args.ragas_max_workers,
                batch_size=args.ragas_batch_size,
                max_retries=args.ragas_max_retries,
                max_wait=args.ragas_max_wait,
            )
            summary["reports"]["ragas"] = str(ragas_path)
            summary["metrics"]["ragas"] = _ragas_summary(ragas_result)
        else:
            summary["metrics"]["ragas"] = {"skipped": "no_answerable_records"}

    if not args.skip_deepeval:
        deepeval_path = report_dir / "deepeval_results.csv"
        if judge_records:
            deepeval_result = evaluate_records_with_deepeval(
                judge_records,
                model=args.deepeval_model or config.evaluator_model,
                threshold=(
                    args.deepeval_threshold
                    if args.deepeval_threshold is not None
                    else config.deepeval_threshold
                ),
                metric_names=tuple(args.deepeval_metrics),
                provider_timeout=args.deepeval_provider_timeout,
                task_timeout=args.deepeval_task_timeout,
            )
            deepeval_rows = _deepeval_result_rows(deepeval_result)
            _write_deepeval_rows(deepeval_rows, deepeval_path)
            summary["reports"]["deepeval"] = str(deepeval_path)
            summary["metrics"]["deepeval"] = _deepeval_summary(deepeval_rows)
        else:
            summary["metrics"]["deepeval"] = {"skipped": "no_answerable_records"}

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Eval suite complete")
    print(f"Summary: {summary_path}")
    _print_summary(summary)


def _ragas_summary(result: Any) -> dict[str, float]:
    if hasattr(result, "to_pandas"):
        dataframe = result.to_pandas()
        numeric_columns = dataframe.select_dtypes(include="number")
        return {
            column: _clean_float(value)
            for column, value in numeric_columns.mean().to_dict().items()
        }

    if hasattr(result, "items"):
        return {
            str(key): _clean_float(value)
            for key, value in result.items()
            if isinstance(value, int | float)
        }

    return {}


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


def _write_deepeval_rows(rows: list[dict[str, Any]], path: Path) -> None:
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


def _deepeval_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
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


def _print_summary(summary: dict[str, Any]) -> None:
    print(
        "Record split: "
        f"answerable={summary['answerable_examples']}, "
        f"refusal={summary['refusal_examples']}, "
        f"judged={summary['judge_examples']} "
        f"({summary['judge_scope']})"
    )

    for report_name, path in summary["reports"].items():
        print(f"{report_name.title()} report: {path}")

    deterministic = summary["metrics"].get("deterministic")
    if deterministic:
        print(
            "Deterministic: "
            f"expected_source={deterministic['expected_source_retrieval_pass_rate']:.1%}, "
            f"source_purity={deterministic['source_filter_purity_pass_rate']:.1%}, "
            f"refusal={deterministic['refusal_behavior_pass_rate']:.1%}"
        )

    ragas = summary["metrics"].get("ragas")
    if ragas:
        if "skipped" in ragas:
            print(f"RAGAS skipped: {ragas['skipped']}")
        else:
            ragas_text = ", ".join(
                f"{metric}={score:.4f}" for metric, score in ragas.items()
            )
            print(f"RAGAS: {ragas_text}")

    deepeval = summary["metrics"].get("deepeval")
    if deepeval:
        if "skipped" in deepeval:
            print(f"DeepEval skipped: {deepeval['skipped']}")
        else:
            for metric_name, metric_summary in deepeval.items():
                print(
                    f"DeepEval {metric_name}: "
                    f"avg={metric_summary['average_score']:.4f}, "
                    f"pass_rate={metric_summary['pass_rate']:.1%}"
                )


def _clean_float(value: Any) -> float:
    return round(float(value), 4)


if __name__ == "__main__":
    main()
