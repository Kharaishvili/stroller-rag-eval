#!/usr/bin/env python
"""Run RAGAS evaluation over the configured RAG pipeline"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from stroller_rag_eval.config import get_config
from stroller_rag_eval.evaluation.dataset import load_eval_examples
from stroller_rag_eval.evaluation.ragas_eval import evaluate_records_with_ragas
from stroller_rag_eval.evaluation.runner import run_rag_over_examples, write_records_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS.")
    parser.add_argument("--dataset", type=Path, help="Evaluation CSV path.")
    parser.add_argument("--output", type=Path, help="RAGAS results output path.")
    parser.add_argument(
        "--records-output",
        type=Path,
        help="JSONL path for normalized RAG inputs sent to RAGAS.",
    )
    parser.add_argument("--top-k", type=int, help="Number of chunks to retrieve.")
    parser.add_argument("--limit", type=int, help="Limit evaluation rows for smoke tests.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable RAGAS progress output.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum concurrent RAGAS jobs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of RAGAS jobs to submit per batch.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Maximum RAGAS retry attempts for transient provider errors.",
    )
    parser.add_argument(
        "--max-wait",
        type=int,
        default=60,
        help="Maximum RAGAS retry wait in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()

    if args.top_k:
        config = replace(config, top_k=args.top_k)

    dataset_path = args.dataset or config.eval_dataset_path
    output_path = args.output or config.eval_reports_dir / "ragas" / "results.csv"
    records_output = (
        args.records_output
        or config.eval_reports_dir / "ragas" / "normalized_records.jsonl"
    )

    examples = load_eval_examples(dataset_path, limit=args.limit)
    records = run_rag_over_examples(examples, config)
    write_records_jsonl(records, records_output)
    result = evaluate_records_with_ragas(
        records,
        evaluator_model=config.evaluator_model,
        embedding_model=config.embedding_model,
        output_path=output_path,
        show_progress=not args.no_progress,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        max_wait=args.max_wait,
    )

    print("RAGAS evaluation complete")
    print(f"Examples evaluated: {len(records)}")
    print(f"Normalized records: {records_output}")
    print(f"RAGAS results: {output_path}")
    print(result)


if __name__ == "__main__":
    main()
