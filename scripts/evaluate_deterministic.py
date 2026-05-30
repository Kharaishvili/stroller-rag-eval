#!/usr/bin/env python
"""Run deterministic checks over the configured RAG pipeline"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from stroller_rag_eval.config import get_config
from stroller_rag_eval.evaluation.dataset import load_eval_examples
from stroller_rag_eval.evaluation.deterministic_eval import evaluate_records_deterministically
from stroller_rag_eval.evaluation.runner import run_rag_over_examples, write_records_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate source retrieval and refusal behavior deterministically."
    )
    parser.add_argument("--dataset", type=Path, help="Evaluation CSV path.")
    parser.add_argument("--output", type=Path, help="Deterministic CSV output path.")
    parser.add_argument(
        "--records-output",
        type=Path,
        help="JSONL path for normalized RAG records.",
    )
    parser.add_argument("--top-k", type=int, help="Number of chunks to retrieve.")
    parser.add_argument("--limit", type=int, help="Limit evaluation rows for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()

    if args.top_k:
        config = replace(config, top_k=args.top_k)

    dataset_path = args.dataset or config.eval_dataset_path
    output_path = args.output or config.eval_reports_dir / "deterministic" / "results.csv"
    records_output = (
        args.records_output
        or config.eval_reports_dir / "deterministic" / "normalized_records.jsonl"
    )

    examples = load_eval_examples(dataset_path, limit=args.limit)
    records = run_rag_over_examples(examples, config)
    write_records_jsonl(records, records_output)
    results = evaluate_records_deterministically(records, output_path=output_path)

    source_pass_rate = results["retrieved_expected_source"].mean()
    source_purity_rate = results["retrieved_only_expected_sources"].mean()
    refusal_pass_rate = results["refusal_behavior_ok"].mean()

    print("Deterministic evaluation complete")
    print(f"Examples evaluated: {len(records)}")
    print(f"Normalized records: {records_output}")
    print(f"Deterministic results: {output_path}")
    print(f"Expected source retrieval pass rate: {source_pass_rate:.1%}")
    print(f"Source filter purity pass rate: {source_purity_rate:.1%}")
    print(f"Refusal behavior pass rate: {refusal_pass_rate:.1%}")


if __name__ == "__main__":
    main()
