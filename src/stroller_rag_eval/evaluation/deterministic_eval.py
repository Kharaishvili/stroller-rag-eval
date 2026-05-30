"""Deterministic checks over normalized RAG evaluation records"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stroller_rag_eval.evaluation.runner import RagEvalRecord


def evaluate_records_deterministically(
    records: list[RagEvalRecord],
    *,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Score source retrieval and refusal behavior without an LLM judge"""

    rows = []
    for record in records:
        expected_files = sorted(_source_file_names(record.expected_sources))
        retrieved_files = sorted(set(record.sources))
        expected_file_set = set(expected_files)
        retrieved_file_set = set(retrieved_files)

        retrieved_expected_source = (
            expected_file_set.issubset(retrieved_file_set)
            if expected_file_set
            else True
        )
        retrieved_only_expected_sources = (
            retrieved_file_set.issubset(expected_file_set)
            if expected_file_set
            else True
        )
        answer_refused = _answer_refused(record.answer)

        rows.append(
            {
                "id": record.example_id,
                "manual": record.manual or "",
                "question": record.question,
                "expected_sources": json.dumps(expected_files, ensure_ascii=False),
                "retrieved_sources": json.dumps(retrieved_files, ensure_ascii=False),
                "retrieved_expected_source": retrieved_expected_source,
                "retrieved_only_expected_sources": retrieved_only_expected_sources,
                "must_refuse": record.must_refuse,
                "answer_refused": answer_refused,
                "refusal_behavior_ok": answer_refused == record.must_refuse,
            }
        )

    dataframe = pd.DataFrame(rows)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)

    return dataframe


def _source_file_names(sources: list[str]) -> set[str]:
    return {
        Path(source.split("#", 1)[0]).name
        for source in sources
        if source.strip()
    }


def _answer_refused(answer: str) -> bool:
    normalized = " ".join(answer.lower().split())
    refusal_markers = (
        "not found in the manual",
        "not found in the provided context",
        "not mentioned in the manual",
        "not mentioned in the provided context",
        "do not know",
        "don't know",
        "insufficient context",
        "context is insufficient",
    )
    return any(marker in normalized for marker in refusal_markers)
