"""Shared RAG evaluation runner and report helpers"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import pandas as pd
from langchain_core.documents import Document

from stroller_rag_eval.config import RagConfig
from stroller_rag_eval.evaluation.dataset import EvalExample
from stroller_rag_eval.rag.pipeline import answer_question
from stroller_rag_eval.rag.retriever import MetadataFilter


@dataclass(frozen=True)
class RagEvalRecord:
    """Normalized output from one RAG pipeline evaluation run"""

    example_id: str
    manual: str | None
    question: str
    answer: str
    contexts: list[str]
    sources: list[str]
    ground_truth: str | None
    expected_sources: list[str]
    tags: list[str]
    must_refuse: bool
    retrieval_filter: MetadataFilter | None


def run_rag_over_examples(
    examples: list[EvalExample],
    config: RagConfig,
) -> list[RagEvalRecord]:
    """Run the RAG pipeline for each evaluation example"""

    if not examples:
        raise ValueError("No evaluation examples were loaded.")

    records: list[RagEvalRecord] = []
    for example in examples:
        retrieval_filter = metadata_filter_for_example(example)
        response = answer_question(
            example.question,
            config,
            metadata_filter=retrieval_filter,
        )
        records.append(
            RagEvalRecord(
                example_id=example.example_id,
                manual=example.manual,
                question=example.question,
                answer=response.answer,
                contexts=[document.page_content for document in response.retrieved_documents],
                sources=_document_sources(response.retrieved_documents),
                ground_truth=example.ground_truth,
                expected_sources=list(example.expected_sources),
                tags=list(example.tags),
                must_refuse=example.must_refuse,
                retrieval_filter=retrieval_filter,
            )
        )

    return records


def records_to_dataframe(records: list[RagEvalRecord]) -> pd.DataFrame:
    """Represent normalized evaluation records as a flat dataframe"""

    rows = []
    for record in records:
        rows.append(
            {
                "id": record.example_id,
                "manual": record.manual or "",
                "question": record.question,
                "answer": record.answer,
                "ground_truth": record.ground_truth or "",
                "contexts": json.dumps(record.contexts, ensure_ascii=False),
                "sources": json.dumps(record.sources, ensure_ascii=False),
                "expected_sources": json.dumps(
                    record.expected_sources, ensure_ascii=False
                ),
                "tags": json.dumps(record.tags, ensure_ascii=False),
                "must_refuse": record.must_refuse,
                "retrieval_filter": json.dumps(
                    record.retrieval_filter or {}, ensure_ascii=False
                ),
            }
        )
    return pd.DataFrame(rows)


def write_records_jsonl(records: list[RagEvalRecord], path: Path) -> None:
    """Write normalized RAG outputs for debugging and reproducibility"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _document_sources(documents: list[Document]) -> list[str]:
    sources: list[str] = []
    for document in documents:
        source = document.metadata.get(
            "file_name",
            document.metadata.get("source_path", document.metadata.get("source")),
        )
        sources.append(_source_file_name(str(source)) if source else "unknown")
    return sources


def metadata_filter_for_example(example: EvalExample) -> MetadataFilter | None:
    """Build a Chroma metadata filter from expected source file names"""

    file_names = sorted(
        {
            _source_file_name(source)
            for source in example.expected_sources
            if source.strip()
        }
    )
    if not file_names:
        return None
    if len(file_names) == 1:
        return cast(MetadataFilter, {"file_name": file_names[0]})

    return cast(MetadataFilter, {"file_name": {"$in": file_names}})


def _source_file_name(source: str) -> str:
    source_without_anchor = source.split("#", 1)[0]
    return Path(source_without_anchor).name
