"""RAGAS adapter for normalized RAG evaluation records"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from stroller_rag_eval.evaluation.runner import RagEvalRecord


def records_to_ragas_rows(records: list[RagEvalRecord]) -> list[dict[str, Any]]:
    """Convert normalized records into RAGAS-friendly rows"""

    return [
        {
            "question": record.question,
            "answer": record.answer,
            "contexts": record.contexts,
            "ground_truth": record.ground_truth or "",
        }
        for record in records
    ]


def evaluate_records_with_ragas(
    records: list[RagEvalRecord],
    *,
    evaluator_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    output_path: Path | None = None,
    show_progress: bool = True,
    max_workers: int = 2,
    batch_size: int | None = 10,
    max_retries: int = 10,
    max_wait: int = 60,
) -> Any:
    """Run RAGAS over normalized RAG evaluation records"""

    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluation dependencies are not installed. "
            'Run `pip install -e ".[eval]"` first.'
        ) from exc

    dataset = Dataset.from_list(records_to_ragas_rows(records))
    result = evaluate(
        dataset=dataset,
        metrics=[
            AnswerRelevancy(strictness=1),
            ContextPrecision(),
            Faithfulness(),
            ContextRecall(),
        ],
        llm=ChatOpenAI(model=evaluator_model, temperature=0),
        embeddings=OpenAIEmbeddings(model=embedding_model),
        run_config=RunConfig(
            max_workers=max_workers,
            max_retries=max_retries,
            max_wait=max_wait,
        ),
        batch_size=batch_size,
        show_progress=show_progress,
    )

    if output_path is not None:
        write_ragas_result(result, output_path)

    return result


def write_ragas_result(result: Any, output_path: Path) -> None:
    """Persist a RAGAS result object as CSV when possible"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(result, "to_pandas"):
        result.to_pandas().to_csv(output_path, index=False)
        return

    if isinstance(result, dict):
        pd.DataFrame([result]).to_csv(output_path, index=False)
        return

    output_path.write_text(str(result), encoding="utf-8")
