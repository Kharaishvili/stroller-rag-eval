"""Evaluation dataset loading utilities"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EvalExample:
    """A single question used to evaluate the RAG pipeline"""

    example_id: str
    manual: str | None
    question: str
    ground_truth: str | None
    expected_sources: tuple[str, ...]
    tags: tuple[str, ...]
    must_refuse: bool


def load_eval_examples(path: Path, *, limit: int | None = None) -> list[EvalExample]:
    """Load evaluation questions from a CSV file"""

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found at {path}. "
            "Create it from data/eval/stroller_qa.csv."
        )

    dataframe = pd.read_csv(path).fillna("")
    if "question" not in dataframe.columns:
        raise ValueError("Evaluation dataset must include a 'question' column.")

    examples: list[EvalExample] = []
    for row_number, (_, row) in enumerate(dataframe.iterrows(), start=1):
        question = str(row["question"]).strip()
        if not question:
            continue

        examples.append(
            EvalExample(
                example_id=_value_or_default(row.get("id"), f"row-{row_number}"),
                manual=_optional_string(row.get("manual")),
                question=question,
                ground_truth=_optional_string(row.get("ground_truth")),
                expected_sources=_split_list(row.get("expected_sources")),
                tags=_split_list(row.get("tags")),
                must_refuse=_parse_bool(row.get("must_refuse")),
            )
        )

        if limit is not None and len(examples) >= limit:
            break

    return examples


def _optional_string(value: object) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _value_or_default(value: object, default: str) -> str:
    if value is None:
        return default

    text = str(value).strip()
    return text or default


def _split_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()

    text = str(value).strip()
    if not text:
        return ()

    return tuple(item.strip() for item in text.split(";") if item.strip())


def _parse_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default

    text = str(value).strip().lower()
    if not text:
        return default

    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False

    raise ValueError(f"Expected a boolean value, got {value!r}.")
