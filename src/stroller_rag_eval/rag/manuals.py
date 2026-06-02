"""Manual identification helpers for product-specific retrieval"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import cast

from stroller_rag_eval.rag.retriever import MetadataFilter


@dataclass(frozen=True)
class ProductManual:
    """Known product/manual mapping for the synthetic stroller corpus"""

    label: str
    file_name: str
    aliases: tuple[str, ...]


PRODUCT_MANUALS = (
    ProductManual(
        label="citylite",
        file_name="citylite_manual.md",
        aliases=("citylite", "city lite", "cl-100", "cl100"),
    ),
    ProductManual(
        label="duoride",
        file_name="duoride_double_manual.md",
        aliases=(
            "duoride",
            "duoride double",
            "duo ride",
            "duo ride double",
            "dr-300",
            "dr300",
        ),
    ),
    ProductManual(
        label="trailpro",
        file_name="trailpro_jogger_manual.md",
        aliases=(
            "trailpro",
            "trailpro jogger",
            "trail pro",
            "trail pro jogger",
            "tj-200",
            "tj200",
        ),
    ),
)

_FUZZY_MATCH_THRESHOLD = 0.85


def manual_choices() -> tuple[str, ...]:
    """Return supported manual labels for CLI choices"""

    return tuple(manual.label for manual in PRODUCT_MANUALS)


def metadata_filter_for_manual(manual_name: str | None) -> MetadataFilter | None:
    """Build a Chroma metadata filter for a known manual label or file name"""

    if not manual_name:
        return None

    normalized_manual_name = _normalize_phrase(_source_file_name(manual_name))
    for manual in PRODUCT_MANUALS:
        identifiers = (
            manual.label,
            manual.file_name,
            Path(manual.file_name).stem,
            *manual.aliases,
        )
        normalized_identifiers = {
            _normalize_phrase(identifier) for identifier in identifiers
        }
        if normalized_manual_name in normalized_identifiers:
            return _metadata_filter_for_file_name(manual.file_name)

    raise ValueError(
        "Unknown manual "
        f"{manual_name!r}. Expected one of: {', '.join(manual_choices())}."
    )


def infer_metadata_filter_from_question(question: str) -> MetadataFilter | None:
    """Infer a product metadata filter when a question names one known manual"""

    normalized_question = _normalize_phrase(question)
    if not normalized_question:
        return None

    exact_matches = [
        manual
        for manual in PRODUCT_MANUALS
        if _question_contains_alias(normalized_question, manual)
    ]
    if len(exact_matches) == 1:
        return _metadata_filter_for_file_name(exact_matches[0].file_name)
    if len(exact_matches) > 1:
        return None

    fuzzy_matches = [
        manual
        for manual in PRODUCT_MANUALS
        if _question_contains_fuzzy_label(normalized_question, manual.label)
    ]
    if len(fuzzy_matches) == 1:
        return _metadata_filter_for_file_name(fuzzy_matches[0].file_name)

    return None


def _question_contains_alias(normalized_question: str, manual: ProductManual) -> bool:
    for alias in manual.aliases:
        normalized_alias = _normalize_phrase(alias)
        if re.search(rf"(^|\s){re.escape(normalized_alias)}($|\s)", normalized_question):
            return True
    return False


def _question_contains_fuzzy_label(normalized_question: str, label: str) -> bool:
    for token in normalized_question.split():
        if len(token) < 6:
            continue
        if SequenceMatcher(None, token, label).ratio() >= _FUZZY_MATCH_THRESHOLD:
            return True
    return False


def _metadata_filter_for_file_name(file_name: str) -> MetadataFilter:
    return cast(MetadataFilter, {"file_name": file_name})


def _normalize_phrase(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _source_file_name(source: str) -> str:
    source_without_anchor = source.split("#", 1)[0]
    return Path(source_without_anchor).name
