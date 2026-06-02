"""Retrieval helpers"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Protocol, TypeAlias

from chromadb.types import Where
from langchain_core.documents import Document


MetadataFilter: TypeAlias = Where


class DocumentRetriever(Protocol):
    """Retriever interface used by hybrid retrieval"""

    def invoke(self, question: str) -> list[Document]:
        ...


class VectorStore(Protocol):
    """Vector-store interface used by hybrid retrieval"""

    def as_retriever(self, search_kwargs: dict[str, object]) -> DocumentRetriever:
        ...

    def get(
        self,
        *,
        where: MetadataFilter | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        ...

STOPWORDS = {
    "about",
    "after",
    "allow",
    "also",
    "and",
    "any",
    "are",
    "can",
    "child",
    "children",
    "citylite",
    "does",
    "double",
    "duoride",
    "for",
    "from",
    "have",
    "how",
    "into",
    "jogger",
    "manual",
    "may",
    "one",
    "should",
    "stroller",
    "the",
    "this",
    "trailpro",
    "use",
    "used",
    "what",
    "when",
    "with",
}
MIN_KEYWORD_SCORE = 2.0


def retrieve_documents(
    question: str,
    vector_store: VectorStore,
    *,
    top_k: int,
    metadata_filter: MetadataFilter | None = None,
) -> list[Document]:
    """Retrieve relevant chunks with vector search plus keyword fallback"""

    search_kwargs: dict[str, object] = {"k": top_k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    vector_documents = list(retriever.invoke(question))
    keyword_documents = _keyword_fallback_documents(
        question,
        vector_store,
        metadata_filter=metadata_filter,
        top_k=top_k,
    )
    return _merge_documents(keyword_documents, vector_documents, top_k=top_k)


def _keyword_fallback_documents(
    question: str,
    vector_store: VectorStore,
    *,
    metadata_filter: MetadataFilter | None,
    top_k: int,
) -> list[Document]:
    query_terms = _keyword_terms(question)
    if not query_terms:
        return []

    try:
        result = vector_store.get(
            where=metadata_filter,
            include=["documents", "metadatas"],
        )
    except Exception:
        return []

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or [{} for _ in documents]
    ids = result.get("ids") or [str(index) for index, _ in enumerate(documents)]

    scored_documents = []
    for document_id, page_content, metadata in zip(ids, documents, metadatas):
        if not page_content:
            continue
        score = _keyword_score(query_terms, page_content)
        if score >= MIN_KEYWORD_SCORE:
            scored_documents.append(
                (
                    score,
                    str(document_id),
                    Document(
                        page_content=page_content,
                        metadata=dict(metadata or {}),
                    ),
                )
            )

    scored_documents.sort(key=lambda item: (-item[0], item[1]))
    return [document for _, _, document in scored_documents[:top_k]]


def _merge_documents(
    keyword_documents: list[Document],
    vector_documents: list[Document],
    *,
    top_k: int,
) -> list[Document]:
    merged_documents: list[Document] = []
    seen_keys: set[tuple[Any, ...]] = set()

    for document in [*keyword_documents, *vector_documents]:
        key = _document_key(document)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_documents.append(document)
        if len(merged_documents) >= top_k:
            break

    return merged_documents


def _document_key(document: Document) -> tuple[Any, ...]:
    metadata = document.metadata
    return (
        metadata.get("source_path") or metadata.get("source") or metadata.get("file_name"),
        metadata.get("chunk_index"),
        document.page_content[:80],
    )


def _keyword_score(query_terms: Counter[str], content: str) -> float:
    content_terms = _keyword_terms(content)
    score = 0.0
    for term, query_count in query_terms.items():
        content_count = content_terms.get(term, 0)
        if not content_count:
            continue
        score += min(content_count, 3) * query_count
    return score


def _keyword_terms(text: str) -> Counter[str]:
    terms = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        normalized = _normalize_token(token)
        if normalized and normalized not in STOPWORDS:
            terms.append(normalized)
    return Counter(terms)


def _normalize_token(token: str) -> str:
    if len(token) < 3:
        return ""
    if len(token) > 5 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token
