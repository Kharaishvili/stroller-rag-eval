"""High-level ingestion and RAG query orchestration"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from stroller_rag_eval.rag.chunking import chunk_documents
from stroller_rag_eval.config import RagConfig, require_openai_api_key
from stroller_rag_eval.rag.document_loader import load_markdown_documents
from stroller_rag_eval.rag.generator import generate_answer
from stroller_rag_eval.rag.retriever import retrieve_documents
from stroller_rag_eval.rag.vector_store import build_vector_store, load_vector_store


@dataclass(frozen=True)
class IngestResult:
    """Summary of an ingestion run"""

    document_count: int
    chunk_count: int
    persist_dir: str
    collection_name: str


@dataclass(frozen=True)
class RagResponse:
    """Question answering result with retrieved evidence"""

    question: str
    answer: str
    retrieved_documents: list[Document]


def ingest_documents(config: RagConfig, *, reset: bool = True) -> IngestResult:
    """Load markdown docs, chunk them, and persist a Chroma vector store"""

    require_openai_api_key(config)
    documents = load_markdown_documents(config.source_dirs)
    if not documents:
        source_dirs = ", ".join(str(path) for path in config.source_dirs)
        raise ValueError(f"No markdown documents found in: {source_dirs}")

    chunks = chunk_documents(
        documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    build_vector_store(chunks, config, reset=reset)

    return IngestResult(
        document_count=len(documents),
        chunk_count=len(chunks),
        persist_dir=str(config.chroma_persist_dir),
        collection_name=config.collection_name,
    )


def answer_question(
    question: str,
    config: RagConfig,
    *,
    metadata_filter: dict[str, object] | None = None,
) -> RagResponse:
    """Retrieve evidence and generate an answer for a question"""

    require_openai_api_key(config)
    vector_store = load_vector_store(config)
    retrieved_documents = retrieve_documents(
        question,
        vector_store,
        top_k=config.top_k,
        metadata_filter=metadata_filter,
    )
    answer = generate_answer(question, retrieved_documents, config)

    return RagResponse(
        question=question,
        answer=answer,
        retrieved_documents=retrieved_documents,
    )
