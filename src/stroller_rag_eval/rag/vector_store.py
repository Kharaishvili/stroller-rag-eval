"""Chroma vector store helpers"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from stroller_rag_eval.config import RagConfig
from stroller_rag_eval.rag.embeddings import build_embeddings


def build_vector_store(
    documents: list[Document],
    config: RagConfig,
    *,
    reset: bool = True,
) -> Chroma:
    """Create and persist a Chroma vector store from chunked documents"""

    if not documents:
        raise ValueError("No document chunks were provided for vector store creation.")

    if reset and config.chroma_persist_dir.exists():
        shutil.rmtree(config.chroma_persist_dir)

    config.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=documents,
        embedding=build_embeddings(config.embedding_model),
        collection_name=config.collection_name,
        persist_directory=str(config.chroma_persist_dir),
        client_settings=_chroma_client_settings(config),
        ids=_stable_document_ids(documents),
    )


def load_vector_store(config: RagConfig) -> Chroma:
    """Load an existing Chroma vector store"""

    if not config.chroma_persist_dir.exists():
        raise FileNotFoundError(
            f"Vector store not found at {config.chroma_persist_dir}. "
            "Run `python scripts/ingest_docs.py` first."
        )

    return Chroma(
        collection_name=config.collection_name,
        embedding_function=build_embeddings(config.embedding_model),
        persist_directory=str(config.chroma_persist_dir),
        client_settings=_chroma_client_settings(config),
    )


def _stable_document_ids(documents: list[Document]) -> list[str]:
    ids: list[str] = []
    for index, document in enumerate(documents):
        source = document.metadata.get("source_path", "unknown-source")
        chunk_index = document.metadata.get("chunk_index", index)
        digest = hashlib.sha256(
            f"{source}:{chunk_index}:{document.page_content}".encode("utf-8")
        ).hexdigest()
        ids.append(digest[:32])
    return ids


def _chroma_client_settings(config: RagConfig) -> Settings:
    return Settings(
        anonymized_telemetry=False,
        chroma_product_telemetry_impl="stroller_rag_eval.chroma_telemetry.NoOpTelemetryClient",
        chroma_telemetry_impl="stroller_rag_eval.chroma_telemetry.NoOpTelemetryClient",
        is_persistent=True,
        persist_directory=str(config.chroma_persist_dir),
    )
