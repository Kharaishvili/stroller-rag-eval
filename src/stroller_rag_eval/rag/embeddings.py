"""Embedding client factory"""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings


def build_embeddings(model: str) -> OpenAIEmbeddings:
    """Create the embedding model used by Chroma"""

    return OpenAIEmbeddings(model=model)
