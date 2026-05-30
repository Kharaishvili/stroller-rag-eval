"""Markdown document discovery and loading"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


MARKDOWN_EXTENSIONS = {".md", ".markdown"}


def discover_markdown_files(source_dirs: tuple[Path, ...]) -> list[Path]:
    """Return markdown files from existing source directories"""

    files: list[Path] = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue

        for path in source_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in MARKDOWN_EXTENSIONS:
                files.append(path.resolve())

    return sorted(files)


def load_markdown_documents(source_dirs: tuple[Path, ...]) -> list[Document]:
    """Load markdown files as LangChain documents"""

    documents: list[Document] = []
    for path in discover_markdown_files(source_dirs):
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
        loaded_documents = loader.load()

        for document in loaded_documents:
            document.metadata.update(
                {
                    "source": str(path),
                    "source_path": str(path),
                    "file_name": path.name,
                }
            )
            documents.append(document)

    return documents
