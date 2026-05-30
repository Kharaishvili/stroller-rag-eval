"""Runtime configuration for the RAG pipeline"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIRS = (
    PROJECT_ROOT / "data",
)


@dataclass(frozen=True)
class RagConfig:
    """Configuration shared by ingestion, retrieval, and generation"""

    project_root: Path
    source_dirs: tuple[Path, ...]
    chroma_persist_dir: Path
    collection_name: str
    embedding_model: str
    chat_model: str
    temperature: float
    chunk_size: int
    chunk_overlap: int
    top_k: int
    openai_api_key: str | None
    eval_dataset_path: Path = PROJECT_ROOT / "data" / "eval" / "stroller_qa.csv"
    eval_reports_dir: Path = PROJECT_ROOT / "reports"
    evaluator_model: str = "gpt-4o-mini"
    deepeval_threshold: float = 0.7


def get_config(env_file: Path | None = None) -> RagConfig:
    """Load configuration from .env and environment variables"""

    load_dotenv(env_file or PROJECT_ROOT / ".env")

    return RagConfig(
        project_root=PROJECT_ROOT,
        source_dirs=_parse_source_dirs(os.getenv("RAG_SOURCE_DIRS")),
        chroma_persist_dir=_resolve_path(
            os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
        ),
        collection_name=os.getenv("RAG_COLLECTION_NAME", "portfolio_rag_eval"),
        embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "120")),
        top_k=int(os.getenv("RAG_TOP_K", "8")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        eval_dataset_path=_resolve_path(
            os.getenv("EVAL_DATASET_PATH", "data/eval/stroller_qa.csv")
        ),
        eval_reports_dir=_resolve_path(os.getenv("EVAL_REPORTS_DIR", "reports")),
        evaluator_model=os.getenv("EVAL_MODEL", "gpt-4o-mini"),
        deepeval_threshold=float(os.getenv("DEEPEVAL_THRESHOLD", "0.7")),
    )


def require_openai_api_key(config: RagConfig) -> None:
    """Fail early with a readable message when OpenAI credentials are missing"""

    if not config.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add a key."
        )


def _parse_source_dirs(raw_value: str | None) -> tuple[Path, ...]:
    if not raw_value:
        return DEFAULT_SOURCE_DIRS

    paths = [item.strip() for item in raw_value.split(",") if item.strip()]
    return tuple(_resolve_path(path) for path in paths)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()
