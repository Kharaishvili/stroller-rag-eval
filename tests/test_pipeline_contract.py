from pathlib import Path

from langchain_core.documents import Document

from stroller_rag_eval.config import RagConfig
from stroller_rag_eval.rag.manuals import infer_metadata_filter_from_question
from stroller_rag_eval.rag.pipeline import answer_question
from stroller_rag_eval.rag.retriever import MetadataFilter, retrieve_documents


class FakeDocument:
    def __init__(self, page_content: str, metadata: dict[str, str]) -> None:
        self.page_content = page_content
        self.metadata = metadata


def test_answer_question_returns_answer_and_retrieved_documents(monkeypatch, tmp_path):
    config = RagConfig(
        project_root=tmp_path,
        source_dirs=(tmp_path / "data",),
        chroma_persist_dir=tmp_path / "chroma",
        collection_name="test_collection",
        embedding_model="fake-embedding-model",
        chat_model="fake-chat-model",
        temperature=0.0,
        chunk_size=800,
        chunk_overlap=120,
        top_k=2,
        openai_api_key="test-key",
    )
    fake_vector_store = object()
    fake_documents = [
        FakeDocument(
            page_content="A retrieved chunk.",
            metadata={"source_path": str(Path("data/example.md"))},
        )
    ]

    monkeypatch.setattr(
        "stroller_rag_eval.rag.pipeline.load_vector_store",
        lambda loaded_config: fake_vector_store,
    )
    monkeypatch.setattr(
        "stroller_rag_eval.rag.pipeline.retrieve_documents",
        lambda question, vector_store, top_k, metadata_filter=None: fake_documents,
    )
    monkeypatch.setattr(
        "stroller_rag_eval.rag.pipeline.generate_answer",
        lambda question, documents, loaded_config: "A grounded answer.",
    )

    response = answer_question("What is covered?", config)

    assert response.question == "What is covered?"
    assert response.answer == "A grounded answer."
    assert response.retrieved_documents == fake_documents


def test_answer_question_passes_metadata_filter(monkeypatch, tmp_path):
    config = RagConfig(
        project_root=tmp_path,
        source_dirs=(tmp_path / "data",),
        chroma_persist_dir=tmp_path / "chroma",
        collection_name="test_collection",
        embedding_model="fake-embedding-model",
        chat_model="fake-chat-model",
        temperature=0.0,
        chunk_size=800,
        chunk_overlap=120,
        top_k=2,
        openai_api_key="test-key",
    )
    observed_filter = None

    monkeypatch.setattr(
        "stroller_rag_eval.rag.pipeline.load_vector_store",
        lambda loaded_config: object(),
    )

    def fake_retrieve_documents(question, vector_store, top_k, metadata_filter=None):
        nonlocal observed_filter
        observed_filter = metadata_filter
        return []

    monkeypatch.setattr(
        "stroller_rag_eval.rag.pipeline.retrieve_documents",
        fake_retrieve_documents,
    )
    monkeypatch.setattr(
        "stroller_rag_eval.rag.pipeline.generate_answer",
        lambda question, documents, loaded_config: "A grounded answer.",
    )

    answer_question(
        "What is covered?",
        config,
        metadata_filter={"file_name": "citylite_manual.md"},
    )

    assert observed_filter == {"file_name": "citylite_manual.md"}


def test_answer_question_infers_manual_filter_from_question(monkeypatch, tmp_path):
    config = RagConfig(
        project_root=tmp_path,
        source_dirs=(tmp_path / "data",),
        chroma_persist_dir=tmp_path / "chroma",
        collection_name="test_collection",
        embedding_model="fake-embedding-model",
        chat_model="fake-chat-model",
        temperature=0.0,
        chunk_size=800,
        chunk_overlap=120,
        top_k=2,
        openai_api_key="test-key",
    )
    observed_filter = None

    monkeypatch.setattr(
        "stroller_rag_eval.rag.pipeline.load_vector_store",
        lambda loaded_config: object(),
    )

    def fake_retrieve_documents(question, vector_store, top_k, metadata_filter=None):
        nonlocal observed_filter
        observed_filter = metadata_filter
        return []

    monkeypatch.setattr(
        "stroller_rag_eval.rag.pipeline.retrieve_documents",
        fake_retrieve_documents,
    )
    monkeypatch.setattr(
        "stroller_rag_eval.rag.pipeline.generate_answer",
        lambda question, documents, loaded_config: "A grounded answer.",
    )

    answer_question("Can I attach a bassinet to DuoRire?", config)

    assert observed_filter == {"file_name": "duoride_double_manual.md"}


def test_infer_metadata_filter_from_question_matches_known_manual_aliases():
    assert infer_metadata_filter_from_question(
        "What is the maximum child weight for TrailPro?"
    ) == {"file_name": "trailpro_jogger_manual.md"}
    assert infer_metadata_filter_from_question(
        "Can I attach a bassinet to DuoRire?"
    ) == {"file_name": "duoride_double_manual.md"}


def test_retrieve_documents_adds_keyword_fallback_hit():
    vector_document = Document(
        page_content="## 13. Canopy Use\n\nUse the canopy for shade.",
        metadata={"file_name": "duoride_double_manual.md", "chunk_index": 13},
    )
    keyword_document = Document(
        page_content=(
            "### Handlebar Bags and Hanging Items\n\n"
            "Do not hang bags, purses, backpacks, or shopping items from the "
            "handlebar."
        ),
        metadata={"file_name": "duoride_double_manual.md", "chunk_index": 5},
    )

    class FakeRetriever:
        def invoke(self, question: str) -> list[Document]:
            assert question == "Can I hang a backpack from the DuoRide Double handlebar?"
            return [vector_document]

    class FakeVectorStore:
        def as_retriever(self, search_kwargs: dict[str, object]) -> FakeRetriever:
            assert search_kwargs == {
                "k": 2,
                "filter": {"file_name": "duoride_double_manual.md"},
            }
            return FakeRetriever()

        def get(
            self,
            *,
            where: MetadataFilter | None = None,
            include: list[str] | None = None,
        ) -> dict[str, object]:
            assert where == {"file_name": "duoride_double_manual.md"}
            assert include == ["documents", "metadatas"]
            return {
                "ids": ["vector", "keyword"],
                "documents": [
                    vector_document.page_content,
                    keyword_document.page_content,
                ],
                "metadatas": [
                    vector_document.metadata,
                    keyword_document.metadata,
                ],
            }

    documents = retrieve_documents(
        "Can I hang a backpack from the DuoRide Double handlebar?",
        FakeVectorStore(),
        top_k=2,
        metadata_filter={"file_name": "duoride_double_manual.md"},
    )

    assert documents[0].metadata["chunk_index"] == 5
    assert documents[1].metadata["chunk_index"] == 13
