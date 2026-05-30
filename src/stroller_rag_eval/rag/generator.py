"""Answer generation from retrieved context"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from stroller_rag_eval.config import RagConfig


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful product-manual question-answering assistant. "
            "Answer only from the provided context. If the context is "
            "insufficient, say you do not know.",
        ),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:",
        ),
    ]
)


def generate_answer(
    question: str,
    documents: list[Document],
    config: RagConfig,
) -> str:
    """Generate an answer grounded in retrieved document chunks"""

    llm = ChatOpenAI(model=config.chat_model, temperature=config.temperature)
    response = (ANSWER_PROMPT | llm).invoke(
        {
            "question": question,
            "context": format_context(documents),
        }
    )
    return response.content if isinstance(response.content, str) else str(response.content)


def format_context(documents: list[Document]) -> str:
    """Format retrieved chunks for the generation prompt"""

    if not documents:
        return "No context was retrieved."

    formatted_chunks = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source_path", document.metadata.get("source", "unknown"))
        formatted_chunks.append(
            f"[Chunk {index} | Source: {source}]\n{document.page_content.strip()}"
        )

    return "\n\n".join(formatted_chunks)
