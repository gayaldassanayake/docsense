"""
Generation: build a grounded prompt from retrieved chunks and call Claude.
"""

from dataclasses import dataclass

import anthropic

from docsense.config import settings
from docsense.retrieval.retriever import RetrievedChunk

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

SYSTEM_PROMPT = """\
You are a Ballerina integration assistant. You answer developer questions about \
Ballerina connector configuration, usage, and code patterns.

Answer using ONLY the provided documentation excerpts. If the excerpts do not \
contain enough information to answer confidently, say so explicitly — do not guess.

For each claim in your answer, cite the source using [connector/section] notation. \
Always include a "Sources" section at the end listing each connector and section referenced.\
"""


@dataclass
class QueryResponse:
    answer: str
    sources: list[dict]   # [{connector, section, url, score}]
    chunks_used: int


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] {chunk.connector} / {chunk.section}\n"
            f"URL: {chunk.source_url}\n"
            f"Score: {chunk.score:.3f}\n\n"
            f"{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def generate(query: str, chunks: list[RetrievedChunk]) -> QueryResponse:
    """
    Build a grounded prompt from the retrieved chunks and call Claude.

    Args:
        query:  The original user question.
        chunks: Retrieved chunks from the retriever.

    Returns:
        QueryResponse with the answer, source list, and chunk count.
    """
    context = _build_context(chunks)

    user_message = (
        f"Documentation excerpts:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}"
    )

    response = _client.messages.create(
        model=settings.claude_model,
        max_tokens=settings.claude_max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text

    sources = [
        {
            "connector": c.connector,
            "section": c.section,
            "url": c.source_url,
            "score": round(c.score, 3),
        }
        for c in chunks
    ]

    return QueryResponse(answer=answer, sources=sources, chunks_used=len(chunks))
