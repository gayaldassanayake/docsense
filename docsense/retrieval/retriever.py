"""
Retriever: embed a query and return the most relevant chunks from Qdrant.
"""

from dataclasses import dataclass

from docsense.config import settings
from docsense.vectorstore.qdrant import search


@dataclass
class RetrievedChunk:
    chunk_id: str
    connector: str
    section: str
    source_url: str
    text: str
    score: float


def retrieve(
    query: str,
    top_k: int | None = None,
    connector_filter: str | None = None,
    mode: str | None = None,
) -> list[RetrievedChunk]:
    """
    Embed the query and return the top-k most relevant chunks.

    Args:
        query:            Natural language question.
        top_k:            Number of chunks to return (defaults to config value).
        connector_filter: If given, restrict results to this connector
                          (e.g. "ballerinax/kafka").
        mode:             Retrieval mode override: "dense", "sparse", or "hybrid".
                          Defaults to settings.retrieval_mode.

    Returns:
        List of RetrievedChunk objects ordered by descending similarity score.
    """
    top_k = top_k or settings.default_top_k

    raw = search(query, top_k=top_k, connector_filter=connector_filter, mode=mode)

    return [
        RetrievedChunk(
            chunk_id=r.get("chunk_id", ""),
            connector=r.get("connector", ""),
            section=r.get("section", ""),
            source_url=r.get("source_url", ""),
            text=r.get("text", ""),
            score=r.get("score", 0.0),
        )
        for r in raw
    ]
