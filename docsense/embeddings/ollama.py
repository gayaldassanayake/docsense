import httpx
from docsense.config import settings


def _embed(texts: list[str]) -> list[list[float]]:
    """
    Call Ollama's /api/embed endpoint for a batch of texts.
    Returns a list of 768-dimensional float vectors.
    """
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{settings.ollama_host}/api/embed",
            json={"model": settings.ollama_embed_model, "input": texts},
        )
        response.raise_for_status()
        return response.json()["embeddings"]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of document chunks for indexing.
    Uses the 'search_document:' prefix that nomic-embed-text expects.
    """
    prefixed = [f"search_document: {t}" for t in texts]
    return _embed(prefixed)


def embed_query(text: str) -> list[float]:
    """
    Embed a single query string for retrieval.
    Uses the 'search_query:' prefix that nomic-embed-text expects.
    """
    return _embed([f"search_query: {text}"])[0]
