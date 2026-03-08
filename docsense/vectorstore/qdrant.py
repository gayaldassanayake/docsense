from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from docsense.config import settings
from docsense.embeddings.ollama import embed_documents, embed_query

VECTOR_DIM = 768


def _client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collection() -> None:
    """Create the Qdrant collection if it doesn't exist yet."""
    client = _client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"Created collection '{settings.qdrant_collection}'")
    else:
        print(f"Collection '{settings.qdrant_collection}' already exists")


def upsert(chunks: list[dict]) -> int:
    """
    Embed and upsert a list of chunk dicts into Qdrant.

    Each chunk must have: chunk_id, text, and any metadata fields
    (connector, section, source_url, token_count).

    Returns the number of chunks upserted.
    """
    client = _client()
    texts = [c["text"] for c in chunks]
    vectors = embed_documents(texts)

    points = [
        PointStruct(
            id=abs(hash(chunk["chunk_id"])) % (2**63),
            vector=vector,
            payload=chunk,
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)


def delete_by_connector(connector: str) -> None:
    """Delete all points in the collection that belong to a given connector."""
    client = _client()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="connector", match=MatchValue(value=connector))]
        ),
    )


def search(
    query: str,
    top_k: int = 5,
    connector_filter: str | None = None,
) -> list[dict]:
    """
    Embed the query and return the top-k most similar chunks.

    Optionally filter by connector name (exact match on payload field).
    Returns a list of payload dicts with an added 'score' key.
    """
    client = _client()
    query_vector = embed_query(query)

    qdrant_filter = None
    if connector_filter:
        qdrant_filter = Filter(
            must=[FieldCondition(key="connector", match=MatchValue(value=connector_filter))]
        )

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    return [{**hit.payload, "score": hit.score} for hit in results.points]
