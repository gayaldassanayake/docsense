from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    models,
)

from docsense.config import settings
from docsense.embeddings.ollama import embed_documents, embed_query
from docsense.embeddings.sparse import sparse_embed_documents, sparse_embed_query

VECTOR_DIM = 768
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def _client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collection() -> None:
    """Create the Qdrant collection with named dense + sparse vectors if it doesn't exist."""
    client = _client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )
        print(f"Created collection '{settings.qdrant_collection}'")
    else:
        print(f"Collection '{settings.qdrant_collection}' already exists")


def recreate_collection() -> None:
    """Delete and recreate the collection. Use when the schema changes (e.g., adding sparse vectors)."""
    client = _client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in existing:
        client.delete_collection(collection_name=settings.qdrant_collection)
        print(f"Deleted old collection '{settings.qdrant_collection}'")
    ensure_collection()


def upsert(chunks: list[dict]) -> int:
    """
    Embed and upsert a list of chunk dicts into Qdrant.

    Each chunk must have: chunk_id, text, and any metadata fields
    (connector, section, source_url, token_count).

    Stores both dense (nomic-embed-text) and sparse (BM25) vectors
    for hybrid retrieval.

    Returns the number of chunks upserted.
    """
    client = _client()
    texts = [c["text"] for c in chunks]

    dense_vectors = embed_documents(texts)
    sparse_vectors = sparse_embed_documents(texts)

    points = [
        PointStruct(
            id=abs(hash(chunk["chunk_id"])) % (2**63),
            vector={
                DENSE_VECTOR_NAME: dense_vector,
                SPARSE_VECTOR_NAME: SparseVector(indices=sparse[0], values=sparse[1]),
            },
            payload=chunk,
        )
        for chunk, dense_vector, sparse in zip(chunks, dense_vectors, sparse_vectors)
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
    mode: str | None = None,
) -> list[dict]:
    """
    Embed the query and return the top-k most similar chunks.

    Args:
        query:            Natural language question.
        top_k:            Number of results to return.
        connector_filter: If given, restrict results to this connector.
        mode:             Retrieval mode: "dense", "sparse", or "hybrid".
                          Defaults to settings.retrieval_mode.

    Returns a list of payload dicts with an added 'score' key.
    """
    mode = mode or settings.retrieval_mode
    client = _client()

    qdrant_filter = None
    if connector_filter:
        qdrant_filter = Filter(
            must=[FieldCondition(key="connector", match=MatchValue(value=connector_filter))]
        )

    if mode == "dense":
        query_vector = embed_query(query)
        results = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

    elif mode == "sparse":
        indices, values = sparse_embed_query(query)
        results = client.query_points(
            collection_name=settings.qdrant_collection,
            query=SparseVector(indices=indices, values=values),
            using=SPARSE_VECTOR_NAME,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

    else:  # hybrid — prefetch from both, fuse with RRF
        query_vector = embed_query(query)
        indices, values = sparse_embed_query(query)

        prefetch_dense = Prefetch(
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            limit=top_k * 2,
            filter=qdrant_filter,
        )
        prefetch_sparse = Prefetch(
            query=SparseVector(indices=indices, values=values),
            using=SPARSE_VECTOR_NAME,
            limit=top_k * 2,
            filter=qdrant_filter,
        )

        results = client.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[prefetch_dense, prefetch_sparse],
            query=models.FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

    return [{**hit.payload, "score": hit.score} for hit in results.points]
