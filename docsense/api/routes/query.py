import time

from fastapi import APIRouter
from pydantic import BaseModel, Field

from docsense.generation.claude import generate
from docsense.retrieval.retriever import retrieve

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    connector_filter: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class SourceItem(BaseModel):
    connector: str
    section: str
    url: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    chunks_used: int
    latency_ms: int


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    t0 = time.perf_counter()

    chunks = retrieve(req.question, top_k=req.top_k, connector_filter=req.connector_filter)
    response = generate(req.question, chunks)

    latency_ms = int((time.perf_counter() - t0) * 1000)

    return QueryResponse(
        answer=response.answer,
        sources=[SourceItem(**s) for s in response.sources],
        chunks_used=response.chunks_used,
        latency_ms=latency_ms,
    )
