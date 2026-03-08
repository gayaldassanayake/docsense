import time

from fastapi import APIRouter
from pydantic import BaseModel

from docsense.ingestion.pipeline import run

router = APIRouter()


class IngestRequest(BaseModel):
    connectors: list[str] | None = None   # e.g. ["ballerinax/kafka", "ballerinax/mysql"]
    force: bool = False


class IngestResponse(BaseModel):
    status: str
    chunks_ingested: int
    connectors_processed: int
    duration_sec: float


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    connectors = None
    if req.connectors:
        connectors = [tuple(c.split("/", 1)) for c in req.connectors]

    t0 = time.perf_counter()
    result = run(connectors=connectors, force_scrape=req.force)
    duration = round(time.perf_counter() - t0, 1)

    return IngestResponse(
        status="ok",
        chunks_ingested=result["chunks_ingested"],
        connectors_processed=result["connectors_processed"],
        duration_sec=duration,
    )
