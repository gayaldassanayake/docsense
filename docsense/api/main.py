import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from docsense.api.routes.ingest import router as ingest_router
from docsense.api.routes.query import router as query_router
from docsense.config import settings

app = FastAPI(title="DocSense", description="RAG-powered Ballerina connector documentation assistant")

app.include_router(query_router)
app.include_router(ingest_router)


@app.get("/health")
def health() -> JSONResponse:
    status = {"qdrant": "ok", "ollama": "ok"}

    try:
        httpx.get(f"http://{settings.qdrant_host}:{settings.qdrant_port}/healthz", timeout=3.0)
    except Exception as e:
        status["qdrant"] = f"error: {e}"

    try:
        httpx.get(f"{settings.ollama_host}/api/tags", timeout=3.0)
    except Exception as e:
        status["ollama"] = f"error: {e}"

    overall = "ok" if all(v == "ok" for v in status.values()) else "degraded"
    return JSONResponse({"status": overall, **status})
