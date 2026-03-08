Implementation Plan

  Phase 0 — Project scaffold

  - pyproject.toml with dependencies (fastapi, qdrant-client, anthropic, pydantic-settings, httpx, beautifulsoup4, tiktoken)
  - .env.example and docker-compose.yml for Qdrant
  - Package structure under docsense/

  Phase 1 — Infrastructure layer (independently testable)

  1. docker-compose.yml — Qdrant at localhost:6333
  2. docsense/config.py — All settings via pydantic-settings + .env
  3. docsense/embeddings/ollama.py — Wrap Ollama's POST /api/embeddings with nomic-embed-text; verify with a smoke test
  4. docsense/vectorstore/qdrant.py — Create collection, upsert(), search(); verify with dummy vectors

  Phase 2 — Ingestion pipeline

  5. docsense/ingestion/scraper.py — Fetch connector docs from apiDocURL + readme fields in connectors.json; cache to data/raw/
  6. docsense/ingestion/chunker.py — Heading-based chunking (default), fixed-size, with metadata attachment
  7. docsense/ingestion/pipeline.py — Wire scrape → chunk → embed → upsert end-to-end for one connector

  Phase 3 — Retrieval + Generation

  8. docsense/retrieval/retriever.py — Embed query with search_query: prefix → Qdrant search → RetrievedChunk dataclass
  9. docsense/generation/claude.py — Build prompt + call claude-sonnet-4-20250514 → QueryResponse with citations

  Phase 4 — API

  10. docsense/api/main.py + routes — POST /query, POST /ingest, GET /health
  11. scripts/ingest.py + scripts/query.py — CLI wrappers

  Phase 5 — Eval harness

  12. tests/eval/questions.json — 15 hand-crafted Q&A pairs (kafka, rabbitmq, mysql, etc.)
  13. tests/eval/run_eval.py — Keyword hit rate + retrieval section hit rate
  14. tests/test_chunker.py, tests/test_retriever.py — Unit tests

  Baseline results (heading strategy, top_k=5):
    Retrieval hit rate: 15/15 (100%)
    Keyword hit rate:   9/15  (60%)

---

  Phase 6 — Hybrid Search (BM25 + vector)

  Problem it solves: pure vector search misses exact identifiers (e.g. `maxOpenConnections`,
  `sendSms`). BM25 keyword search excels at exact terms but fails at synonyms/paraphrasing.
  Combining both covers both cases.

  Plan:
  - Run vector search and BM25 in parallel, combine scores: final = α×vector + (1-α)×bm25
  - Qdrant has native sparse vector support for BM25 — store sparse + dense vectors per chunk
  - Target metric: keyword hit rate > 80%

---

  Phase 7 — Reranking

  Problem it solves: the embedding model (nomic-embed-text) is fast but coarse. It ranks
  chunks by vector proximity, not by actual relevance to the specific question.

  Plan:
  - Stage 1: retrieve top-20 chunks with the existing fast vector search (wide net)
  - Stage 2: pass all 20 + the query to a cross-encoder reranker (reads both together,
    scores relevance precisely), keep top-5
  - Cross-encoders to consider: ms-marco-MiniLM, bge-reranker, or via Cohere Rerank API
  - Target metric: improved context precision in RAGAS evaluation

---

  Phase 8 — Agentic Multi-hop

  Problem it solves: single-hop retrieval can't answer questions that span multiple connectors
  or require sequential lookups (e.g. "which connector for Kafka→MySQL and what type is the
  consumer group ID?").

  Plan:
  - Give Claude a search(query, connector?) tool via the Anthropic tool use API
  - Claude decides how many retrieval steps to take and in what order
  - Implement a simple loop: Claude calls tool → retriever runs → result fed back → repeat
    until Claude produces a final answer
  - Architecture shift: question → agent plans → search → (search again?) → generate

---

  Phase 9 — RAGAS Evaluation

  Problem it solves: keyword hit rate is binary and misses semantically correct answers that
  paraphrase rather than quote exact terms. Need richer, multi-dimensional metrics.

  Four RAGAS metrics (each scored by an LLM judge):
  - Faithfulness:       does the answer contain claims not in the retrieved chunks? (hallucination)
  - Answer relevancy:   does the answer actually address the question?
  - Context precision:  are the retrieved chunks useful, or is there noise?
  - Context recall:     did retrieval find all chunks needed to answer fully?

  Plan:
  - Integrate the `ragas` Python library
  - Replace/supplement keyword eval in run_eval.py with RAGAS scores
  - Use results to pinpoint whether low scores are a retrieval problem (precision/recall)
    or a generation problem (faithfulness/relevancy)
