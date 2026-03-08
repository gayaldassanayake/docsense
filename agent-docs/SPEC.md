# DocSense — Project Specification
### RAG-Powered Ballerina Connector Documentation Assistant

---

## Overview

DocSense is a RAG (Retrieval-Augmented Generation) API that 
1. ingests Ballerina connector documentation from,
    a. Ballerina Central (central.ballerina.io)
    b. WSO2 Integrator documentation (bi.docs.wso2.com)
2. indexes it with semantic embeddings, 
3.  answers natural language questions about connector usage, configuration properties, and code examples with cited sources in every response.

**Stack:**
- Language: Python 3.11+
- Embeddings: `nomic-embed-text` via Ollama (local, free)
- Vector DB: Qdrant (Docker)
- LLM: Claude API (`claude-sonnet-4-20250514`)
- Interface: REST API via FastAPI

---

## Goals

1. Learn every meaningful layer of a RAG pipeline hands-on
2. Produce a portfolio piece with a clear problem statement, measurable output, and clean README
3. Lay the groundwork for a future retrieval layer that could augment BI Copilot

---

## Repository Structure

```
docsense/
├── README.md
├── docker-compose.yml          # Qdrant
├── pyproject.toml
├── .env.example
│
├── docsense/
│   ├── __init__.py
│   ├── config.py               # Settings via pydantic-settings
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── scraper.py          # Fetch docs from Ballerina Central
│   │   ├── chunker.py          # Split docs into chunks
│   │   └── pipeline.py         # Orchestrates scrape → chunk → embed → upsert
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── ollama.py           # nomic-embed-text client wrapper
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── qdrant.py           # Qdrant client wrapper (upsert, search)
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py        # Query → embed → search → return chunks
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   └── claude.py           # Prompt builder + Claude API call
│   │
│   └── api/
│       ├── __init__.py
│       ├── main.py             # FastAPI app
│       └── routes/
│           ├── query.py        # POST /query
│           └── ingest.py       # POST /ingest (trigger re-ingestion)
│
├── scripts/
│   ├── ingest.py               # CLI: python scripts/ingest.py
│   └── query.py                # CLI: python scripts/query.py "your question"
│
├── data/
│   └── raw/                    # Scraped markdown/HTML cached locally
│
└── tests/
    ├── test_chunker.py
    ├── test_retriever.py
    └── eval/
        ├── questions.json       # Hand-crafted Q&A eval set
        └── run_eval.py         # Evaluate retrieval quality
```

---

## Phase 1 Scope (v1)

### 1. Doc Ingestion

**Source:** Ballerina Central connector documentation pages. Later we can add bi.docs.wso2.com

I have added the response of api.central.ballerina.io/2.0/registry/packages/ballerinax in @connectors.json, which has the package documentation and metadata.
But for initial cut, start with 5–10 connectors you know well (e.g. `ballerinax/kafka`,
`ballerinax/rabbitmq`, `ballerinax/twilio`, `ballerinax/java.jdbc`,
`ballerinax/mysql`). 

You can find the API documentation url from the `apiDocURL` field. You can find the overview, setup guide and some example codes in the `readme` section.


**`scraper.py` responsibilities:**
- Accept a list of connector doc URLs (or a manifest file)
- Fetch HTML, extract the main content body (strip nav/footer)
- Save raw content to `data/raw/<connector-name>.md`
- Respect rate limits; cache locally so you don't re-fetch on every run

---

### 2. Chunking

**`chunker.py` responsibilities:**
- Load raw doc files
- Split by headings first (H2/H3 boundary chunking)
- Apply a sliding window with overlap within each section if it exceeds the
  max token limit (target: ~400 tokens per chunk, 50-token overlap)
- Attach metadata to every chunk:

```python
{
  "chunk_id": "kafka-001",
  "connector": "ballerinax/kafka",
  "source_url": "https://central.ballerina.io/ballerinax/kafka/...",
  "section": "Configurations",
  "text": "...",
  "token_count": 387
}
```

**What to learn here:** Chunking strategy is the single biggest lever on
retrieval quality. Bad chunks = bad answers, regardless of your model or
vector DB. You will iterate on this.

**Chunking strategies to implement and compare (label them in code):**
- `strategy=fixed` — naive fixed-size splits
- `strategy=heading` — split on H2/H3 (default)
- `strategy=semantic` — placeholder for Phase 2 (sentence-transformer similarity)

---

### 3. Embeddings

**`ollama.py` responsibilities:**
- Wrap the Ollama HTTP API (`POST /api/embeddings`)
- Model: `nomic-embed-text` (768-dim, strong on technical content)
- Batch embed a list of strings, return list of float vectors
- Handle retries on transient errors

**Prerequisites for the developer:**
```bash
ollama pull nomic-embed-text
```

**What to learn here:** Embedding latency, batching, and the difference between
embedding a *query* vs embedding a *document* (nomic-embed-text uses prefixes
`search_query:` and `search_document:` — this matters for retrieval quality).

---

### 4. Vector Store

**`qdrant.py` responsibilities:**
- Connect to local Qdrant via Docker
- Create a collection `docsense_connectors` with cosine similarity, 768 dims
- `upsert(chunks)` — embed + store with metadata as payload
- `search(query_text, top_k=5)` — embed query + return top-k chunks with scores

**docker-compose.yml:**
```yaml
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

**What to learn here:** How vector DBs store and index high-dimensional vectors.
The Qdrant payload (metadata) system. Why cosine similarity for text.

---

### 5. Retrieval

**`retriever.py` responsibilities:**
- Accept a natural language query string
- Embed with `search_query:` prefix (nomic-embed-text convention)
- Query Qdrant, return top-k chunks with their metadata and similarity scores
- Filter by connector name if specified (Qdrant payload filter)

**Return format:**
```python
@dataclass
class RetrievedChunk:
    chunk_id: str
    connector: str
    section: str
    source_url: str
    text: str
    score: float
```

---

### 6. Generation

**`claude.py` responsibilities:**
- Accept a query + list of `RetrievedChunk` objects
- Build a prompt using the system prompt below
- Call Claude API (`claude-sonnet-4-20250514`)
- Return structured response with answer + citations

**System prompt (starting point — iterate on this):**
```
You are a Ballerina integration assistant. You answer developer questions about
Ballerina connector configuration, usage, and code patterns.

Answer using ONLY the provided documentation excerpts. If the excerpts do not
contain enough information to answer confidently, say so explicitly.

For each claim in your answer, cite the source using [connector/section] notation.
Always include a "Sources" section at the end listing the connectors referenced.
```

**Response schema:**
```python
@dataclass
class QueryResponse:
    answer: str
    sources: list[dict]   # [{connector, section, url, score}]
    chunks_used: int
```

---

### 7. API

**Two endpoints:**

`POST /query`
```json
// Request
{
  "question": "How do I configure SASL authentication for the Kafka connector?",
  "connector_filter": "ballerinax/kafka",  // optional
  "top_k": 5  // optional, default 5
}

// Response
{
  "answer": "...",
  "sources": [
    {
      "connector": "ballerinax/kafka",
      "section": "Security Configurations",
      "url": "https://...",
      "score": 0.87
    }
  ],
  "chunks_used": 4,
  "latency_ms": 1240
}
```

`POST /ingest`
```json
// Request
{
  "connectors": ["ballerinax/kafka", "ballerinax/rabbitmq"]  // optional, defaults to all
}

// Response
{
  "status": "ok",
  "chunks_ingested": 312,
  "duration_sec": 18.4
}
```

`GET /health` — returns Qdrant + Ollama connectivity status.

---

## Eval Harness

Create `tests/eval/questions.json` with at least 15 hand-crafted Q&A pairs.
Focus on questions that require connector-specific knowledge Claude wouldn't
know from training alone.

**Example entries:**
```json
[
  {
    "id": "kafka-001",
    "connector": "ballerinax/kafka",
    "question": "What is the correct type for the securityProtocol field in Kafka auth config?",
    "expected_keywords": ["SecurityProtocol", "SASL_SSL", "PLAINTEXT"],
    "expected_source_section": "Security"
  },
  {
    "id": "rabbitmq-002",
    "connector": "ballerinax/rabbitmq",
    "question": "How do I acknowledge a message manually in RabbitMQ?",
    "expected_keywords": ["basicAck", "deliveryTag", "channel"],
    "expected_source_section": "Message Acknowledgment"
  }
]
```

`run_eval.py` should:
1. Run each question through the full RAG pipeline
2. Check if expected keywords appear in the answer
3. Check if the expected source section appears in retrieved chunks
4. Print a retrieval hit rate and a keyword hit rate
5. Output a summary table to stdout

This gives you a concrete before/after metric when you later change chunking
strategies or add reranking.

---

## Configuration (`config.py`)

All settings via environment variables with pydantic-settings:

```python
class Settings(BaseSettings):
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "docsense_connectors"

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"

    # Claude
    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-20250514"
    claude_max_tokens: int = 1024

    # Retrieval
    default_top_k: int = 5
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 50
    chunking_strategy: str = "heading"

    class Config:
        env_file = ".env"
```

---

## Implementation Order

Build in this order — each step is independently testable:

1. `docker-compose.yml` → confirm Qdrant is up at localhost:6333
2. `embeddings/ollama.py` → test with `print(embed("hello world"))`
3. `vectorstore/qdrant.py` → create collection, upsert dummy vectors, search
4. `ingestion/scraper.py` → fetch one connector doc, save to `data/raw/`
5. `ingestion/chunker.py` → chunk it, print chunk count and sample chunks
6. `ingestion/pipeline.py` → wire 4+5+2+3 together, ingest one connector end-to-end
7. `retrieval/retriever.py` → run a test query, inspect returned chunks
8. `generation/claude.py` → pass chunks to Claude, inspect answer
9. `api/` → wrap in FastAPI, test with curl
10. `tests/eval/` → build eval set, run baseline eval, record results in README

---

## README Requirements (portfolio checklist)

Your README must include:

- [ ] Problem statement (1 paragraph — why does this exist?)
- [ ] Architecture diagram (even a simple ASCII one is fine)
- [ ] Quick start (docker-compose up, pip install, ingest, query — under 5 commands)
- [ ] Example query + response (copy-paste output from a real run)
- [ ] Eval results table (baseline retrieval hit rate %)
- [ ] "What I learned" section — this is what makes it a portfolio piece

---

## What's explicitly out of scope for v1

- Hybrid search (BM25 + vector) — Phase 2
- Reranking — Phase 2
- Agentic multi-hop — Phase 3
- RAGAS evaluation — Phase 4
- UI — future

These are called out so Claude Code doesn't over-engineer v1. Ship the
core loop first, measure it, then layer on complexity.

---

## Key learning checkpoints

After building each layer, pause and ask yourself:

- **After chunking:** Can you tell which chunk strategy produces more coherent
  units? Print 10 random chunks and read them.
- **After ingestion:** How many chunks does a single connector produce?
  What's the distribution of chunk sizes?
- **After first retrieval:** Are the top-5 chunks actually relevant to your
  query? Score them manually before looking at the LLM answer.
- **After generation:** Does the answer ever hallucinate details not in the
  retrieved chunks? This is the point of RAG — the answer should be
  traceable to a specific chunk.
- **After eval:** What's your baseline keyword hit rate? 60%? 80%?
  Write it in the README. This is your benchmark for Phase 2.