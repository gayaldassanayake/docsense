# DocSense

A RAG pipeline that answers natural language questions about Ballerina connector usage by ingesting official documentation and returning grounded, cited answers.

## Why

Ballerina connector docs are spread across Ballerina Central and the WSO2 BI docs site. Finding the right configuration field or code pattern requires bouncing between multiple pages. DocSense ingests those docs once, indexes them semantically, and answers questions with exact citations so you can trace every claim back to its source.

## Architecture

```
                        ┌─────────────────────────────────────┐
  INGESTION             │                                     │
                        │  connectors.json  ──►  scraper.py   │
                        │                           │         │
                        │                      chunker.py     │
                        │                    (hierarchical)    │
                        │                           │         │
                        │              ┌────────────┴───────┐ │
                        │       nomic-embed-text      BM25   │ │
                        │        (dense 768d)       (sparse) │ │
                        │              └────────────┬───────┘ │
                        │                        Qdrant       │
                        │                    (Docker, local)  │
                        └───────────────────────────┬─────────┘
                                                    │
                        ┌───────────────────────────▼─────────┐
  QUERY                 │                                     │
                        │  question  ──►  embed (dense+sparse)│
                        │                           │         │
                        │            hybrid search (RRF fusion│
                        │              dense + BM25 prefetch) │
                        │                           │         │
                        │            Claude (claude-sonnet-4) │
                        │          grounded prompt + citations│
                        │                           │         │
                        │              answer + sources[]     │
                        └─────────────────────────────────────┘
```

**Stack:** Python 3.11 · FastAPI · Qdrant · Ollama (`nomic-embed-text`) · Claude API

## How It Works

### Ingestion

Two API calls per connector fetch the docs from Ballerina Central:

1. **Registry API** — returns the package readme markdown and version metadata
2. **Docs API** — returns structured API reference JSON (records, clients, functions, types, enums, errors)

Both are rendered into a single markdown file per connector and cached locally in `data/raw/`. Currently ingests 5 connectors: kafka, rabbitmq, mysql, java.jdbc, and twilio.

### Chunking

Three strategies are available (set via `CHUNKING_STRATEGY` env var):

- **Hierarchical (default)** — splits on H1–H4 boundaries, prefixing every chunk with its full ancestor breadcrumb (e.g. `[H1: kafka] [H2: Records] [H3: ConsumerConfiguration]`). Also emits **field-level chunks** from markdown tables so each record field gets its own chunk. Oversized sections are split via sliding window (400 tokens, 50-token overlap).
  - *Why:* breadcrumb prefixes preserve parent context — a chunk for "ConsumerConfiguration" retains that it's under "Records" under "kafka". Field-level chunks enable precise retrieval of individual record fields.
- **Heading** — splits on H1/H2/H3 boundaries with sliding window for oversized sections. Simpler but loses parent context.
- **Fixed** — naive fixed-size token splits. Baseline for comparison.

### Retrieval

Three retrieval modes (set via `RETRIEVAL_MODE` env var):

- **Dense** — semantic similarity via `nomic-embed-text` (768-dim vectors, cosine distance). Queries are prefixed with `search_query:` and documents with `search_document:` per Nomic's recommended usage. Good for natural language queries like *"how do I authenticate?"*.
- **Sparse (BM25)** — custom tokenizer that splits on non-alphanumeric boundaries and does camelCase splitting (e.g. `decoupleProcessing` → `["decouple", "processing", "decoupleprocessing"]`). Tokens are hashed via MurmurHash3 into 2^18 (262k) buckets. Uses log-normalized TF (`1 + log(tf)`); Qdrant applies IDF server-side. Good for exact API identifiers like `ProducerRecord` or `sendSms`.
- **Hybrid (default)** — prefetches top `2×k` results from both dense and sparse, then fuses them with **Reciprocal Rank Fusion (RRF)**. Gets the best of both worlds.
  - *Why hybrid:* pure vector search misses exact API identifiers; pure keyword search misses semantic intent. Hybrid closes this gap.

### Generation

Retrieved chunks are formatted into a numbered context block, each showing the connector, section, URL, and relevance score. The system prompt constrains Claude to:

- Answer **only** from the provided documentation excerpts — no guessing
- Cite sources using `[connector/section]` notation inline
- Include a **Sources** list at the end with URLs for traceability

## Quick Start

**Prerequisites:** Docker, [Ollama](https://ollama.com), an Anthropic API key, and `uv`.

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# → fill in ANTHROPIC_API_KEY in .env

# 3. Start Qdrant
docker compose up -d

# 4. Pull the embedding model
ollama pull nomic-embed-text

# 5. Ingest connector docs (scrapes + embeds 5 connectors, ~30s)
uv run python scripts/ingest.py

# 6. Ask a question
uv run python scripts/query.py "How do I manually acknowledge a RabbitMQ message?"
```

Or run the API server and use the interactive docs at `http://localhost:8000/docs`:

```bash
uv run uvicorn docsense.api.main:app --reload
```

## Example

```
$ uv run python scripts/query.py \
    "How do I manually acknowledge a RabbitMQ message?" \
    --show-chunks

Retrieved 5 chunks:
  [0.856] ballerinax/rabbitmq / Advanced usage
  [0.757] ballerinax/rabbitmq / Message Acknowledgment
  ...

=== Answer ===
To manually acknowledge a RabbitMQ message:

1. Set `autoAck: false` in the service config [rabbitmq/Advanced usage]
2. Call `caller->basicAck()` to positively acknowledge [rabbitmq/Advanced usage]
3. Call `caller->basicNack(true, requeue = false)` to reject [rabbitmq/Advanced usage]

@rabbitmq:ServiceConfig { queueName: "MyQueue", autoAck: false }
service rabbitmq:Service on channelListener {
    remote function onMessage(rabbitmq:BytesMessage message, rabbitmq:Caller caller) {
        rabbitmq:Error? result = caller->basicAck();
    }
}

Sources: rabbitmq/Advanced usage, rabbitmq/Message Acknowledgment

(chunks_used=5, latency=6842ms)
```

## Eval Results (hierarchical strategy · hybrid retrieval · top_k=5)

| Metric | Score |
|---|---|
| Retrieval hit rate | 15/15 (100%) |
| Keyword hit rate | 9/15 (60%) |

| Connector | Retrieval | Keywords |
|---|---|---|
| ballerinax/kafka | 3/3 | 2/3 |
| ballerinax/rabbitmq | 4/4 | 3/4 |
| ballerinax/mysql | 4/4 | 3/4 |
| ballerinax/java.jdbc | 2/2 | 1/2 |
| ballerinax/twilio | 2/2 | 0/2 |

The retriever always finds the right section. Keyword misses are mostly exact API identifiers (e.g. `sendSms`, `ProducerRecord`) that the answer paraphrases — hybrid search helps close this gap, and re-running evals with updated prompts is next.

## Roadmap

- ~~**Phase 6** — Hybrid search (BM25 + vector) to improve keyword hit rate~~ ✅
- **Phase 7** — Reranking with a cross-encoder for better context precision
- **Phase 8** — Agentic multi-hop for cross-connector questions
- **Phase 9** — RAGAS evaluation for richer, multi-dimensional metrics
