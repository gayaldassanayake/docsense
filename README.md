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
                        │                     (heading split) │
                        │                           │         │
                        │                   nomic-embed-text  │
                        │                     (Ollama local)  │
                        │                           │         │
                        │                        Qdrant       │
                        │                    (Docker, local)  │
                        └───────────────────────────┬─────────┘
                                                    │
                        ┌───────────────────────────▼─────────┐
  QUERY                 │                                     │
                        │  question  ──►  embed (search_query:│
                        │                    prefix)          │
                        │                           │         │
                        │               vector search top-k   │
                        │                           │         │
                        │            Claude (claude-sonnet-4) │
                        │          grounded prompt + citations│
                        │                           │         │
                        │              answer + sources[]     │
                        └─────────────────────────────────────┘
```

**Stack:** Python 3.11 · FastAPI · Qdrant · Ollama (`nomic-embed-text`) · Claude API

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

## Eval Results (baseline · heading strategy · top_k=5)

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

The retriever always finds the right section. Keyword misses are mostly exact API identifiers (e.g. `sendSms`, `ProducerRecord`) that the answer paraphrases — a hybrid BM25 + vector search would close this gap.

## Roadmap

- **Phase 6** — Hybrid search (BM25 + vector) to improve keyword hit rate
- **Phase 7** — Reranking with a cross-encoder for better context precision
- **Phase 8** — Agentic multi-hop for cross-connector questions
- **Phase 9** — RAGAS evaluation for richer, multi-dimensional metrics
