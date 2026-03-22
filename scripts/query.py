"""
CLI: run a question through the full RAG pipeline and print the answer.

Usage:
    uv run python scripts/query.py "How do I configure SASL auth for Kafka?"
    uv run python scripts/query.py "How do I ack a RabbitMQ message?" --connector ballerinax/rabbitmq
    uv run python scripts/query.py "..." --top-k 8
    uv run python scripts/query.py "..." --show-chunks   # print full retrieved chunks before the answer
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

from docsense.generation.claude import generate
from docsense.retrieval.retriever import retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the DocSense RAG pipeline")
    parser.add_argument("question", help="Natural language question")
    parser.add_argument("--connector", "-c", default=None, help="Filter by connector (e.g. ballerinax/kafka)")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--show-chunks", action="store_true", help="Print retrieved chunks before the answer")
    parser.add_argument("--mode", "-m", choices=["dense", "sparse", "hybrid"], default=None,
                        help="Retrieval mode (default: from config)")
    args = parser.parse_args()

    print(f"\nQuestion: {args.question}")
    if args.connector:
        print(f"Filter:   {args.connector}")
    print()

    t0 = time.perf_counter()

    chunks = retrieve(args.question, top_k=args.top_k, connector_filter=args.connector, mode=args.mode)

    if not chunks:
        print("No relevant chunks found. Has the ingestion pipeline been run?")
        print("  uv run python scripts/ingest.py")
        sys.exit(1)

    if args.show_chunks:
        print("=== Retrieved Chunks ===")
        for i, c in enumerate(chunks, 1):
            print(f"[{i}] score={c.score:.3f}  {c.connector} / {c.section}")
            print(c.text.strip())
            print()
            print("---")

    response = generate(args.question, chunks)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    print("=== Answer ===")
    print(response.answer)
    print(f"\n(chunks_used={response.chunks_used}, latency={latency_ms}ms)")


if __name__ == "__main__":
    main()
