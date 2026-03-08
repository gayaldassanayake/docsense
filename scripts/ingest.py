"""
CLI: run the ingestion pipeline to scrape, chunk, embed, and upsert connector docs.

Usage:
    uv run python scripts/ingest.py
    uv run python scripts/ingest.py --connectors ballerinax/kafka ballerinax/mysql
    uv run python scripts/ingest.py --force   # re-fetch even if cached
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

from docsense.ingestion.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest connector docs into DocSense")
    parser.add_argument(
        "--connectors", "-c", nargs="+", metavar="ORG/NAME",
        help="Connectors to ingest (e.g. ballerinax/kafka). Defaults to all 5.",
    )
    parser.add_argument("--force", "-f", action="store_true", help="Re-fetch even if cached")
    args = parser.parse_args()

    connectors = None
    if args.connectors:
        connectors = []
        for s in args.connectors:
            parts = s.split("/", 1)
            if len(parts) != 2:
                print(f"Error: connector must be in ORG/NAME format, got {s!r}")
                sys.exit(1)
            connectors.append(tuple(parts))

    t0 = time.perf_counter()
    result = run(connectors=connectors, force_scrape=args.force)
    duration = time.perf_counter() - t0

    print(f"\nDone — {result['chunks_ingested']} chunks ingested from {result['connectors_processed']} connectors in {duration:.1f}s")


if __name__ == "__main__":
    main()
