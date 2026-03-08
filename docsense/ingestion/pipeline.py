"""
Ingestion pipeline: scrape → chunk → embed → upsert.

Orchestrates the three ingestion modules for one or more connectors.
"""

from pathlib import Path

from docsense.ingestion.chunker import chunk_file
from docsense.ingestion.scraper import DEFAULT_CONNECTORS, scrape
from docsense.vectorstore.qdrant import delete_by_connector, ensure_collection, upsert


def _source_url(org: str, name: str, all_packages: list[dict]) -> str:
    """Extract the canonical apiDocURL for citations."""
    import json
    for pkg in all_packages:
        if pkg.get("organization") == org and pkg.get("name") == name:
            modules = pkg.get("modules", [])
            if modules:
                return modules[0].get("apiDocURL", "")
    return f"https://central.ballerina.io/{org}/{name}"


def run(
    connectors: list[tuple[str, str]] | None = None,
    force_scrape: bool = False,
    chunking_strategy: str | None = None,
) -> dict:
    """
    Run the full ingestion pipeline.

    Args:
        connectors:        list of (org, name) tuples; defaults to DEFAULT_CONNECTORS
        force_scrape:      re-fetch docs even if cached
        chunking_strategy: override the config strategy ("heading" | "fixed")

    Returns:
        summary dict with chunks_ingested, connectors_processed
    """
    import json
    with open("resources/connectors.json") as f:
        all_packages = json.load(f)

    if connectors is None:
        connectors = DEFAULT_CONNECTORS

    ensure_collection()

    cached_paths: list[Path] = scrape(connectors=connectors, force=force_scrape)

    total_chunks = 0
    for path, (org, name) in zip(cached_paths, connectors):
        connector_id = f"{org}/{name}"
        source_url = _source_url(org, name, all_packages)

        print(f"\n[pipeline] chunking {connector_id} …")
        chunks = chunk_file(
            path,
            connector=connector_id,
            source_url=source_url,
            strategy=chunking_strategy,
        )
        print(f"  {len(chunks)} chunks produced")

        print(f"[pipeline] deleting old chunks for {connector_id} …")
        delete_by_connector(connector_id)

        print(f"[pipeline] upserting {connector_id} …")
        n = upsert(chunks)
        print(f"  {n} chunks upserted")
        total_chunks += n

    return {
        "connectors_processed": len(cached_paths),
        "chunks_ingested": total_chunks,
    }
