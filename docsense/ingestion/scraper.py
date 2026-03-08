"""
Scraper: loads connector data from connectors.json and fetches API doc pages.

Two content sources per connector:
  1. `readme`      — markdown already in the JSON (overview, setup, examples)
  2. `apiDocURL`   — HTML page on lib.ballerina.io (types, records, functions)

Both are combined and saved to data/raw/<org>-<name>.md.
Subsequent runs skip connectors whose cache file already exists.
"""

import json
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

CONNECTORS_FILE = Path("resources/connectors.json")
RAW_DIR = Path("data/raw")

# The 5 connectors we start with (spec §1)
DEFAULT_CONNECTORS = [
    ("ballerinax", "kafka"),
    ("ballerinax", "rabbitmq"),
    ("ballerinax", "twilio"),
    ("ballerinax", "java.jdbc"),
    ("ballerinax", "mysql"),
]


def _fetch_api_docs(url: str) -> str:
    """Fetch the lib.ballerina.io API doc page and extract readable text."""
    try:
        response = httpx.get(url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as e:
        print(f"  Warning: could not fetch {url}: {e}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")

    # The main content lives in <main> or a div with class "api-doc-content"
    main = soup.find("main") or soup.find("div", class_="api-doc-content") or soup.body
    if not main:
        return ""

    # Remove nav, footer, script, style noise
    for tag in main.find_all(["nav", "footer", "script", "style", "button"]):
        tag.decompose()

    return main.get_text(separator="\n", strip=True)


def _load_connector_data(org: str, name: str, all_packages: list[dict]) -> dict | None:
    """Find a connector in connectors.json and return its combined content."""
    for pkg in all_packages:
        if pkg.get("organization") == org and pkg.get("name") == name:
            return pkg
    return None


def scrape(
    connectors: list[tuple[str, str]] | None = None,
    force: bool = False,
) -> list[Path]:
    """
    Scrape and cache documentation for the given connectors.

    Args:
        connectors: list of (org, name) tuples. Defaults to DEFAULT_CONNECTORS.
        force:      re-fetch even if cache file exists.

    Returns:
        list of Path objects for the cached files.
    """
    if connectors is None:
        connectors = DEFAULT_CONNECTORS

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONNECTORS_FILE) as f:
        all_packages = json.load(f)

    cached_paths: list[Path] = []

    for org, name in connectors:
        cache_file = RAW_DIR / f"{org}-{name}.md"

        if cache_file.exists() and not force:
            print(f"[cache] {org}/{name} → {cache_file}")
            cached_paths.append(cache_file)
            continue

        print(f"[fetch] {org}/{name} …")
        pkg = _load_connector_data(org, name, all_packages)
        if not pkg:
            print(f"  Warning: {org}/{name} not found in connectors.json")
            continue

        sections: list[str] = []

        # Source 1: readme (already markdown)
        readme = pkg.get("readme", "").strip()
        if readme:
            sections.append(f"# {org}/{name} — Overview & Setup\n\n{readme}")

        # Source 2: API doc page (HTML → plain text) from the first module
        modules = pkg.get("modules", [])
        if modules:
            api_doc_url = modules[0].get("apiDocURL", "")
            if api_doc_url:
                print(f"  Fetching API docs: {api_doc_url}")
                api_text = _fetch_api_docs(api_doc_url)
                if api_text:
                    sections.append(f"# {org}/{name} — API Reference\n\n{api_text}")

                time.sleep(0.5)  # be polite to the server

        if not sections:
            print(f"  Warning: no content found for {org}/{name}")
            continue

        content = "\n\n---\n\n".join(sections)
        cache_file.write_text(content, encoding="utf-8")
        print(f"  Saved {len(content):,} chars → {cache_file}")
        cached_paths.append(cache_file)

    return cached_paths
