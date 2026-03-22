"""
Scraper: fetches Ballerina connector docs from the Central API.

Two API calls per connector:
  1. Registry API  → package metadata + readme markdown
  2. Docs API      → structured API reference (records, clients, functions, types, enums)

Both are combined and saved to data/raw/<org>-<name>.md.
Subsequent runs skip connectors whose cache file already exists.
"""

from pathlib import Path

import httpx

from docsense.config import settings

RAW_DIR = Path("data/raw")

# The 5 connectors we start with (spec §1)
DEFAULT_CONNECTORS = [
    ("ballerinax", "kafka"),
    ("ballerinax", "rabbitmq"),
    ("ballerinax", "twilio"),
    ("ballerinax", "java.jdbc"),
    ("ballerinax", "mysql"),
]


# ---------------------------------------------------------------------------
# Low-level API helpers
# ---------------------------------------------------------------------------


def _get(url: str) -> dict:
    """Perform a GET request and return the parsed JSON, or {} on error."""
    try:
        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Warning: could not fetch {url}: {e}")
        return {}


def _type_name(type_obj: dict) -> str:
    """Extract a human-readable type name from a type descriptor object."""
    if not type_obj:
        return "any"
    name = type_obj.get("name", "any")
    if type_obj.get("isArrayType"):
        name = f"{name}[]"
    if type_obj.get("isNullable") or type_obj.get("isOptional"):
        name = f"{name}?"
    return name


# ---------------------------------------------------------------------------
# Markdown rendering from docs API JSON
# ---------------------------------------------------------------------------


def _render_fields_table(fields: list[dict]) -> str:
    """Render a list of record/object fields as a markdown table."""
    if not fields:
        return ""
    rows = ["| Field | Type | Default | Description |",
            "|-------|------|---------|-------------|"]
    for f in fields:
        fname = f.get("name", "")
        ftype = _type_name(f.get("type", {}))
        fdefault = f.get("defaultValue") or "—"
        fdesc = (f.get("description") or "").replace("\n", " ").strip()
        rows.append(f"| `{fname}` | `{ftype}` | {fdefault} | {fdesc} |")
    return "\n".join(rows)


def _render_params(params: list[dict]) -> str:
    """Render method parameters as a compact inline signature."""
    parts = []
    for p in params:
        pname = p.get("name", "")
        ptype = _type_name(p.get("type", {}))
        parts.append(f"{ptype} {pname}")
    return ", ".join(parts)


def _render_method(method: dict, level: int = 4) -> str:
    """Render a single client method/function to markdown."""
    heading = "#" * level
    name = method.get("name", "")
    desc = (method.get("description") or "").strip()
    params = method.get("parameters", [])
    returns = method.get("returnParameters", [])

    lines = [f"{heading} `{name}`"]
    if desc:
        lines.append(desc)
    if params:
        lines.append(f"**Parameters:** `{_render_params(params)}`")
    if returns:
        ret_types = ", ".join(_type_name(r.get("type", {})) for r in returns)
        lines.append(f"**Returns:** `{ret_types}`")
    return "\n\n".join(lines)


def _render_module_docs(module: dict) -> list[str]:
    """Convert one module's docs JSON into a list of markdown section strings."""
    sections: list[str] = []

    # --- Records ---
    records = module.get("records", [])
    if records:
        rec_parts = ["## Records"]
        for rec in records:
            name = rec.get("name", "")
            desc = (rec.get("description") or "").strip()
            rec_parts.append(f"### {name}")
            if desc:
                rec_parts.append(desc)
            table = _render_fields_table(rec.get("fields", []))
            if table:
                rec_parts.append(table)
        sections.append("\n\n".join(rec_parts))

    # --- Clients ---
    clients = module.get("clients", [])
    if clients:
        client_parts = ["## Clients"]
        for client in clients:
            name = client.get("name", "")
            desc = (client.get("description") or "").strip()
            client_parts.append(f"### {name}")
            if desc:
                client_parts.append(desc)

            all_methods = (
                client.get("remoteMethods", [])
                + client.get("methods", [])
                + client.get("otherMethods", [])
            )
            for method in all_methods:
                client_parts.append(_render_method(method, level=4))
        sections.append("\n\n".join(client_parts))

    # --- Functions ---
    functions = module.get("functions", [])
    if functions:
        fn_parts = ["## Functions"]
        for fn in functions:
            fn_parts.append(_render_method(fn, level=3))
        sections.append("\n\n".join(fn_parts))

    # --- Types ---
    types = module.get("types", [])
    if types:
        type_parts = ["## Types"]
        for t in types:
            name = t.get("name", "")
            desc = (t.get("description") or "").strip()
            type_parts.append(f"### {name}")
            if desc:
                type_parts.append(desc)
        sections.append("\n\n".join(type_parts))

    # --- Enums ---
    enums = module.get("enums", [])
    if enums:
        enum_parts = ["## Enums"]
        for enum in enums:
            name = enum.get("name", "")
            desc = (enum.get("description") or "").strip()
            enum_parts.append(f"### {name}")
            if desc:
                enum_parts.append(desc)
            members = enum.get("members", [])
            if members:
                member_lines = ["| Member | Description |", "|--------|-------------|"]
                for mem in members:
                    mname = mem.get("name", "")
                    mdesc = (mem.get("description") or "").replace("\n", " ").strip()
                    member_lines.append(f"| `{mname}` | {mdesc} |")
                enum_parts.append("\n".join(member_lines))
        sections.append("\n\n".join(enum_parts))

    # --- Errors ---
    errors = module.get("errors", [])
    if errors:
        err_parts = ["## Errors"]
        for err in errors:
            name = err.get("name", "")
            desc = (err.get("description") or "").strip()
            err_parts.append(f"### {name}")
            if desc:
                err_parts.append(desc)
        sections.append("\n\n".join(err_parts))

    return sections


# ---------------------------------------------------------------------------
# Public scrape function
# ---------------------------------------------------------------------------


def scrape(
    connectors: list[tuple[str, str]] | None = None,
    force: bool = False,
) -> list[Path]:
    """
    Fetch and cache documentation for the given connectors.

    For each connector:
      1. Calls the registry API to get the readme + version.
      2. Calls the docs API to get structured API reference.
      3. Combines both into one markdown file saved to data/raw/<org>-<name>.md.

    Args:
        connectors: list of (org, name) tuples. Defaults to DEFAULT_CONNECTORS.
        force:      re-fetch even if cache file exists.

    Returns:
        list of Path objects for the cached files.
    """
    if connectors is None:
        connectors = DEFAULT_CONNECTORS

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    base = settings.ballerina_central_api

    cached_paths: list[Path] = []

    for org, name in connectors:
        cache_file = RAW_DIR / f"{org}-{name}.md"

        if cache_file.exists() and not force:
            print(f"[cache] {org}/{name} → {cache_file}")
            cached_paths.append(cache_file)
            continue

        print(f"[fetch] {org}/{name} …")

        # Step 1: registry API — readme + version
        registry_data = _get(f"{base}/registry/packages/{org}/{name}/latest")
        if not registry_data:
            print(f"  Warning: registry API returned nothing for {org}/{name}")
            continue

        version = registry_data.get("version", "latest")
        readme = (registry_data.get("readme") or "").strip()
        summary = (registry_data.get("summary") or "").strip()
        source_url = (registry_data.get("sourceCodeLocation") or "").strip()

        # Step 2: docs API — structured API reference
        docs_data = _get(f"{base}/docs/{org}/{name}/{version}")
        modules = docs_data.get("docsData", {}).get("modules", [])

        sections: list[str] = []

        # Overview section from registry API
        header_lines = [f"# {org}/{name} — Overview & Setup"]
        if summary:
            header_lines.append(f"**Summary:** {summary}")
        if source_url:
            header_lines.append(f"**Source:** {source_url}")
        if readme:
            header_lines.append(readme)
        sections.append("\n\n".join(header_lines))

        # API reference sections from docs API
        if modules:
            api_header = f"# {org}/{name} — API Reference (v{version})"
            module_sections = []
            for module in modules:
                module_sections.extend(_render_module_docs(module))
            if module_sections:
                sections.append(api_header + "\n\n" + "\n\n---\n\n".join(module_sections))

        if not sections:
            print(f"  Warning: no content found for {org}/{name}")
            continue

        content = "\n\n---\n\n".join(sections)
        cache_file.write_text(content, encoding="utf-8")
        print(f"  Saved {len(content):,} chars → {cache_file}")
        cached_paths.append(cache_file)

    return cached_paths
