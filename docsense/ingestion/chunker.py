"""
Chunker: splits raw connector docs into chunks ready for embedding.

Four strategies (set via config or per-call):
  - "heading" (default): split on H1/H2/H3 boundaries, then apply a
    sliding window within any section that exceeds chunk_size_tokens.
  - "fixed": naive fixed-size splits with overlap, ignoring headings.
  - "hierarchical": split on H1-H4, prefix every chunk with its full ancestor
    breadcrumb (e.g. "[H1: kafka] [H2: Records] [H3: ConsumerConfiguration]"),
    and emit additional field-level chunks for markdown tables.
  - "semantic": placeholder for Phase 2.

Each chunk is a dict:
  {
    "chunk_id":    str,   # e.g. "ballerinax-kafka-003"
    "connector":   str,   # e.g. "ballerinax/kafka"
    "source_url":  str,
    "section":     str,   # heading path, e.g. "Security Configurations"
    "text":        str,
    "token_count": int,
  }
"""

import re
from pathlib import Path

import tiktoken

from docsense.config import settings

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _sliding_window(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into token-bounded chunks with overlap."""
    tokens = _enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_enc.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """
    Split markdown text on H1/H2/H3 boundaries.
    Returns a list of (heading, body) tuples.
    """
    pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        return [("Document", text.strip())]

    sections = []
    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((heading, body))

    return sections if sections else [("Document", text.strip())]


def chunk_heading(
    text: str,
    connector: str,
    source_url: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """Split by headings, then apply sliding window to oversized sections."""
    sections = _split_by_headings(text)
    chunks = []
    idx = 0

    for heading, body in sections:
        token_count = _count_tokens(body)
        if token_count <= chunk_size:
            chunks.append({
                "chunk_id": f"{connector.replace('/', '-')}-{idx:04d}",
                "connector": connector,
                "source_url": source_url,
                "section": heading,
                "text": f"{heading}\n\n{body}",
                "token_count": token_count,
            })
            idx += 1
        else:
            # Section is too large — slide a window through it
            sub_chunks = _sliding_window(body, chunk_size, overlap)
            for j, sub in enumerate(sub_chunks):
                chunks.append({
                    "chunk_id": f"{connector.replace('/', '-')}-{idx:04d}",
                    "connector": connector,
                    "source_url": source_url,
                    "section": f"{heading} (part {j + 1})",
                    "text": f"{heading}\n\n{sub}",
                    "token_count": _count_tokens(sub),
                })
                idx += 1

    return chunks


def chunk_fixed(
    text: str,
    connector: str,
    source_url: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """Naive fixed-size splits, ignoring heading structure."""
    sub_chunks = _sliding_window(text, chunk_size, overlap)
    return [
        {
            "chunk_id": f"{connector.replace('/', '-')}-fixed-{i:04d}",
            "connector": connector,
            "source_url": source_url,
            "section": f"chunk-{i}",
            "text": sub,
            "token_count": _count_tokens(sub),
        }
        for i, sub in enumerate(sub_chunks)
    ]


def _parse_hierarchy(text: str) -> list[dict]:
    """
    Parse markdown into sections tracking the full ancestor breadcrumb.

    Each returned item:
      {
        "level":      int,   # 1–4
        "heading":    str,   # just this node's heading text
        "breadcrumb": str,   # "[H1: x] [H2: y] [H3: z]" up to this level
        "body":       str,   # text between this heading and the next heading
      }

    Why we need this: the plain `_split_by_headings` discards parent context,
    so a chunk for "ConsumerConfiguration" has no indication it lives under
    "Records" under "kafka". The breadcrumb restores that signal for the
    embedding model.
    """
    pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        return [{"level": 1, "heading": "Document", "breadcrumb": "[H1: Document]", "body": text.strip()}]

    # Stack tracks the current heading at each depth: stack[0]=H1, stack[1]=H2, …
    stack: list[str | None] = [None, None, None, None]

    sections = []
    for i, match in enumerate(matches):
        level = len(match.group(1))   # number of '#' chars
        heading = match.group(2).strip()

        # Update the stack: set this level, clear all deeper levels
        stack[level - 1] = heading
        for deeper in range(level, 4):
            stack[deeper] = None

        # Build breadcrumb from H1 up to the current level (skip empty slots)
        crumb_parts = [f"[H{lvl + 1}: {stack[lvl]}]" for lvl in range(level) if stack[lvl]]
        breadcrumb = " ".join(crumb_parts)

        # Body = text from after this heading to the start of the next heading
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        if body:
            sections.append({
                "level": level,
                "heading": heading,
                "breadcrumb": breadcrumb,
                "body": body,
            })

    return sections if sections else [{"level": 1, "heading": "Document", "breadcrumb": "[H1: Document]", "body": text.strip()}]


def _parse_field_rows(body: str) -> list[dict]:
    """
    Extract markdown table rows as field-level dicts.

    Expects tables with at least 'Field' and 'Type' header columns.
    Returns one dict per data row with keys: field, type, default, description.
    Returns [] if no such table is found.

    Why field-level chunks: a single H3 section for a record type may contain
    10+ fields. Splitting each field into its own chunk makes exact-match
    retrieval far more precise — "what does decoupleProcessing do?" will hit
    the field chunk directly rather than a large section blob.
    """
    lines = body.splitlines()

    # Find the header line that contains both "Field" and "Type"
    header_idx = None
    for i, line in enumerate(lines):
        if "|" in line and "Field" in line and "Type" in line:
            header_idx = i
            break

    if header_idx is None:
        return []

    # Parse column names from the header row
    header_cells = [c.strip() for c in lines[header_idx].split("|") if c.strip()]
    col_names = [h.lower() for h in header_cells]

    def _get(cells: list[str], name: str) -> str:
        """Return cell value for a column name, or '' if absent."""
        try:
            return cells[col_names.index(name)].strip(" `")
        except (ValueError, IndexError):
            return ""

    rows = []
    for line in lines[header_idx + 1:]:
        # Skip separator rows (e.g. |---|---|)
        if re.match(r"^\s*\|[\s\-|]+\|\s*$", line):
            continue
        if "|" not in line:
            break  # table ended
        cells = [c.strip() for c in line.split("|") if c.strip() or line.strip().startswith("|")]
        # Re-split cleanly: split on | and drop first/last empty strings
        cells = [c.strip() for c in line.split("|")]
        cells = cells[1:-1] if len(cells) > 2 else cells  # strip leading/trailing empty

        if not any(cells):
            break

        rows.append({
            "field":       _get(cells, "field"),
            "type":        _get(cells, "type"),
            "default":     _get(cells, "default"),
            "description": _get(cells, "description"),
        })

    return rows


def chunk_hierarchical(
    text: str,
    connector: str,
    source_url: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """
    Hierarchical chunking: breadcrumb-prefixed section chunks + field-level chunks.

    Two types of output chunks:
      1. Section chunks — one per heading section (or split via sliding window
         if the body exceeds chunk_size). Text format:
           {breadcrumb}

           {body}

      2. Field chunks — one per markdown table row inside a section. Text format:
           {breadcrumb}

           Field: {field} | Type: {type} | {description}

    The `section` field stores the full breadcrumb instead of just the leaf
    heading name, so downstream retrieval logging shows full context.
    """
    sections = _parse_hierarchy(text)
    chunks = []
    idx = 0

    connector_slug = connector.replace("/", "-")

    for sec in sections:
        breadcrumb = sec["breadcrumb"]
        body = sec["body"]

        full_text = f"{breadcrumb}\n\n{body}"
        token_count = _count_tokens(full_text)

        if token_count <= chunk_size:
            chunks.append({
                "chunk_id":    f"{connector_slug}-hierarchical-{idx:04d}",
                "connector":   connector,
                "source_url":  source_url,
                "section":     breadcrumb,
                "text":        full_text,
                "token_count": token_count,
            })
            idx += 1
        else:
            # Body too large — slide a window through it, re-prepend breadcrumb
            sub_chunks = _sliding_window(body, chunk_size, overlap)
            for j, sub in enumerate(sub_chunks):
                sub_text = f"{breadcrumb}\n\n{sub}"
                chunks.append({
                    "chunk_id":    f"{connector_slug}-hierarchical-{idx:04d}",
                    "connector":   connector,
                    "source_url":  source_url,
                    "section":     f"{breadcrumb} (part {j + 1})",
                    "text":        sub_text,
                    "token_count": _count_tokens(sub_text),
                })
                idx += 1

        # Field-level chunks from any markdown table in this section's body
        field_rows = _parse_field_rows(body)
        for row in field_rows:
            if not row["field"]:
                continue
            desc = row["description"] or ""
            type_str = row["type"] or ""
            field_text = (
                f"{breadcrumb}\n\n"
                f"Field: {row['field']} | Type: {type_str} | {desc}"
            )
            chunks.append({
                "chunk_id":    f"{connector_slug}-hierarchical-{idx:04d}",
                "connector":   connector,
                "source_url":  source_url,
                "section":     breadcrumb,
                "text":        field_text,
                "token_count": _count_tokens(field_text),
            })
            idx += 1

    return chunks


def chunk_file(
    path: Path,
    connector: str,
    source_url: str,
    strategy: str | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[dict]:
    """
    Chunk a cached raw doc file.

    Args:
        path:       Path to the .md file in data/raw/.
        connector:  e.g. "ballerinax/kafka"
        source_url: canonical URL for citations
        strategy:   "heading" | "fixed" (defaults to config value)
        chunk_size: max tokens per chunk (defaults to config value)
        overlap:    overlap tokens for sliding window (defaults to config value)
    """
    strategy = strategy or settings.chunking_strategy
    chunk_size = chunk_size or settings.chunk_size_tokens
    overlap = overlap or settings.chunk_overlap_tokens

    text = path.read_text(encoding="utf-8")

    if strategy == "heading":
        return chunk_heading(text, connector, source_url, chunk_size, overlap)
    elif strategy == "fixed":
        return chunk_fixed(text, connector, source_url, chunk_size, overlap)
    elif strategy == "hierarchical":
        return chunk_hierarchical(text, connector, source_url, chunk_size, overlap)
    elif strategy == "semantic":
        raise NotImplementedError("Semantic chunking is planned for Phase 2")
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")
