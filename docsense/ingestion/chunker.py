"""
Chunker: splits raw connector docs into chunks ready for embedding.

Three strategies (set via config or per-call):
  - "heading" (default): split on H1/H2/H3 boundaries, then apply a
    sliding window within any section that exceeds chunk_size_tokens.
  - "fixed": naive fixed-size splits with overlap, ignoring headings.
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
    elif strategy == "semantic":
        raise NotImplementedError("Semantic chunking is planned for Phase 2")
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")
