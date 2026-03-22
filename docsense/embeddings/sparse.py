"""
Lightweight BM25 sparse embedding for hybrid retrieval.

Produces sparse vectors (token_id → term_frequency) that Qdrant combines
with server-side IDF (via Modifier.IDF on the sparse vector config).

Tokenization approach:
  1. Lowercase the text
  2. Split on non-alphanumeric characters (preserving camelCase as one token)
  3. Also split camelCase into sub-tokens so both the full name and parts are indexed
  4. Hash each token to a 32-bit integer index using MurmurHash3

This avoids fastembed's Qdrant/bm25 model which crashes due to a
py-rust-stemmers segfault on macOS ARM + Python 3.13+.
"""

import math
import re
from collections import Counter

import mmh3

# Vocabulary size — sparse vector indices are hashed into this range.
# 2^18 = 262144 buckets. Large enough to avoid excessive collisions.
VOCAB_SIZE = 2**18

# camelCase / PascalCase boundary: split before an uppercase letter
# that follows a lowercase letter (e.g., "decoupleProcessing" → ["decouple", "Processing"])
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")

# Tokenization: split on anything that isn't alphanumeric or underscore
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

# English stopwords — filtered out to prevent common words from inflating
# scores for long documents. This is standard BM25 practice.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "must", "can", "could", "it", "its",
    "this", "that", "these", "those", "i", "we", "you", "he", "she", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how", "not",
    "no", "nor", "if", "then", "else", "so", "as", "up", "out", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "once", "here", "there", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "than", "too", "very", "just", "also",
})


def _tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 sparse embedding.

    Extracts raw tokens, then also splits camelCase into sub-tokens.
    Both the original token and its camelCase parts are kept so that
    a query for "decoupleProcessing" matches both the exact field name
    and queries mentioning "decouple" or "processing" separately.
    """
    raw_tokens = _TOKEN_RE.findall(text)
    tokens: list[str] = []
    for tok in raw_tokens:
        lower = tok.lower()
        if lower not in _STOPWORDS:
            tokens.append(lower)
        # Split camelCase and add sub-tokens
        parts = _CAMEL_RE.split(tok)
        if len(parts) > 1:
            for part in parts:
                part_lower = part.lower()
                if part_lower != lower and part_lower not in _STOPWORDS:
                    tokens.append(part_lower)
    return tokens


def _to_sparse_vector(text: str) -> tuple[list[int], list[float]]:
    """
    Convert text to a sparse vector of (indices, values).

    Indices are MurmurHash3 hashes of tokens mod VOCAB_SIZE.
    Values are raw term frequencies (Qdrant applies IDF server-side).
    """
    tokens = _tokenize(text)
    if not tokens:
        return ([], [])

    counts = Counter(mmh3.hash(t, signed=False) % VOCAB_SIZE for t in tokens)
    indices = sorted(counts.keys())
    # Log-normalized TF: 1 + log(tf). Dampens the advantage of long documents
    # where common terms appear many times. Qdrant multiplies by IDF server-side.
    values = [1.0 + math.log(counts[i]) for i in indices]
    return (indices, values)


def sparse_embed_documents(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    """
    Batch-embed documents for indexing.

    Returns a list of (indices, values) tuples — one per input text.
    Each tuple represents a sparse vector where:
      - indices: hashed token IDs that appear in the text
      - values: term frequency weights for those tokens
    """
    return [_to_sparse_vector(text) for text in texts]


def sparse_embed_query(text: str) -> tuple[list[int], list[float]]:
    """
    Embed a single query string.

    Returns (indices, values) tuple. Same tokenization as documents —
    IDF weighting is handled by Qdrant server-side.
    """
    return _to_sparse_vector(text)
