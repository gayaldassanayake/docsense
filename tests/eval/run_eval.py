"""
Eval harness: run each question through the RAG pipeline and measure quality.

Two metrics:
  - Retrieval hit rate: was the expected section in the top-k retrieved chunks?
  - Keyword hit rate:   did all expected keywords appear in the generated answer?

Usage:
    uv run python tests/eval/run_eval.py
    uv run python tests/eval/run_eval.py --no-generate   # retrieval only (faster)
    uv run python tests/eval/run_eval.py --top-k 8
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from docsense.generation.claude import generate
from docsense.retrieval.retriever import retrieve


def run_eval(top_k: int = 5, skip_generate: bool = False) -> None:
    questions_path = Path("tests/eval/questions.json")
    questions = json.loads(questions_path.read_text())

    results = []

    for q in questions:
        qid = q["id"]
        question = q["question"]
        connector = q["connector"]
        expected_section = q["expected_source_section"].lower()
        expected_keywords = q["expected_keywords"]

        # --- Retrieval ---
        chunks = retrieve(question, top_k=top_k, connector_filter=connector)
        retrieved_sections = [c.section.lower() for c in chunks]
        top_score = chunks[0].score if chunks else 0.0

        # Detect fixed chunking by checking if section names are all "chunk-N".
        # Fixed chunks have no heading metadata, so fall back to checking whether
        # expected keywords appear in the retrieved chunk text instead.
        _fixed_pattern = re.compile(r"^chunk-\d+$")
        using_fixed_chunks = all(_fixed_pattern.match(s) for s in retrieved_sections)

        if using_fixed_chunks:
            # Retrieval hit = at least one retrieved chunk contains an expected keyword
            retrieval_hit = any(
                kw.lower() in c.text.lower()
                for c in chunks
                for kw in expected_keywords
            )
            retrieval_method = "keyword-in-text"
        else:
            # Retrieval hit = expected section name found in retrieved chunk sections
            retrieval_hit = any(expected_section in s for s in retrieved_sections)
            retrieval_method = "section-name"

        # --- Generation ---
        keyword_hit = None
        missing_keywords = []
        if not skip_generate:
            response = generate(question, chunks)
            answer = response.answer.lower()
            missing_keywords = [kw for kw in expected_keywords if kw.lower() not in answer]
            keyword_hit = len(missing_keywords) == 0

        results.append({
            "id": qid,
            "connector": connector,
            "question": question,
            "retrieval_hit": retrieval_hit,
            "retrieval_method": retrieval_method,
            "top_score": top_score,
            "keyword_hit": keyword_hit,
            "missing_keywords": missing_keywords,
            "expected_section": q["expected_source_section"],
            "retrieved_sections": [c.section for c in chunks[:3]],
        })

        # Live progress
        r_mark = "✓" if retrieval_hit else "✗"
        k_mark = ("✓" if keyword_hit else "✗") if keyword_hit is not None else "-"
        print(f"[{qid}] retrieval={r_mark}  keywords={k_mark}  score={top_score:.3f}  [{retrieval_method}]")
        if not retrieval_hit:
            if retrieval_method == "section-name":
                print(f"  expected section: {q['expected_source_section']!r}")
                print(f"  got sections:     {[c.section for c in chunks[:3]]}")
            else:
                print(f"  expected keywords in chunks: {expected_keywords}")
                print(f"  got sections:                {[c.section for c in chunks[:3]]}")
        if missing_keywords:
            print(f"  missing keywords: {missing_keywords}")

    # --- Summary ---
    n = len(results)
    retrieval_hits = sum(1 for r in results if r["retrieval_hit"])
    retrieval_rate = retrieval_hits / n * 100

    print(f"\n{'='*55}")
    print(f"{'EVAL RESULTS':^55}")
    print(f"{'='*55}")
    print(f"  Questions:          {n}")
    print(f"  Top-k:              {top_k}")
    print(f"  Retrieval hit rate: {retrieval_hits}/{n}  ({retrieval_rate:.0f}%)")

    if not skip_generate:
        keyword_results = [r for r in results if r["keyword_hit"] is not None]
        keyword_hits = sum(1 for r in keyword_results if r["keyword_hit"])
        keyword_rate = keyword_hits / len(keyword_results) * 100
        print(f"  Keyword hit rate:   {keyword_hits}/{len(keyword_results)}  ({keyword_rate:.0f}%)")

    print(f"{'='*55}")

    # --- Per-connector breakdown ---
    connectors = sorted(set(r["connector"] for r in results))
    print(f"\n{'Connector':<30} {'Retrieval':>10} {'Keywords':>10}")
    print("-" * 52)
    for conn in connectors:
        conn_results = [r for r in results if r["connector"] == conn]
        r_hits = sum(1 for r in conn_results if r["retrieval_hit"])
        r_total = len(conn_results)
        r_str = f"{r_hits}/{r_total}"

        if not skip_generate:
            k_hits = sum(1 for r in conn_results if r["keyword_hit"])
            k_str = f"{k_hits}/{r_total}"
        else:
            k_str = "-"

        print(f"  {conn:<28} {r_str:>10} {k_str:>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DocSense eval harness")
    parser.add_argument("--top-k", "-k", type=int, default=5)
    parser.add_argument("--no-generate", action="store_true", help="Skip LLM generation (retrieval only)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    run_eval(top_k=args.top_k, skip_generate=args.no_generate)
    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s")
