import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
from search import hybrid_search, get_document
from reranker import rerank
from generation import generate_answer, validate_citations, build_prompt


def test_full_generation():
    query = "chair makes noise when leaning back"
    top_k = 20
    top_n = 5

    print("=" * 80)
    print(f"FULL GENERATION TEST")
    print(f"Query: {query}")
    print("=" * 80)

    # Phase 2: Hybrid search
    print(f"\n[1/4] Running hybrid_search(top_k={top_k})...")
    start = time.perf_counter()
    candidates = hybrid_search(query, top_k=top_k)
    print(f"       Got {len(candidates)} candidates in {(time.perf_counter()-start)*1000:.0f}ms")

    # Fetch full content for candidates
    for c in candidates:
        doc = get_document(c["doc_id"])
        if doc:
            c["content"] = doc.get("content", "")
            c["sku"] = doc.get("sku", "unknown")
            c["part_numbers"] = doc.get("part_numbers", [])
            c["product_name"] = doc.get("product_name", "unknown")

    # Phase 3: Rerank
    print(f"\n[2/4] Running rerank(top_n={top_n})...")
    start = time.perf_counter()
    reranked = rerank(query, candidates, top_n=top_n)
    print(f"       Reranked in {(time.perf_counter()-start)*1000:.0f}ms")

    print(f"\nTop {top_n} docs passed to LLM:")
    for i, r in enumerate(reranked):
        preview = r["content"][:120].replace("\n", " ")
        print(f"  {i+1}. {r['doc_id']} (rerank={r['rerank_score']:.2f}) - {preview}...")

    # Phase 4: Generate answer
    print(f"\n[3/4] Generating answer via Omniroute...")
    start = time.perf_counter()
    answer = generate_answer(query, reranked)
    gen_time = time.perf_counter() - start
    print(f"       Generated in {gen_time*1000:.0f}ms")

    print(f"\n[4/4] Validating citations...")
    valid_doc_ids = [r["doc_id"] for r in reranked]
    citation_result = validate_citations(answer, valid_doc_ids)

    print(f"\n{'='*80}")
    print("FINAL ANSWER")
    print(f"{'='*80}")
    print(answer)
    print(f"{'='*80}")

    print(f"\nCITATION VALIDATION:")
    print(f"  Total citations found: {citation_result['total_citations_found']}")
    print(f"  Valid citations: {citation_result['valid_citations']}")
    print(f"  Invalid citations: {citation_result['invalid_citations']}")
    print(f"  Has valid citation: {citation_result['has_valid_citation']}")
    print(f"  All valid: {citation_result['all_valid']}")

    if not citation_result["has_valid_citation"]:
        print(f"\n[WARNING] No valid citations found! The answer may not be grounded.")
        print(f"Valid doc_ids were: {valid_doc_ids}")
    elif citation_result["invalid_citations"]:
        print(f"\n[WARNING] Some citations reference non-existent docs: {citation_result['invalid_citations']}")


if __name__ == "__main__":
    test_full_generation()