import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
from search import hybrid_search, get_document
from reranker import rerank


def test_sku_query():
    query = "SKU CHAIR-ERG-X99"
    top_k = 20
    top_n = 5

    print("=" * 80)
    print(f"TEST: Exact SKU query -> '{query}'")
    print("=" * 80)

    start = time.perf_counter()
    candidates = hybrid_search(query, top_k=top_k)
    hybrid_time = time.perf_counter() - start
    print(f"hybrid_search: {hybrid_time*1000:.1f}ms, {len(candidates)} candidates")

    for c in candidates:
        doc = get_document(c["doc_id"])
        if doc:
            c["content"] = doc.get("content", "")
            c["sku"] = doc.get("sku", "unknown")
            c["part_numbers"] = doc.get("part_numbers", [])
            c["product_name"] = doc.get("product_name", "unknown")

    start = time.perf_counter()
    reranked = rerank(query, candidates, top_n=top_n)
    rerank_time = time.perf_counter() - start

    print(f"\nTop {top_n} after rerank (rerank: {rerank_time*1000:.1f}ms):")
    print(f"{'Rank':<4} {'Rerank':<10} {'RRF':<10} {'Doc ID':<40} {'SKU'}")
    print("-" * 80)
    for i, r in enumerate(reranked):
        print(f"{i+1:<4} {r['rerank_score']:<10.4f} {r['combined_score']:<10.4f} {r['doc_id']:<40} {r['sku']}")


if __name__ == "__main__":
    test_sku_query()