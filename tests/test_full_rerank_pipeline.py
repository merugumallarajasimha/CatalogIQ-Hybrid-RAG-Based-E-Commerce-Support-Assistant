import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import time
from search import hybrid_search, get_document
from reranker import rerank


def test_full_rerank_pipeline():
    query = "chair makes noise when leaning back"
    top_k = 20
    top_n = 5

    print("=" * 80)
    print("FULL PIPELINE TEST: hybrid_search -> rerank")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Fetching {top_k} candidates from hybrid_search...")

    start = time.perf_counter()
    candidates = hybrid_search(query, top_k=top_k)
    hybrid_time = time.perf_counter() - start
    print(f"hybrid_search took {hybrid_time*1000:.1f}ms, returned {len(candidates)} candidates")

    print("\n--- Phase 2 (Hybrid RRF) Top 10 ---")
    print(f"{'Rank':<4} {'Score':<10} {'Doc ID':<20} {'Sparse':<8} {'Dense':<8} {'SKU'}")
    print("-" * 80)
    for i, c in enumerate(candidates[:10]):
        doc = get_document(c["doc_id"])
        sku = doc.get("sku", "unknown") if doc else "?"
        sparse_r = c["rank_in_sparse"] if c["rank_in_sparse"] else "—"
        dense_r = c["rank_in_dense"] if c["rank_in_dense"] else "—"
        print(f"{i+1:<4} {c['combined_score']:<10.4f} {c['doc_id']:<20} {str(sparse_r):<8} {str(dense_r):<8} {sku}")

    for c in candidates:
        doc = get_document(c["doc_id"])
        if doc:
            c["content"] = doc.get("content", "")
            c["sku"] = doc.get("sku", "unknown")
            c["part_numbers"] = doc.get("part_numbers", [])
            c["product_name"] = doc.get("product_name", "unknown")
            c["category"] = doc.get("category", "unknown")

    print(f"\n--- Running rerank() on {len(candidates)} candidates ---")
    start = time.perf_counter()
    reranked = rerank(query, candidates, top_n=top_n)
    rerank_time = time.perf_counter() - start

    print(f"\n{'='*80}")
    print(f"FINAL TOP {top_n} AFTER RERANKING (rerank took {rerank_time*1000:.1f}ms)")
    print(f"{'='*80}")
    print(f"{'Rank':<4} {'Rerank':<10} {'RRF':<10} {'Doc ID':<20} {'SKU':<20} Content Preview")
    print("-" * 80)
    for i, r in enumerate(reranked):
        preview = r["content"][:100].replace("\n", " ")
        print(f"{i+1:<4} {r['rerank_score']:<10.4f} {r['combined_score']:<10.4f} {r['doc_id']:<20} {r['sku']:<20} {preview}...")

    print("\n" + "=" * 80)
    print("COMPARISON: Did reranking improve the order?")
    print("-" * 80)
    print("Check if the chair squeaking doc (should contain 'SQUEAKING' or 'S-4012-SCRW')")
    print("moved to rank 1 in reranked results vs its position in hybrid results.")
    print("=" * 80)


if __name__ == "__main__":
    test_full_rerank_pipeline()