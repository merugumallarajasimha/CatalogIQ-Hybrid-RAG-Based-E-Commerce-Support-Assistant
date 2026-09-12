import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from search import hybrid_search, get_document


def print_results(name: str, results: list, top: int = 5):
    print(f"\n{'='*70}")
    print(f"{name} - Top {top} Fused Results")
    print(f"{'='*70}")
    for i, r in enumerate(results[:top]):
        doc = get_document(r["doc_id"])
        if doc:
            sku = doc.get("sku", "unknown")
            product = doc.get("product_name", "unknown")
            content_preview = doc.get("content", "")[:150].replace("\n", " ")
            sparse_rank = r["rank_in_sparse"] if r["rank_in_sparse"] else "—"
            dense_rank = r["rank_in_dense"] if r["rank_in_dense"] else "—"
            print(f"  {i+1}. Score: {r['combined_score']:.4f} | Sparse: {sparse_rank} | Dense: {dense_rank}")
            print(f"     Doc: {r['doc_id']}")
            print(f"     SKU: {sku} | Product: {product}")
            print(f"     Content: {content_preview}...")
        else:
            print(f"  {i+1}. Score: {r['combined_score']:.4f} | Doc: {r['doc_id']} (not found)")


def main():
    print("Testing Hybrid Search (RRF Fusion)")
    print("=" * 70)
    
    test_queries = [
        "SKU CHAIR-ERG-X99",
        "chair makes noise when leaning back",
    ]
    
    for query in test_queries:
        print(f"\n>>> hybrid_search('{query}')")
        results = hybrid_search(query, top_k=20)
        print_results("Hybrid Search", results)


if __name__ == "__main__":
    main()