import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from search import sparse_search, dense_search, get_document


def print_results(name: str, results: list, top: int = 5):
    print(f"\n{'='*60}")
    print(f"{name} - Top {top} Results")
    print(f"{'='*60}")
    for i, (doc_id, rank) in enumerate(results[:top]):
        doc = get_document(doc_id)
        if doc:
            sku = doc.get("sku", "unknown")
            product = doc.get("product_name", "unknown")
            content_preview = doc.get("content", "")[:150].replace("\n", " ")
            print(f"  {i+1}. Rank: {rank} | Doc: {doc_id}")
            print(f"     SKU: {sku} | Product: {product}")
            print(f"     Content: {content_preview}...")
        else:
            print(f"  {i+1}. Rank: {rank} | Doc: {doc_id} (not found)")


def main():
    print("Testing Individual Search Functions")
    print("=" * 60)
    
    test_queries = [
        ("SKU CHAIR-ERG-X99", "sparse"),
        ("chair makes noise when leaning back", "dense"),
    ]
    
    for query, search_type in test_queries:
        if search_type == "sparse":
            print(f"\n>>> sparse_search('{query}')")
            results = sparse_search(query, top_k=20)
            print_results("Sparse Search", results)
        else:
            print(f"\n>>> dense_search('{query}')")
            results = dense_search(query, top_k=20)
            print_results("Dense Search", results)


if __name__ == "__main__":
    main()