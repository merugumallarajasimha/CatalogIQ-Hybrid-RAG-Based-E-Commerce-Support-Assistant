import os
import time
from typing import List, Dict, Any, Optional

# pyrefly: ignore [missing-import]
from sentence_transformers import CrossEncoder


RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"Loading reranker model: {RERANKER_MODEL}...")
        start = time.perf_counter()
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        elapsed = time.perf_counter() - start
        print(f"Reranker loaded in {elapsed:.3f}s")
    return _reranker


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = 5
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    reranker = get_reranker()

    pairs = [(query, c.get("content", "")) for c in candidates]

    start = time.perf_counter()
    scores = reranker.predict(pairs, batch_size=len(pairs))
    elapsed = time.perf_counter() - start
    print(f"Reranker scored {len(pairs)} pairs in {elapsed*1000:.1f}ms")

    reranked = []
    for i, candidate in enumerate(candidates):
        doc = {
            "doc_id": candidate["doc_id"],
            "rerank_score": float(scores[i]),
            "combined_score": candidate.get("combined_score", 0.0),
            "content": candidate.get("content", ""),
            "sku": candidate.get("sku", "unknown"),
            "part_numbers": candidate.get("part_numbers", []),
            "product_name": candidate.get("product_name", "unknown"),
            "category": candidate.get("category", "unknown"),
        }
        reranked.append(doc)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]


if __name__ == "__main__":
    print("Reranker module ready. Run tests/test_reranker_alone.py")