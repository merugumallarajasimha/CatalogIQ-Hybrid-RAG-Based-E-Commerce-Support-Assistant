import os
import time
import logging
from typing import List, Dict, Any, Optional

# pyrefly: ignore [missing-import]
from sentence_transformers import CrossEncoder

logger = logging.getLogger("reranker")

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {RERANKER_MODEL}...")
        start = time.perf_counter()
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        elapsed = time.perf_counter() - start
        logger.info(f"Reranker loaded in {elapsed:.3f}s")
    return _reranker


def preload_reranker() -> None:
    """Explicit call to load model at server boot to prevent query latency."""
    get_reranker()


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    **kwargs: Any
) -> List[Dict[str, Any]]:
    """
    Reranks candidate documents using the CrossEncoder model.
    """
    if not candidates:
        return []

    # Handle parameter name variations (top_k vs top_n)
    if "top_n" in kwargs:
        top_k = kwargs["top_n"]

    reranker_model = get_reranker()

    # Safely retrieve content text across different document models
    pairs = [
        (query, c.get("content") or c.get("text", ""))
        for c in candidates
    ]

    start = time.perf_counter()
    scores = reranker_model.predict(pairs, batch_size=len(pairs))
    elapsed = time.perf_counter() - start
    logger.info(f"Reranker scored {len(pairs)} pairs in {elapsed*1000:.1f}ms")

    reranked = []
    for i, candidate in enumerate(candidates):
        doc = dict(candidate)
        doc["rerank_score"] = float(scores[i])
        doc["content"] = candidate.get("content") or candidate.get("text", "")
        doc["sku"] = candidate.get("sku", "unknown")
        doc["product_name"] = candidate.get("product_name", "unknown")
        doc["category"] = candidate.get("category", "unknown")
        reranked.append(doc)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# Module aliases for compatibility across services
rerank_results = rerank
rerank_candidates = rerank
rerank_documents = rerank


if __name__ == "__main__":
    print("Pre-loading reranker model for testing...")
    preload_reranker()
    print("Reranker module ready.")