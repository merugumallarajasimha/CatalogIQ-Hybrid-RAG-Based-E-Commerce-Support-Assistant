import os
import time
import sys
import json
from typing import List, Tuple, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient

# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from exact_search import has_identifier, exact_qdrant_lookup


COLLECTION_NAME = "support_docs"

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

DENSE_TOP_K = 20
BM25_TOP_K = 20
RRF_TOP_K = 15
FINAL_TOP_K = 5
RRF_K = 60


_client: Optional[QdrantClient] = None
_embedder: Optional[SentenceTransformer] = None
_bm25_search_instance = None


def get_client() -> QdrantClient:

    global _client

    if _client is None:
        _client = QdrantClient(
            url=QDRANT_URL,
            timeout=30
        )

    return _client


def get_embedder() -> SentenceTransformer:

    global _embedder

    if _embedder is None:
        _embedder = SentenceTransformer(
            EMBEDDING_MODEL
        )

    return _embedder


def get_bm25_search():
    """Get or create BM25Search instance (cached)."""
    global _bm25_search_instance
    if _bm25_search_instance is None:
        from bm25_search import BM25Search
        _bm25_search_instance = BM25Search()
        records_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "normalized_records.json",
        )
        if os.path.exists(records_path):
            with open(records_path, encoding="utf-8", errors="replace") as f:
                normalized_records = json.load(f)
            _bm25_search_instance.build_from_records(normalized_records)
    return _bm25_search_instance


def dense_search(
    query: str,
    top_k: int = DENSE_TOP_K
) -> List[Tuple[str, int]]:

    client = get_client()

    embedder = get_embedder()

    query_vec = embedder.encode(
        query
    ).tolist()

    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vec,

        using="dense_vector",

        limit=top_k,

        with_payload=["doc_id"]
    )

    return [
        (
            hit.payload["doc_id"],
            rank + 1
        )

        for rank, hit
        in enumerate(results.points)

        if hit.payload
        and "doc_id" in hit.payload
    ]


def bm25_search(
    query: str,
    top_k: int = BM25_TOP_K
) -> List[Tuple[str, float]]:

    bm25 = get_bm25_search()
    return bm25.search(query, top_k=top_k)


def hybrid_search(
    query: str,
    top_k: int = RRF_TOP_K,
    k: int = RRF_K,
) -> List[Dict[str, Any]]:

    print("\n" + "=" * 60)
    print("HYBRID SEARCH DEBUG (Dense + BM25 + Exact)")
    print("=" * 60)

    # -----------------------------
    # Step 1: Check for exact identifier match
    # -----------------------------
    if has_identifier(query):
        print(f"Exact identifier detected in query: {query}")
        exact_results = exact_qdrant_lookup(query, top_k=RRF_TOP_K)
        if exact_results:
            print(f"Exact lookup found {len(exact_results)} results, skipping semantic search")
            # Format exact results to match hybrid_search output format
            formatted = []
            for r in exact_results:
                formatted.append({
                    "doc_id": r["doc_id"],
                    "rrf_score": r["score"],
                    "dense_rank": None,
                    "bm25_rank": None,
                    "retrieval_sources": ["exact"],
                    "exact_match": True,
                    "matched_identifier": r.get("matched_identifier"),
                    "match_type": r.get("match_type"),
                })
            print("=" * 60)
            return formatted[:RRF_TOP_K]

    # -----------------------------
    # Dense search
    # -----------------------------
    start = time.perf_counter()

    dense_results = dense_search(query, DENSE_TOP_K)

    dense_time = (time.perf_counter() - start) * 1000
    print(f"Dense search: {dense_time:.2f} ms")

    # -----------------------------
    # BM25 search
    # -----------------------------
    start = time.perf_counter()

    bm25_results_raw = bm25_search(query, BM25_TOP_K)
    bm25_results = [doc_id for doc_id, _ in bm25_results_raw]
    bm25_time = (time.perf_counter() - start) * 1000
    print(f"BM25 search: {bm25_time:.2f} ms")

    # -----------------------------
    # RRF Fusion
    # -----------------------------
    start = time.perf_counter()

    dense_ranks = {doc_id: rank for doc_id, rank in dense_results}
    bm25_ranks = {doc_id: i + 1 for i, doc_id in enumerate(bm25_results)}

    all_doc_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())

    fused = []
    for doc_id in all_doc_ids:
        score = 0.0
        retrieval_sources = []

        if doc_id in dense_ranks:
            score += 1.0 / (k + dense_ranks[doc_id])
            retrieval_sources.append("dense")

        if doc_id in bm25_ranks:
            score += 1.0 / (k + bm25_ranks[doc_id])
            retrieval_sources.append("bm25")

        fused.append({
            "doc_id": doc_id,
            "rrf_score": score,
            "dense_rank": dense_ranks.get(doc_id),
            "bm25_rank": bm25_ranks.get(doc_id),
            "retrieval_sources": retrieval_sources,
            "exact_match": False,
        })

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    rrf_time = (time.perf_counter() - start) * 1000
    print(f"RRF fusion: {rrf_time:.2f} ms")
    print(f"Total hybrid search: {dense_time + bm25_time + rrf_time:.2f} ms")
    print("=" * 60)

    return fused[:RRF_TOP_K]


def get_document(
    doc_id: str
) -> Optional[Dict[str, Any]]:

    client = get_client()

    results = client.scroll(

        collection_name=COLLECTION_NAME,

        scroll_filter={
            "must": [
                {
                    "key": "doc_id",
                    "match": {
                        "value": doc_id
                    }
                }
            ]
        },

        limit=1,

        with_payload=True
    )

    points, next_page = results

    if points:
        return points[0].payload

    return None


if __name__ == "__main__":

    print("Testing search module...")

    print("Run test_individual_search.py after Qdrant is running and data is ingested.")