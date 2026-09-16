import os
import time
import sys
import json
import re
from typing import List, Tuple, Dict, Any, Optional, Set

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
    "sentence-transformers/all-MiniLM-L6-v2"
)

DENSE_TOP_K = 20
BM25_TOP_K = 20
RRF_TOP_K = 15
FINAL_TOP_K = 5
RRF_K = 60


_client: Optional[QdrantClient] = None
_embedder: Optional[SentenceTransformer] = None
_bm25_search_instance = None

# Known SKUs and part numbers for identifier-aware routing
ALL_VALID_SKUS = {
    "CHAIR-ERG-X99", "DESK-STD-MOTO", "MON-ARM-DUAL",
    "KB-ERGO-SPLIT", "MOUSE-VERT-PRO", "HEADSET-WL-PRO",
}

# Part number pattern: X-XXXX-XXX or similar
PART_NUMBER_PATTERN = re.compile(r'\b[A-Z]{1,3}-\d{4}-[A-Z]{2,4}\b|\b[A-Z]{1,3}-\d{4}\b|\b[A-Z]{1,3}-\d{4}-[A-Z]{3}\b')

# Load part numbers from parts catalog
_PART_NUMBERS: Optional[Set[str]] = None


def load_part_numbers() -> Set[str]:
    """Load all known part numbers from normalized records."""
    global _PART_NUMBERS
    if _PART_NUMBERS is not None:
        return _PART_NUMBERS
    
    _PART_NUMBERS = set()
    records_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "normalized_records.json"
    )
    if os.path.exists(records_path):
        with open(records_path, encoding="utf-8", errors="replace") as f:
            records = json.load(f)
        for r in records:
            # SKU field
            sku = r.get("sku")
            if sku and sku in ALL_VALID_SKUS:
                _PART_NUMBERS.add(sku)
            # Part numbers in text
            text = r.get("text", "")
            for match in PART_NUMBER_PATTERN.findall(text):
                _PART_NUMBERS.add(match)
            # Explicit part_number field
            pn = r.get("part_number")
            if pn:
                _PART_NUMBERS.add(pn)
    return _PART_NUMBERS


def detect_identifiers(query: str) -> Dict[str, List[str]]:
    """Detect SKUs and part numbers in query for routing."""
    part_numbers = load_part_numbers()
    found_skus = []
    found_parts = []
    
    query_upper = query.upper()
    
    # Check for exact SKU mentions
    for sku in ALL_VALID_SKUS:
        if sku in query_upper:
            found_skus.append(sku)
    
    # Check for part number mentions
    for pn in part_numbers:
        if pn and pn.upper() in query_upper:
            found_parts.append(pn)
    
    # Also check regex pattern for any missed
    for match in PART_NUMBER_PATTERN.findall(query):
        if match.upper() not in [p.upper() for p in found_parts]:
            found_parts.append(match)
    
    return {"skus": found_skus, "parts": found_parts}


def route_query(query: str) -> str:
    """
    Determine retrieval strategy based on query content.
    Returns: 'exact', 'bm25_only', 'hybrid', 'dense_only'
    """
    identifiers = detect_identifiers(query)
    
    # Multiple SKUs mentioned -> comparison query, use hybrid to get both
    if len(identifiers["skus"]) > 1:
        return "hybrid"
    
    # Single SKU + part numbers -> exact lookup for that product
    if identifiers["skus"] and len(identifiers["skus"]) == 1:
        return "exact"
    
    # Part numbers only (no SKU) -> BM25 only (fast, enriched chunks map to SKU)
    if identifiers["parts"] and not identifiers["skus"]:
        return "bm25_only"
    
    # Check for comparison queries (without explicit SKUs)
    comparison_keywords = ["compare", "vs", "versus", "difference", "better", "which is"]
    if any(kw in query.lower() for kw in comparison_keywords):
        return "hybrid"
    
    # Check for troubleshooting/symptom queries
    trouble_keywords = ["error", "fix", "repair", "broken", "not working", "issue", "problem", 
                        "squeak", "sag", "tilt", "disconnect", "jitter", "double",
                        "won't", "wont", "doesn't", "doesnt", "not moving", "slow"]
    if any(kw in query.lower() for kw in trouble_keywords):
        return "hybrid"
    
    # Default: full hybrid for semantic queries
    return "hybrid"


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
    print("HYBRID SEARCH (Simplified + Identifier-Aware)")
    print("=" * 60)

    # -----------------------------
    # Step 1: Route query to optimal strategy
    # -----------------------------
    route = route_query(query)
    identifiers = detect_identifiers(query)
    print(f"Route: {route} | Identifiers: SKUs={identifiers['skus']}, Parts={identifiers['parts']}")

    # -----------------------------
    # Strategy: Exact identifier -> BM25 + exact lookup
    # -----------------------------
    if route == "exact":
        print("Using BM25 + Exact lookup for single-SKU identifier query")
        exact_results = exact_qdrant_lookup(query, top_k=RRF_TOP_K)
        if exact_results:
            print(f"Exact lookup found {len(exact_results)} results")
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
            return formatted[:top_k]

    # -----------------------------
    # Strategy: Part number only -> BM25 only (fast, no dense)
    # -----------------------------
    if route == "bm25_only":
        print("Using BM25 only for part number query")
        start = time.perf_counter()
        bm25_raw = bm25_search(query, BM25_TOP_K)
        bm25_time = (time.perf_counter() - start) * 1000
        
        results = []
        for doc_id, score in bm25_raw[:top_k]:
            results.append({
                "doc_id": doc_id,
                "rrf_score": score,
                "dense_rank": None,
                "bm25_rank": len(results) + 1,
                "retrieval_sources": ["bm25"],
                "exact_match": False,
            })
        print(f"BM25 only: {bm25_time:.2f} ms")
        print("=" * 60)
        return results

    # -----------------------------
    # Strategy: Hybrid (dense + BM25 + RRF) for semantic/troubleshooting/comparison
    # -----------------------------
    print("Using Dense + BM25 + RRF hybrid")
    
    # Dense search
    start = time.perf_counter()
    dense_results = dense_search(query, DENSE_TOP_K)
    dense_time = (time.perf_counter() - start) * 1000
    print(f"Dense search: {dense_time:.2f} ms")

    # BM25 search
    start = time.perf_counter()
    bm25_results_raw = bm25_search(query, BM25_TOP_K)
    bm25_results = [doc_id for doc_id, _ in bm25_results_raw]
    bm25_time = (time.perf_counter() - start) * 1000
    print(f"BM25 search: {bm25_time:.2f} ms")

    # RRF Fusion
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

    return fused[:top_k]


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