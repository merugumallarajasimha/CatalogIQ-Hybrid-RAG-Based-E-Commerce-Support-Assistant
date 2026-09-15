"""
Exact Search Module (Step 3)
Retrieves documents using exact identifier matching
for SKUs, ASINs, part numbers, product names, and model codes.
"""
import json
import os
import re
import sys
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from extract_ids import extract_ids

# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "normalized_records.json",
)

COLLECTION_NAME = "support_docs"

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

_exact_client: Optional[QdrantClient] = None


def get_exact_client() -> QdrantClient:
    global _exact_client
    if _exact_client is None:
        _exact_client = QdrantClient(url=QDRANT_URL, timeout=30)
    return _exact_client


def load_records() -> List[Dict[str, Any]]:
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, encoding="utf-8", errors="replace") as f:
        return json.load(f)


def exact_match_score(
    query_ids: Dict[str, List[str]],
    record: Dict[str, Any],
) -> float:
    sku = str(record.get("sku", ""))
    text = str(record.get("text", ""))
    product_name = str(record.get("product_name", ""))

    score = 0.0

    all_ids = []
    for id_type in ("skus", "asins", "part_numbers"):
        all_ids.extend(query_ids.get(id_type, []))

    for identifier in all_ids:
        if sku.upper() == identifier.upper():
            score += 100.0
        if identifier.upper() in text.upper():
            score += 10.0
        if identifier.upper() in product_name.upper():
            score += 5.0

    return score


def exact_search(
    query: str,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    records = load_records()
    if not records:
        return []

    query_ids = extract_ids(query)

    scored = []
    for record in records:
        score = exact_match_score(query_ids, record)
        if score > 0:
            doc_id = str(record.get("sku", record.get("part_number", "")))
            scored.append((doc_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def exact_search_with_metadata(
    query: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    records = load_records()
    if not records:
        return []

    query_ids = extract_ids(query)

    scored = []
    for record in records:
        score = exact_match_score(query_ids, record)
        if score > 0:
            scored.append({
                "doc_id": str(record.get("sku", record.get("part_number", ""))),
                "score": score,
                "sku": record.get("sku", ""),
                "product_name": record.get("product_name", ""),
                "category": record.get("category", ""),
                "matched_identifiers": [
                    ident
                    for ident in (
                        query_ids.get("skus", [])
                        + query_ids.get("part_numbers", [])
                    )
                    if ident.upper() in str(record.get("sku", "")).upper()
                    or ident.upper() in str(record.get("text", "")).upper()
                    or ident.upper() in str(record.get("product_name", "")).upper()
                ],
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def exact_qdrant_lookup(
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Exact Qdrant payload lookup for identifiers.
    Skips semantic search entirely when an identifier is detected.
    Returns results in the same format as hybrid_search for seamless integration.
    """
    client = get_exact_client()
    query_ids = extract_ids(query)

    all_identifiers = []
    for id_type in ("skus", "asins", "part_numbers"):
        all_identifiers.extend(query_ids.get(id_type, []))

    if not all_identifiers:
        return []

    results = []
    seen_doc_ids = set()
    for identifier in all_identifiers:
        # Search by sku field
        sku_results = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="sku", match=MatchValue(value=identifier))
                ]
            ),
            limit=top_k,
            with_payload=True
        )
        points, _ = sku_results
        for point in points:
            payload = point.payload
            doc_id = payload.get("doc_id", identifier)
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                results.append({
                    "doc_id": doc_id,
                    "score": 100.0,  # Exact match gets highest score
                    "sku": payload.get("sku", ""),
                    "product_name": payload.get("product_name", ""),
                    "category": payload.get("category", ""),
                    "content": payload.get("content", payload.get("text", "")),
                    "matched_identifier": identifier,
                    "match_type": "sku"
                })

        # Search by part_number field (if different from sku)
        pn_results = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="part_number", match=MatchValue(value=identifier))
                ]
            ),
            limit=top_k,
            with_payload=True
        )
        points, _ = pn_results
        for point in points:
            payload = point.payload
            doc_id = payload.get("doc_id", identifier)
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                results.append({
                    "doc_id": doc_id,
                    "score": 100.0,
                    "sku": payload.get("sku", ""),
                    "product_name": payload.get("product_name", ""),
                    "category": payload.get("category", ""),
                    "content": payload.get("content", payload.get("text", "")),
                    "matched_identifier": identifier,
                    "match_type": "part_number"
                })

    return results[:top_k]


def has_identifier(query: str) -> bool:
    """Check if query contains any identifiable SKU, ASIN, or part number."""
    query_ids = extract_ids(query)
    for id_type in ("skus", "asins", "part_numbers"):
        if query_ids.get(id_type):
            return True
    return False


if __name__ == "__main__":
    test_queries = [
        "B07DP4LM9H",
        "S-4012-SCRW",
        "CHAIR-ERG-X99",
        "B-3301-BLT",
        "SN-8801-SNS",
        "C-7721-GL",
        "PN C-7721-GL",
        "Model No. 45-AB-789",
    ]

    print("=" * 80)
    print("EXACT SEARCH TEST")
    print("=" * 80)

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = exact_search(query, top_k=5)
        for doc_id, score in results[:5]:
            print(f"  {doc_id}: {score:.1f}")

        detailed = exact_search_with_metadata(query, top_k=3)
        for item in detailed[:3]:
            print(
                f"  [{item['doc_id']}] "
                f"score={item['score']:.1f} "
                f"matched={item['matched_identifiers']}"
            )
