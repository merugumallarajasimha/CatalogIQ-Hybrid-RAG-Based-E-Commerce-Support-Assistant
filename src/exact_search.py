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

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "normalized_records.json",
)


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
