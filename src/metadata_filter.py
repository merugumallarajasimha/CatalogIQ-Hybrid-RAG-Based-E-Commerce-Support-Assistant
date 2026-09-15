import re
import os
import sys
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from extract_ids import extract_ids

METADATA_FIELDS = [
    "sku", "brand", "category", "product_type",
    "color", "size", "availability", "doc_type",
]

COLOR_KEYWORDS = [
    "black", "white", "red", "blue", "green", "yellow",
    "silver", "gray", "grey", "brown", "tan", "beige",
    "navy", "dark", "light",
]

SIZE_KEYWORDS = [
    "small", "medium", "large", "xl", "l size", "m size",
    "s size", "xlarge", "tiny", "huge", "compact",
]


def extract_metadata_filter(query: str) -> Dict[str, str]:
    filters = {}
    query_lower = query.lower()

    for field in METADATA_FIELDS:
        if field in query_lower:
            pattern = re.escape(field) + r'\s*[:=]\s*(\w+)'
            match = re.search(pattern, query_lower)
            if match:
                filters[field] = match.group(1)

    color_match = re.search(
        r'\b(black|white|red|blue|green|yellow|silver|gray|grey|brown|tan|beige|navy|dark|light)\b',
        query_lower
    )
    if color_match:
        filters["color"] = color_match.group(1)

    size_match = re.search(
        r'\b(small|medium|large|xl|s size|m size|l size|xlarge|tiny|huge|compact)\b',
        query_lower
    )
    if size_match:
        filters["size"] = size_match.group(1)

    return filters


def has_enough_signal(query: str, filters: Dict[str, str]) -> Tuple[bool, str]:
    if not filters:
        return False, "no_filters"

    if len(filters) >= 2:
        return True, "multiple_filters"

    field = list(filters.keys())[0]
    value = filters[field]

    if field == "color" and len(value) >= 3:
        return True, "clear_color"
    if field == "size" and len(value) >= 3:
        return True, "clear_size"
    if field == "category" and len(value) >= 3:
        return True, "clear_category"
    if field == "product_type" and len(value) >= 3:
        return True, "clear_product_type"
    if field == "sku" or field == "brand":
        return True, f"clear_{field}"

    return False, "ambiguous_value"


def apply_metadata_filter(
    candidates: List[Dict[str, Any]],
    filters: Dict[str, str]
) -> List[Dict[str, Any]]:
    if not filters:
        return candidates

    filtered = []
    for candidate in candidates:
        match = True
        for field, value in filters.items():
            candidate_value = str(
                candidate.get(field, "")
            ).lower()
            if value not in candidate_value:
                match = False
                break
        if match:
            filtered.append(candidate)

    return filtered


def metadata_aware_filter(
    query: str,
    candidates: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    filters = extract_metadata_filter(query)
    has_signal, reason = has_enough_signal(query, filters)

    evidence = {
        "query": query,
        "filters_extracted": filters,
        "has_enough_signal": has_signal,
        "reason": reason,
        "candidates_before": len(candidates),
    }

    if has_signal:
        filtered = apply_metadata_filter(candidates, filters)
        evidence["candidates_after"] = len(filtered)
        evidence["filtered"] = True
        return filtered, evidence
    else:
        evidence["candidates_after"] = len(candidates)
        evidence["filtered"] = False
        return candidates, evidence


if __name__ == "__main__":
    test_queries = [
        "Show me black hoodies",
        "Show me black products",
        "What is the warranty",
        "Find small chairs",
        "Red chair under $50",
        "Blue large desk",
        "How do I fix this",
    ]

    sample_candidates = [
        {"doc_id": "1", "sku": "CHAIR-ERG-X99", "color": "black", "size": "large", "category": "Furniture", "product_type": "chair"},
        {"doc_id": "2", "sku": "DESK-STD-MOTO", "color": "white", "size": "large", "category": "Furniture", "product_type": "desk"},
        {"doc_id": "3", "sku": "KB-ERGO-SPLIT", "color": "black", "size": "medium", "category": "Electronics", "product_type": "keyboard"},
        {"doc_id": "4", "sku": "MOUSE-VERT-PRO", "color": "gray", "size": "small", "category": "Electronics", "product_type": "mouse"},
    ]

    print("=" * 80)
    print("METADATA-AWARE FILTERING")
    print("=" * 80)

    for query in test_queries:
        filtered, evidence = metadata_aware_filter(query, sample_candidates)
        print(f"\nQuery: {query}")
        print(f"  Filters: {evidence['filters_extracted']}")
        print(f"  Has signal: {evidence['has_enough_signal']} ({evidence['reason']})")
        print(f"  Candidates: {evidence['candidates_before']} -> {evidence['candidates_after']}")
        print(f"  Filtered: {evidence['filtered']}")
        for c in filtered:
            print(f"    {c['doc_id']}: {c.get('sku')}")
