import re
import sys
import os
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from extract_ids import extract_ids


CATEGORY_EXACT_LOOKUP = "exact_lookup"
CATEGORY_PRODUCT_SEARCH = "product_search"
CATEGORY_PRODUCT_COMPARISON = "product_comparison"
CATEGORY_TROUBLESHOOTING = "troubleshooting"
CATEGORY_GENERAL_QUESTION = "general_question"
CATEGORY_UNSUPPORTED = "unsupported"

ALL_CATEGORIES = [
    CATEGORY_EXACT_LOOKUP,
    CATEGORY_PRODUCT_SEARCH,
    CATEGORY_PRODUCT_COMPARISON,
    CATEGORY_TROUBLESHOOTING,
    CATEGORY_GENERAL_QUESTION,
    CATEGORY_UNSUPPORTED,
]

COMPARISON_PATTERNS = [
    re.compile(r'\bvs\.?\b', re.IGNORECASE),
    re.compile(r'\bversus\b', re.IGNORECASE),
    re.compile(r'\bcompare\b', re.IGNORECASE),
    re.compile(r'\bdifference\b', re.IGNORECASE),
    re.compile(r'\bbetter\b', re.IGNORECASE),
    re.compile(r'\bworse\b', re.IGNORECASE),
    re.compile(r'\bwhich.*(more|better)\b', re.IGNORECASE),
]

TROUBLESHOOTING_KEYWORDS = [
    'not working', 'broken', 'stuck', 'jammed', 'failing',
    'error e', 'problem', 'won', "won't", 'does not work',
    'squeak', 'loud', 'smoke', 'overheat', 'disconnect',
    'jitter', 'double press', 'lag', 'slow', 'stuck',
]

TROUBLESHOOTING_ACTIONS = [
    'how do i fix', 'how to fix', 'how to repair',
    'how to replace', 'how to reset', 'how to calibrate',
    'how to adjust', 'fix', 'repair', 'replace',
]

PRODUCT_SEARCH_SIGNALS = [
    "product", "model", "specification", "specs", "feature", "dimension",
    "weight", "size", "color", "material", "brand", "price", "warranty",
    "compatible", "parts", "accessories", "battery", "motor", "blade",
    "wheel", "mount", "base", "arm", "stand", "desk", "chair",
]


def classify_query(query: str) -> Tuple[str, Dict[str, Any]]:
    evidence = {
        "query": query,
        "has_identifier": False,
        "identifiers": {},
        "comparison_signals": [],
        "troubleshooting_signals": [],
        "product_signals": [],
    }

    identifiers = extract_ids(query)
    evidence["identifiers"] = identifiers

    all_identifiers = []
    for id_type in ("skus", "asins", "part_numbers"):
        all_identifiers.extend(identifiers.get(id_type, []))
    evidence["has_identifier"] = len(all_identifiers) > 0

    query_lower = query.lower()

    for pattern in COMPARISON_PATTERNS:
        if pattern.search(query):
            evidence["comparison_signals"].append(pattern.pattern)

    for kw in TROUBLESHOOTING_KEYWORDS:
        if kw in query_lower:
            evidence["troubleshooting_signals"].append(kw)

    for action in TROUBLESHOOTING_ACTIONS:
        if action in query_lower:
            evidence["troubleshooting_signals"].append(action)

    for signal in PRODUCT_SEARCH_SIGNALS:
        if signal in query_lower:
            evidence["product_signals"].append(signal)

    if evidence["comparison_signals"]:
        return CATEGORY_PRODUCT_COMPARISON, evidence

    if evidence["has_identifier"]:
        return CATEGORY_EXACT_LOOKUP, evidence

    if evidence["troubleshooting_signals"]:
        return CATEGORY_TROUBLESHOOTING, evidence

    if evidence["product_signals"] or len(query.split()) >= 4:
        return CATEGORY_PRODUCT_SEARCH, evidence

    return CATEGORY_GENERAL_QUESTION, evidence


def classify_queries(queries: List[str]) -> List[Dict[str, Any]]:
    results = []
    for query in queries:
        category, evidence = classify_query(query)
        results.append({
            "query": query,
            "category": category,
            "evidence": evidence,
        })
    return results


if __name__ == "__main__":
    test_queries = [
        "What is the torque spec for screw S-4012-SCRW?",
        "Which part number is the gas lift cylinder for CHAIR-ERG-X99?",
        "What does error E02 mean on DESK-STD-MOTO?",
        "Tell me about part S-4012-SCRW",
        "My chair squeaks when I lean back, how do I fix it?",
        "Standing desk won't move up or down, shows error E01",
        "What are the specifications of B07DP4LM9H?",
        "Compare CHAIR-ERG-X99 vs DESK-STD-MOTO",
        "What is the warranty on CHAIR-ERG-X99?",
        "How do I cook a turkey?",
        "Left side of desk lags behind right when raising",
        "Keyboard switch SW-2201-SWT actuation force?",
        "Which is better: mouse A or mouse B?",
        "Mouse cursor jumps around on glass desk",
        "What color is the monitor arm?",
    ]

    print("=" * 80)
    print("QUERY ROUTER CLASSIFICATION")
    print("=" * 80)

    results = classify_queries(test_queries)
    for r in results:
        print(f"\nQuery: {r['query']}")
        print(f"  Category: {r['category']}")
        ev = r['evidence']
        if ev['has_identifier']:
            print(f"  Identifiers: {ev['identifiers']}")
        if ev['comparison_signals']:
            print(f"  Comparison signals: {ev['comparison_signals']}")
        if ev['troubleshooting_signals']:
            print(f"  Troubleshooting signals: {ev['troubleshooting_signals']}")
        if ev['product_signals']:
            print(f"  Product signals: {ev['product_signals']}")
