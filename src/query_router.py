import re
import sys
import os
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
try:
    from extract_ids import extract_ids
except ImportError:
    from .extract_ids import extract_ids


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

APPAREL_AND_CATALOG_SIGNALS = [
    "hoodie", "pullover", "sweater", "sweatshirt", "shirt", "t-shirt", 
    "jacket", "coat", "pants", "shorts", "jeans", "apparel", "clothing",
    "wear", "outfit", "size", "fit", "sleeve", "zipper", "fleece", "fabric",
    "tell me about", "details", "overview", "info", "information"
]

PRODUCT_SEARCH_SIGNALS = [
    "product", "model", "specification", "specs", "feature", "dimension",
    "weight", "size", "color", "material", "brand", "price", "warranty",
    "compatible", "parts", "accessories", "battery", "motor", "blade",
    "wheel", "mount", "base", "arm", "stand", "desk", "chair", "monitor",
    "keyboard", "mouse", "headset", "armrest", "ergonomic", "ergoflex",
    "part number", "sku", "torque", "assembly", "installation", "manual",
    "troubleshooting", "repair", "replace", "reset", "calibrate", "adjust",
    "error code", "error e", "cylinder", "caster", "switch", "sensor",
    "driver", "bolt", "screw",
] + APPAREL_AND_CATALOG_SIGNALS

CATALOG_SIGNALS = [
    "specification", "specs", "feature", "dimension", "weight", "size",
    "color", "material", "warranty", "compatible", "battery", "motor",
    "blade", "wheel", "mount", "stand", "desk", "chair", "monitor",
    "keyboard", "mouse", "headset", "armrest", "ergonomic", "ergoflex",
    "part number", "sku", "torque", "assembly", "installation", "manual",
    "troubleshooting", "repair", "replace", "reset", "calibrate", "adjust",
    "error code", "error e", "cylinder", "caster", "switch", "sensor",
    "driver", "bolt", "screw",
] + APPAREL_AND_CATALOG_SIGNALS

IRRELEVANT_PATTERNS = [
    re.compile(r'\b(cook|recipe|turkey|bake|dinner|lunch|breakfast)\b', re.IGNORECASE),
    re.compile(r'\b(weather|forecast|rain|temperature)\b', re.IGNORECASE),
    re.compile(r'\b(movie|film|actor|song|music|sports|game score)\b', re.IGNORECASE),
    re.compile(r'\b(hi|hello|hey|how are you|what\'s up|whats up|greetings|good morning|good afternoon|good evening)\b', re.IGNORECASE),
    re.compile(r'\b(thank you|thanks|bye|goodbye|see you)\b', re.IGNORECASE),
]

# Known valid SKUs from catalog
VALID_SKUS = {
    "B-3301-BLT", "C-5500-CAST", "C-7721-GL", "CB-2201-CTL", "CHAIR-ERG-X99",
    "DESK-STD-MOTO", "DR-2201-DRV", "GR-9901-GRS", "GS-9901-GSP", "HEADSET-WL-PRO",
    "KB-ERGO-SPLIT", "M-8841-MOT", "MON-ARM-DUAL", "MOUSE-VERT-PRO", "P-9912-ARM",
    "P-9913-BASE", "PT-1101-PVT", "S-4012-SCRW", "SN-1101-HGT", "SN-8801-SNS",
    "SP-4402-SPND", "SW-2201-SWT", "WM-4401-WLS",
}


def _find_signals(query_lower: str, signals: List[str]) -> List[str]:
    return [
        signal
        for signal in signals
        if signal in query_lower
    ]


def _has_catalog_context(evidence: Dict[str, Any]) -> bool:
    if evidence.get("valid_identifiers", False) or evidence["catalog_signals"]:
        return True
    
    query_words = evidence["query"].strip().split()
    if len(query_words) >= 2:
        for pattern in IRRELEVANT_PATTERNS:
            if pattern.search(evidence["query"]):
                return False
        return True

    return False


def classify_query(query: str) -> Tuple[str, Dict[str, Any]]:
    evidence = {
        "query": query,
        "has_identifier": False,
        "identifiers": {},
        "comparison_signals": [],
        "troubleshooting_signals": [],
        "product_signals": [],
        "catalog_signals": [],
        "valid_identifiers": False,
    }

    identifiers = extract_ids(query)
    evidence["identifiers"] = identifiers

    all_identifiers = []
    for id_type in ("skus", "asins", "part_numbers"):
        all_identifiers.extend(identifiers.get(id_type, []))
    evidence["has_identifier"] = len(all_identifiers) > 0

    # Validate identifiers against known catalog
    valid_ids = []
    for sku in identifiers.get("skus", []):
        if sku in VALID_SKUS:
            valid_ids.append(sku)
    for pn in identifiers.get("part_numbers", []):
        if pn in VALID_SKUS:
            valid_ids.append(pn)
    evidence["valid_identifiers"] = len(valid_ids) > 0

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

    product_signals = _find_signals(query_lower, PRODUCT_SEARCH_SIGNALS)
    catalog_signals = _find_signals(query_lower, CATALOG_SIGNALS)
    
    evidence["product_signals"] = product_signals
    evidence["catalog_signals"] = catalog_signals

    has_catalog_context = _has_catalog_context(evidence)

    if not has_catalog_context:
        return CATEGORY_UNSUPPORTED, evidence

    # Invalid SKU mentioned - treat as unsupported
    if evidence["has_identifier"] and not evidence["valid_identifiers"]:
        return CATEGORY_UNSUPPORTED, evidence

    if evidence["comparison_signals"]:
        return CATEGORY_PRODUCT_COMPARISON, evidence

    if evidence["valid_identifiers"]:
        return CATEGORY_EXACT_LOOKUP, evidence

    if evidence["troubleshooting_signals"]:
        return CATEGORY_TROUBLESHOOTING, evidence

    return CATEGORY_PRODUCT_SEARCH, evidence


def is_out_of_scope(query: str) -> Tuple[bool, str]:
    query = query.strip()
    category, evidence = classify_query(query)

    if category == CATEGORY_UNSUPPORTED:
        return True, "Query has no catalog, product, part, or support signal."

    if not _has_catalog_context(evidence):
        return True, "Query is not grounded in the supported catalog domain."

    return False, f"Query classified as {category}."


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
        "Tell me about the 11 Degrees Core Pull Over Hoodie",
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