import re
from typing import List, Dict, Set


SKU_PATTERN = re.compile(
    r'\b(?:SKU|sku|Sku)\s*[:#]?\s*([A-Z0-9][A-Z0-9\-_]{2,})\b',
    re.IGNORECASE
)

PART_NUMBER_PATTERN = re.compile(
    r'\b(?:part|Part|PART)\s*(?:number|#|no\.?)\s*[:#]?\s*([A-Z0-9][A-Z0-9\-_\.]{2,})\b',
    re.IGNORECASE
)

MODEL_PATTERN = re.compile(
    r'\b(?:model|Model|MODEL)\s*(?:number|#|no\.?|num\.?)?\s*[:#]?\s*([A-Z0-9][A-Z0-9\-_\.]{2,})\b',
    re.IGNORECASE
)

ALPHANUMERIC_CODE_PATTERN = re.compile(
    r'\b([A-Z]{2,4}[-_][A-Z0-9]{2,}[-_][A-Z0-9]{2,})\b'
)

HYPHENATED_CODE_PATTERN = re.compile(
    r'\b([A-Z0-9]+(?:[-_][A-Z0-9]+){2,})\b'
)

PN_PREFIX_PATTERN = re.compile(
    r'\bPN\s*[:#]\s*([A-Z0-9][A-Z0-9\-_\.]{2,})\b',
    re.IGNORECASE
)


def extract_skus(text: str) -> List[str]:
    skus: Set[str] = set()
    for match in SKU_PATTERN.finditer(text):
        skus.add(match.group(1).upper())
    return sorted(skus)


def extract_part_numbers(text: str) -> List[str]:
    parts: Set[str] = set()
    
    for match in PART_NUMBER_PATTERN.finditer(text):
        parts.add(match.group(1).upper())
    
    for match in MODEL_PATTERN.finditer(text):
        parts.add(match.group(1).upper())
    
    for match in PN_PREFIX_PATTERN.finditer(text):
        parts.add(match.group(1).upper())
    
    for match in ALPHANUMERIC_CODE_PATTERN.finditer(text):
        parts.add(match.group(1).upper())
    
    for match in HYPHENATED_CODE_PATTERN.finditer(text):
        candidate = match.group(1).upper()
        if any(c.isdigit() for c in candidate) and any(c.isalpha() for c in candidate):
            parts.add(candidate)
    
    return sorted(parts)


def extract_ids(text: str) -> Dict[str, List[str]]:
    return {
        "skus": extract_skus(text),
        "part_numbers": extract_part_numbers(text)
    }


if __name__ == "__main__":
    test_text = (
        "Replacement filter for model X-9921-A, compatible with SKU CHAIR-ERG-X99.\n"
        "The part number is P-9913-BASE and also uses screw S-4012-SCRW.\n"
        "PN: C-7721-GL for the gas lift. Model No. 45-AB-789 is the old format."
    )
    result = extract_ids(test_text)
    print("Input:", test_text.strip())
    print("Extracted:", result)