import ast
import csv
import json
import os
import re
import uuid
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd


# =========================================================
# PATH CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATASET_DIR / "normalized_records.json"


# =========================================================
# FILE EXCLUSION PATTERNS
# =========================================================

EXCLUDED_FILE_PATTERNS = [
    "shopping_queries_dataset",
]


# =========================================================
# CATEGORY TAXONOMY AND MAPPING
# =========================================================

CATEGORY_MAP = {
    "furniture/seating": "Furniture/Seating",
    "seating": "Furniture/Seating",
    "chairs": "Furniture/Seating",
    "furniture/desks": "Furniture/Desks",
    "desks": "Furniture/Desks",
    "furniture/mounts": "Furniture/Mounts",
    "mounts": "Furniture/Mounts",
    "furniture/storage": "Furniture/Storage",
    "storage": "Furniture/Storage",
    "electronics": "Electronics",
    "fasteners": "Fasteners",
    "screws": "Fasteners",
    "peripherals/audio": "Peripherals/Audio",
    "audio": "Peripherals/Audio",
    "peripherals/keyboards": "Peripherals/Keyboards",
    "keyboards": "Peripherals/Keyboards",
    "peripherals/mice": "Peripherals/Mice",
    "mice": "Peripherals/Mice",
    "peripherals/displays": "Peripherals/Displays",
    "displays": "Peripherals/Displays",
    "monitors": "Peripherals/Displays",
    "peripherals/networking": "Peripherals/Networking",
    "networking": "Peripherals/Networking",
    "maintenance": "Maintenance",
    "tools": "Tools",
    "accessories": "Accessories",
}

DEFAULT_CATEGORY = "uncategorized"


# =========================================================
# HTML TAG STRIPPING
# =========================================================

HTML_TAG_RE = re.compile(r'<[^>]+>')
HTML_ENTITY_MAP = {
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&quot;': '"',
    '&#39;': "'",
    '&nbsp;': ' ',
}


def strip_html(text: str) -> str:
    for entity, replacement in HTML_ENTITY_MAP.items():
        text = text.replace(entity, replacement)
    text = HTML_TAG_RE.sub(' ', text)
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(?<=\d)\.(?=\d)', '.', text)
    text = re.sub(r'\.\s+(?=[A-Z0-9])', '. ', text)
    return text.strip()


def clean_text_content(text: str) -> str:
    text = strip_html(text)
    return normalize_whitespace(text)


# =========================================================
# GENERAL HELPERS
# =========================================================

def clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def get_first_value(record: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for key in keys:
        value = clean_value(record.get(key))
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
        else:
            return value
    return None


def parse_list_value(value: Any) -> List[str]:
    value = clean_value(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str):
        return [str(value).strip()]
    
    value = value.strip()
    if not value:
        return []
        
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed_value = ast.literal_eval(value)
            if isinstance(parsed_value, (list, tuple)):
                return [str(item).strip() for item in parsed_value if str(item).strip()]
        except (ValueError, SyntaxError):
            value = value[1:-1]
            return [item.strip().strip("'\"") for item in value.split(",") if item.strip()]
            
    if ";" in value:
        return [item.strip() for item in value.split(";") if item.strip()]
        
    return [value]


# =========================================================
# FILE LOADERS
# =========================================================

def load_json_file(filepath: Path) -> List[Dict[str, Any]]:
    with filepath.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Expected a JSON list or dictionary, received {type(data).__name__}")


def load_csv_file(filepath: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with filepath.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            records.append(dict(row))
    return records


def load_parquet_file(filepath: Path) -> List[Dict[str, Any]]:
    dataframe = pd.read_parquet(filepath)
    dataframe = dataframe.where(pd.notnull(dataframe), None)
    records = dataframe.to_dict(orient="records")
    return [{key: clean_value(value) for key, value in record.items()} for record in records]


def find_dataset_directory() -> Path:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Data directory does not exist:\n{DATASET_DIR}")
    if not DATASET_DIR.is_dir():
        raise NotADirectoryError(f"Expected a directory but found:\n{DATASET_DIR}")
    supported_files = [
        file for file in DATASET_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in {".csv", ".json", ".parquet"}
    ]
    if not supported_files:
        raise FileNotFoundError(f"No CSV, JSON, or Parquet files found in:\n{DATASET_DIR}")
    return DATASET_DIR


def is_excluded_file(filename: str) -> bool:
    filename_lower = filename.lower()
    return any(pattern.lower() in filename_lower for pattern in EXCLUDED_FILE_PATTERNS)


# =========================================================
# TEXT FORMATTING & DEDUPLICATION
# =========================================================

def format_product_name(name: str) -> str:
    return clean_text_content(name)


def format_text_for_chunking(text: str) -> str:
    text = clean_text_content(text)
    text = re.sub(r'\s*\.\s*\.', '.', text)
    text = re.sub(r'\s*,\s*,', ',', text)
    text = re.sub(r'^\s*[.,;:]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\.\s*$', '.', text)
    return text.strip()


def compute_text_fingerprint(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower())


def detect_exact_duplicates(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    text_to_indices: Dict[str, List[int]] = {}
    for i, record in enumerate(records):
        fp = compute_text_fingerprint(record.get("text", ""))
        text_to_indices.setdefault(fp, []).append(i)

    exact_dup_indices: Set[int] = set()
    exact_dup_groups = 0
    for indices in text_to_indices.values():
        if len(indices) > 1:
            exact_dup_groups += 1
            for idx in indices[1:]:
                exact_dup_indices.add(idx)

    kept = [record for i, record in enumerate(records) if i not in exact_dup_indices]
    removed = [record for i, record in enumerate(records) if i in exact_dup_indices]
    return kept, removed, exact_dup_groups, len(exact_dup_indices)


def detect_near_duplicates(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    sku_groups: Dict[str, List[int]] = {}
    for i, record in enumerate(records):
        sku = record.get("sku", "unknown")
        if sku != "unknown":
            sku_groups.setdefault(sku, []).append(i)

    near_dup_indices: Set[int] = set()
    groups_count = 0
    for indices in sku_groups.values():
        if len(indices) <= 1:
            continue
        
        # Check text fingerprints within same SKU
        text_fps: Dict[str, List[int]] = {}
        for idx in indices:
            fp = compute_text_fingerprint(records[idx].get("text", ""))
            text_fps.setdefault(fp, []).append(idx)
            
        for fp, idx_list in text_fps.items():
            if len(idx_list) > 1:
                groups_count += 1
                for idx in idx_list[1:]:
                    near_dup_indices.add(idx)

    kept = [record for i, record in enumerate(records) if i not in near_dup_indices]
    flagged = [record for i, record in enumerate(records) if i in near_dup_indices]
    return kept, flagged, groups_count


# =========================================================
# RECORD NORMALIZATION
# =========================================================

def normalize_record(record: Dict[str, Any], source_file: str) -> Optional[Dict[str, Any]]:
    # SKU Extraction
    sku = get_first_value(record, ["sku", "SKU", "product_id", "productId", "productID", "part_number", "part_no", "partNumber", "id", "asin", "ASIN"])
    
    # Product Name Extraction
    product_name = get_first_value(record, ["product_name", "product_title", "title", "name", "productName", "item_name"])
    
    # Category Extraction
    category_raw = get_first_value(record, ["category", "product_category", "category_name", "department", "productCategory", "main_category"])

    # Fallback to compatible SKUs if SKU is missing
    if not sku:
        compatible_skus = parse_list_value(record.get("compatible_skus"))
        if compatible_skus:
            sku = compatible_skus[0]

    sku = str(sku).strip() if sku else "unknown"

    if not product_name:
        product_name = get_first_value(record, ["description", "product_description", "body", "content"]) or "unknown"

    # Standardize Category Mapping
    category = DEFAULT_CATEGORY
    if category_raw:
        cat_lower = str(category_raw).strip().lower()
        category = CATEGORY_MAP.get(cat_lower, CATEGORY_MAP.get(str(category_raw).strip(), DEFAULT_CATEGORY))

    # Searchable Content Assembly
    text_parts: List[str] = []
    main_text_keys = ["content", "text", "product_description", "description", "body", "details", "feature", "features"]
    
    main_text = ""
    for key in main_text_keys:
        value = clean_value(record.get(key))
        if value:
            main_text = str(value)
            break

    if main_text:
        text_parts.append(format_text_for_chunking(main_text))

    if product_name != "unknown":
        text_parts.append(f"Product name: {format_product_name(product_name)}")

    if category != DEFAULT_CATEGORY:
        text_parts.append(f"Category: {category}")

    # Troubleshooting & Ticket Details
    customer_desc = clean_value(record.get("customer_description"))
    if customer_desc:
        text_parts.append(f"Customer issue: {format_text_for_chunking(customer_desc)}")

    agent_res = clean_value(record.get("agent_resolution"))
    if agent_res:
        text_parts.append(f"Resolution: {format_text_for_chunking(agent_res)}")

    replacement_proc = clean_value(record.get("replacement_procedure"))
    if replacement_proc:
        text_parts.append(f"Replacement procedure: {format_text_for_chunking(replacement_proc)}")

    resolution_steps = parse_list_value(record.get("resolution_steps"))
    if resolution_steps:
        text_parts.append(f"Resolution steps: {', '.join(resolution_steps)}")

    # Part Numbers & Compatible SKUs
    part_numbers = parse_list_value(record.get("part_numbers"))
    singular_part = clean_value(record.get("part_number"))
    if singular_part:
        part_numbers.append(str(singular_part).strip())
    if part_numbers:
        text_parts.append(f"Part numbers: {', '.join(set(part_numbers))}")

    compatible_skus = parse_list_value(record.get("compatible_skus"))
    if compatible_skus:
        text_parts.append(f"Compatible SKUs: {', '.join(set(compatible_skus))}")

    text = "\n\n".join(part for part in text_parts if part.strip()).strip()
    if not text:
        return None

    # Stable, Deterministic Document ID via MD5 Hash
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{sku}-{text_hash}"))

    confidence_flags = []
    if sku == "unknown":
        confidence_flags.append("sku_defaulted")
    if category == DEFAULT_CATEGORY:
        confidence_flags.append("category_defaulted")
    if not main_text:
        confidence_flags.append("no_content")

    return {
        "doc_id": doc_id,
        "sku": sku,
        "product_name": str(product_name).strip(),
        "category": str(category).strip(),
        "text": text,
        "source_file": source_file,
        "confidence_flags": confidence_flags,
    }


# =========================================================
# DATASET LOADER
# =========================================================

def load_dataset() -> List[Dict[str, Any]]:
    dataset_dir = find_dataset_directory()
    print(f"\nUsing dataset directory: {dataset_dir}")

    all_records: List[Dict[str, Any]] = []
    stats = {
        "files_processed": 0,
        "total_records_before_dedup": 0,
        "records_skipped": 0,
        "excluded_files": [],
        "exact_duplicates_removed": 0,
        "near_duplicates_flagged": 0,
    }

    for filepath in sorted(dataset_dir.iterdir()):
        if not filepath.is_file():
            continue

        extension = filepath.suffix.lower()
        if extension not in {".csv", ".json", ".parquet"}:
            continue

        filename = filepath.name
        if is_excluded_file(filename) or "normalized" in filename.lower():
            stats["excluded_files"].append(filename)
            continue

        stats["files_processed"] += 1
        print(f"Loading {filename}...")

        try:
            if extension == ".csv":
                records = load_csv_file(filepath)
            elif extension == ".json":
                records = load_json_file(filepath)
            elif extension == ".parquet":
                records = load_parquet_file(filepath)
            else:
                records = []
        except Exception as error:
            print(f"ERROR loading {filename}: {error}")
            continue

        for record in records:
            if not isinstance(record, dict):
                stats["records_skipped"] += 1
                continue

            normalized = normalize_record(record, filename)
            if normalized:
                all_records.append(normalized)
                stats["total_records_before_dedup"] += 1
            else:
                stats["records_skipped"] += 1

    # Deduplication
    all_records, _, _, exact_count = detect_exact_duplicates(all_records)
    stats["exact_duplicates_removed"] = exact_count

    all_records, _, near_count = detect_near_duplicates(all_records)
    stats["near_duplicates_flagged"] = near_count

    print(f"\nTotal loaded: {len(all_records)} records")
    return all_records


def save_normalized(records: List[Dict[str, Any]], output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)
    print(f"Saved {len(records)} normalized records to: {output_path}")


if __name__ == "__main__":
    records = load_dataset()
    save_normalized(records)