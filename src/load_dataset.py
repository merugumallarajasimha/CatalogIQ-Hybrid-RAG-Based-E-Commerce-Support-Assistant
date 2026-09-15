import ast
import csv
import json
import os
import re
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
    "Furniture/Seating": "Furniture/Seating",
    "Furniture/Desks": "Furniture/Desks",
    "Furniture/Mounts": "Furniture/Mounts",
    "Furniture/Storage": "Furniture/Storage",
    "Electronics": "Electronics",
    "Fasteners": "Fasteners",
    "Peripherals/Audio": "Peripherals/Audio",
    "Peripherals/Keyboards": "Peripherals/Keyboards",
    "Peripherals/Mice": "Peripherals/Mice",
    "Peripherals/Displays": "Peripherals/Displays",
    "Peripherals/Networking": "Peripherals/Networking",
    "Maintenance": "Maintenance",
    "Tools": "Tools",
    "Accessories": "Accessories",
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
    text = HTML_ENTITY_MAP.get(text, text)
    for entity, replacement in HTML_ENTITY_MAP.items():
        text = text.replace(entity, replacement)
    text = HTML_TAG_RE.sub(' ', text)
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(?<=\d)\.(?=\d)', '.', text)
    text = re.sub(r'\.\s+(?=[A-Z0-9])', '. ', text)
    text = text.strip()
    return text


def clean_text_content(text: str) -> str:
    text = strip_html(text)
    text = normalize_whitespace(text)
    return text


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


def get_first_value(
    record: Dict[str, Any],
    keys: List[str],
) -> Optional[Any]:
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


def parse_list_value(value: Any) -> Any:
    value = clean_value(value)
    if value is None:
        return []
    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]
    if isinstance(value, tuple):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return []
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed_value = ast.literal_eval(value)
            if isinstance(parsed_value, (list, tuple)):
                return [
                    str(item).strip()
                    for item in parsed_value
                    if str(item).strip()
                ]
        except (ValueError, SyntaxError):
            value = value[1:-1]
        return [
            item.strip().strip("'\"")
            for item in value.split(",")
            if item.strip()
        ]
    if ";" in value:
        return [
            item.strip()
            for item in value.split(";")
            if item.strip()
        ]
    return value


# =========================================================
# FILE LOADERS
# =========================================================

def load_json_file(filepath: Path) -> List[Dict[str, Any]]:
    with filepath.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        return [
            record
            for record in data
            if isinstance(record, dict)
        ]
    if isinstance(data, dict):
        return [data]
    raise ValueError(
        f"Expected a JSON list or dictionary, "
        f"but received {type(data).__name__}"
    )


def load_csv_file(filepath: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with filepath.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        for row in reader:
            records.append(dict(row))
    return records


def load_parquet_file(filepath: Path) -> List[Dict[str, Any]]:
    dataframe = pd.read_parquet(filepath)
    dataframe = dataframe.where(
        pd.notnull(dataframe),
        None,
    )
    records = dataframe.to_dict(orient="records")
    return [
        {
            key: clean_value(value)
            for key, value in record.items()
        }
        for record in records
    ]


# =========================================================
# DATASET DIRECTORY
# =========================================================

def find_dataset_directory() -> Path:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Data directory does not exist:\n{DATASET_DIR}"
        )
    if not DATASET_DIR.is_dir():
        raise NotADirectoryError(
            f"Expected a directory but found:\n{DATASET_DIR}"
        )
    supported_files = [
        file
        for file in DATASET_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in {
            ".csv",
            ".json",
            ".parquet",
        }
    ]
    if not supported_files:
        raise FileNotFoundError(
            f"No CSV, JSON, or Parquet files found in:\n"
            f"{DATASET_DIR}"
        )
    return DATASET_DIR


def is_excluded_file(filename: str) -> bool:
    filename_lower = filename.lower()
    for pattern in EXCLUDED_FILE_PATTERNS:
        if pattern.lower() in filename_lower:
            return True
    return False


# =========================================================
# TEXT FORMATTING
# =========================================================

def format_product_name(name: str) -> str:
    name = clean_text_content(name)
    return name


def format_text_for_chunking(text: str) -> str:
    text = clean_text_content(text)
    text = re.sub(r'\s*\.\s*\.', '.', text)
    text = re.sub(r'\s*,\s*,', ',', text)
    text = re.sub(r'^\s*[.,;:]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\.\s*$', '.', text)
    text = text.strip()
    return text


# =========================================================
# DUPLICATE DETECTION
# =========================================================

def compute_text_fingerprint(text: str) -> str:
    cleaned = re.sub(r'\s+', ' ', text.strip().lower())
    return cleaned


def detect_exact_duplicates(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    text_to_indices: Dict[str, List[int]] = {}
    for i, record in enumerate(records):
        fp = compute_text_fingerprint(record.get("text", ""))
        if fp not in text_to_indices:
            text_to_indices[fp] = []
        text_to_indices[fp].append(i)

    exact_dup_indices: Set[int] = set()
    exact_dup_groups = []
    for fp, indices in text_to_indices.items():
        if len(indices) > 1:
            exact_dup_groups.append(indices)
            for idx in indices[1:]:
                exact_dup_indices.add(idx)

    kept = []
    removed = []
    for i, record in enumerate(records):
        if i in exact_dup_indices:
            removed.append(record)
        else:
            kept.append(record)

    return kept, removed, len(exact_dup_groups), len(exact_dup_indices)


def detect_near_duplicates(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    sku_groups: Dict[str, List[int]] = {}
    for i, record in enumerate(records):
        sku = record.get("sku", "unknown")
        if sku not in sku_groups:
            sku_groups[sku] = []
        sku_groups[sku].append(i)

    near_dup_indices: Set[int] = set()
    near_dup_info = []
    for sku, indices in sku_groups.items():
        if len(indices) <= 1:
            continue
        text_fps = {}
        for idx in indices:
            fp = compute_text_fingerprint(records[idx].get("text", ""))
            if fp not in text_fps:
                text_fps[fp] = []
            text_fps[fp].append(idx)
        for fp, idx_list in text_fps.items():
            if len(idx_list) > 1:
                for idx in idx_list[1:]:
                    near_dup_indices.add(idx)
                near_dup_info.append({
                    "sku": sku,
                    "count": len(idx_list),
                    "indices": idx_list,
                })

    kept = []
    flagged = []
    for i, record in enumerate(records):
        if i in near_dup_indices:
            flagged.append(record)
        else:
            kept.append(record)

    return kept, flagged, len(near_dup_info)


# =========================================================
# DATASET LOADER
# =========================================================

def load_dataset() -> List[Dict[str, Any]]:
    dataset_dir = find_dataset_directory()

    print("\nUsing dataset directory:")
    print(dataset_dir)

    all_records: List[Dict[str, Any]] = []

    stats = {
        "files_processed": 0,
        "total_records_before_dedup": 0,
        "records_skipped": 0,
        "defaults_used": {
            "sku": 0,
            "product_name": 0,
            "category": 0,
        },
        "excluded_files": [],
        "exact_duplicates_removed": 0,
        "near_duplicates_flagged": 0,
        "near_duplicates_merged": 0,
        "category_defaulted": 0,
        "sku_defaulted": 0,
    }

    for filepath in sorted(dataset_dir.iterdir()):
        if not filepath.is_file():
            continue

        extension = filepath.suffix.lower()
        if extension not in {".csv", ".json", ".parquet"}:
            continue

        filename = filepath.name

        if is_excluded_file(filename):
            print(f"Skipping excluded file: {filename}")
            stats["excluded_files"].append(filename)
            continue

        if "normalized" in filename.lower():
            print(f"Skipping generated file: {filename}")
            continue

        stats["files_processed"] += 1

        print(f"\nLoading {filename}...")

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
            print(
                f"ERROR loading {filename}: "
                f"{type(error).__name__}: {error}"
            )
            continue

        print(f"Records found in {filename}: {len(records)}")

        for record in records:
            if not isinstance(record, dict):
                stats["records_skipped"] += 1
                continue

            normalized_record = normalize_record(
                record=record,
                source_file=filename,
            )

            if normalized_record:
                all_records.append(normalized_record)
                stats["total_records_before_dedup"] += 1

                if normalized_record["sku"] == "unknown":
                    stats["defaults_used"]["sku"] += 1
                    stats["sku_defaulted"] += 1
                if normalized_record["product_name"] == "unknown":
                    stats["defaults_used"]["product_name"] += 1
                if normalized_record["category"] == DEFAULT_CATEGORY:
                    stats["defaults_used"]["category"] += 1
                    stats["category_defaulted"] += 1
            else:
                stats["records_skipped"] += 1

    print("\n--- Deduplication ---")
    before_dedup = len(all_records)

    all_records, exact_removed, exact_groups, exact_count = detect_exact_duplicates(all_records)
    stats["exact_duplicates_removed"] = exact_count
    stats["exact_duplicate_groups"] = exact_groups
    print(
        f"Exact duplicate groups: {exact_groups}, "
        f"records removed: {exact_count}"
    )

    all_records, near_flagged, near_groups = detect_near_duplicates(all_records)
    stats["near_duplicates_flagged"] = len(near_flagged)
    stats["near_duplicate_groups"] = near_groups
    print(
        f"Near-duplicate groups flagged: {near_groups}, "
        f"records flagged: {len(near_flagged)}"
    )

    after_dedup = len(all_records)
    print(
        f"\nTotal records before dedup: {before_dedup}"
    )
    print(f"Total records after dedup: {after_dedup}")
    print(
        f"Net removed: {before_dedup - after_dedup}"
    )

    print_summary(stats)

    return all_records


# =========================================================
# RECORD NORMALIZATION
# =========================================================

def normalize_record(
    record: Dict[str, Any],
    source_file: str,
) -> Optional[Dict[str, Any]]:

    # -----------------------------------------------------
    # SKU
    # -----------------------------------------------------
    sku = get_first_value(
        record,
        [
            "sku",
            "SKU",
            "product_id",
            "productId",
            "productID",
            "part_number",
            "part_no",
            "partNumber",
            "id",
            "asin",
            "ASIN",
        ],
    )

    # -----------------------------------------------------
    # PRODUCT NAME
    # -----------------------------------------------------
    product_name = get_first_value(
        record,
        [
            "product_name",
            "product_title",
            "title",
            "name",
            "productName",
            "item_name",
        ],
    )

    # -----------------------------------------------------
    # CATEGORY
    # -----------------------------------------------------
    category = get_first_value(
        record,
        [
            "category",
            "product_category",
            "category_name",
            "department",
            "productCategory",
            "main_category",
        ],
    )

    # -----------------------------------------------------
    # TRY TO DERIVE SKU FROM COMPATIBLE SKUS
    # -----------------------------------------------------
    if not sku:
        compatible_skus = record.get("compatible_skus")
        compatible_skus = parse_list_value(compatible_skus)
        if isinstance(compatible_skus, list):
            if compatible_skus:
                sku = compatible_skus[0]
        elif compatible_skus:
            sku = str(compatible_skus).split(",")[0].strip()

    if not sku:
        sku = "unknown"

    if not product_name:
        product_name = get_first_value(
            record,
            [
                "description",
                "product_description",
                "body",
                "content",
            ],
        ) or "unknown"

    if not category:
        category = DEFAULT_CATEGORY

    # Apply category mapping
    category = CATEGORY_MAP.get(category, DEFAULT_CATEGORY)

    # -----------------------------------------------------
    # BUILD SEARCHABLE TEXT
    # -----------------------------------------------------
    text_parts: List[str] = []

    main_text_keys = [
        "content",
        "text",
        "product_description",
        "description",
        "body",
        "details",
        "feature",
        "features",
    ]

    main_text_source = None
    main_text = ""
    for key in main_text_keys:
        value = clean_value(record.get(key))
        if value:
            main_text = str(value)
            main_text_source = key
            break

    if main_text:
        main_text = format_text_for_chunking(main_text)
        text_parts.append(main_text)

    if product_name != "unknown":
        product_name_clean = re.sub(r'\s+', ' ', str(product_name).strip().lower())
        main_text_clean = re.sub(r'\s+', ' ', str(main_text).strip().lower())
        if product_name_clean != main_text_clean:
            text_parts.append(
                f"Product name: {format_product_name(product_name)}"
            )

    if category != DEFAULT_CATEGORY:
        text_parts.append(
            f"Category: {category}"
        )

    resolution_steps = parse_list_value(
        record.get("resolution_steps")
    )
    if resolution_steps:
        if isinstance(resolution_steps, list):
            steps = [
                str(step).strip()
                for step in resolution_steps
                if str(step).strip()
            ]
            if steps:
                text_parts.append(
                    "Resolution steps:\n"
                    + "\n".join(
                        f"- {step}" for step in steps
                    )
                )
        else:
            text_parts.append(
                f"Resolution steps: {resolution_steps}"
            )

    customer_description = clean_value(
        record.get("customer_description")
    )
    if customer_description:
        text_parts.append(
            f"Customer issue: {format_text_for_chunking(customer_description)}"
        )

    agent_resolution = clean_value(record.get("agent_resolution"))
    if agent_resolution:
        text_parts.append(
            f"Resolution: {format_text_for_chunking(agent_resolution)}"
        )

    replacement_procedure = clean_value(
        record.get("replacement_procedure")
    )
    if replacement_procedure:
        text_parts.append(
            f"Replacement procedure: {format_text_for_chunking(replacement_procedure)}"
        )

    issue_type = clean_value(record.get("issue_type"))
    if issue_type:
        text_parts.append(f"Issue type: {issue_type}")

    # -----------------------------------------------------
    # PART NUMBERS - from BOTH part_numbers (list) AND part_number (singular)
    # -----------------------------------------------------
    part_numbers_text = ""

    part_numbers_list = parse_list_value(record.get("part_numbers"))
    if part_numbers_list and isinstance(part_numbers_list, list):
        part_numbers_text = ", ".join(
            str(part).strip()
            for part in part_numbers_list
            if str(part).strip()
        )

    part_number_singular = clean_value(record.get("part_number"))
    if part_number_singular and str(part_number_singular).strip():
        if part_numbers_text:
            part_numbers_text = part_numbers_text + ", " + str(part_number_singular).strip()
        else:
            part_numbers_text = str(part_number_singular).strip()

    if part_numbers_text:
        text_parts.append(
            f"Part numbers: {part_numbers_text}"
        )

    compatible_skus = parse_list_value(record.get("compatible_skus"))
    if compatible_skus:
        if isinstance(compatible_skus, list):
            compatible_skus_text = ", ".join(
                str(item).strip()
                for item in compatible_skus
                if str(item).strip()
            )
        else:
            compatible_skus_text = str(compatible_skus)
        if compatible_skus_text:
            text_parts.append(
                f"Compatible SKUs: {compatible_skus_text}"
            )

    related_parts = parse_list_value(record.get("related_parts"))
    if related_parts:
        if isinstance(related_parts, list):
            related_parts_text = ", ".join(
                str(item).strip()
                for item in related_parts
                if str(item).strip()
            )
        else:
            related_parts_text = str(related_parts)
        if related_parts_text:
            text_parts.append(
                f"Related parts: {related_parts_text}"
            )

    parts_used = parse_list_value(record.get("parts_used"))
    if parts_used:
        if isinstance(parts_used, list):
            parts_used_text = ", ".join(
                str(item).strip()
                for item in parts_used
                if str(item).strip()
            )
        else:
            parts_used_text = str(parts_used)
        if parts_used_text:
            text_parts.append(
                f"Parts used: {parts_used_text}"
            )

    # -----------------------------------------------------
    # EXTRACTION METHOD TRACKING
    # -----------------------------------------------------
    extraction_method = []
    if main_text:
        extraction_method.append("content_field")
    if part_numbers_list:
        extraction_method.append("part_numbers_field")
    if part_number_singular:
        extraction_method.append("part_number_field")
    if compatible_skus:
        extraction_method.append("compatible_skus_field")
    if related_parts:
        extraction_method.append("related_parts_field")
    if not extraction_method:
        extraction_method.append("none")

    # -----------------------------------------------------
    # FINAL TEXT
    # -----------------------------------------------------
    text = "\n\n".join(
        str(part).strip()
        for part in text_parts
        if part and str(part).strip()
    ).strip()

    if not text:
        return None

    # -----------------------------------------------------
    # EXTRACTION CONFIDENCE
    # -----------------------------------------------------
    has_sku = sku != "unknown"
    has_category = category != DEFAULT_CATEGORY
    confidence_flags = []
    if not has_sku:
        confidence_flags.append("sku_defaulted")
    if not has_category:
        confidence_flags.append("category_defaulted")
    if not main_text:
        confidence_flags.append("no_content")

    return {
        "sku": str(sku).strip(),
        "product_name": str(product_name).strip(),
        "category": str(category).strip(),
        "text": text,
        "source_file": source_file,
        "extraction_method": "+".join(extraction_method),
        "confidence_flags": confidence_flags,
        "has_sku": has_sku,
        "has_category": has_category,
    }


# =========================================================
# SUMMARY
# =========================================================

def print_summary(stats: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("DATASET LOADING SUMMARY")
    print("=" * 60)

    print(f"Files processed: {stats['files_processed']}")
    print(f"Total records loaded (before dedup): {stats['total_records_before_dedup']}")
    print(f"Records skipped: {stats['records_skipped']}")

    if stats.get("excluded_files"):
        print(f"\nExcluded files ({len(stats['excluded_files'])}):")
        for f in stats["excluded_files"]:
            print(f"  - {f}")

    print(f"\nExact duplicates removed: {stats.get('exact_duplicates_removed', 0)}")
    print(f"Near-duplicate records flagged: {stats.get('near_duplicates_flagged', 0)}")

    print(f"\nRecords using defaults:")
    print(f"  SKU: {stats['defaults_used']['sku']}")
    print(f"  Product name: {stats['defaults_used']['product_name']}")
    print(f"  Category: {stats['defaults_used']['category']}")

    print("=" * 60)


# =========================================================
# SAVE NORMALIZED RECORDS
# =========================================================

def save_normalized(
    records: List[Dict[str, Any]],
    output_path: Path = OUTPUT_PATH,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            records,
            file,
            indent=2,
            ensure_ascii=False,
        )
    print(
        f"\nSaved {len(records)} normalized records to:"
    )
    print(output_path)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    records = load_dataset()

    save_normalized(records)

    print("\nFirst 5 records:")

    for record in records[:5]:
        preview_text = (
            record["text"][:150]
            .replace("\n", " ")
        )
        print(
            f"\nSKU: {record['sku']}\n"
            f"Product: {record['product_name']}\n"
            f"Category: {record['category']}\n"
            f"Source: {record['source_file']}\n"
            f"Confidence flags: {record.get('confidence_flags', [])}\n"
            f"Extraction: {record.get('extraction_method', 'none')}\n"
            f"Text: {preview_text}..."
        )
