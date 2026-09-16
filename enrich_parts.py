"""
Enrich part catalog chunks with parent product SKU.
This ensures BM25 can map part numbers to their parent products.
"""
import json
import os

# Load normalized records
records_path = "data/normalized_records.json"
with open(records_path, 'r', encoding='utf-8') as f:
    records = json.load(f)

# Build SKU -> product info mapping
sku_info = {}
for r in records:
    sku = r.get("sku")
    if sku and sku in {"CHAIR-ERG-X99", "DESK-STD-MOTO", "MON-ARM-DUAL", 
                        "KB-ERGO-SPLIT", "MOUSE-VERT-PRO", "HEADSET-WL-PRO"}:
        if sku not in sku_info:
            sku_info[sku] = {
                "product_name": r.get("product_name"),
                "category": r.get("category"),
            }

print("Known SKUs:", list(sku_info.keys()))

# Build part_number -> compatible SKUs mapping from parts catalog
part_to_skus = {}
parts_path = "data/parts_catalog.csv"
import csv
with open(parts_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pn = row.get("part_number", "").strip()
        compat = row.get("compatible_skus", "").strip()
        if pn and compat:
            try:
                skus = eval(compat) if compat.startswith("[") else [compat]
                part_to_skus[pn] = [s for s in skus if s in sku_info]
            except:
                part_to_skus[pn] = []

print(f"Part-to-SKU mappings: {len(part_to_skus)}")

# Enrich each record
for r in records:
    text = r.get("text", "")
    sku = r.get("sku")
    
    # If this is a part catalog entry, find its parent SKUs
    if r.get("source_file") == "parts_catalog.csv":
        pn = r.get("sku")  # In normalized records, parts have their part_number as sku field
        if pn in part_to_skus:
            parent_skus = part_to_skus[pn]
            if parent_skus:
                # Add parent SKU info to text
                parent_names = [sku_info[s]["product_name"] for s in parent_skus if s in sku_info]
                if parent_names:
                    r["parent_skus"] = parent_skus
                    r["parent_product_names"] = parent_names
                    # Prepend parent info to text for better BM25 matching
                    prefix = f"Part of: {', '.join(parent_names)} (SKUs: {', '.join(parent_skus)}). "
                    if not text.startswith(prefix):
                        r["text"] = prefix + text
                    print(f"  Enriched {pn} -> {parent_skus}")

# Save enriched records
with open(records_path, 'w', encoding='utf-8') as f:
    json.dump(records, f, indent=2)

print(f"\nSaved enriched records to {records_path}")

# Also rebuild BM25 index
print("\nRebuilding BM25 index...")
from src.bm25_search import BM25Search
bm25 = BM25Search()
bm25.build_from_records(records)
print("BM25 index rebuilt successfully!")