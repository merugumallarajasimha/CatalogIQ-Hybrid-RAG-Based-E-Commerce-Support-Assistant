import json
import csv
import os
from typing import List, Dict, Any, Optional


def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_csv_file(filepath: str) -> List[Dict[str, Any]]:
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ['part_numbers', 'compatible_skus', 'related_parts']:
                if key in row and isinstance(row[key], str):
                    val = row[key].strip()
                    if val.startswith('[') and val.endswith(']'):
                        try:
                            import ast
                            row[key] = ast.literal_eval(val)
                        except:
                            row[key] = [v.strip().strip("'\"") for v in val[1:-1].split(',')]
                    elif val:
                        row[key] = [v.strip() for v in val.split(';') if v.strip()]
            
            if 'resolution_steps' in row and isinstance(row['resolution_steps'], str):
                row['resolution_steps'] = [s.strip() for s in row['resolution_steps'].split('|') if s.strip()]
            if 'parts_used' in row and isinstance(row['parts_used'], str):
                row['parts_used'] = [p.strip() for p in row['parts_used'].split(';') if p.strip()]
            records.append(row)
    return records


def load_dataset(dataset_dir: str = "dataset") -> List[Dict[str, Any]]:
    all_records = []
    stats = {
        "files_processed": 0,
        "total_records": 0,
        "records_skipped": 0,
        "defaults_used": {"sku": 0, "product_name": 0, "category": 0}
    }
    
    for filename in sorted(os.listdir(dataset_dir)):
        filepath = os.path.join(dataset_dir, filename)
        
        if not (filename.endswith('.json') or filename.endswith('.csv')):
            continue
        
        if 'shopping_queries' in filename or 'normalized' in filename:
            continue
        
        stats["files_processed"] += 1
        print(f"Loading {filename}...")
        
        try:
            if filename.endswith('.json'):
                records = load_json_file(filepath)
            else:
                records = load_csv_file(filepath)
        except Exception as e:
            print(f"  ERROR loading {filename}: {e}")
            stats["records_skipped"] += len(records) if 'records' in locals() else 0
            continue
        
        for record in records:
            normalized = normalize_record(record, filename)
            if normalized:
                all_records.append(normalized)
                stats["total_records"] += 1
                
                if normalized["sku"] == "unknown":
                    stats["defaults_used"]["sku"] += 1
                if normalized["product_name"] == "unknown":
                    stats["defaults_used"]["product_name"] += 1
                if normalized["category"] == "unknown":
                    stats["defaults_used"]["category"] += 1
            else:
                stats["records_skipped"] += 1
    
    print_summary(stats)
    return all_records


def normalize_record(record: Dict[str, Any], source_file: str) -> Optional[Dict[str, Any]]:
    sku = record.get("sku") or record.get("product_id")
    product_name = record.get("product_name") or record.get("product_title") or record.get("title")
    category = record.get("category") or record.get("product_category")
    
    if "compatible_skus" in record and record["compatible_skus"] and not sku:
        compatible = record["compatible_skus"]
        if isinstance(compatible, list) and compatible:
            sku = compatible[0]
        elif isinstance(compatible, str):
            sku = compatible.split(",")[0].strip()
    
    if not sku:
        sku = "unknown"
    if not product_name:
        product_name = record.get("description", "unknown")
    if not category:
        category = "unknown"
    
    text_parts = []
    
    if "content" in record and record["content"]:
        text_parts.append(str(record["content"]))
    elif "text" in record and record["text"]:
        text_parts.append(str(record["text"]))
    elif "product_description" in record and record["product_description"]:
        text_parts.append(str(record["product_description"]))
    elif "description" in record and record["description"]:
        text_parts.append(str(record["description"]))
    
    if "resolution_steps" in record and record["resolution_steps"]:
        if isinstance(record["resolution_steps"], list):
            text_parts.extend(record["resolution_steps"])
        else:
            text_parts.append(str(record["resolution_steps"]))
    
    if "customer_description" in record and record["customer_description"]:
        text_parts.append(f"Customer issue: {record['customer_description']}")
    
    if "agent_resolution" in record and record["agent_resolution"]:
        text_parts.append(f"Resolution: {record['agent_resolution']}")
    
    if "replacement_procedure" in record and record["replacement_procedure"]:
        text_parts.append(f"Replacement: {record['replacement_procedure']}")
    
    if "issue_type" in record and record["issue_type"]:
        text_parts.append(f"Issue type: {record['issue_type']}")
    
    if "part_numbers" in record and record["part_numbers"]:
        if isinstance(record["part_numbers"], list):
            text_parts.append(f"Part numbers: {', '.join(record['part_numbers'])}")
        else:
            text_parts.append(f"Part numbers: {record['part_numbers']}")
    
    if "compatible_skus" in record and record["compatible_skus"]:
        if isinstance(record["compatible_skus"], list):
            text_parts.append(f"Compatible SKUs: {', '.join(record['compatible_skus'])}")
        else:
            text_parts.append(f"Compatible SKUs: {record['compatible_skus']}")
    
    text = "\n\n".join(text_parts).strip()
    
    if not text:
        return None
    
    return {
        "sku": str(sku).strip() if sku != "unknown" else "unknown",
        "product_name": str(product_name).strip() if product_name != "unknown" else "unknown",
        "category": str(category).strip() if category != "unknown" else "unknown",
        "text": text,
        "source_file": source_file
    }


def print_summary(stats: Dict[str, Any]) -> None:
    print("\n" + "=" * 50)
    print("DATASET LOADING SUMMARY")
    print("=" * 50)
    print(f"Files processed: {stats['files_processed']}")
    print(f"Total records loaded: {stats['total_records']}")
    print(f"Records skipped: {stats['records_skipped']}")
    print(f"Records using defaults:")
    print(f"  SKU: {stats['defaults_used']['sku']}")
    print(f"  Product name: {stats['defaults_used']['product_name']}")
    print(f"  Category: {stats['defaults_used']['category']}")
    print("=" * 50)


def save_normalized(records: List[Dict[str, Any]], output_path: str = "dataset/normalized_records.json") -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(records)} normalized records to {output_path}")


if __name__ == "__main__":
    records = load_dataset("dataset")
    save_normalized(records)
    print(f"\nFirst 3 records:")
    for r in records[:3]:
        print(f"  SKU: {r['sku']} | Product: {r['product_name']} | Category: {r['category']} | Text: {r['text'][:100]}...")