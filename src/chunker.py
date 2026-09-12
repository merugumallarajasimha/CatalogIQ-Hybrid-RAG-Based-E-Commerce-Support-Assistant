import re
from typing import List, Dict, Any, Optional
from extract_ids import extract_ids


def split_into_chunks(text: str, max_sentences: int = 5, min_sentences: int = 3) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= max_sentences:
        return [" ".join(sentences)]
    
    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sentences = sentences[i:i + max_sentences]
        if len(chunk_sentences) < min_sentences and i + max_sentences < len(sentences):
            chunk_sentences = sentences[i:i + min_sentences]
        chunks.append(" ".join(chunk_sentences))
        i += max_sentences
    
    return chunks


def chunk_record(record: Dict[str, Any], chunk_size: int = 5) -> List[Dict[str, Any]]:
    content = record.get("content", "") or record.get("text", "")
    if not content:
        return []
    
    sku = record.get("sku", "unknown")
    product_name = record.get("product_name", "unknown")
    category = record.get("category", "unknown")
    doc_type = record.get("doc_type", "unknown")
    title = record.get("title", "")
    source_file = record.get("source_file", "")
    
    # Create a short source identifier for unique IDs
    source_short = source_file.replace(".json", "").replace(".csv", "").replace("_", "-")[:20]
    
    chunks_text = split_into_chunks(content, max_sentences=chunk_size)
    
    chunked_records = []
    for idx, chunk_text in enumerate(chunks_text):
        extracted = extract_ids(chunk_text)
        
        chunk = {
            "id": f"{sku}-{source_short}-{idx}",
            "sku": sku,
            "part_numbers": extracted["part_numbers"],
            "product_name": product_name,
            "category": category,
            "doc_type": doc_type,
            "content": chunk_text,
            "source_file": source_file,
            "chunk_index": idx,
            "total_chunks": len(chunks_text)
        }
        chunked_records.append(chunk)
    
    return chunked_records


def chunk_records(records: List[Dict[str, Any]], chunk_size: int = 5) -> List[Dict[str, Any]]:
    all_chunks = []
    for record in records:
        all_chunks.extend(chunk_record(record, chunk_size))
    return all_chunks


if __name__ == "__main__":
    test_record = {
        "sku": "CHAIR-ERG-X99",
        "product_name": "ErgoFlex Pro Office Chair",
        "category": "Furniture/Seating",
        "doc_type": "troubleshooting_guide",
        "content": "TROUBLESHOOTING GUIDE - ErgoFlex Pro Office Chair (SKU: CHAIR-ERG-X99)\n\nCOMMON ISSUES AND RESOLUTIONS:\n\n1. RECLINE SQUEAKING NOISE\nSymptom: Audible squeaking when leaning back in the chair.\nRoot Cause: Loose screw set S-4012-SCRW under base P-9913-BASE or worn pivot bearings.\nResolution:\n  a) Turn chair upside down on protective surface.\n  b) Locate screw set S-4012-SCRW (4 screws) securing base P-9913-BASE to seat frame.\n  c) Tighten all 4 screws to 12.5 Nm torque using 5mm hex key.\n  d) If squeaking persists, inspect pivot bearings P-9912-ARM for wear.\n  e) Replace worn bearings with part number P-9912-ARM (2 per chair)."
    }
    
    chunks = chunk_record(test_record)
    for c in chunks:
        print(f"Chunk {c['chunk_index']}: {c['content'][:100]}...")
        print(f"  SKU: {c['sku']}, Parts: {c['part_numbers']}")
        print()