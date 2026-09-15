"""Step 8: Chunk Statistics"""
import sys, os, json, re, csv, importlib.util
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# ============================================================
# Inline chunking from chunker.py to avoid import issues
# ============================================================

def split_into_chunks(text, max_sentences=5, min_sentences=3):
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

def chunk_record_b(record, chunk_size=5):
    content = record.get("content", "") or record.get("text", "")
    if not content:
        return []
    sku = record.get("sku", "unknown")
    product_name = record.get("product_name", "unknown")
    category = record.get("category", "unknown")
    doc_type = record.get("doc_type", "unknown")
    source_file = record.get("source_file", "")
    source_short = source_file.replace(".json", "").replace(".csv", "").replace("_", "-")[:20]
    from extract_ids import extract_ids
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

DATA_DIR = os.path.join(os.getcwd(), 'data')

# Load normalized records
with open(os.path.join(DATA_DIR, 'normalized_records.json'), encoding='utf-8', errors='replace') as f:
    records = json.load(f)

print("=" * 80)
print("STEP 8: CHUNK STATISTICS")
print("=" * 80)
print()

# ============================================================
# STRATEGY B: Current fixed-size chunking (3-5 sentences)
# ============================================================
print("=" * 80)
print("STRATEGY B: Current chunker.py (3-5 sentence fixed-size)")
print("=" * 80)

all_chunks_b = []
chunk_counts = []
for record in records:
    chunks = chunk_record_b(record, chunk_size=5)
    all_chunks_b.extend(chunks)
    chunk_counts.append(len(chunks))

total_products = len(records)
total_chunks_b = len(all_chunks_b)
avg_chunks = total_chunks_b / total_products if total_products > 0 else 0
min_chunks = min(chunk_counts) if chunk_counts else 0
max_chunks = max(chunk_counts) if chunk_counts else 0

all_lengths = [len(c.get('content', '')) for c in all_chunks_b]
avg_len = sum(all_lengths) / len(all_lengths) if all_lengths else 0
min_len = min(all_lengths) if all_lengths else 0
max_len = max(all_lengths) if all_lengths else 0

empty_chunks = sum(1 for c in all_chunks_b if not c.get('content', '').strip())

chunk_texts = {}
dup_chunks = 0
for c in all_chunks_b:
    txt = c.get('content', '').strip().lower()
    if txt in chunk_texts:
        dup_chunks += 1
    else:
        chunk_texts[txt] = True

chunk_types_b = Counter(c.get('doc_type', 'unknown') for c in all_chunks_b)
zero_chunk_products_b = [(i, r['sku'], r['source_file'], len(r.get('text',''))) for i, (r, count) in enumerate(zip(records, chunk_counts)) if count == 0]

print(f"\nCHECKPOINT B - Strategy B Stats:")
print(f"  Total products: {total_products}")
print(f"  Total chunks: {total_chunks_b}")
print(f"  Avg chunks per product: {avg_chunks:.2f}")
print(f"  Min chunks per product: {min_chunks}")
print(f"  Max chunks per product: {max_chunks}")
print(f"  Avg chunk length (chars): {avg_len:.1f}")
print(f"  Min chunk length (chars): {min_len}")
print(f"  Max chunk length (chars): {max_len}")
print(f"  Empty chunks: {empty_chunks}")
print(f"  Duplicate chunks (exact text): {dup_chunks}")
print(f"  Chunk types: {dict(chunk_types_b)}")
print(f"  Products with ZERO chunks: {len(zero_chunk_products_b)}")
if zero_chunk_products_b:
    print("  ERROR - Zero chunk products:")
    for idx, sku, src, text_len in zero_chunk_products_b:
        print(f"    SKU: {sku} | Source: {src} | Text length: {text_len}")
print()

# ============================================================
# STRATEGY A: One complete document per product (no splitting)
# ============================================================
print("=" * 80)
print("STRATEGY A: No splitting (one chunk per document)")
print("=" * 80)

all_chunks_a = []
for record in records:
    text = record.get('content', '') or record.get('text', '')
    if not text:
        continue
    source_short = record.get('source_file', '').replace('.json','').replace('.csv','').replace('_','-')[:20]
    chunk = {
        'id': f"{record['sku']}-{source_short}-0",
        'sku': record['sku'],
        'part_numbers': [],
        'product_name': record['product_name'],
        'category': record['category'],
        'doc_type': 'product',
        'content': text,
        'source_file': record['source_file'],
        'chunk_index': 0,
        'total_chunks': 1
    }
    all_chunks_a.append(chunk)

total_chunks_a = len(all_chunks_a)
lengths_a = [len(c.get('content', '')) for c in all_chunks_a]
avg_len_a = sum(lengths_a) / len(lengths_a) if lengths_a else 0
min_len_a = min(lengths_a) if lengths_a else 0
max_len_a = max(lengths_a) if lengths_a else 0
dup_a = 0
texts_a = {}
for c in all_chunks_a:
    txt = c.get('content', '').strip().lower()
    if txt in texts_a:
        dup_a += 1
    else:
        texts_a[txt] = True
empty_a = sum(1 for c in all_chunks_a if not c.get('content', '').strip())

print(f"\nCHECKPOINT A - Strategy A Stats:")
print(f"  Total products: {total_products}")
print(f"  Total chunks: {total_chunks_a}")
print(f"  Avg chunks per product: 1.00")
print(f"  Avg chunk length (chars): {avg_len_a:.1f}")
print(f"  Min chunk length (chars): {min_len_a}")
print(f"  Max chunk length (chars): {max_len_a}")
print(f"  Empty chunks: {empty_a}")
print(f"  Duplicate chunks (exact text): {dup_a}")
print()

# ============================================================
# STRATEGY C: Field-aware semantic chunking (probe with raw data)
# ============================================================
print("=" * 80)
print("STRATEGY C: Field-aware semantic chunking (probe)")
print("=" * 80)

strategy_c_error = None
try:
    from src.field_chunker import build_chunks_from_record

    # Test on 5 records from each type
    sample = []
    for r in records:
        if 'product_manuals' in r.get('source_file', '') and len([x for x in sample if 'manual' in x.get('source_file','')]) < 2:
            sample.append(r)
    for r in records:
        if 'parts_catalog' in r.get('source_file', '') and len([x for x in sample if 'parts' in x.get('source_file','')]) < 2:
            sample.append(r)
    for r in records:
        if 'troubleshooting' in r.get('source_file', '') and len([x for x in sample if 'ticket' in x.get('source_file','')]) < 2:
            sample.append(r)

    all_chunks_c = []
    errors = []
    for record in sample:
        try:
            src_file = record.get('source_file', 'unknown')
            sku = record.get('sku', 'unknown')
            chunks = build_chunks_from_record(record, src_file, sku)
            all_chunks_c.extend(chunks)
        except Exception as e:
            errors.append(f"{sku}: {type(e).__name__}: {str(e)[:100]}")

    if errors:
        print(f"\nSTRATEGY C PROBE: Errors (needs canonical_builder field mapping):")
        for e in errors[:5]:
            print(f"  {e}")
        print(f"  Strategy C requires raw source files (not normalized) for canonical text building.")
    elif all_chunks_c:
        types_c = Counter(c.get('chunk_type', 'unknown') for c in all_chunks_c)
        empty_c = sum(1 for c in all_chunks_c if not c.get('content', '').strip())
        lengths_c = [len(c.get('content', '')) for c in all_chunks_c]
        print(f"\nCHECKPOINT C - Strategy C Probe ({len(sample)} records):")
        print(f"  Total chunks: {len(all_chunks_c)}")
        print(f"  Avg chunk length: {sum(lengths_c)/len(lengths_c):.1f}")
        print(f"  Empty chunks: {empty_c}")
        print(f"  Chunk types: {dict(types_c)}")
    else:
        print("  No chunks generated.")
except Exception as e:
    strategy_c_error = f"{type(e).__name__}: {str(e)[:200]}"
    print(f"\nSTRATEGY C PROBE: Import error - {strategy_c_error}")
    print("  Will be built properly in Step 9.")
print()

# ============================================================
# OVERALL SUMMARY
# ============================================================
print("=" * 80)
print("OVERALL SUMMARY - ALL STRATEGIES")
print("=" * 80)
print()
print(f"Total products in dataset: {total_products}")
print(f"Total records: {len(records)}")
print()
print(f"{'Strategy':<35} {'Chunks':>8} {'Avg/Prod':>10} {'AvgLen':>8} {'MinLen':>8} {'MaxLen':>8} {'Empty':>6} {'Dups':>6}")
print("-" * 105)
print(f"{'A: No splitting':<35} {total_chunks_a:>8} {'1.00':>10} {avg_len_a:>8.1f} {min_len_a:>8} {max_len_a:>8} {empty_a:>6} {dup_a:>6}")
print(f"{'B: Fixed 3-5 sentences':<35} {total_chunks_b:>8} {avg_chunks:>10.2f} {avg_len:>8.1f} {min_len:>8} {max_len:>8} {empty_chunks:>6} {dup_chunks:>6}")
c_status = "PROBE" if not strategy_c_error else "ERROR"
print(f"{'C: Field-aware':<35} {'see log':>8} {'TBD':>10} {'TBD':>8} {'TBD':>8} {'TBD':>8} {'TBD':>6} {'TBD':>6}  ({c_status})")
print()

# Zero chunk products
print("=" * 80)
print("ZERO-CHUNK PRODUCTS (ERROR CONDITION)")
print("=" * 80)
if zero_chunk_products_b:
    print(f"  FOUND {len(zero_chunk_products_b)} PRODUCTS WITH ZERO CHUNKS (ERROR):")
    for idx, sku, src, text_len in zero_chunk_products_b:
        print(f"    SKU: {sku} | Source: {src} | Text length: {text_len}")
else:
    print("  All products produced at least 1 chunk. PASS.")
print()

# Breakdown by source file
print("=" * 80)
print("CHUNK BREAKDOWN BY SOURCE FILE (Strategy B)")
print("=" * 80)
source_chunks = defaultdict(int)
source_products = defaultdict(int)
for i, (r, count) in enumerate(zip(records, chunk_counts)):
    src = r.get('source_file', 'unknown')
    source_products[src] += 1
    source_chunks[src] += count
for src in sorted(source_chunks.keys()):
    print(f"  {src}: {source_chunks[src]} chunks from {source_products[src]} products")
print()

# Save results
results = {
    'strategy_a': {'total_chunks': total_chunks_a, 'avg_per_product': 1.0, 'avg_len': avg_len_a, 'min_len': min_len_a, 'max_len': max_len_a, 'empty': empty_a, 'duplicates': dup_a},
    'strategy_b': {'total_chunks': total_chunks_b, 'avg_per_product': avg_chunks, 'avg_len': avg_len, 'min_len': min_len, 'max_len': max_len, 'empty': empty_chunks, 'duplicates': dup_chunks},
    'zero_chunk_products': zero_chunk_products_b,
    'total_products': total_products,
    'total_records': len(records),
    'chunk_types_b': dict(chunk_types_b),
    'source_breakdown': {src: {'chunks': source_chunks[src], 'products': source_products[src]} for src in source_chunks},
}
results_dir = os.path.join(os.getcwd(), 'evaluation', 'chunking_comparison')
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, 'step8_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"Step 8 results saved to: {results_path}")
