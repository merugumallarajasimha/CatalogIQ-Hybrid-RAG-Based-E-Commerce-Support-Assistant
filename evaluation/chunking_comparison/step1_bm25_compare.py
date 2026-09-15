"""Step 1 CHECKPOINT: BM25 before/after comparison - discriminative analysis"""
import sys, os, json
from collections import Counter

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from bm25_search import BM25Search, tokenize, compute_crc32_tf_idf

DATA_DIR = os.path.join(os.getcwd(), 'data')

with open(os.path.join(DATA_DIR, 'normalized_records.json'), encoding='utf-8', errors='replace') as f:
    records = json.load(f)

print("=" * 80)
print("STEP 1 CHECKPOINT: BM25 BEFORE/AFTER COMPARISON")
print("=" * 80)
print()

# Build both indices
bm25 = BM25Search()
bm25.build_from_records(records)

# Find documents containing target SKUs
target_skus = ['CHAIR-ERG-X99', 'B-3301-BLT', 'S-4012-SCRW', 'SN-8801-SNS', 'DESK-STD-MOTO']
for sku in target_skus:
    containing = [(r['sku'], r.get('content', '')[:80]) for r in records if sku in str(r.get('content', '')) or sku == r.get('sku', '')]
    print(f"\n{sku} found in {len(containing)} documents:")
    for doc_sku, text in containing[:3]:
        print(f"  {doc_sku}: {text}...")

print()
print("=" * 80)
print("SCORING COMPARISON ON TEST QUERIES")
print("=" * 80)

test_queries = [
    "CHAIR-ERG-X99 repair warranty",
    "B-3301-BLT bolt replacement",
    "S-4012-SCRW screw M6x20",
    "SN-8801-SNS sensor troubleshooting",
    "DESK-STD-MOTO motor replacement",
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    # OLD: CRC32 TF-IDF sparse search
    old_results = []
    for record in records:
        text = record.get('content', '') or record.get('text', '') or ''
        doc_id = str(record.get('sku', ''))
        score = compute_crc32_tf_idf(query, text)
        if score > 0:
            old_results.append((doc_id, score))
    old_results.sort(key=lambda x: x[1], reverse=True)

    # NEW: BM25 search
    new_results = bm25.search(query, top_k=10)

    print(f"\n  OLD (CRC32 TF-IDF Sparse) - Top 5:")
    for doc_id, score in old_results[:5]:
        print(f"    {doc_id}: {score:.6f}")

    print(f"\n  NEW (BM25) - Top 5:")
    for doc_id, score in new_results[:5]:
        print(f"    {doc_id}: {score:.6f}")

    # Score distribution analysis
    if old_results and new_results:
        old_scores = [s for _, s in old_results]
        new_scores = [s for _, s in new_results]
        old_range = max(old_scores) - min(old_scores) if old_scores else 0
        new_range = max(new_scores) - min(new_scores) if new_scores else 0
        old_cv = (sum((x - sum(old_scores)/len(old_scores))**2 for x in old_scores) / len(old_scores)) ** 0.5 if old_scores else 0
        new_cv = (sum((x - sum(new_scores)/len(new_scores))**2 for x in new_scores) / len(new_scores)) ** 0.5 if new_scores else 0

        print(f"\n  Score Distribution:")
        print(f"    OLD: range={old_range:.6f}, stdev={old_cv:.6f}")
        print(f"    NEW: range={new_range:.6f}, stdev={new_cv:.6f}")
        print(f"    BM25 discrimination improvement: {((new_cv/old_cv)-1)*100:.0f}%" if old_cv > 0 else "    N/A")

print()
print("=" * 80)
print("VERIFICATION")
print("=" * 80)
print("BM25 model: rank_bm25.BM25Okapi (k1=1.5, b=0.75)")
print(f"Corpus size: {len(records)} documents")
print(f"Tokenization: lowercase alphanumeric, min length 2")
print("IDF: computed automatically by BM25Okapi from corpus statistics")
print("Integration: src/bm25_search.py BM25Search class")
