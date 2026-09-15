import sys
import time
import json
sys.path.insert(0, 'src')

# Pre-load all models first
from search import get_client, get_embedder, get_bm25_search, DENSE_TOP_K, BM25_TOP_K, RRF_TOP_K, RRF_K, COLLECTION_NAME
from reranker import get_reranker, rerank
from retrieval_metrics import compute_metrics, load_evaluation_queries, load_normalized_records
from exact_search import extract_ids, exact_qdrant_lookup
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Pre-load everything
print("Pre-loading models...")
_ = get_client()
_ = get_embedder()
_ = get_bm25_search()
_ = get_reranker()
print("Models loaded!")

queries = load_evaluation_queries()
records = load_normalized_records()

bm25 = get_bm25_search()
client = get_client()
embedder = get_embedder()

def dense_search_cached(query, top_k=DENSE_TOP_K):
    query_vec = embedder.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        using="dense_vector",
        limit=top_k,
        with_payload=["doc_id"]
    )
    return [(hit.payload["doc_id"], rank + 1) for rank, hit in enumerate(results.points) if hit.payload and "doc_id" in hit.payload]

def bm25_search_cached(query, top_k=BM25_TOP_K):
    results = bm25.search(query, top_k=top_k)
    return results

def exact_qdrant_lookup_cached(query, top_k=5):
    query_ids = extract_ids(query)
    all_identifiers = []
    for id_type in ("skus", "asins", "part_numbers"):
        all_identifiers.extend(query_ids.get(id_type, []))
    
    if not all_identifiers:
        return []
    
    results = []
    seen_doc_ids = set()
    for identifier in all_identifiers:
        sku_results = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[FieldCondition(key="sku", match=MatchValue(value=identifier))]),
            limit=top_k,
            with_payload=True
        )
        points, _ = sku_results
        for point in points:
            payload = point.payload
            doc_id = payload.get("doc_id", identifier)
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                results.append({"doc_id": doc_id, "score": 100.0})
        
        pn_results = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[FieldCondition(key="part_number", match=MatchValue(value=identifier))]),
            limit=top_k,
            with_payload=True
        )
        points, _ = pn_results
        for point in points:
            payload = point.payload
            doc_id = payload.get("doc_id", identifier)
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                results.append({"doc_id": doc_id, "score": 100.0})
    
    return results[:top_k]

def dense_only_retrieval(query, top_k=5):
    results = dense_search_cached(query, DENSE_TOP_K)
    return [doc_id for doc_id, _ in results[:top_k]]

def bm25_only_retrieval(query, top_k=5):
    results = bm25_search_cached(query, BM25_TOP_K)
    return [doc_id for doc_id, _ in results[:top_k]]

def dense_bm25_concat_retrieval(query, top_k=5):
    dense_results = dense_search_cached(query, DENSE_TOP_K)
    bm25_results_raw = bm25_search_cached(query, BM25_TOP_K)
    bm25_results = [doc_id for doc_id, _ in bm25_results_raw]
    
    seen = set()
    combined = []
    for doc_id, _ in dense_results:
        if doc_id not in seen:
            combined.append(doc_id)
            seen.add(doc_id)
    for doc_id in bm25_results:
        if doc_id not in seen:
            combined.append(doc_id)
            seen.add(doc_id)
    return combined[:top_k]

def dense_bm25_rrf_retrieval(query, top_k=5):
    dense_results = dense_search_cached(query, DENSE_TOP_K)
    bm25_results_raw = bm25_search_cached(query, BM25_TOP_K)
    bm25_results = [doc_id for doc_id, _ in bm25_results_raw]
    
    dense_ranks = {doc_id: rank for doc_id, rank in dense_results}
    bm25_ranks = {doc_id: i + 1 for i, doc_id in enumerate(bm25_results)}
    all_doc_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())
    
    fused = []
    for doc_id in all_doc_ids:
        score = 0.0
        if doc_id in dense_ranks:
            score += 1.0 / (RRF_K + dense_ranks[doc_id])
        if doc_id in bm25_ranks:
            score += 1.0 / (RRF_K + bm25_ranks[doc_id])
        fused.append({"doc_id": doc_id, "rrf_score": score})
    
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return [f["doc_id"] for f in fused[:top_k]]

def full_pipeline_retrieval(query, top_k=5):
    # Check exact lookup first
    exact_results = exact_qdrant_lookup_cached(query, top_k=RRF_TOP_K)
    if exact_results:
        return [r["doc_id"] for r in exact_results[:top_k]]
    
    # Otherwise use RRF
    candidates = dense_bm25_rrf_retrieval(query, RRF_TOP_K)
    # Enrich with mock content for reranker
    enriched = []
    for doc_id in candidates:
        enriched.append({
            "doc_id": doc_id,
            "content": "test content for " + doc_id,
            "sku": doc_id,
            "part_numbers": [],
            "product_name": "Test",
            "category": "Test",
        })
    reranked = rerank(query, enriched, top_n=top_k)
    return [r['doc_id'] for r in reranked]

# Run ablation
configs = [
    ("Dense Only", dense_only_retrieval, 5),
    ("BM25 Only", bm25_only_retrieval, 5),
    ("Dense + BM25 (concat)", dense_bm25_concat_retrieval, 5),
    ("Dense + BM25 + RRF", dense_bm25_rrf_retrieval, 5),
    ("Dense + BM25 + RRF + Reranker", full_pipeline_retrieval, 5),
]

print("=" * 100)
print("RETRIEVAL ABLATION STUDY - 5 Configurations")
print("=" * 100)
print()

results = {name: [] for name, _, _ in configs}

for entry in queries:
    query = entry["query"]
    relevant = entry.get("relevant_skus", [])
    
    print(f"Query: {query}")
    print(f"  Relevant: {relevant}")
    
    for name, func, top_k in configs:
        start = time.perf_counter()
        try:
            retrieved = func(query, top_k)
            elapsed = (time.perf_counter() - start) * 1000
            metrics = compute_metrics(query, relevant, retrieved)
            metrics["latency_ms"] = elapsed
            results[name].append(metrics)
            print(f"  {name}: Recall@5={metrics['recall@5']:.3f}, MRR={metrics['mrr']:.3f}, Latency={elapsed:.1f}ms")
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            print(f"  {name}: ERROR - {e}")
            results[name].append({"recall@5": 0, "mrr": 0, "latency_ms": elapsed, "error": str(e)})
    print()

# Aggregate results
print("=" * 100)
print("AGGREGATED RESULTS")
print("=" * 100)
print()

metrics_to_show = ["recall@5", "recall@10", "mrr", "latency_ms"]

header = f"{'Config':<35}"
for m in metrics_to_show:
    header += f"{m:>15}"
print(header)
print("-" * len(header))

for name, _, _ in configs:
    vals = results[name]
    if not vals:
        continue
    row = f"{name:<35}"
    for m in metrics_to_show:
        avg = sum(v.get(m, 0) for v in vals) / len(vals)
        if m == "latency_ms":
            row += f"{avg:>15.1f}"
        else:
            row += f"{avg:>15.4f}"
    print(row)

print()
print("=" * 100)