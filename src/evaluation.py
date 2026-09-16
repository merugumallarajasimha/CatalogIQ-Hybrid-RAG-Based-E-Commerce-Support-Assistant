"""
CatalogIQ Retrieval Evaluation and Ablation Framework
Runs queries from evaluation/queries.jsonl through retrieval configurations,
computes metrics, generates ablation tables, and performs failure analysis.
"""
import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter

# Set IPv4 explicitly to avoid Windows IPv6 localhost SYN_SENT timeout
os.environ["QDRANT_URL"] = os.getenv("QDRANT_URL", "http://127.0.0.1:6333").replace("localhost", "127.0.0.1")

# Add src to path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from retrieval_metrics import compute_metrics, recall_at_k, mrr, ndcg_at_k, precision_at_k, hit_rate
from search import (
    dense_search,
    bm25_search,
    get_client,
    get_embedder,
    get_bm25_search,
    get_document,
    DENSE_TOP_K,
    BM25_TOP_K,
    RRF_TOP_K,
    RRF_K,
    COLLECTION_NAME,
)
from reranker import get_reranker, rerank
from exact_search import has_identifier, exact_qdrant_lookup


# Global cache for doc_id / part_number to SKU mappings and doc payloads
_DOC_TO_SKU_MAP: Optional[Dict[str, str]] = None
_PART_TO_SKUS_MAP: Optional[Dict[str, List[str]]] = None
_DOC_CACHE: Optional[Dict[str, Dict[str, Any]]] = None

ALL_VALID_SKUS = {
    "CHAIR-ERG-X99",
    "DESK-STD-MOTO",
    "MON-ARM-DUAL",
    "KB-ERGO-SPLIT",
    "MOUSE-VERT-PRO",
    "HEADSET-WL-PRO",
}


def load_mappings():
    """Build fast in-memory mappings from Qdrant doc_id and parts to product SKUs."""
    global _DOC_TO_SKU_MAP, _PART_TO_SKUS_MAP, _DOC_CACHE
    if _DOC_TO_SKU_MAP is not None and _PART_TO_SKUS_MAP is not None and _DOC_CACHE is not None:
        return _DOC_TO_SKU_MAP, _PART_TO_SKUS_MAP, _DOC_CACHE

    _DOC_TO_SKU_MAP = {}
    _PART_TO_SKUS_MAP = {}
    _DOC_CACHE = {}

    # 1. Load from Qdrant in one single fast scroll
    try:
        client = get_client()
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            with_payload=True
        )
        for p in points:
            payload = p.payload or {}
            doc_id = payload.get("doc_id")
            sku = payload.get("sku")
            if doc_id:
                _DOC_CACHE[doc_id] = payload
                if sku:
                    _DOC_TO_SKU_MAP[doc_id] = sku
    except Exception as e:
        print(f"Warning: Could not load points from Qdrant: {e}")

    # 2. Load parts catalog
    parts_path = os.path.join(PROJECT_ROOT, "data", "parts_catalog.json")
    if os.path.exists(parts_path):
        with open(parts_path, "r", encoding="utf-8") as f:
            parts = json.load(f)
            for p in parts:
                pn = p.get("part_number")
                compat = p.get("compatible_skus", [])
                if pn and compat:
                    _PART_TO_SKUS_MAP[pn] = compat

    return _DOC_TO_SKU_MAP, _PART_TO_SKUS_MAP, _DOC_CACHE


def doc_id_to_skus(doc_id: str) -> List[str]:
    """Map a retrieved doc_id or part number to product SKUs."""
    doc_map, part_map, _ = load_mappings()

    # Direct SKU match
    if doc_id in ALL_VALID_SKUS:
        return [doc_id]

    # Qdrant doc_id match
    if doc_id in doc_map:
        return [doc_map[doc_id]]

    # Prefix match (e.g. CHAIR-ERG-X99-product-manuals-0)
    for sku in ALL_VALID_SKUS:
        if doc_id.startswith(sku):
            return [sku]

    # Part number match
    if doc_id in part_map:
        return part_map[doc_id]

    return []


def extract_skus_from_results(doc_ids: List[str]) -> List[str]:
    """Given an ordered list of doc_ids, return ordered list of distinct product SKUs."""
    skus = []
    seen = set()
    for d in doc_ids:
        mapped = doc_id_to_skus(d)
        for s in mapped:
            if s in ALL_VALID_SKUS and s not in seen:
                seen.add(s)
                skus.append(s)
    return skus


def retrieve_dense_only(query: str, top_k: int = 5) -> Tuple[List[str], List[str], float]:
    """Config A: Dense retrieval only."""
    start = time.perf_counter()
    raw = dense_search(query, top_k=DENSE_TOP_K)
    elapsed = (time.perf_counter() - start) * 1000
    doc_ids = [doc_id for doc_id, _ in raw]
    skus = extract_skus_from_results(doc_ids)
    return doc_ids[:top_k], skus[:top_k], elapsed


def retrieve_bm25_only(query: str, top_k: int = 5) -> Tuple[List[str], List[str], float]:
    """Config B: BM25 retrieval only."""
    start = time.perf_counter()
    raw = bm25_search(query, top_k=BM25_TOP_K)
    elapsed = (time.perf_counter() - start) * 1000
    doc_ids = [doc_id for doc_id, _ in raw]
    skus = extract_skus_from_results(doc_ids)
    return doc_ids[:top_k], skus[:top_k], elapsed


def retrieve_dense_bm25_concat(query: str, top_k: int = 5) -> Tuple[List[str], List[str], float]:
    """Config C: Dense + BM25 concatenated & deduplicated, no RRF."""
    start = time.perf_counter()
    dense_raw = dense_search(query, top_k=DENSE_TOP_K)
    bm25_raw = bm25_search(query, top_k=BM25_TOP_K)

    dense_docs = [doc_id for doc_id, _ in dense_raw]
    bm25_docs = [doc_id for doc_id, _ in bm25_raw]

    combined = []
    seen = set()
    # Interleave 1-by-1 from each list to give fair representation without scoring
    max_len = max(len(dense_docs), len(bm25_docs))
    for i in range(max_len):
        if i < len(dense_docs) and dense_docs[i] not in seen:
            seen.add(dense_docs[i])
            combined.append(dense_docs[i])
        if i < len(bm25_docs) and bm25_docs[i] not in seen:
            seen.add(bm25_docs[i])
            combined.append(bm25_docs[i])

    elapsed = (time.perf_counter() - start) * 1000
    skus = extract_skus_from_results(combined)
    return combined[:top_k], skus[:top_k], elapsed


def retrieve_dense_bm25_rrf(query: str, top_k: int = 5, rrf_k: int = RRF_K) -> Tuple[List[str], List[str], float]:
    """Config D: Dense + BM25 + RRF fusion."""
    start = time.perf_counter()
    dense_raw = dense_search(query, top_k=DENSE_TOP_K)
    bm25_raw = bm25_search(query, top_k=BM25_TOP_K)

    dense_ranks = {doc_id: rank for doc_id, rank in dense_raw}
    bm25_ranks = {doc_id: i + 1 for i, (doc_id, _) in enumerate(bm25_raw)}

    all_docs = set(dense_ranks.keys()) | set(bm25_ranks.keys())
    fused = []
    for doc_id in all_docs:
        score = 0.0
        if doc_id in dense_ranks:
            score += 1.0 / (rrf_k + dense_ranks[doc_id])
        if doc_id in bm25_ranks:
            score += 1.0 / (rrf_k + bm25_ranks[doc_id])
        fused.append((doc_id, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    elapsed = (time.perf_counter() - start) * 1000
    doc_ids = [d for d, _ in fused]
    skus = extract_skus_from_results(doc_ids)
    return doc_ids[:top_k], skus[:top_k], elapsed


def retrieve_full_pipeline(query: str, top_k: int = 5) -> Tuple[List[str], List[str], float]:
    """Config E: Dense + BM25 + RRF + Cross-Encoder Reranker."""
    start = time.perf_counter()

    # Step 1: Check exact identifier lookup
    if has_identifier(query):
        exact_results = exact_qdrant_lookup(query, top_k=RRF_TOP_K)
        if exact_results:
            doc_ids = [r["doc_id"] for r in exact_results]
            skus = extract_skus_from_results(doc_ids)
            elapsed = (time.perf_counter() - start) * 1000
            return doc_ids[:top_k], skus[:top_k], elapsed

    # Step 2: Dense + BM25 RRF
    dense_raw = dense_search(query, top_k=DENSE_TOP_K)
    bm25_raw = bm25_search(query, top_k=BM25_TOP_K)

    dense_ranks = {doc_id: rank for doc_id, rank in dense_raw}
    bm25_ranks = {doc_id: i + 1 for i, (doc_id, _) in enumerate(bm25_raw)}

    all_docs = set(dense_ranks.keys()) | set(bm25_ranks.keys())
    fused = []
    for doc_id in all_docs:
        score = 0.0
        if doc_id in dense_ranks:
            score += 1.0 / (RRF_K + dense_ranks[doc_id])
        if doc_id in bm25_ranks:
            score += 1.0 / (RRF_K + bm25_ranks[doc_id])
        fused.append((doc_id, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    top_candidates = fused[:RRF_TOP_K]

    # Step 3: Enrich with document content using fast in-memory cache
    _, _, doc_cache = load_mappings()
    enriched = []
    for doc_id, score in top_candidates:
        doc = doc_cache.get(doc_id)
        if not doc:
            doc = get_document(doc_id)
            if doc:
                doc_cache[doc_id] = doc
        content = doc.get("content", "") if doc else ""
        sku = doc.get("sku", "unknown") if doc else "unknown"
        enriched.append({
            "doc_id": doc_id,
            "content": content,
            "sku": sku,
            "combined_score": score,
        })

    # Step 4: Rerank
    reranked = rerank(query, enriched, top_n=top_k)
    elapsed = (time.perf_counter() - start) * 1000

    doc_ids = [r["doc_id"] for r in reranked]
    skus = extract_skus_from_results(doc_ids)
    return doc_ids[:top_k], skus[:top_k], elapsed


def classify_failure(
    query: str,
    expected_skus: List[str],
    retrieved_skus: List[str],
    dense_rank: Optional[int],
    bm25_rank: Optional[int],
    rrf_rank: Optional[int],
    reranker_rank: Optional[int],
) -> str:
    """
    Classify failure reason based on stage-by-stage evidence into one of:
    - missing_product_in_dataset
    - bad_query_understanding
    - poor_chunking
    - exact_identifier_missed
    - dense_retrieval_failure
    - bm25_failure
    - rrf_ranking_failure
    - reranker_failure
    - metadata_filter_failure
    """
    if not expected_skus:
        return "irrelevant_query_handled" if not retrieved_skus else "false_positive_retrieval"

    for sku in expected_skus:
        if sku not in ALL_VALID_SKUS:
            return "missing_product_in_dataset"

    # Check exact identifier missed
    if has_identifier(query):
        if (dense_rank is None or dense_rank > 5) and (bm25_rank is None or bm25_rank > 5):
            return "exact_identifier_missed"

    # Check reranker failure: good RRF rank but reranker dropped it
    if rrf_rank is not None and rrf_rank <= 5:
        if reranker_rank is None or reranker_rank > 5:
            return "reranker_failure"

    # Check RRF ranking failure: good Dense or BM25 rank, but RRF dropped it
    if (dense_rank is not None and dense_rank <= 5) or (bm25_rank is not None and bm25_rank <= 5):
        if rrf_rank is None or rrf_rank > 5:
            return "rrf_ranking_failure"

    # Check dense retrieval failure: BM25 found it, dense failed
    if (dense_rank is None or dense_rank > 10) and (bm25_rank is not None and bm25_rank <= 10):
        return "dense_retrieval_failure"

    # Check BM25 failure: dense found it, BM25 completely failed on keyword
    if (bm25_rank is None or bm25_rank > 10) and (dense_rank is not None and dense_rank <= 10):
        return "bm25_failure"

    # Both dense and BM25 missed
    if (dense_rank is None or dense_rank > 20) and (bm25_rank is None or bm25_rank > 20):
        return "bad_query_understanding"

    return "poor_chunking"


def run_evaluation(queries_path: str = "evaluation/queries.jsonl", verbose: bool = False):
    """
    Run full retrieval evaluation and ablation across 5 configurations.
    """
    if not os.path.exists(queries_path):
        queries_path = os.path.join(PROJECT_ROOT, queries_path)
    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Queries file not found at {queries_path}")

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(queries)} evaluation queries from {queries_path}")

    # Pre-load models
    print("Pre-loading models and database connections...")
    _ = get_client()
    _ = get_embedder()
    _ = get_bm25_search()
    _ = get_reranker()
    load_mappings()
    print("All models and mappings loaded!\n")

    results_dir = os.path.join(PROJECT_ROOT, "evaluation", "results")
    os.makedirs(results_dir, exist_ok=True)

    configs = [
        ("A. Dense only", retrieve_dense_only),
        ("B. BM25 only", retrieve_bm25_only),
        ("C. Dense + BM25 (concat)", retrieve_dense_bm25_concat),
        ("D. Dense + BM25 + RRF", retrieve_dense_bm25_rrf),
        ("E. Dense + BM25 + RRF + Reranker", retrieve_full_pipeline),
    ]

    all_eval_data: Dict[str, List[Dict[str, Any]]] = {c[0]: [] for c in configs}
    detailed_per_query_results = []
    failures_to_record = []

    print("=" * 85)
    print("RUNNING ABLATION STUDY ACROSS 5 CONFIGURATIONS")
    print("=" * 85)

    for idx, qentry in enumerate(queries, start=1):
        query = qentry["query"]
        relevant = qentry.get("relevant_skus", [])
        qtype = qentry.get("query_type", "unknown")

        stage_debug = {}
        query_record = {
            "query_idx": idx,
            "query": query,
            "query_type": qtype,
            "relevant_skus": relevant,
        }

        # Track per-stage ranks for failure analysis
        dense_skus = []
        bm25_skus = []
        rrf_skus = []
        reranker_skus = []

        for config_name, func in configs:
            doc_ids, retrieved_skus, lat_ms = func(query, top_k=5)
            metrics = compute_metrics(relevant, retrieved_skus, latency_ms=lat_ms)

            entry_data = {
                "query": query,
                "relevant_skus": relevant,
                "retrieved_skus": retrieved_skus,
                "retrieved_doc_ids": doc_ids,
                "hit_at_5": int(metrics["hit_rate"]),
                "reciprocal_rank": metrics["mrr"],
                "recall@5": metrics["recall@5"],
                "recall@10": metrics["recall@10"],
                "precision@5": metrics["precision@5"],
                "ndcg@5": metrics["ndcg@5"],
                "latency_ms": lat_ms,
            }
            all_eval_data[config_name].append(entry_data)
            stage_debug[config_name] = retrieved_skus

            if "Dense only" in config_name:
                dense_skus = retrieved_skus
            elif "BM25 only" in config_name:
                bm25_skus = retrieved_skus
            elif "RRF" in config_name and "Reranker" not in config_name:
                rrf_skus = retrieved_skus
            elif "Reranker" in config_name:
                reranker_skus = retrieved_skus
                query_record["retrieved_skus"] = retrieved_skus
                query_record["hit_at_5"] = int(metrics["hit_rate"])
                query_record["reciprocal_rank"] = metrics["mrr"]
                query_record["recall@5"] = metrics["recall@5"]
                query_record["ndcg@5"] = metrics["ndcg@5"]
                query_record["latency_ms"] = lat_ms

        detailed_per_query_results.append(query_record)

        # Print progress every 10 queries
        if idx % 10 == 0 or idx == len(queries):
            print(f"Evaluated {idx}/{len(queries)} queries...", flush=True)

        # Verbose side-by-side debugging
        if verbose:
            print(f"\n[{idx}/{len(queries)}] Query: {query}")
            print(f"  Type: {qtype} | Relevant: {relevant}")
            print(f"  Dense:    {dense_skus}")
            print(f"  BM25:     {bm25_skus}")
            print(f"  RRF:      {rrf_skus}")
            print(f"  Reranker: {reranker_skus}")

        # Failure Analysis check on final pipeline (Config E)
        is_hit = query_record.get("hit_at_5", 0) == 1
        if not is_hit and relevant:
            # Find rank in each stage
            d_rank = next((i + 1 for i, s in enumerate(dense_skus) if s in relevant), None)
            b_rank = next((i + 1 for i, s in enumerate(bm25_skus) if s in relevant), None)
            r_rank = next((i + 1 for i, s in enumerate(rrf_skus) if s in relevant), None)
            re_rank = next((i + 1 for i, s in enumerate(reranker_skus) if s in relevant), None)

            reason = classify_failure(
                query, relevant, reranker_skus, d_rank, b_rank, r_rank, re_rank
            )
            failures_to_record.append({
                "query": query,
                "query_type": qtype,
                "expected_sku": ", ".join(relevant),
                "retrieved_skus": ", ".join(reranker_skus),
                "dense_rank": d_rank if d_rank else "None",
                "bm25_rank": b_rank if b_rank else "None",
                "rrf_rank": r_rank if r_rank else "None",
                "reranker_rank": re_rank if re_rank else "None",
                "failure_reason": reason,
            })

    # Save detailed per-query JSON and CSV
    json_out = os.path.join(results_dir, "evaluation_results.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(detailed_per_query_results, f, indent=2)

    csv_out = os.path.join(results_dir, "evaluation_results.csv")
    import csv
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query_idx", "query", "query_type", "relevant_skus", "retrieved_skus",
            "hit_at_5", "reciprocal_rank", "recall@5", "ndcg@5", "latency_ms"
        ])
        writer.writeheader()
        for row in detailed_per_query_results:
            row_copy = dict(row)
            row_copy["relevant_skus"] = ";".join(row_copy["relevant_skus"])
            row_copy["retrieved_skus"] = ";".join(row_copy["retrieved_skus"])
            writer.writerow(row_copy)

    # Save Failure Analysis CSV
    failure_csv_out = os.path.join(results_dir, "failure_analysis.csv")
    with open(failure_csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query", "query_type", "expected_sku", "retrieved_skus",
            "dense_rank", "bm25_rank", "rrf_rank", "reranker_rank", "failure_reason"
        ])
        writer.writeheader()
        for frow in failures_to_record:
            writer.writerow(frow)

    # -------------------------------------------------------------
    # Step 4 Ablation Summary Table
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print("RETRIEVAL ABLATION RESULTS SUMMARY")
    print("=" * 85)

    header = f"{'Experiment':<35} | {'Recall@5':<10} | {'MRR':<10} | {'NDCG@5':<10} | {'Latency (ms)':<12}"
    print(header)
    print("-" * len(header))

    ablation_summary = []
    for config_name, _ in configs:
        cdata = all_eval_data[config_name]
        mean_r5 = sum(d["recall@5"] for d in cdata) / len(cdata)
        mean_mrr = sum(d["reciprocal_rank"] for d in cdata) / len(cdata)
        mean_ndcg = sum(d["ndcg@5"] for d in cdata) / len(cdata)
        mean_lat = sum(d["latency_ms"] for d in cdata) / len(cdata)

        print(f"{config_name:<35} | {mean_r5:<10.4f} | {mean_mrr:<10.4f} | {mean_ndcg:<10.4f} | {mean_lat:<12.1f}")
        ablation_summary.append({
            "Experiment": config_name,
            "Recall@5": round(mean_r5, 4),
            "MRR": round(mean_mrr, 4),
            "NDCG@5": round(mean_ndcg, 4),
            "Latency": round(mean_lat, 1),
        })

    print("=" * 85)

    # Print Failure Category Breakdown
    print("\n" + "=" * 85)
    print(f"FAILURE ANALYSIS BREAKDOWN ({len(failures_to_record)} non-irrelevant queries missed in Top 5)")
    print("=" * 85)
    cat_counts = Counter(f["failure_reason"] for f in failures_to_record)
    for cat, cnt in cat_counts.most_common():
        print(f"  {cat:<30}: {cnt}")
    if not failures_to_record:
        print("  Zero retrieval failures recorded! 100% Hit Rate.")
    print("=" * 85)
    print(f"Results saved to:\n  - {csv_out}\n  - {json_out}\n  - {failure_csv_out}\n")

    return ablation_summary, failures_to_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CatalogIQ Retrieval Evaluation Runner")
    parser.add_argument("--queries", default="evaluation/queries.jsonl", help="Path to queries.jsonl")
    parser.add_argument("--verbose", action="store_true", help="Print side-by-side stage ranking per query")
    args = parser.parse_args()

    run_evaluation(queries_path=args.queries, verbose=args.verbose)
