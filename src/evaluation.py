import json
import os
import sys
import time
import traceback
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from retrieval_metrics import (
    load_evaluation_queries,
    load_normalized_records,
    compute_metrics,
)
from bm25_search import BM25Search
from extract_ids import extract_ids


def bm25_retrieval(
    query: str,
    bm25: BM25Search,
    top_k: int = 10,
) -> List[str]:
    results = bm25.search(query, top_k=top_k)
    return [doc_id for doc_id, _ in results]


def exact_retrieval(
    query: str,
    records: List[Dict],
    top_k: int = 10,
) -> List[str]:
    from exact_search import exact_search
    results = exact_search(query, top_k=top_k)
    return [doc_id for doc_id, _ in results]


def hybrid_retrieval(
    query: str,
    bm25: BM25Search,
    records: List[Dict],
    top_k: int = 15,
) -> List[str]:
    bm25_results = [doc_id for doc_id, _ in bm25.search(query, top_k=20)]

    exact_ids = extract_ids(query)
    all_identifiers = []
    for id_type in ("skus", "asins", "part_numbers"):
        all_identifiers.extend(exact_ids.get(id_type, []))

    if all_identifiers:
        exact_results = exact_retrieval(query, records, top_k=5)
        return exact_results[:5]

    seen = set()
    combined = []
    for doc_id in bm25_results:
        if doc_id not in seen:
            combined.append(doc_id)
            seen.add(doc_id)

    return combined[:top_k]


def run_ablation(
    queries: List[Dict],
    records: List[Dict],
) -> Dict[str, List[Dict]]:
    bm25 = BM25Search()
    bm25.build_from_records(records)

    configs = [
        "bm25_only",
        "exact_only",
        "hybrid_bm25",
        "hybrid_exact",
    ]

    all_results = {c: [] for c in configs}

    for entry in queries:
        query = entry["query"]
        relevant = entry.get("relevant_skus", [])

        for config in configs:
            start = time.perf_counter()
            try:
                if config == "bm25_only":
                    retrieved = bm25_retrieval(query, bm25, top_k=5)
                elif config == "exact_only":
                    retrieved = exact_retrieval(query, records, top_k=5)
                elif config == "hybrid_bm25":
                    retrieved = hybrid_retrieval(query, bm25, records, top_k=5)
                elif config == "hybrid_exact":
                    exact_ids = extract_ids(query)
                    all_ids = []
                    for t in ("skus", "asins", "part_numbers"):
                        all_ids.extend(exact_ids.get(t, []))
                    if all_ids:
                        retrieved = exact_retrieval(query, records, top_k=5)
                    else:
                        retrieved = bm25_retrieval(query, bm25, top_k=5)
                else:
                    retrieved = []

                elapsed = time.perf_counter() - start
                metrics = compute_metrics(query, relevant, retrieved)
                metrics["latency_ms"] = elapsed * 1000
                metrics["retrieved"] = retrieved
                all_results[config].append(metrics)

            except Exception as e:
                elapsed = time.perf_counter() - start
                all_results[config].append({
                    "recall@5": 0.0,
                    "recall@10": 0.0,
                    "precision@5": 0.0,
                    "mrr": 0.0,
                    "ndcg@5": 0.0,
                    "hit_rate": 0.0,
                    "latency_ms": elapsed * 1000,
                    "error": str(e),
                })

    return all_results


def aggregate_results(
    all_results: Dict[str, List[Dict]],
    queries: List[Dict],
) -> Dict[str, Dict[str, float]]:
    aggregated = {}
    for config, results in all_results.items():
        if not results:
            continue
        n = len(results)
        agg = {}
        for metric in ["recall@5", "recall@10", "precision@5", "mrr", "ndcg@5", "hit_rate", "latency_ms"]:
            values = [r.get(metric, 0.0) for r in results]
            agg[metric] = sum(values) / len(values) if values else 0.0
        aggregated[config] = agg
    return aggregated


if __name__ == "__main__":
    queries = load_evaluation_queries()
    records = load_normalized_records()

    print(f"Loaded {len(queries)} evaluation queries")
    print(f"Loaded {len(records)} normalized records")
    print()

    print("Running ablation study (4 configurations)...")
    print("This may take a few minutes...")
    print()

    all_results = run_ablation(queries, records)
    aggregated = aggregate_results(all_results, queries)

    print("=" * 100)
    print("RETRIEVAL ABLATION RESULTS")
    print("=" * 100)
    print()

    config_labels = {
        "bm25_only": "BM25 Only",
        "exact_only": "Exact Only",
        "hybrid_bm25": "Dense + BM25 + RRF",
        "hybrid_exact": "Exact + BM25 Hybrid",
    }

    metric_labels = {
        "recall@5": "Recall@5",
        "recall@10": "Recall@10",
        "precision@5": "Precision@5",
        "mrr": "MRR",
        "ndcg@5": "NDCG@5",
        "hit_rate": "Hit Rate",
        "latency_ms": "Latency (ms)",
    }

    metric_keys = ["recall@5", "recall@10", "precision@5", "mrr", "ndcg@5", "hit_rate", "latency_ms"]

    header = f"{'Metric':<15}" + "".join(f"{config_labels.get(c, c):>20}" for c in aggregated.keys())
    print(header)
    print("-" * len(header))

    for mk in metric_keys:
        row = f"{metric_labels.get(mk, mk):<15}"
        for config in aggregated.keys():
            val = aggregated[config].get(mk, 0.0)
            if mk == "latency_ms":
                row += f"{val:>20.1f}"
            else:
                row += f"{val:>20.4f}"
        print(row)

    print()
    print("=" * 100)

    for config, results in all_results.items():
        print(f"\n{config}: {len(results)} queries evaluated")
        for r in results[:3]:
            print(f"  Query: {r.get('retrieved', [])}")
