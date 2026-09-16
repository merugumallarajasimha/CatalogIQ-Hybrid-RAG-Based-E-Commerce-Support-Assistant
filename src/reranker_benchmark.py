import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sentence_transformers import CrossEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:6333")

import evaluation
import search
from retrieval_metrics import compute_metrics

BENCHMARK_EMBEDDING_MODEL = os.getenv(
    "BENCHMARK_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
search.EMBEDDING_MODEL = BENCHMARK_EMBEDDING_MODEL
search._embedder = None

DEFAULT_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
    "mixedbread-ai/mxbai-rerank-base-v1",
]


def load_queries(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_rrf_cache(
    queries: List[Dict[str, Any]],
    output_path: Path,
) -> List[Dict[str, Any]]:
    evaluation.load_mappings()
    cache = []
    for index, query_entry in enumerate(queries, start=1):
        query = query_entry["query"]
        doc_ids, _, _ = evaluation.retrieve_dense_bm25_rrf(query, top_k=15)
        docs = []
        for doc_id in doc_ids:
            doc = evaluation._DOC_CACHE.get(doc_id)
            if doc is None:
                doc = evaluation.get_document(doc_id)
                if doc:
                    evaluation._DOC_CACHE[doc_id] = doc
            if doc:
                docs.append({
                    "doc_id": doc_id,
                    "content": str(doc.get("content", "")),
                })
        cache.append({
            "query_idx": index,
            "query": query,
            "relevant_skus": list(query_entry.get("relevant_skus", [])),
            "doc_ids": doc_ids,
            "docs": docs,
        })
        if index % 10 == 0 or index == len(queries):
            print(f"Prepared RRF candidates {index}/{len(queries)}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2)
    return cache


def percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction) - 1))
    return ordered[index]


def benchmark_model(
    model_name: str,
    cache: List[Dict[str, Any]],
    output_path: Path,
    warmup_queries: int,
) -> Dict[str, Any]:
    print(f"Loading reranker: {model_name}", flush=True)
    load_start = time.perf_counter()
    model = CrossEncoder(model_name, max_length=512)
    model_load_seconds = time.perf_counter() - load_start
    print(f"Reranker loaded in {model_load_seconds:.3f}s", flush=True)

    for entry in cache[:warmup_queries]:
        pairs = [(entry["query"], doc["content"]) for doc in entry["docs"]]
        if pairs:
            model.predict(pairs, batch_size=len(pairs))

    per_query = []
    latencies = []
    for entry in cache:
        pairs = [(entry["query"], doc["content"]) for doc in entry["docs"]]
        start = time.perf_counter()
        scores = model.predict(pairs, batch_size=len(pairs))
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        scored = [
            (doc["doc_id"], float(score))
            for doc, score in zip(entry["docs"], scores)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        top_ids = [doc_id for doc_id, _ in scored[:5]]
        retrieved_skus = evaluation.extract_skus_from_results(top_ids)
        metrics = compute_metrics(entry["relevant_skus"], retrieved_skus, latency_ms=latency_ms)
        per_query.append({
            "query_idx": entry["query_idx"],
            "query": entry["query"],
            "relevant_skus": entry["relevant_skus"],
            "candidate_doc_ids": entry["doc_ids"],
            "reranked_doc_ids": top_ids,
            "retrieved_skus": retrieved_skus,
            **metrics,
        })
        if entry["query_idx"] % 10 == 0 or entry["query_idx"] == len(cache):
            print(f"{model_name}: evaluated {entry['query_idx']}/{len(cache)}", flush=True)

    summary = {
        "model": model_name,
        "retrieval_collection": evaluation.COLLECTION_NAME,
        "retrieval_embedder": search.EMBEDDING_MODEL,
        "rrf_top_k": 15,
        "rerank_top_k": 5,
        "queries": len(cache),
        "warmup_queries": warmup_queries,
        "model_load_seconds": round(model_load_seconds, 3),
        "recall@5": sum(row["recall@5"] for row in per_query) / len(per_query),
        "mrr": sum(row["mrr"] for row in per_query) / len(per_query),
        "ndcg@5": sum(row["ndcg@5"] for row in per_query) / len(per_query),
        "rerank_latency_ms": sum(latencies) / len(latencies),
        "rerank_latency_p50_ms": percentile(latencies, 0.50),
        "rerank_latency_p95_ms": percentile(latencies, 0.95),
    }
    result = {**summary, "per_query": per_query}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    del model
    gc.collect()
    return summary


def print_table(rows: List[Dict[str, Any]]) -> None:
    print("\nRERANKER BENCHMARK")
    print("=" * 110)
    print(f"{'Model':<42} {'R@5':>8} {'MRR':>8} {'NDCG@5':>10} {'Mean ms':>10} {'p95 ms':>10}")
    print("-" * 110)
    for row in rows:
        print(
            f"{row['model']:<42} "
            f"{row['recall@5']:>8.4f} "
            f"{row['mrr']:>8.4f} "
            f"{row['ndcg@5']:>10.4f} "
            f"{row['rerank_latency_ms']:>10.2f} "
            f"{row['rerank_latency_p95_ms']:>10.2f}"
        )
    print("=" * 110)


def main() -> None:
    parser = argparse.ArgumentParser(description="CatalogIQ reranker benchmark")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--queries", default="evaluation/queries.jsonl")
    parser.add_argument("--output-dir", default="evaluation/benchmarks/rerankers")
    parser.add_argument("--cache-path")
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    queries_path = Path(args.queries)
    if not queries_path.is_absolute():
        queries_path = PROJECT_ROOT / queries_path
    output_dir = Path(args.output_dir)
    queries = load_queries(queries_path)
    cache_path = Path(args.cache_path) if args.cache_path else output_dir / "rrf_top15_candidates.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            cache = json.load(handle)
        print(f"Reusing {len(cache)} cached RRF candidate sets", flush=True)
    else:
        cache = build_rrf_cache(queries, cache_path)
    rows = []
    for model_name in args.models:
        slug = model_name.replace("/", "_").replace("-", "_")
        rows.append(
            benchmark_model(
                model_name=model_name,
                cache=cache,
                output_path=output_dir / f"{slug}.json",
                warmup_queries=args.warmup,
            )
        )
    print_table(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
