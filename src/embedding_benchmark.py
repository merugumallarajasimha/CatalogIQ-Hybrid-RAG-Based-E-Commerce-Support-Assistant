import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, SparseVectorParams, VectorParams
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from retrieval_metrics import compute_metrics

VALID_SKUS = {
    "CHAIR-ERG-X99",
    "DESK-STD-MOTO",
    "MON-ARM-DUAL",
    "KB-ERGO-SPLIT",
    "MOUSE-VERT-PRO",
    "HEADSET-WL-PRO",
}
DENSE_TOP_K = 20


def load_queries(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scroll_all_points(
    client: QdrantClient,
    collection_name: str,
) -> List[Any]:
    points: List[Any] = []
    offset = None
    while True:
        page, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(page)
        if offset is None:
            return points


def collection_names(client: QdrantClient) -> set:
    return {item.name for item in client.get_collections().collections}


def create_target_collection(
    client: QdrantClient,
    collection_name: str,
    dimension: int,
    replace: bool,
) -> None:
    names = collection_names(client)
    if collection_name in names:
        if not replace:
            raise RuntimeError(
                f"Collection {collection_name!r} already exists; pass --replace to rebuild it."
            )
        client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense_vector": VectorParams(
                size=dimension,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse_vector": SparseVectorParams()
        },
    )


def load_part_mapping() -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    path = PROJECT_ROOT / "data" / "parts_catalog.json"
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    for record in records:
        part_number = record.get("part_number")
        compatible = record.get("compatible_skus", [])
        if not part_number or not compatible:
            continue
        targets = [sku for sku in compatible if sku in VALID_SKUS]
        if targets:
            mapping[str(part_number)] = targets
    return mapping


def add_unique(values: List[str], seen: set, candidate: Any) -> None:
    if candidate in VALID_SKUS and candidate not in seen:
        values.append(candidate)
        seen.add(candidate)


def doc_ids_to_skus(
    doc_ids: List[str],
    payload_by_doc_id: Dict[str, Dict[str, Any]],
    part_mapping: Dict[str, List[str]],
) -> List[str]:
    values: List[str] = []
    seen = set()
    for doc_id in doc_ids:
        add_unique(values, seen, doc_id)
        payload = payload_by_doc_id.get(doc_id, {})
        if not payload:
            continue
        add_unique(values, seen, payload.get("sku"))
        for valid_sku in VALID_SKUS:
            if str(doc_id).startswith(valid_sku):
                add_unique(values, seen, valid_sku)
        identifiers = [doc_id, payload.get("sku")]
        identifiers.extend(payload.get("part_numbers", []) or [])
        for identifier in identifiers:
            if not identifier:
                continue
            add_unique(values, seen, identifier)
            for sku in part_mapping.get(str(identifier), []):
                add_unique(values, seen, sku)
    return values


def encode_batch(
    embedder: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> Tuple[Any, float]:
    start = time.perf_counter()
    vectors = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    elapsed = time.perf_counter() - start
    return vectors, elapsed


def measure_embedding_latency(
    model_name: str,
    source_collection: str,
    output_path: Path,
    batch_size: int,
) -> Dict[str, Any]:
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"), timeout=120)
    source_points = scroll_all_points(client, source_collection)
    if not source_points:
        raise RuntimeError(f"Source collection {source_collection!r} is empty.")
    print(f"Loading embedding model: {model_name}", flush=True)
    load_start = time.perf_counter()
    embedder = SentenceTransformer(model_name)
    model_load_seconds = time.perf_counter() - load_start
    dimension = int(embedder.get_sentence_embedding_dimension())
    texts = [str(point.payload.get("content", "")) for point in source_points]
    _, elapsed = encode_batch(embedder, texts, batch_size=batch_size)
    result = {
        "model": model_name,
        "source_collection": source_collection,
        "dimension": dimension,
        "points": len(source_points),
        "model_load_seconds": round(model_load_seconds, 3),
        "embedding_seconds": round(elapsed, 3),
        "embedding_ms_per_doc": round(elapsed * 1000 / len(source_points), 3),
        "batch_size": batch_size,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return result


def ingest_benchmark_collection(
    model_name: str,
    source_collection: str,
    target_collection: str,
    output_path: Path,
    batch_size: int,
    upload_batch_size: int,
    replace: bool,
) -> Dict[str, Any]:
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"), timeout=120)
    source_points = scroll_all_points(client, source_collection)
    if not source_points:
        raise RuntimeError(f"Source collection {source_collection!r} is empty.")

    print(f"Loading embedding model: {model_name}", flush=True)
    load_start = time.perf_counter()
    embedder = SentenceTransformer(model_name)
    model_load_seconds = time.perf_counter() - load_start
    dimension = int(embedder.get_sentence_embedding_dimension())
    print(f"Embedding dimension: {dimension}", flush=True)

    if target_collection not in collection_names(client) or replace:
        create_target_collection(client, target_collection, dimension, replace)
    target_info = client.get_collection(target_collection)
    target_dimension = int(
        target_info.config.params.vectors["dense_vector"].size
    )
    if target_dimension != dimension:
        raise RuntimeError(
            f"Existing collection has dimension {target_dimension}; expected {dimension}."
        )

    embedding_seconds = 0.0
    upload_seconds = 0.0
    ingest_start = time.perf_counter()
    uploaded = 0
    for start in range(0, len(source_points), upload_batch_size):
        batch_points = source_points[start:start + upload_batch_size]
        texts = [str(point.payload.get("content", "")) for point in batch_points]
        vectors, elapsed = encode_batch(embedder, texts, batch_size)
        embedding_seconds += elapsed
        points = []
        for point, vector in zip(batch_points, vectors):
            sparse = point.vector.get("sparse_vector")
            if sparse is None:
                raise RuntimeError(f"Point {point.id} has no sparse vector.")
            points.append(
                PointStruct(
                    id=point.id,
                    vector={
                        "dense_vector": vector.tolist(),
                        "sparse_vector": sparse,
                    },
                    payload=point.payload,
                )
            )
        upload_start = time.perf_counter()
        client.upsert(
            collection_name=target_collection,
            points=points,
            wait=True,
        )
        upload_seconds += time.perf_counter() - upload_start
        uploaded += len(points)
        print(f"Uploaded {uploaded}/{len(source_points)} points", flush=True)

    total_seconds = time.perf_counter() - ingest_start
    target_info = client.get_collection(target_collection)
    if target_info.points_count != len(source_points):
        raise RuntimeError(
            f"Target has {target_info.points_count} points; expected {len(source_points)}."
        )

    result = {
        "model": model_name,
        "source_collection": source_collection,
        "collection": target_collection,
        "dimension": dimension,
        "source_points": len(source_points),
        "target_points": target_info.points_count,
        "model_load_seconds": round(model_load_seconds, 3),
        "embedding_seconds": round(embedding_seconds, 3),
        "embedding_ms_per_doc": round(embedding_seconds * 1000 / len(source_points), 3),
        "upload_seconds": round(upload_seconds, 3),
        "total_ingest_seconds": round(total_seconds, 3),
        "batch_size": batch_size,
        "upload_batch_size": upload_batch_size,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return result


def evaluate_dense_only(
    model_name: str,
    collection_name: str,
    queries_path: Path,
    output_path: Path,
    warmup_count: int = 0,
) -> Dict[str, Any]:
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"), timeout=60)
    print(f"Loading embedding model: {model_name}", flush=True)
    embedder = SentenceTransformer(model_name)
    queries = load_queries(queries_path)
    part_mapping = load_part_mapping()
    source_points = scroll_all_points(client, collection_name)
    payload_by_doc_id = {
        str(point.payload.get("doc_id")): point.payload
        for point in source_points
        if point.payload and point.payload.get("doc_id")
    }

    for warmup_entry in queries[:warmup_count]:
        warmup_vector = embedder.encode(
            warmup_entry["query"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        client.query_points(
            collection_name=collection_name,
            query=warmup_vector.tolist(),
            using="dense_vector",
            limit=DENSE_TOP_K,
            with_payload=False,
        )

    per_query = []
    query_latencies = []
    for index, query_entry in enumerate(queries, start=1):
        query = query_entry["query"]
        relevant = list(query_entry.get("relevant_skus", []))
        start = time.perf_counter()
        query_vector = embedder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            using="dense_vector",
            limit=DENSE_TOP_K,
            with_payload=True,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        query_latencies.append(latency_ms)
        doc_ids = [
            hit.payload["doc_id"]
            for hit in response.points
            if hit.payload and hit.payload.get("doc_id")
        ]
        retrieved_skus = doc_ids_to_skus(doc_ids, payload_by_doc_id, part_mapping)
        metrics = compute_metrics(relevant, retrieved_skus, latency_ms=latency_ms)
        per_query.append({
            "query_idx": index,
            "query": query,
            "relevant_skus": relevant,
            "retrieved_doc_ids": doc_ids,
            "retrieved_skus": retrieved_skus,
            **metrics,
        })
        if index % 10 == 0 or index == len(queries):
            print(f"Evaluated {index}/{len(queries)} queries", flush=True)

    summary = {
        "model": model_name,
        "collection": collection_name,
        "queries": len(queries),
        "warmup_queries": warmup_count,
        "recall@5": sum(row["recall@5"] for row in per_query) / len(per_query),
        "recall@10": sum(row["recall@10"] for row in per_query) / len(per_query),
        "mrr": sum(row["mrr"] for row in per_query) / len(per_query),
        "ndcg@5": sum(row["ndcg@5"] for row in per_query) / len(per_query),
        "query_latency_ms": sum(query_latencies) / len(query_latencies),
        "query_latency_p50_ms": sorted(query_latencies)[len(query_latencies) // 2],
        "query_latency_p95_ms": sorted(query_latencies)[int(len(query_latencies) * 0.95) - 1],
    }
    result = {
        **summary,
        "per_query": per_query,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    print("\nDENSE-ONLY RETRIEVAL")
    print("=" * 100)
    print(f"{'Model':<42} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'NDCG@5':>10} {'Query ms':>10}")
    print("-" * 100)
    for row in rows:
        print(
            f"{row['model']:<42} "
            f"{row['recall@5']:>8.4f} "
            f"{row['recall@10']:>8.4f} "
            f"{row['mrr']:>8.4f} "
            f"{row['ndcg@5']:>10.4f} "
            f"{row['query_latency_ms']:>10.2f}"
        )
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="CatalogIQ dense embedder benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("ingest", "evaluate", "both", "measure-embedding"), default="both")
    parser.add_argument("--source-collection", default="support_docs")
    parser.add_argument("--collection")
    parser.add_argument("--queries", default="evaluation/queries.jsonl")
    parser.add_argument("--output-dir", default="evaluation/benchmarks")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--upload-batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    queries_path = Path(args.queries)
    if not queries_path.is_absolute():
        queries_path = PROJECT_ROOT / queries_path
    collection_name = args.collection or f"support_docs_{args.model.replace('/', '_').replace('-', '_')}"
    if collection_name == args.source_collection:
        slug = "baseline"
    else:
        slug = collection_name.removeprefix("support_docs_") or "model"
    ingest_path = output_dir / f"{slug}_ingest.json"
    eval_path = output_dir / f"{slug}_dense_only.json"

    if args.mode == "measure-embedding":
        measure_embedding_latency(
            model_name=args.model,
            source_collection=collection_name,
            output_path=output_dir / f"{slug}_embedding_latency.json",
            batch_size=args.batch_size,
        )
        return

    if args.mode in ("ingest", "both"):
        ingest_benchmark_collection(
            model_name=args.model,
            source_collection=args.source_collection,
            target_collection=collection_name,
            output_path=ingest_path,
            batch_size=args.batch_size,
            upload_batch_size=args.upload_batch_size,
            replace=args.replace,
        )
    if args.mode in ("evaluate", "both"):
        summary = evaluate_dense_only(
            model_name=args.model,
            collection_name=collection_name,
            queries_path=queries_path,
            output_path=eval_path,
            warmup_count=args.warmup,
        )
        print_summary_table([summary])


if __name__ == "__main__":
    main()
