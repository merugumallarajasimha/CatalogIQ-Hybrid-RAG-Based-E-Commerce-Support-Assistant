import argparse
import csv
import io
import json
import os
import re
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:6333")

import evaluation
import generation
import reranker
import search
from retrieval_metrics import compute_metrics

DEFAULT_MODELS = [
    "auto",
    "groq/openai/gpt-oss-120b",
    "groq/llama-3.3-70b-versatile",
]
CITATION_PATTERN = re.compile(r"\[Doc:\s*([^\]]+)\]")


def load_queries(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_failure_map() -> Dict[str, str]:
    path = PROJECT_ROOT / "evaluation" / "results" / "failure_analysis.csv"
    if not path.exists():
        return {}
    result = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row.get("query", "")] = row.get("failure_reason", "")
    return result


def extract_cited_doc_ids(answer: str) -> List[str]:
    return CITATION_PATTERN.findall(answer)


def check_no_answer_appropriateness(
    query: str,
    answer: str,
    relevant_skus: List[str],
    failure_by_query: Dict[str, str],
) -> Dict[str, Any]:
    answer_lower = answer.lower()
    decline_phrases = [
        "does not provide",
        "doesn't provide",
        "not provide",
        "not available",
        "not in the document",
        "not in the documentation",
        "no information",
        "cannot answer",
        "unable to answer",
        "don't know",
        "do not know",
        "insufficient information",
        "not mentioned",
        "not found",
    ]
    is_decline = any(phrase in answer_lower for phrase in decline_phrases)
    should_decline = len(relevant_skus) == 0 or query in failure_by_query
    correct = is_decline == should_decline
    if correct:
        decline_type = "correct"
    elif is_decline:
        decline_type = "false_decline"
    else:
        decline_type = "missed_decline"
    return {
        "is_decline": is_decline,
        "should_decline": should_decline,
        "correct_no_answer": correct,
        "decline_type": decline_type,
    }


def check_unsupported_claims(answer: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    sentences = re.split(r"[.!?]+", answer)
    sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 10]
    cited = extract_cited_doc_ids(answer)
    factual_indicators = [
        "is",
        "has",
        "supports",
        "includes",
        "features",
        "specifies",
        "requires",
        "uses",
        "contains",
        "provides",
        "offers",
        "rated",
        "capacity",
        "weight",
        "size",
        "dimension",
        "voltage",
        "torque",
        "dpi",
        "impedance",
        "warranty",
        "angle",
        "speed",
        "range",
    ]
    unsupported = 0
    total_claims = 0
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if "doc:" in sentence_lower:
            continue
        if any(indicator in sentence_lower for indicator in factual_indicators):
            total_claims += 1
    unsupported = max(0, total_claims - len(cited))
    return {
        "total_sentences": len(sentences),
        "total_citations": len(cited),
        "factual_claims_est": total_claims,
        "unsupported_claims_est": unsupported,
        "citation_coverage": len(cited) / max(1, total_claims) if total_claims > 0 else 1.0,
    }


def check_faithfulness(answer: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    doc_content = " ".join([doc.get("content", "") for doc in docs])
    doc_content_lower = doc_content.lower()
    answer_sentences = re.split(r"[.!?]+", answer)
    hallucinated = 0
    supported = 0
    for sentence in answer_sentences:
        sentence = sentence.strip()
        if len(sentence) < 15 or "doc:" in sentence.lower():
            continue
        words = set(re.findall(r"\b[A-Z0-9-]{3,}\b", sentence))
        if words:
            matches = sum(1 for word in words if word.lower() in doc_content_lower)
            if matches == 0:
                hallucinated += 1
            else:
                supported += 1
    score = supported / max(1, supported + hallucinated)
    return {
        "supported_statements": supported,
        "potentially_hallucinated": hallucinated,
        "faithfulness_score": score,
    }


def check_answer_relevance(query: str, answer: str) -> Dict[str, Any]:
    query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
    answer_words = set(re.findall(r"\b\w{3,}\b", answer.lower()))
    stop = {
        "the",
        "and",
        "for",
        "with",
        "what",
        "how",
        "does",
        "is",
        "are",
        "not",
        "you",
        "your",
    }
    query_words = query_words - stop
    answer_words = answer_words - stop
    overlap = query_words & answer_words
    return {
        "query_keywords": len(query_words),
        "answer_keywords": len(answer_words),
        "overlap": len(overlap),
        "relevance_score": round(len(overlap) / max(1, len(query_words)), 3),
    }


def build_retrieval_cache(
    queries: List[Dict[str, Any]],
    output_path: Path,
) -> List[Dict[str, Any]]:
    search.EMBEDDING_MODEL = os.getenv(
        "BENCHMARK_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    search._embedder = None
    reranker.RERANKER_MODEL = os.getenv(
        "BENCHMARK_RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    reranker._reranker = None
    evaluation.load_mappings()
    cache = []
    for index, query_entry in enumerate(queries, start=1):
        query = query_entry["query"]
        with redirect_stdout(io.StringIO()):
            doc_ids, _, retrieval_ms = evaluation.retrieve_full_pipeline(query, top_k=5)
        docs = []
        for doc_id in doc_ids:
            doc = evaluation._DOC_CACHE.get(doc_id)
            if doc is None:
                doc = evaluation.get_document(doc_id)
                if doc:
                    evaluation._DOC_CACHE[doc_id] = doc
            if doc:
                docs.append({
                    "doc_id": doc.get("doc_id", doc_id),
                    "content": str(doc.get("content", "")),
                    "sku": doc.get("sku", "unknown"),
                    "part_numbers": doc.get("part_numbers", []),
                    "product_name": doc.get("product_name", "unknown"),
                    "category": doc.get("category", "unknown"),
                })
        cache.append({
            "query_idx": index,
            "query": query,
            "query_type": query_entry.get("query_type", "unknown"),
            "relevant_skus": list(query_entry.get("relevant_skus", [])),
            "doc_ids": doc_ids,
            "docs": docs,
            "retrieval_latency_ms": round(retrieval_ms, 3),
        })
        if index % 10 == 0 or index == len(queries):
            print(f"Prepared generation context {index}/{len(queries)}", flush=True)
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
    output_dir: Path,
    failure_by_query: Dict[str, str],
    warmup_queries: int,
    limit: Optional[int],
) -> Dict[str, Any]:
    generation.OMNIROUTE_MODEL = model_name
    generation._client = None
    model_cache = cache if limit is None else cache[:limit]
    for entry in model_cache[:warmup_queries]:
        with redirect_stdout(io.StringIO()):
            generation.generate_answer(entry["query"], entry["docs"])

    per_query = []
    latencies = []
    errors = 0
    for entry in model_cache:
        start = time.perf_counter()
        with redirect_stdout(io.StringIO()):
            answer = generation.generate_answer(entry["query"], entry["docs"])
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        if answer.startswith("[ERROR]"):
            errors += 1
        valid_doc_ids = [doc["doc_id"] for doc in entry["docs"]]
        citation_info = generation.validate_citations(answer, valid_doc_ids)
        no_answer = check_no_answer_appropriateness(
            entry["query"],
            answer,
            entry["relevant_skus"],
            failure_by_query,
        )
        unsupported = check_unsupported_claims(answer, entry["docs"])
        faithfulness = check_faithfulness(answer, entry["docs"])
        relevance = check_answer_relevance(entry["query"], answer)
        retrieved_skus = evaluation.extract_skus_from_results(valid_doc_ids)
        per_query.append({
            "query_idx": entry["query_idx"],
            "query": entry["query"],
            "query_type": entry["query_type"],
            "relevant_skus": entry["relevant_skus"],
            "retrieved_doc_ids": valid_doc_ids,
            "retrieved_skus": retrieved_skus,
            "answer": answer,
            "generation_time_ms": round(latency_ms, 3),
            "answer_length": len(answer),
            "is_error": answer.startswith("[ERROR]"),
            "citations_found": citation_info["total_citations_found"],
            "valid_citations": citation_info["has_valid_citation"],
            "all_citations_valid": citation_info["all_valid"],
            "invalid_citations": list(citation_info["invalid_citations"]),
            "no_answer_correct": no_answer["correct_no_answer"],
            "no_answer_type": no_answer["decline_type"],
            "unsupported_claims_est": unsupported["unsupported_claims_est"],
            "factual_claims_est": unsupported["factual_claims_est"],
            "citation_coverage": unsupported["citation_coverage"],
            "faithfulness_score": faithfulness["faithfulness_score"],
            "potentially_hallucinated": faithfulness["potentially_hallucinated"],
            "relevance_score": relevance["relevance_score"],
        })
        if entry["query_idx"] % 10 == 0 or entry["query_idx"] == len(model_cache):
            print(f"{model_name}: generated {entry['query_idx']}/{len(model_cache)}", flush=True)

    relevant = [row for row in per_query if row["query_type"] != "irrelevant"]
    total_unsupported = sum(row["unsupported_claims_est"] for row in relevant)
    total_claims = sum(row["factual_claims_est"] for row in relevant)
    summary = {
        "model": model_name,
        "retrieval_collection": search.COLLECTION_NAME,
        "retrieval_embedder": search.EMBEDDING_MODEL,
        "reranker": reranker.RERANKER_MODEL,
        "queries": len(per_query),
        "relevant_queries": len(relevant),
        "irrelevant_queries": len(per_query) - len(relevant),
        "warmup_queries": warmup_queries,
        "generation_errors": errors,
        "citation_validity_rate": sum(1 for row in relevant if row["all_citations_valid"]) / max(1, len(relevant)),
        "citation_completeness": sum(row["citation_coverage"] for row in relevant) / max(1, len(relevant)),
        "answer_relevance": sum(row["relevance_score"] for row in relevant) / max(1, len(relevant)),
        "faithfulness": sum(row["faithfulness_score"] for row in relevant) / max(1, len(relevant)),
        "no_answer_accuracy": sum(1 for row in per_query if row["no_answer_correct"]) / max(1, len(per_query)),
        "unsupported_claim_rate": total_unsupported / max(1, total_claims),
        "unsupported_claims_per_query": total_unsupported / max(1, len(relevant)),
        "generation_latency_ms": sum(latencies) / max(1, len(latencies)),
        "generation_latency_p50_ms": percentile(latencies, 0.50),
        "generation_latency_p95_ms": percentile(latencies, 0.95),
    }
    result = {**summary, "per_query": per_query}
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = model_name.replace("/", "_").replace(":", "_").replace("-", "_")
    with (output_dir / f"{slug}.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    with (output_dir / f"{slug}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_query[0].keys()))
        writer.writeheader()
        writer.writerows(per_query)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def print_table(rows: List[Dict[str, Any]]) -> None:
    print("\nLLM GENERATION BENCHMARK")
    print("=" * 150)
    print(f"{'Model':<34} {'Cite valid':>10} {'Cite complete':>13} {'Relevance':>10} {'Faithful':>10} {'No-answer':>10} {'Unsup rate':>11} {'Gen ms':>10} {'p95 ms':>10}")
    print("-" * 150)
    for row in rows:
        print(
            f"{row['model']:<34} "
            f"{row['citation_validity_rate']:>10.4f} "
            f"{row['citation_completeness']:>13.4f} "
            f"{row['answer_relevance']:>10.4f} "
            f"{row['faithfulness']:>10.4f} "
            f"{row['no_answer_accuracy']:>10.4f} "
            f"{row['unsupported_claim_rate']:>11.4f} "
            f"{row['generation_latency_ms']:>10.2f} "
            f"{row['generation_latency_p95_ms']:>10.2f}"
        )
    print("=" * 150)


def main() -> None:
    parser = argparse.ArgumentParser(description="CatalogIQ LLM generation benchmark")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--queries", default="evaluation/queries.jsonl")
    parser.add_argument("--output-dir", default="evaluation/benchmarks/llms")
    parser.add_argument("--retrieval-cache")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    queries_path = Path(args.queries)
    if not queries_path.is_absolute():
        queries_path = PROJECT_ROOT / queries_path
    output_dir = Path(args.output_dir)
    queries = load_queries(queries_path)
    cache_path = Path(args.retrieval_cache) if args.retrieval_cache else output_dir / "retrieval_context.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            cache = json.load(handle)
        print(f"Reusing {len(cache)} cached retrieval contexts", flush=True)
    else:
        cache = build_retrieval_cache(queries, cache_path)
    failure_by_query = load_failure_map()
    rows = []
    for model_name in args.models:
        rows.append(
            benchmark_model(
                model_name=model_name,
                cache=cache,
                output_dir=output_dir,
                failure_by_query=failure_by_query,
                warmup_queries=args.warmup,
                limit=args.limit,
            )
        )
    print_table(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
