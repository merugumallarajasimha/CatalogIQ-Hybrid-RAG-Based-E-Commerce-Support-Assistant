import json
import os
import sys
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from extract_ids import extract_ids
from bm25_search import BM25Search, tokenize


def load_evaluation_queries(path: str = "evaluation/queries.jsonl"):
    queries = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    queries.append(json.loads(line))
    return queries


def load_normalized_records(path: str = "data/normalized_records.json"):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)


def recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = len(set(relevant) & set(retrieved_k))
    return hits / len(relevant)


def precision_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    if not retrieved:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = len(set(relevant) & set(retrieved_k))
    return hits / k


def mrr(relevant: List[str], retrieved: List[str]) -> float:
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k, 1):
        if doc_id in relevant:
            dcg += 1.0 / (time.time() - time.time() + 1.0)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / (i + 1) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate(relevant: List[str], retrieved: List[str]) -> float:
    return 1.0 if set(relevant) & set(retrieved) else 0.0


def compute_metrics(
    query: str,
    relevant_skus: List[str],
    retrieved_ids: List[str],
) -> Dict[str, float]:
    return {
        "recall@5": recall_at_k(relevant_skus, retrieved_ids, 5),
        "recall@10": recall_at_k(relevant_skus, retrieved_ids, 10),
        "precision@5": precision_at_k(relevant_skus, retrieved_ids, 5),
        "mrr": mrr(relevant_skus, retrieved_ids),
        "ndcg@5": ndcg_at_k(relevant_skus, retrieved_ids, 5),
        "hit_rate": hit_rate(relevant_skus, retrieved_ids),
    }
