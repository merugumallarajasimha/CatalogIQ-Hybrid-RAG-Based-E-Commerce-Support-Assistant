"""
CatalogIQ Retrieval Metrics Module
Pure, independently testable functions for evaluation:
Recall@k, Precision@k, MRR, NDCG@k, Hit Rate, and latency.
"""
import math
import time
from typing import Dict, List, Any, Optional, Set


def recall_at_k(relevant: List[str], retrieved: List[str], k: int = 5) -> float:
    """
    Calculate Recall@k: fraction of relevant items retrieved in top k.
    Pure function: input (relevant, retrieved, k) -> float in [0.0, 1.0].
    """
    if not relevant or k <= 0:
        return 0.0
    rel_set: Set[str] = set(relevant)
    retrieved_k = retrieved[:k]
    hits = len(rel_set & set(retrieved_k))
    return hits / len(rel_set)


def precision_at_k(relevant: List[str], retrieved: List[str], k: int = 5) -> float:
    """
    Calculate Precision@k: fraction of top k retrieved items that are relevant.
    Pure function: input (relevant, retrieved, k) -> float in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = len(set(relevant) & set(retrieved_k))
    return hits / k


def mrr(relevant: List[str], retrieved: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR): 1 / rank of first relevant item (1-indexed).
    Returns 0.0 if no relevant items are retrieved.
    """
    if not relevant or not retrieved:
        return 0.0
    rel_set: Set[str] = set(relevant)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in rel_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: List[str], retrieved: List[str], k: int = 5) -> float:
    """
    Calculate NDCG@k with standard binary relevance and logarithmic discount:
    DCG@k = sum_{i=1}^k (rel_i / log2(i + 1))
    IDCG@k = sum_{i=1}^{min(|relevant|, k)} (1 / log2(i + 1))
    NDCG@k = DCG@k / IDCG@k
    """
    if not relevant or not retrieved or k <= 0:
        return 0.0
    rel_set: Set[str] = set(relevant)
    retrieved_k = retrieved[:k]

    dcg = 0.0
    for i, doc in enumerate(retrieved_k, start=1):
        if doc in rel_set:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0.0 else 0.0


def hit_rate(relevant: List[str], retrieved: List[str], k: int = 5) -> float:
    """
    Calculate Hit Rate@k: 1.0 if at least one relevant item appears in top k, else 0.0.
    """
    if not relevant or not retrieved or k <= 0:
        return 0.0
    rel_set: Set[str] = set(relevant)
    retrieved_k = retrieved[:k]
    return 1.0 if bool(rel_set & set(retrieved_k)) else 0.0


def compute_metrics(
    relevant: List[str],
    retrieved: List[str],
    latency_ms: float = 0.0
) -> Dict[str, float]:
    """
    Compute full suite of retrieval metrics for a single query.
    """
    return {
        "recall@5": recall_at_k(relevant, retrieved, k=5),
        "recall@10": recall_at_k(relevant, retrieved, k=10),
        "precision@5": precision_at_k(relevant, retrieved, k=5),
        "mrr": mrr(relevant, retrieved),
        "ndcg@5": ndcg_at_k(relevant, retrieved, k=5),
        "hit_rate": hit_rate(relevant, retrieved, k=5),
        "latency_ms": latency_ms,
    }


def run_unit_tests():
    """
    Unit test against 3 hand-worked examples.
    """
    print("=" * 70)
    print("RUNNING RETRIEVAL METRICS UNIT TESTS (3 HAND-WORKED EXAMPLES)")
    print("=" * 70)

    # Example 1: Single relevant at rank 2
    rel1 = ["SKU-A"]
    ret1 = ["SKU-B", "SKU-A", "SKU-C", "SKU-D", "SKU-E"]
    m1 = compute_metrics(rel1, ret1)
    print("\n--- Example 1: Single relevant at rank 2 ---")
    print(f"Relevant:  {rel1}")
    print(f"Retrieved: {ret1}")
    for k, v in m1.items():
        if k != "latency_ms":
            print(f"  {k:<12}: {v:.4f}")
    assert math.isclose(m1["recall@5"], 1.0), f"Expected 1.0, got {m1['recall@5']}"
    assert math.isclose(m1["recall@10"], 1.0), f"Expected 1.0, got {m1['recall@10']}"
    assert math.isclose(m1["precision@5"], 0.2), f"Expected 0.2, got {m1['precision@5']}"
    assert math.isclose(m1["mrr"], 0.5), f"Expected 0.5, got {m1['mrr']}"
    assert math.isclose(m1["ndcg@5"], 1.0 / math.log2(3)), f"Expected {1.0/math.log2(3):.4f}, got {m1['ndcg@5']}"
    assert math.isclose(m1["hit_rate"], 1.0), f"Expected 1.0, got {m1['hit_rate']}"

    # Example 2: Multiple relevant with partial retrieval at ranks 2 and 4
    rel2 = ["SKU-A", "SKU-B", "SKU-C"]
    ret2 = ["SKU-X", "SKU-B", "SKU-Y", "SKU-A", "SKU-Z", "SKU-W", "SKU-C"]
    m2 = compute_metrics(rel2, ret2)
    print("\n--- Example 2: Multiple relevant with hits at rank 2 & 4 ---")
    print(f"Relevant:  {rel2}")
    print(f"Retrieved: {ret2}")
    for k, v in m2.items():
        if k != "latency_ms":
            print(f"  {k:<12}: {v:.4f}")
    assert math.isclose(m2["recall@5"], 2.0 / 3.0), f"Expected 0.6667, got {m2['recall@5']}"
    assert math.isclose(m2["recall@10"], 1.0), f"Expected 1.0, got {m2['recall@10']}"
    assert math.isclose(m2["precision@5"], 0.4), f"Expected 0.4, got {m2['precision@5']}"
    assert math.isclose(m2["mrr"], 0.5), f"Expected 0.5, got {m2['mrr']}"
    expected_dcg = (1.0 / math.log2(3)) + (1.0 / math.log2(5))
    expected_idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3)) + (1.0 / math.log2(4))
    assert math.isclose(m2["ndcg@5"], expected_dcg / expected_idcg), f"NDCG mismatch: {m2['ndcg@5']} vs {expected_dcg/expected_idcg}"
    assert math.isclose(m2["hit_rate"], 1.0), f"Expected 1.0, got {m2['hit_rate']}"

    # Example 3: Total miss
    rel3 = ["SKU-TARGET"]
    ret3 = ["SKU-1", "SKU-2", "SKU-3", "SKU-4", "SKU-5", "SKU-6"]
    m3 = compute_metrics(rel3, ret3)
    print("\n--- Example 3: Complete miss ---")
    print(f"Relevant:  {rel3}")
    print(f"Retrieved: {ret3}")
    for k, v in m3.items():
        if k != "latency_ms":
            print(f"  {k:<12}: {v:.4f}")
    assert math.isclose(m3["recall@5"], 0.0), f"Expected 0.0, got {m3['recall@5']}"
    assert math.isclose(m3["recall@10"], 0.0), f"Expected 0.0, got {m3['recall@10']}"
    assert math.isclose(m3["precision@5"], 0.0), f"Expected 0.0, got {m3['precision@5']}"
    assert math.isclose(m3["mrr"], 0.0), f"Expected 0.0, got {m3['mrr']}"
    assert math.isclose(m3["ndcg@5"], 0.0), f"Expected 0.0, got {m3['ndcg@5']}"
    assert math.isclose(m3["hit_rate"], 0.0), f"Expected 0.0, got {m3['hit_rate']}"

    print("\n" + "=" * 70)
    print("ALL 3 UNIT TESTS PASSED WITH 100% PRECISION!")
    print("=" * 70)


if __name__ == "__main__":
    run_unit_tests()
