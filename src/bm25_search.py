"""
BM25 Search Module
Real BM25 scoring for SKUs, ASINs, part numbers, product names,
technical specs, error codes, and rare keywords.
"""
import re
import json
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import numpy as np
import rank_bm25

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from extract_ids import extract_ids


BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r'[a-z0-9\-_]+', text)
    return [t for t in tokens if len(t) >= 2]


class BM25Search:
    def __init__(self):
        self.corpus: List[str] = []
        self.doc_ids: List[str] = []
        self.bm25_model: Optional[rank_bm25.BM25Okapi] = None
        self._built = False

    def build_from_records(self, records: List[Dict[str, Any]]) -> None:
        self.corpus = []
        self.doc_ids = []
        for record in records:
            doc_id = str(record.get('sku', record.get('doc_id', record.get('part_number', ''))))
            text = record.get('content', '') or record.get('text', '') or ''
            self.corpus.append(text)
            self.doc_ids.append(doc_id)

        if not self.corpus:
            return

        tokenized = [tokenize(doc) for doc in self.corpus]
        self.bm25_model = rank_bm25.BM25Okapi(tokenized, k1=BM25_K1, b=BM25_B)
        self._built = True

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        if not self._built or not self.corpus:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25_model.get_scores(query_tokens)

        scored = []
        for i, score in enumerate(scores):
            if score > 0:
                scored.append((self.doc_ids[i], float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_with_extract_ids(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        ids = extract_ids(query)
        boosted_query = query
        for sku in ids.get('skus', []):
            boosted_query += f" {sku}"
        for pn in ids.get('part_numbers', []):
            boosted_query += f" {pn}"
        return self.search(boosted_query, top_k)


def compute_crc32_tf_idf(query: str, document: str) -> float:
    query_tokens = set(tokenize(query))
    doc_tokens = tokenize(document)
    if not doc_tokens:
        return 0.0

    doc_counter = Counter(doc_tokens)
    total = len(doc_tokens)

    score = 0.0
    for token in query_tokens:
        if token in doc_counter:
            tf = doc_counter[token]
            score += tf / total

    return score


if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

    with open(os.path.join(DATA_DIR, 'normalized_records.json'), encoding='utf-8', errors='replace') as f:
        records = json.load(f)

    print("Building BM25 index from normalized records...")
    bm25 = BM25Search()
    bm25.build_from_records(records)
    print(f"Indexed {len(bm25.corpus)} documents")
    print()

    test_queries = [
        "B07DP4LM9H",
        "S-4012-SCRW",
        "CHAIR-ERG-X99",
        "B-3301-BLT",
        "SN-8801-SNS",
    ]

    print("=" * 80)
    print("BM25 SEARCH TEST")
    print("=" * 80)
    for query in test_queries:
        results = bm25.search(query, top_k=5)
        print(f"\nQuery: {query}")
        for doc_id, score in results[:5]:
            print(f"  {doc_id}: {score:.4f}")
