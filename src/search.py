import os
import zlib
import time
import sys
import json
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

from dotenv import load_dotenv

load_dotenv()

# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
# pyrefly: ignore [missing-import]
from qdrant_client.models import SparseVector

# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


COLLECTION_NAME = "support_docs"

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

DENSE_TOP_K = 20
BM25_TOP_K = 20
RRF_TOP_K = 15
FINAL_TOP_K = 5


_client: Optional[QdrantClient] = None
_embedder: Optional[SentenceTransformer] = None


def get_client() -> QdrantClient:

    global _client

    if _client is None:
        _client = QdrantClient(
            url=QDRANT_URL,
            timeout=30
        )

    return _client


def get_embedder() -> SentenceTransformer:

    global _embedder

    if _embedder is None:
        _embedder = SentenceTransformer(
            EMBEDDING_MODEL
        )

    return _embedder


def token_to_index(token: str) -> int:
    """
    Must be identical to the function used during ingestion.
    """

    return zlib.crc32(
        token.encode("utf-8")
    )


def compute_sparse_vector(
    text: str
) -> SparseVector:

    words = text.lower().split()

    if not words:
        return SparseVector(
            indices=[],
            values=[]
        )

    word_counts = Counter(words)

    total_words = sum(
        word_counts.values()
    )

    indices = []
    values = []

    for word, count in word_counts.items():

        index = token_to_index(word)

        score = count / total_words

        indices.append(index)
        values.append(score)

    return SparseVector(
        indices=indices,
        values=values
    )


def sparse_search(
    query: str,
    top_k: int = 20
) -> List[Tuple[str, int]]:

    client = get_client()

    sparse_vec = compute_sparse_vector(query)

    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=sparse_vec,

        using="sparse_vector",

        limit=top_k,

        with_payload=["doc_id"]
    )

    return [
        (
            hit.payload["doc_id"],
            rank + 1
        )

        for rank, hit
        in enumerate(results.points)

        if hit.payload
        and "doc_id" in hit.payload
    ]


def dense_search(
    query: str,
    top_k: int = 20
) -> List[Tuple[str, int]]:

    client = get_client()

    embedder = get_embedder()

    query_vec = embedder.encode(
        query
    ).tolist()

    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vec,

        using="dense_vector",

        limit=top_k,

        with_payload=["doc_id"]
    )

    return [
        (
            hit.payload["doc_id"],
            rank + 1
        )

        for rank, hit
        in enumerate(results.points)

        if hit.payload
        and "doc_id" in hit.payload
    ]


def hybrid_search(
    query: str,
    top_k: int = RRF_TOP_K,
    k: int = RRF_TOP_K,
) -> List[Dict[str, Any]]:

    print("\n" + "=" * 60)
    print("HYBRID SEARCH DEBUG")
    print("=" * 60)

    # -----------------------------
    # Sparse search
    # -----------------------------
    start = time.perf_counter()

    sparse_results = sparse_search(
        query,
        DENSE_TOP_K
    )

    sparse_time = (
        time.perf_counter() - start
    ) * 1000

    print(
        f"Sparse search: {sparse_time:.2f} ms"
    )

    # -----------------------------
    # Dense search
    # -----------------------------
    start = time.perf_counter()

    dense_results = dense_search(
        query,
        DENSE_TOP_K
    )

    dense_time = (
        time.perf_counter() - start
    ) * 1000

    print(
        f"Dense search: {dense_time:.2f} ms"
    )

    # -----------------------------
    # BM25 search (offline/local)
    # -----------------------------
    start = time.perf_counter()

    try:
        from bm25_search import BM25Search

        bm25 = BM25Search()
        records_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "normalized_records.json",
        )
        if os.path.exists(records_path):
            with open(records_path, encoding="utf-8", errors="replace") as f:
                normalized_records = json.load(f)
            bm25.build_from_records(normalized_records)
        bm25_results_raw = bm25.search(query, top_k=BM25_TOP_K)
        bm25_results = [doc_id for doc_id, _ in bm25_results_raw]
        bm25_time = (time.perf_counter() - start) * 1000
        print(f"BM25 search: {bm25_time:.2f} ms")
    except Exception as bm25_error:
        bm25_results = []
        bm25_time = 0.0
        print(f"BM25 search: SKIPPED ({bm25_error})")

    # -----------------------------
    # RRF
    # -----------------------------
    start = time.perf_counter()

    sparse_ranks = {
        doc_id: rank
        for doc_id, rank in sparse_results
    }

    dense_ranks = {
        doc_id: rank
        for doc_id, rank in dense_results
    }

    all_doc_ids = (
        set(sparse_ranks.keys())
        |
        set(dense_ranks.keys())
        |
        set(bm25_results)
    )

    fused = []

    for doc_id in all_doc_ids:

        score = 0.0

        if doc_id in sparse_ranks:
            score += 1.0 / (
                k + sparse_ranks[doc_id]
            )

        if doc_id in dense_ranks:
            score += 1.0 / (
                k + dense_ranks[doc_id]
            )

        if doc_id in bm25_results:
            bm25_rank = bm25_results.index(doc_id) + 1
            score += 1.0 / (k + bm25_rank)

        retrieval_sources = []
        if doc_id in sparse_ranks:
            retrieval_sources.append("sparse")
        if doc_id in dense_ranks:
            retrieval_sources.append("dense")
        if doc_id in bm25_results:
            retrieval_sources.append("bm25")

        fused.append({
            "doc_id": doc_id,
            "rrf_score": score,
            "dense_rank": dense_ranks.get(doc_id),
            "bm25_rank": (
                bm25_results.index(doc_id) + 1
                if doc_id in bm25_results
                else None
            ),
            "retrieval_sources": retrieval_sources,
        })

    fused.sort(
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    rrf_time = (
        time.perf_counter() - start
    ) * 1000

    print(
        f"RRF fusion: {rrf_time:.2f} ms"
    )

    print(
        f"Total hybrid search: "
        f"{sparse_time + dense_time + bm25_time + rrf_time:.2f} ms"
    )

    print("=" * 60)

    return fused[:RRF_TOP_K]


def get_document(
    doc_id: str
) -> Optional[Dict[str, Any]]:

    client = get_client()

    results = client.scroll(

        collection_name=COLLECTION_NAME,

        scroll_filter={
            "must": [
                {
                    "key": "doc_id",
                    "match": {
                        "value": doc_id
                    }
                }
            ]
        },

        limit=1,

        with_payload=True
    )

    points, next_page = results

    if points:

        return points[0].payload

    return None


if __name__ == "__main__":

    print(
        "Testing search module..."
    )

    print(
        "Run test_individual_search.py "
        "after Qdrant is running and data is ingested."
    )