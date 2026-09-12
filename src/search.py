import os
import zlib

from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "support_docs"

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


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
    top_k: int = 20,
    k: int = 60
) -> List[Dict[str, Any]]:

    sparse_results = sparse_search(
        query,
        top_k * 2
    )

    dense_results = dense_search(
        query,
        top_k * 2
    )

    sparse_ranks = {
        doc_id: rank
        for doc_id, rank
        in sparse_results
    }

    dense_ranks = {
        doc_id: rank
        for doc_id, rank
        in dense_results
    }

    all_doc_ids = (
        set(sparse_ranks.keys())
        |
        set(dense_ranks.keys())
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

        fused.append({
            "doc_id": doc_id,
            "combined_score": score,
            "rank_in_sparse":
                sparse_ranks.get(doc_id),
            "rank_in_dense":
                dense_ranks.get(doc_id)
        })

    fused.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )

    return fused[:top_k]


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