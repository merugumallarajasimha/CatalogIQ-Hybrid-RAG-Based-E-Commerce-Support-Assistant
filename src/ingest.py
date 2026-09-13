import os
import uuid
import zlib
from typing import List, Dict, Any, Optional
from collections import Counter

# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
# pyrefly: ignore [missing-import]
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
)
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
COLLECTION_NAME = "support_docs"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

_embedder: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    global _embedder

    if _embedder is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        print(
            f"Model loaded. Embedding dimension: "
            f"{_embedder.get_sentence_embedding_dimension()}"
        )

    return _embedder


def token_to_index(token: str) -> int:
    """
    Convert a token into a deterministic uint32 sparse-vector index.

    The same token always gets the same index across
    ingestion and search.
    """
    return zlib.crc32(token.encode("utf-8"))


def compute_sparse_vector(text: str) -> SparseVector:
    words = text.lower().split()

    if not words:
        return SparseVector(indices=[], values=[])

    word_counts = Counter(words)
    total_words = sum(word_counts.values())

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


def create_collection(client: QdrantClient) -> None:
    collections = client.get_collections().collections

    exists = any(c.name == COLLECTION_NAME for c in collections)

    if exists:
        info = client.get_collection(COLLECTION_NAME)

        print(f"Collection '{COLLECTION_NAME}' already exists:")
        print(f"  Status: {info.status}")
        print(f"  Vectors: {info.config.params.vectors}")
        print(f"  Sparse vectors: {info.config.params.sparse_vectors}")

        return

    print(f"Creating collection '{COLLECTION_NAME}'...")

    client.create_collection(
        collection_name=COLLECTION_NAME,

        vectors_config={
            "dense_vector": VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        },

        sparse_vectors_config={
            "sparse_vector": SparseVectorParams()
        }
    )

    print(
        f"Collection created with dense_vector "
        f"({EMBEDDING_DIM}d) and sparse_vector"
    )


def prepare_points(
    chunks: List[Dict[str, Any]],
    embedder: SentenceTransformer
) -> List[PointStruct]:

    points = []

    for chunk in chunks:

        content = chunk.get("content", "") or chunk.get("text", "")

        if not content:
            continue

        dense_vec = embedder.encode(content).tolist()

        sparse_vec = compute_sparse_vector(content)

        payload = {
            "doc_id": chunk.get("id", str(uuid.uuid4())),
            "sku": chunk.get("sku", "unknown"),
            "part_numbers": chunk.get("part_numbers", []),
            "product_name": chunk.get("product_name", "unknown"),
            "category": chunk.get("category", "unknown"),
            "doc_type": chunk.get("doc_type", "unknown"),
            "content": content,
            "source_file": chunk.get("source_file", ""),
            "chunk_index": chunk.get("chunk_index", 0),
            "total_chunks": chunk.get("total_chunks", 1)
        }

        point = PointStruct(
            id=str(uuid.uuid4()),

            vector={
                "dense_vector": dense_vec,
                "sparse_vector": sparse_vec
            },

            payload=payload
        )

        points.append(point)

    return points


def ingest_chunks(
    chunks: List[Dict[str, Any]],
    batch_size: int = 64
) -> None:

    client = QdrantClient(
        url=QDRANT_URL,
        timeout=60
    )

    create_collection(client)

    embedder = get_embedder()

    points = prepare_points(
        chunks,
        embedder
    )

    print(
        f"Uploading {len(points)} points "
        f"in batches of {batch_size}..."
    )

    for i in range(0, len(points), batch_size):

        batch = points[i:i + batch_size]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )

        print(
            f"  Uploaded "
            f"{min(i + batch_size, len(points))}/{len(points)}"
        )

    info = client.get_collection(COLLECTION_NAME)

    print(
        f"Ingestion complete. "
        f"Collection now has {info.points_count} points."
    )


def run_ingestion(
    dataset_dir: str = "data",
    chunk_size: int = 5
) -> None:

    from .load_dataset import load_dataset
    from .chunker import chunk_records

    print("Loading dataset...")

    records = load_dataset(dataset_dir)

    print(
        f"Chunking {len(records)} records..."
    )

    chunks = chunk_records(
        records,
        chunk_size
    )

    print(
        f"Created {len(chunks)} chunks"
    )

    ingest_chunks(chunks)


if __name__ == "__main__":
    run_ingestion()