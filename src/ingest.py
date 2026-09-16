

import os
import uuid
import zlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

from dotenv import load_dotenv

# Load .env
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
)
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

NORMALIZED_FILE = DATA_DIR / "normalized_records.json"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_DIM = 384

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "support_docs"
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

# Start with only 1000 records.
# Set to 0 later if you want to ingest everything.
MAX_RECORDS = int(
    os.getenv(
        "MAX_INGEST_RECORDS",
        "1000"
    )
)

# Number of texts sent to the embedding model at once.
EMBED_BATCH_SIZE = int(
    os.getenv(
        "EMBED_BATCH_SIZE",
        "16"
    )
)

# Number of Qdrant points uploaded at once.
UPLOAD_BATCH_SIZE = int(
    os.getenv(
        "UPLOAD_BATCH_SIZE",
        "64"
    )
)


# Global embedding model
_embedder: Optional[SentenceTransformer] = None


# ============================================================
# EMBEDDING MODEL
# ============================================================

def get_embedder() -> SentenceTransformer:
    """
    Load the embedding model only once.
    """

    global _embedder

    if _embedder is None:

        print()
        print(
            f"Loading embedding model: {EMBEDDING_MODEL}"
        )

        _embedder = SentenceTransformer(
            EMBEDDING_MODEL
        )

        dimension = (
            _embedder
            .get_sentence_embedding_dimension()
        )

        print(
            f"Embedding model loaded."
        )

        print(
            f"Embedding dimension: {dimension}"
        )

        if dimension != EMBEDDING_DIM:

            raise ValueError(
                f"Expected embedding dimension "
                f"{EMBEDDING_DIM}, "
                f"but model returned "
                f"{dimension}."
            )

    return _embedder


# ============================================================
# SPARSE VECTOR
# ============================================================

def token_to_index(token: str) -> int:
    """
    Convert a token into a deterministic
    sparse-vector index.
    """

    return zlib.crc32(
        token.encode("utf-8")
    )


def compute_sparse_vector(
    text: str
) -> SparseVector:
    """
    Create a simple TF-based sparse vector.
    """

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

        values.append(
            float(score)
        )

    return SparseVector(
        indices=indices,
        values=values
    )


# ============================================================
# LOAD NORMALIZED DATA
# ============================================================

def load_normalized_records(
    filepath: Path = NORMALIZED_FILE,
    max_records: int = MAX_RECORDS
) -> List[Dict[str, Any]]:
    """
    Load records from normalized_records.json.

    We use the normalized JSON instead of reading the
    original Parquet files again.
    """

    if not filepath.exists():

        raise FileNotFoundError(
            f"\nNormalized dataset not found:\n"
            f"{filepath}\n\n"
            f"Run this first:\n"
            f".\\venv\\Scripts\\python.exe "
            f"-m src.load_dataset"
        )

    print()
    print(
        "Loading normalized records from:"
    )

    print(filepath)

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as file:

        records = json.load(file)

    if not isinstance(records, list):

        raise ValueError(
            "normalized_records.json "
            "must contain a list of records."
        )

    original_count = len(records)

    print(
        f"Total normalized records available: "
        f"{original_count}"
    )

    if max_records > 0:

        records = records[:max_records]

    print(
        f"Records selected for ingestion: "
        f"{len(records)}"
    )

    return records


# ============================================================
# QDRANT COLLECTION
# ============================================================

def create_collection(
    client: QdrantClient
) -> None:
    """
    Create the Qdrant collection if it doesn't exist.
    """

    collections = (
        client
        .get_collections()
        .collections
    )

    exists = any(
        collection.name == COLLECTION_NAME
        for collection in collections
    )

    if exists:

        info = client.get_collection(
            COLLECTION_NAME
        )

        print()
        print(
            f"Collection '{COLLECTION_NAME}' "
            f"already exists."
        )

        print(
            f"Status: {info.status}"
        )

        print(
            f"Points: {info.points_count}"
        )

        return

    print()
    print(
        f"Creating collection "
        f"'{COLLECTION_NAME}'..."
    )

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
        "Collection created successfully."
    )

    print(
        f"Dense vector dimension: "
        f"{EMBEDDING_DIM}"
    )

    print(
        "Sparse vector: enabled"
    )


# ============================================================
# TEXT EXTRACTION
# ============================================================

def get_record_text(
    record: Dict[str, Any]
) -> str:
    """
    Extract searchable text from a record.
    """

    content = (

        record.get("content")

        or record.get("text")

        or record.get("description")

        or record.get("product_name")

        or record.get("title")

        or ""
    )

    return str(content).strip()


# ============================================================
# PAYLOAD
# ============================================================

def create_payload(
    record: Dict[str, Any],
    content: str
) -> Dict[str, Any]:
    """
    Create the payload stored in Qdrant.
    """

    return {

        "doc_id": str(
            record.get("id")
            or record.get("doc_id")
            or uuid.uuid4()
        ),

        "sku": str(
            record.get("sku")
            or record.get("product_id")
            or "unknown"
        ),

        "part_numbers": record.get(
            "part_numbers",
            []
        ),

        "product_name": str(
            record.get("product_name")
            or record.get("title")
            or "unknown"
        ),

        "category": str(
            record.get("category")
            or "unknown"
        ),

        "doc_type": str(
            record.get("doc_type")
            or "product"
        ),

        "content": content,

        "source_file": str(
            record.get("source_file")
            or ""
        ),

        "chunk_index": int(
            record.get(
                "chunk_index",
                0
            )
            or 0
        ),

        "total_chunks": int(
            record.get(
                "total_chunks",
                1
            )
            or 1
        ),

        "confidence_flags": record.get(
            "confidence_flags",
            []
        ),

        "extraction_method": str(
            record.get("extraction_method")
            or "none"
        ),
    }


# ============================================================
# PREPARE POINTS
# ============================================================

def prepare_points(
    records: List[Dict[str, Any]],
    embedder: SentenceTransformer
) -> List[PointStruct]:
    """
    Convert records into Qdrant points.
    """

    valid_records = []

    texts = []

    for record in records:

        content = get_record_text(
            record
        )

        if not content:
            continue

        valid_records.append(
            record
        )

        texts.append(
            content
        )

    if not texts:

        print(
            "No valid text found."
        )

        return []

    print()
    print(
        f"Generating dense embeddings "
        f"for {len(texts)} records..."
    )

    dense_embeddings = embedder.encode(

        texts,

        batch_size=EMBED_BATCH_SIZE,

        show_progress_bar=True,

        convert_to_numpy=True,

        normalize_embeddings=True
    )

    points = []

    for index, (
        record,
        content
    ) in enumerate(
        zip(
            valid_records,
            texts
        )
    ):

        dense_vector = (
            dense_embeddings[index]
            .tolist()
        )

        sparse_vector = (
            compute_sparse_vector(
                content
            )
        )

        payload = create_payload(
            record,
            content
        )

        stable_key = (

            f"{payload['sku']}_"

            f"{payload['source_file']}_"

            f"{payload['chunk_index']}_"

            f"{index}"
        )

        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                stable_key
            )
        )

        point = PointStruct(

            id=point_id,

            vector={

                "dense_vector":
                    dense_vector,

                "sparse_vector":
                    sparse_vector
            },

            payload=payload
        )

        points.append(
            point
        )

    print()
    print(
        f"Prepared {len(points)} "
        f"Qdrant points."
    )

    return points


# ============================================================
# UPLOAD TO QDRANT
# ============================================================

def upload_points(
    client: QdrantClient,
    points: List[PointStruct]
) -> None:
    """
    Upload points to Qdrant in batches.
    """

    total_points = len(points)

    if total_points == 0:

        print(
            "No points to upload."
        )

        return

    print()
    print(
        f"Uploading {total_points} points "
        f"in batches of "
        f"{UPLOAD_BATCH_SIZE}..."
    )

    for start in range(
        0,
        total_points,
        UPLOAD_BATCH_SIZE
    ):

        end = min(
            start + UPLOAD_BATCH_SIZE,
            total_points
        )

        batch = points[
            start:end
        ]

        client.upsert(

            collection_name=COLLECTION_NAME,

            points=batch,

            wait=True
        )

        print(
            f"Uploaded "
            f"{end}/{total_points}"
        )

    print()
    print(
        "All points uploaded successfully."
    )


# ============================================================
# INGESTION PIPELINE
# ============================================================

def ingest_records(
    records: List[Dict[str, Any]]
) -> None:
    """
    Complete ingestion pipeline.
    """

    print()
    print(
        "Connecting to Qdrant..."
    )

    print(
        f"Qdrant URL: {QDRANT_URL}"
    )

    client = QdrantClient(
        url=QDRANT_URL,
        timeout=120
    )

    # Test connection
    client.get_collections()

    print(
        "Connected to Qdrant successfully."
    )

    create_collection(
        client
    )

    embedder = get_embedder()

    points = prepare_points(
        records,
        embedder
    )

    upload_points(
        client,
        points
    )

    info = client.get_collection(
        COLLECTION_NAME
    )

    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Points in collection: "
        f"{info.points_count}"
    )

    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Qdrant URL: "
        f"{QDRANT_URL}"
    )

    print("=" * 60)


# ============================================================
# RUN INGESTION
# ============================================================

def run_ingestion() -> None:
    """
    Main ingestion entry point.
    """

    records = load_normalized_records(
        filepath=NORMALIZED_FILE,
        max_records=MAX_RECORDS
    )

    print()
    print(
        f"Starting ingestion for "
        f"{len(records)} records..."
    )

    ingest_records(
        records
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_ingestion()