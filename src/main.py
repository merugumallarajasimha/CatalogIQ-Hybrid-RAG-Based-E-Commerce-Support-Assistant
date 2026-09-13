import os
import sys
import time
from typing import Dict, Any, List

# Add the src directory to Python path so we can import sibling modules
sys.path.insert(0, os.path.dirname(__file__))

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, field_validator


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(ENV_PATH)


# ============================================================
# 2. VALIDATE REQUIRED ENVIRONMENT VARIABLES
# ============================================================

REQUIRED_ENV_VARS = [
    "QDRANT_URL",
    "OMNIROUTE_API_KEY",
]

for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        raise RuntimeError(
            f"Missing required environment variable: {var}"
        )


# ============================================================
# 3. IMPORT RAG COMPONENTS
# ============================================================

from .search import hybrid_search, get_document
from .reranker import rerank
from .generation import generate_answer, validate_citations


# ============================================================
# 4. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Hybrid RAG Support API",
    description="Hybrid RAG-based E-Commerce Customer Support Assistant",
    version="1.0.0",
)


# ============================================================
# 5. REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")

        if len(v.strip()) < 3:
            raise ValueError(
                "Question must be at least 3 characters"
            )

        return v.strip()


# ============================================================
# 6. RESPONSE MODELS
# ============================================================

class SourceInfo(BaseModel):
    doc_id: str
    sku: str
    product_name: str


class CitationCheck(BaseModel):
    total_citations_found: int
    valid_citations: List[str]
    invalid_citations: List[str]
    has_valid_citation: bool
    all_valid: bool


class TimingInfo(BaseModel):
    sparse_dense_search_ms: float
    rerank_ms: float
    generation_ms: float
    total_ms: float


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    citation_check: CitationCheck
    timing_ms: TimingInfo


class HealthResponse(BaseModel):
    status: str


# ============================================================
# 7. ENRICH SEARCH RESULTS WITH DOCUMENT CONTENT
# ============================================================

def enrich_candidates_with_content(
    candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    enriched = []

    for candidate in candidates:
        doc_id = candidate.get("doc_id")

        if not doc_id:
            continue

        doc = get_document(doc_id)

        if doc:
            enriched_candidate = dict(candidate)

            enriched_candidate["content"] = doc.get(
                "content",
                ""
            )

            enriched_candidate["sku"] = doc.get(
                "sku",
                "unknown"
            )

            enriched_candidate["part_numbers"] = doc.get(
                "part_numbers",
                []
            )

            enriched_candidate["product_name"] = doc.get(
                "product_name",
                "unknown"
            )

            enriched_candidate["category"] = doc.get(
                "category",
                "unknown"
            )

            enriched.append(enriched_candidate)

    return enriched


# ============================================================
# 8. ASK ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=AskResponse
)
def ask(request: AskRequest) -> AskResponse:

    question = request.question

    total_start = time.perf_counter()

    try:

        # ----------------------------------------------------
        # STEP 1: HYBRID SEARCH
        # ----------------------------------------------------

        search_start = time.perf_counter()

        candidates = hybrid_search(
            question,
            top_k=20
        )

        search_elapsed = (
            time.perf_counter() - search_start
        ) * 1000

        if not candidates:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No candidates found — "
                    "check that the Qdrant collection "
                    "contains data"
                ),
            )

        # ----------------------------------------------------
        # STEP 2: GET DOCUMENT CONTENT
        # ----------------------------------------------------

        candidates = enrich_candidates_with_content(
            candidates
        )

        if not candidates:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Search returned results, but "
                    "documents could not be retrieved "
                    "from Qdrant"
                ),
            )

        # ----------------------------------------------------
        # STEP 3: RERANK
        # ----------------------------------------------------

        rerank_start = time.perf_counter()

        top_docs = rerank(
            question,
            candidates,
            top_n=5
        )

        rerank_elapsed = (
            time.perf_counter() - rerank_start
        ) * 1000

        if not top_docs:
            raise HTTPException(
                status_code=503,
                detail="Reranker returned no results",
            )

        # ----------------------------------------------------
        # STEP 4: GENERATE ANSWER
        # ----------------------------------------------------

        generation_start = time.perf_counter()

        answer = generate_answer(
            question,
            top_docs
        )

        generation_elapsed = (
            time.perf_counter() - generation_start
        ) * 1000

        # ----------------------------------------------------
        # STEP 5: VALIDATE CITATIONS
        # ----------------------------------------------------

        valid_doc_ids = [
            document["doc_id"]
            for document in top_docs
            if document.get("doc_id")
        ]

        citation_result = validate_citations(
            answer,
            valid_doc_ids
        )

        # ----------------------------------------------------
        # STEP 6: BUILD SOURCE INFORMATION
        # ----------------------------------------------------

        sources = []

        for document in top_docs:
            sources.append(
                SourceInfo(
                    doc_id=document.get(
                        "doc_id",
                        "unknown"
                    ),
                    sku=document.get(
                        "sku",
                        "unknown"
                    ),
                    product_name=document.get(
                        "product_name",
                        "unknown"
                    ),
                )
            )

        # ----------------------------------------------------
        # STEP 7: TOTAL TIMING
        # ----------------------------------------------------

        total_elapsed = (
            time.perf_counter() - total_start
        ) * 1000

        # ----------------------------------------------------
        # STEP 8: RETURN RESPONSE
        # ----------------------------------------------------

        return AskResponse(
            answer=answer,
            sources=sources,
            citation_check=CitationCheck(
                **citation_result
            ),
            timing_ms=TimingInfo(
                sparse_dense_search_ms=search_elapsed,
                rerank_ms=rerank_elapsed,
                generation_ms=generation_elapsed,
                total_ms=total_elapsed,
            ),
        )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except HTTPException:
        raise

    except ConnectionError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Dependency unavailable: "
                f"{type(e).__name__}: {e}"
            ),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Internal error: "
                f"{type(e).__name__}: {e}"
            ),
        )


# ============================================================
# 9. HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health() -> HealthResponse:

    return HealthResponse(
        status="ok"
    )


# ============================================================
# 10. RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    # pyrefly: ignore [missing-import]
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )