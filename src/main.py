import os
import sys
import time
from typing import Dict, Any, List, Optional

# Add the parent and current directory to Python path for seamless imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
ROOT_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
elif os.path.exists(ROOT_ENV_PATH):
    load_dotenv(ROOT_ENV_PATH)
else:
    load_dotenv()


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

try:
    from src.catalog_service import CatalogResult, answer_catalog_question
    from src.search import get_document
except ImportError:
    from catalog_service import CatalogResult, answer_catalog_question
    try:
        from search import get_document
    except ImportError:
        get_document = None


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
    out_of_scope: bool = False
    scope_reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str


# ============================================================
# 7. ENRICH SEARCH RESULTS WITH DOCUMENT CONTENT
# ============================================================

def enrich_candidates_with_content(
    candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if get_document is None:
        return candidates

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

    result = answer_catalog_question(request.question)

    if getattr(result, "error", None):
        status_code = getattr(result, "status_code", 500) or 500
        raise HTTPException(
            status_code=status_code,
            detail=result.error,
        )

    return AskResponse(
        answer=result.answer,
        sources=[
            SourceInfo(**source)
            for source in result.sources
        ],
        citation_check=CitationCheck(
            **result.citation_check
        ),
        timing_ms=TimingInfo(
            **result.timing_ms
        ),
        out_of_scope=getattr(result, "out_of_scope", False),
        scope_reason=getattr(result, "scope_reason", None),
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

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )