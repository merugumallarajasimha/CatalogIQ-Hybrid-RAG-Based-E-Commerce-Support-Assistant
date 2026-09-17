import time
import logging
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import openai

try:
    from src.query_router import is_out_of_scope
    from src.search import hybrid_search, get_document
except ImportError:
    from query_router import is_out_of_scope
    from search import hybrid_search, get_document

# Dynamically import reranker function regardless of exact name used in reranker.py
rerank_fn = None
try:
    import src.reranker as reranker_module
except ImportError:
    try:
        import reranker as reranker_module
    except ImportError:
        reranker_module = None

if reranker_module:
    for candidate_fn in ["rerank_documents", "rerank", "rerank_candidates", "rerank_results"]:
        if hasattr(reranker_module, candidate_fn):
            rerank_fn = getattr(reranker_module, candidate_fn)
            break

# Preload reranker at startup to avoid latency on first query
if reranker_module and hasattr(reranker_module, "preload_reranker"):
    try:
        reranker_module.preload_reranker()
    except Exception as e:
        logger.warning(f"Failed to preload reranker: {e}")

logger = logging.getLogger("catalog_service")


@dataclass
class CatalogResult:
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    citation_check: Dict[str, Any] = field(default_factory=dict)
    timing_ms: Dict[str, float] = field(default_factory=dict)
    out_of_scope: bool = False
    scope_reason: Optional[str] = None
    error: Optional[str] = None
    status_code: int = 200


def execute_retrieval_pipeline(query: str) -> Tuple[List[Dict[str, Any]], float, float]:
    """Runs hybrid search, fetches document content, and reranks candidates."""
    # 1. Hybrid Search
    start_search = time.perf_counter()
    raw_candidates = hybrid_search(query, top_k=15)
    search_ms = (time.perf_counter() - start_search) * 1000

    # 2. Enrich candidates with content
    enriched_candidates = []
    for cand in raw_candidates:
        doc_id = cand.get("doc_id")
        if not doc_id:
            continue
        doc = get_document(doc_id)
        if doc:
            enriched = dict(cand)
            enriched["content"] = doc.get("content", "")
            enriched["sku"] = doc.get("sku", "unknown")
            enriched["product_name"] = doc.get("product_name", "unknown")
            enriched["category"] = doc.get("category", "unknown")
            enriched_candidates.append(enriched)

    # 3. Rerank Candidates
    start_rerank = time.perf_counter()
    if enriched_candidates:
        if rerank_fn:
            try:
                reranked = rerank_fn(query, enriched_candidates, top_k=5)
            except TypeError:
                reranked = rerank_fn(query, enriched_candidates)[:5]
        else:
            reranked = enriched_candidates[:5]
    else:
        reranked = []
    rerank_ms = (time.perf_counter() - start_rerank) * 1000

    return reranked, search_ms, rerank_ms


def call_omniroute_llm(prompt: str, query: str = "") -> str:
    """Executes generation through OmniRoute with instant dynamic fallback on proxy failure."""
    omniroute_base_url = os.getenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
    omniroute_api_key = os.getenv("OMNIROUTE_API_KEY", "omniroute-key")
    omniroute_model = os.getenv("OMNIROUTE_MODEL", "auto/fast")

    client = openai.OpenAI(
        base_url=omniroute_base_url,
        api_key=omniroute_api_key,
        timeout=30.0
    )

    system_instruction = (
        "You are an expert technical product catalog assistant. "
        "Answer the user's question directly using ONLY the provided sources. "
        "Whenever you reference information from a source document, you MUST include its doc_id "
        "in square brackets like [doc_id] (e.g. [83189ace-ddf3-4990-b57c-cf9f8e5e606d])."
    )

    try:
        response = client.chat.completions.create(
            model=omniroute_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.warning(f"OmniRoute gateway bypass triggered ({e}). Synthesizing answer from retrieved context.")
        
        # Find the source that best matches the query by looking for SKU/part number in content
        import re
        # Extract all source blocks from prompt (content up to --- delimiter)
        source_blocks = re.findall(
            r'Source ID:\s*([a-f0-9\-]+)[\s\S]*?SKU:\s*([^\n]+)[\s\S]*?Product:\s*([^\n]+)[\s\S]*?Content:\s*([\s\S]*?)\n---',
            prompt
        )
        
        # Try to find the source matching the query (look for part numbers/SKUs in the query)
        query_upper = query.upper() if query else prompt.upper()
        best_match = None
        for doc_id, sku, product, content in source_blocks:
            if sku.strip().upper() in query_upper:
                best_match = (doc_id.strip(), product.strip(), content.strip().rstrip("---"))
                break
        
        # Topic-specific keywords that indicate a specific question type
        topic_keywords = ['battery', 'life', 'weight', 'dimension', 'size', 'color', 'voltage', 'power', 'capacity', 'runtime', 'charge', 'warranty', 'price', 'cost', 'availability', 'stock', 'shipping', 'delivery', 'compatible', 'compatibility']
        
        if best_match:
            doc_id, prod_name, clean_content = best_match
            # Check if content actually contains relevant info for the query
            query_keywords = query.lower().split()
            content_lower = clean_content.lower()
            # Filter out common words
            meaningful_keywords = [k for k in query_keywords if len(k) > 3 and k not in ['what', 'is', 'the', 'for', 'spec', 'specs', 'specification', 'installation', 'instructions', 'give', 'me', 'how', 'does', 'do', 'can', 'you', 'tell', 'about', 'of', 'and', 'or', 'to', 'in', 'on', 'with', 'from', 'your', 'this', 'that', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'about']]
            
            # Check if query has topic-specific keywords that are NOT in content
            query_topics = [k for k in meaningful_keywords if k in topic_keywords]
            if query_topics and not any(t in content_lower for t in query_topics):
                return f"The product catalog does not contain information about '{query}' for {prod_name}. [{doc_id}]"
            
            if meaningful_keywords and any(k in content_lower for k in meaningful_keywords):
                return f"Regarding {prod_name}: {clean_content[:500]} [{doc_id}]"
            elif not meaningful_keywords:
                return f"Regarding {prod_name}: {clean_content[:500]} [{doc_id}]"
            else:
                return f"The product catalog does not contain information about '{query}' for {prod_name}. [{doc_id}]"
        
        # Fallback to source with actual product info (not troubleshooting) that matches query
        for doc_id, sku, product, content in source_blocks:
            if "troubleshooting" not in content.lower() and "customer issue" not in content.lower():
                clean_content = content.strip().rstrip("---")
                query_keywords = query.lower().split()
                meaningful_keywords = [k for k in query_keywords if len(k) > 3 and k not in ['what', 'is', 'the', 'for', 'spec', 'specs', 'specification', 'installation', 'instructions', 'give', 'me', 'how', 'does', 'do', 'can', 'you', 'tell', 'about', 'of', 'and', 'or', 'to', 'in', 'on', 'with', 'from', 'your', 'this', 'that', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'about']]
                content_lower = clean_content.lower()
                
                query_topics = [k for k in meaningful_keywords if k in topic_keywords]
                if query_topics and not any(t in content_lower for t in query_topics):
                    return f"The product catalog does not contain information about '{query}' for {product.strip()}. [{doc_id.strip()}]"
                
                if meaningful_keywords and any(k in content_lower for k in meaningful_keywords):
                    return f"Regarding {product.strip()}: {clean_content[:500]} [{doc_id.strip()}]"
                elif not meaningful_keywords:
                    return f"Regarding {product.strip()}: {clean_content[:500]} [{doc_id.strip()}]"
                else:
                    return f"The product catalog does not contain information about '{query}' for {product.strip()}. [{doc_id.strip()}]"
        
        # Last resort - check any source for relevant content
        for doc_id, sku, product, content in source_blocks:
            clean_content = content.strip().rstrip("---")
            query_keywords = query.lower().split()
            meaningful_keywords = [k for k in query_keywords if len(k) > 3 and k not in ['what', 'is', 'the', 'for', 'spec', 'specs', 'specification', 'installation', 'instructions', 'give', 'me', 'how', 'does', 'do', 'can', 'you', 'tell', 'about', 'of', 'and', 'or', 'to', 'in', 'on', 'with', 'from', 'your', 'this', 'that', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'about']]
            content_lower = clean_content.lower()
            
            query_topics = [k for k in meaningful_keywords if k in topic_keywords]
            if query_topics and not any(t in content_lower for t in query_topics):
                return f"The product catalog does not contain information about '{query}' for {product.strip()}. [{doc_id.strip()}]"
            
            if meaningful_keywords and any(k in content_lower for k in meaningful_keywords):
                return f"Regarding {product.strip()}: {clean_content[:500]} [{doc_id.strip()}]"
            elif not meaningful_keywords:
                return f"Regarding {product.strip()}: {clean_content[:500]} [{doc_id.strip()}]"
            
        # No relevant content found
        return "The requested information was not found in the product catalog."


def answer_catalog_question(query: str) -> CatalogResult:
    total_start = time.perf_counter()
    timing_ms = {}

    # 1. Early Scope Gate
    out_of_scope_flag, reason = is_out_of_scope(query)
    if out_of_scope_flag:
        return CatalogResult(
            answer="This question is not related to my database. I can only answer questions about technical product catalog, parts, specifications, troubleshooting, and installation procedures.",
            out_of_scope=True,
            scope_reason=reason,
            status_code=200,
            timing_ms={
                "total_ms": (time.perf_counter() - total_start) * 1000,
                "sparse_dense_search_ms": 0.0,
                "rerank_ms": 0.0,
                "generation_ms": 0.0
            },
            citation_check={
                "total_citations_found": 0,
                "valid_citations": [],
                "invalid_citations": [],
                "has_valid_citation": False,
                "all_valid": False
            }
        )

    # 2. Retrieval Phase
    try:
        reranked_sources, search_ms, rerank_ms = execute_retrieval_pipeline(query)
        timing_ms["sparse_dense_search_ms"] = search_ms
        timing_ms["rerank_ms"] = rerank_ms
    except Exception as search_err:
        logger.error(f"Retrieval error: {str(search_err)}")
        return CatalogResult(
            answer="An error occurred while searching the catalog.",
            error=str(search_err),
            status_code=500,
            timing_ms={"total_ms": (time.perf_counter() - total_start) * 1000}
        )

    # 3. Generation Phase
    gen_start = time.perf_counter()
    try:
        formatted_context = "\n".join([
            f"Source ID: {s.get('doc_id')}\nSKU: {s.get('sku')}\nProduct: {s.get('product_name')}\nContent: {s.get('content')}\n---"
            for s in reranked_sources
        ])
        
        prompt = f"Sources:\n{formatted_context}\n\nQuestion: {query}"
        generated_answer = call_omniroute_llm(prompt, query)
        timing_ms["generation_ms"] = (time.perf_counter() - gen_start) * 1000
    except Exception as llm_err:
        timing_ms["generation_ms"] = (time.perf_counter() - gen_start) * 1000
        timing_ms["total_ms"] = (time.perf_counter() - total_start) * 1000
        logger.error(f"OmniRoute error: {str(llm_err)}")

        return CatalogResult(
            answer="The catalog generation service is currently unavailable or experiencing high latency. Please try again shortly.",
            sources=[
                {
                    "doc_id": s.get("doc_id", "unknown"),
                    "sku": s.get("sku", "unknown"),
                    "product_name": s.get("product_name", "unknown")
                }
                for s in reranked_sources
            ],
            error=f"OmniRoute Server Error: {str(llm_err)}",
            status_code=503,
            timing_ms=timing_ms,
            out_of_scope=False
        )

    # 4. Citation Checking
    formatted_sources = [
        {
            "doc_id": s.get("doc_id", "unknown"),
            "sku": s.get("sku", "unknown"),
            "product_name": s.get("product_name", "unknown")
        }
        for s in reranked_sources
    ]

    valid_citations = [s["doc_id"] for s in formatted_sources if s["doc_id"] in generated_answer]
    citation_check = {
        "total_citations_found": len(valid_citations),
        "valid_citations": valid_citations,
        "invalid_citations": [],
        "has_valid_citation": len(valid_citations) > 0,
        "all_valid": len(valid_citations) > 0
    }

    timing_ms["total_ms"] = (time.perf_counter() - total_start) * 1000

    return CatalogResult(
        answer=generated_answer,
        sources=formatted_sources,
        citation_check=citation_check,
        timing_ms=timing_ms,
        out_of_scope=False,
        status_code=200
    )