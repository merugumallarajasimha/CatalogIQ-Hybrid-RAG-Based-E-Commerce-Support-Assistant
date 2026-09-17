import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.catalog_service import answer_catalog_question
from src.generation import OUT_OF_SCOPE_MESSAGE


def test_out_of_scope_query_short_circuits_pipeline():
    calls = []

    def fail_search(*args, **kwargs):
        calls.append("search")
        raise AssertionError("search should not run")

    result = answer_catalog_question(
        "How do I cook a turkey?",
        search_fn=fail_search,
    )

    assert result.out_of_scope is True
    assert result.answer == OUT_OF_SCOPE_MESSAGE
    assert result.sources == []
    assert calls == []


def test_low_relevance_results_are_declined_before_generation():
    def search(*args, **kwargs):
        return [{"doc_id": "DOC_OTHER", "content": "Unrelated document."}]

    def enrich(candidates):
        return candidates

    def rerank(*args, **kwargs):
        return [
            {
                "doc_id": "DOC_OTHER",
                "content": "Unrelated document.",
                "rerank_score": -3.2,
                "exact_match": False,
            }
        ]

    def fail_generate(*args, **kwargs):
        raise AssertionError("generation should not run")

    result = answer_catalog_question(
        "What is the warranty on a gaming laptop?",
        search_fn=search,
        enrich_fn=enrich,
        rerank_fn=rerank,
        generate_fn=fail_generate,
    )

    assert result.out_of_scope is True
    assert result.answer == OUT_OF_SCOPE_MESSAGE
    assert "relevance threshold" in result.scope_reason


def test_valid_catalog_answer_returns_sources_and_citations():
    def search(*args, **kwargs):
        return [{"doc_id": "DOC_CHAIR", "content": "The chair supports 150 kg."}]

    def enrich(candidates):
        return [
            {
                **candidate,
                "sku": "CHAIR-ERG-X99",
                "product_name": "ErgoFlex Pro Office Chair",
            }
            for candidate in candidates
        ]

    def rerank(*args, **kwargs):
        return [
            {
                "doc_id": "DOC_CHAIR",
                "content": "The chair supports 150 kg.",
                "sku": "CHAIR-ERG-X99",
                "product_name": "ErgoFlex Pro Office Chair",
                "rerank_score": 4.1,
                "exact_match": False,
            }
        ]

    def generate(*args, **kwargs):
        return "The chair supports 150 kg. [Doc: DOC_CHAIR]"

    def validate(answer, doc_ids):
        return {
            "total_citations_found": 1,
            "valid_citations": ["DOC_CHAIR"],
            "invalid_citations": [],
            "has_valid_citation": True,
            "all_valid": True,
        }

    result = answer_catalog_question(
        "What is the chair capacity?",
        search_fn=search,
        enrich_fn=enrich,
        rerank_fn=rerank,
        generate_fn=generate,
        validate_fn=validate,
    )

    assert result.error is None
    assert result.out_of_scope is False
    assert result.answer.startswith("The chair supports 150 kg.")
    assert result.sources[0]["doc_id"] == "DOC_CHAIR"
    assert result.citation_check["all_valid"] is True


def test_answer_without_a_valid_citation_is_replaced_with_refusal():
    def search(*args, **kwargs):
        return [{"doc_id": "DOC_CHAIR", "content": "The chair supports 150 kg."}]

    def enrich(candidates):
        return candidates

    def rerank(*args, **kwargs):
        return [
            {
                "doc_id": "DOC_CHAIR",
                "content": "The chair supports 150 kg.",
                "rerank_score": 4.1,
                "exact_match": False,
            }
        ]

    def generate(*args, **kwargs):
        return "The chair supports 150 kg."

    def validate(answer, doc_ids):
        return {
            "total_citations_found": 0,
            "valid_citations": [],
            "invalid_citations": [],
            "has_valid_citation": False,
            "all_valid": False,
        }

    result = answer_catalog_question(
        "What is the chair capacity?",
        search_fn=search,
        enrich_fn=enrich,
        rerank_fn=rerank,
        generate_fn=generate,
        validate_fn=validate,
    )

    assert result.answer == OUT_OF_SCOPE_MESSAGE
