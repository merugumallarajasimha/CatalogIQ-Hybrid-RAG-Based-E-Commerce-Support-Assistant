import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.query_router import (
    CATEGORY_PRODUCT_COMPARISON,
    CATEGORY_TROUBLESHOOTING,
    CATEGORY_UNSUPPORTED,
    classify_query,
    is_out_of_scope,
)


def test_obvious_out_of_scope_queries_are_declined():
    queries = [
        "How do I cook a turkey for Thanksgiving?",
        "What is the weather forecast for tomorrow morning?",
        "How should beginners invest in index funds and stocks?",
        "What is the capital city of Australia?",
        "Can you write a poem about space travel?",
    ]

    for query in queries:
        out_of_scope, reason = is_out_of_scope(query)
        assert out_of_scope, query
        assert reason


def test_catalog_queries_are_allowed():
    queries = [
        "What is the torque spec for screw S-4012-SCRW?",
        "How do I replace the armrest bracket on the ErgoFlex chair?",
        "What does error E02 mean on DESK-STD-MOTO?",
        "Compare CHAIR-ERG-X99 and DESK-STD-MOTO",
        "What are the dimensions of the monitor arm?",
    ]

    for query in queries:
        out_of_scope, _ = is_out_of_scope(query)
        assert not out_of_scope, query


def test_generic_questions_do_not_bypass_the_guard():
    assert classify_query("How do I cook a turkey?")[0] == CATEGORY_UNSUPPORTED
    assert classify_query("What is the best phone?")[0] == CATEGORY_UNSUPPORTED
    assert classify_query("What is the product of 2 and 3?")[0] == CATEGORY_UNSUPPORTED
    assert classify_query("What is the price?")[0] == CATEGORY_UNSUPPORTED
    assert classify_query("Compare gaming laptops under $1000")[0] == CATEGORY_UNSUPPORTED
    assert classify_query("How do I fix my car?")[0] == CATEGORY_UNSUPPORTED


def test_router_keeps_catalog_categories():
    assert classify_query("What is the warranty on CHAIR-ERG-X99?")[0] == "exact_lookup"
    assert classify_query("My chair squeaks when I lean back")[0] == CATEGORY_TROUBLESHOOTING
    assert classify_query("Compare CHAIR-ERG-X99 vs DESK-STD-MOTO")[0] == CATEGORY_PRODUCT_COMPARISON
