import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.generation import OUT_OF_SCOPE_MESSAGE, build_prompt


def test_generation_prompt_uses_catalogiq_grounding_instructions():
    prompt = build_prompt(
        "What is the torque spec?",
        [
            {
                "doc_id": "DOC_B3301_BLT",
                "content": "Torque the M8 flange bolt to 15 Nm.",
            }
        ],
    )

    assert "You are CatalogIQ" in prompt
    assert "RELEVANT CONTEXT:" in prompt
    assert "[Doc: DOC_B3301_BLT]" in prompt
    assert "Do NOT use outside knowledge" in prompt
    assert OUT_OF_SCOPE_MESSAGE in prompt
    assert "USER QUESTION: What is the torque spec?" in prompt


def test_generation_prompt_formats_each_document_chunk():
    prompt = build_prompt(
        "Compare the parts.",
        [
            {"doc_id": "DOC_ONE", "content": "First document."},
            {"doc_id": "DOC_TWO", "content": "Second document."},
        ],
    )

    assert "[Doc: DOC_ONE]\nFirst document." in prompt
    assert "[Doc: DOC_TWO]\nSecond document." in prompt
