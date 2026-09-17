import os
import re
import time
from typing import List, Dict, Any, Optional

# pyrefly: ignore [missing-import]
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

OMNIROUTE_BASE_URL = os.getenv(
    "OMNIROUTE_BASE_URL",
    "http://localhost:20128/v1"
)

OMNIROUTE_MODEL = os.getenv(
    "OMNIROUTE_MODEL",
    "ddgw/gpt-4o-mini"
)

# Keep answers short for customer-support use.
MAX_TOKENS = 512

# Low temperature = more factual / deterministic RAG answers.
TEMPERATURE = 0.1

OUT_OF_SCOPE_MESSAGE = (
    "I'm sorry, but I could not find information regarding your query "
    "in the technical catalog."
)


# ============================================================
# CLIENT
# ============================================================

_client: Optional[OpenAI] = None


def get_omniroute_client() -> OpenAI:
    """
    Create the OmniRoute OpenAI-compatible client once and reuse it.
    """

    global _client

    if _client is None:

        api_key = os.environ.get("OMNIROUTE_API_KEY")

        if not api_key:
            raise ValueError(
                "OMNIROUTE_API_KEY environment variable not set"
            )

        try:
            _client = OpenAI(
                api_key=api_key,
                base_url=OMNIROUTE_BASE_URL,
                timeout=30.0,
            )

        except Exception as e:
            raise ConnectionError(
                f"Failed to create OmniRoute client: {e}"
            ) from e

    return _client


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    query: str,
    top_docs: List[Dict[str, Any]]
) -> str:
    """
    Build a compact grounded RAG prompt.

    The model must:
    1. Use only retrieved documents.
    2. Never invent product information.
    3. Cite every factual statement.
    4. Explicitly say when information is unavailable.
    """

    doc_blocks = []

    for i, doc in enumerate(top_docs):

        doc_id = doc.get(
            "doc_id",
            f"unknown_{i}"
        )

        content = str(
            doc.get("content", "")
        ).strip()

        # Prevent unnecessarily large prompts.
        if len(content) > 2500:
            content = (
                content[:2500]
                + "... [truncated]"
            )

        doc_blocks.append(
            f"[Doc: {doc_id}]\n"
            f"{content}"
        )

    docs_text = "\n\n".join(doc_blocks)

    prompt = f"""You are CatalogIQ, an enterprise technical support and product catalog assistant.

Your task is to answer the user's question accurately using ONLY the provided document chunks below.

---
RELEVANT CONTEXT:
{docs_text}
---

CRITICAL INSTRUCTIONS:
1. Grounding: Rely STRICTLY on the facts contained within the provided context. Do NOT use outside knowledge, speculate, or make assumptions. Do NOT fill gaps with pre-trained knowledge. If the context does not contain the answer, you MUST refuse.
2. Missing Information / Out of Domain: If the answer cannot be found in the provided context, state EXACTLY: "{OUT_OF_SCOPE_MESSAGE}" Do NOT attempt to answer using pre-trained external knowledge. Do NOT hedge, paraphrase, or soften the refusal. Do NOT say "based on the documents" or "according to available information" and then give an answer -- if the facts are not in the context, refuse.
3. Citations: For EVERY statement or fact you provide, cite the source document ID using the exact format `[Doc: <ID>]` (e.g., `[Doc: DOC_B3301_BLT]`). Every factual claim must have at least one citation. Do NOT write a sentence without a citation if it states a fact. Introductory or connective phrases without facts do not need citations.
4. Irrelevant Context: If the retrieved context is about a different product, part, or topic than the user's question, treat it as missing information and refuse. Do NOT force a connection.
5. Tone: Provide a direct, professional, and concise technical answer. No fluff, no disclaimers about being an AI.

USER QUESTION: {query}
ANSWER:
""".strip()

    return prompt


# ============================================================
# RESPONSE EXTRACTION
# ============================================================

def extract_text_from_content(content: Any) -> str:
    """
    Robustly extract text from different OpenAI-compatible
    response formats.

    Some providers return:
        string

    Others may return:
        list[str]

    or:
        list[dict]

    or SDK content objects.
    """

    if content is None:
        return ""

    # --------------------------------------------------------
    # Normal case: plain string
    # --------------------------------------------------------

    if isinstance(content, str):
        return content.strip()

    # --------------------------------------------------------
    # List / array response
    # --------------------------------------------------------

    if isinstance(content, list):

        parts: List[str] = []

        for item in content:

            # Example:
            # ["hello", "world"]
            if isinstance(item, str):
                parts.append(item)
                continue

            # Example:
            # {"type": "text", "text": "hello"}
            if isinstance(item, dict):

                text = item.get("text")

                if isinstance(text, str):
                    parts.append(text)
                    continue

                # Some providers may nest text.
                if isinstance(text, dict):

                    nested_text = text.get("value")

                    if nested_text:
                        parts.append(
                            str(nested_text)
                        )

                continue

            # SDK object:
            # item.text
            text = getattr(
                item,
                "text",
                None
            )

            if isinstance(text, str):
                parts.append(text)
                continue

            # SDK object may contain value.
            value = getattr(
                item,
                "value",
                None
            )

            if isinstance(value, str):
                parts.append(value)

        return "\n".join(parts).strip()

    # --------------------------------------------------------
    # Other object types
    # --------------------------------------------------------

    text = getattr(
        content,
        "text",
        None
    )

    if isinstance(text, str):
        return text.strip()

    value = getattr(
        content,
        "value",
        None
    )

    if isinstance(value, str):
        return value.strip()

    return str(content).strip()


# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    query: str,
    top_docs: List[Dict[str, Any]]
) -> str:

    prompt = build_prompt(
        query,
        top_docs
    )

    print("\n" + "=" * 60)
    print("GENERATION DEBUG")
    print("=" * 60)

    print(
        f"Prompt characters: {len(prompt):,}"
    )

    print(
        f"Prompt words:      {len(prompt.split()):,}"
    )

    print(
        f"Documents:         {len(top_docs)}"
    )

    print(
        f"Model:             {OMNIROUTE_MODEL}"
    )

    print("=" * 60)

    client = get_omniroute_client()

    generation_start = time.perf_counter()

    try:

        response = client.chat.completions.create(
            model=OMNIROUTE_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            max_tokens=MAX_TOKENS,

            temperature=TEMPERATURE,

            stream=False,
        )

        generation_time = (
            time.perf_counter()
            - generation_start
        ) * 1000

        print(
            f"OmniRoute request: "
            f"{generation_time:.1f} ms"
        )

        # ----------------------------------------------------
        # Validate choices
        # ----------------------------------------------------

        if not response.choices:

            print(
                "WARNING: Model returned "
                "no choices."
            )

            return (
                "The model returned no answer."
            )

        message = response.choices[0].message

        # ----------------------------------------------------
        # Extract content safely
        # ----------------------------------------------------

        content = getattr(
            message,
            "content",
            None
        )

        answer = extract_text_from_content(
            content
        )

        # ----------------------------------------------------
        # Debug response structure
        # ----------------------------------------------------

        print(
            f"Response content type: "
            f"{type(content).__name__}"
        )

        print(
            f"Extracted answer length: "
            f"{len(answer):,} characters"
        )

        # Print only a small preview.
        if answer:

            preview = answer[:300]

            print(
                "Answer preview:"
            )

            print(preview)

        else:

            print(
                "WARNING: Empty answer "
                "returned by model."
            )

        # ----------------------------------------------------
        # Empty response protection
        # ----------------------------------------------------

        if not answer:

            return (
                "The model returned an empty answer."
            )

        # ----------------------------------------------------
        # IMPORTANT:
        # Do NOT convert to ASCII.
        #
        # Product data may contain:
        # Spanish
        # French
        # accents
        # Unicode product names
        # ----------------------------------------------------

        return answer.strip()

    except Exception as e:

        generation_time = (
            time.perf_counter()
            - generation_start
        ) * 1000

        print(
            f"OmniRoute failed after "
            f"{generation_time:.1f} ms"
        )

        error_msg = (
            f"[ERROR] OmniRoute generation failed: "
            f"{type(e).__name__}: {e}"
        )

        print(error_msg)

        return error_msg


# ============================================================
# CITATION VALIDATION
# ============================================================

CITATION_PATTERN = re.compile(
    r"\[Doc:\s*([^\]]+)\]"
)


def validate_citations(
    answer: str,
    valid_doc_ids: List[str]
) -> Dict[str, Any]:

    found_citations = (
        CITATION_PATTERN.findall(answer)
    )

    valid_set = set(
        valid_doc_ids
    )

    valid_citations = [
        citation
        for citation in found_citations
        if citation in valid_set
    ]

    invalid_citations = [
        citation
        for citation in found_citations
        if citation not in valid_set
    ]

    return {
        "total_citations_found": len(
            found_citations
        ),

        "valid_citations": valid_citations,

        "invalid_citations": invalid_citations,

        "has_valid_citation": (
            len(valid_citations) > 0
        ),

        "all_valid": (
            len(found_citations) > 0
            and len(invalid_citations) == 0
        ),
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Generation module ready."
    )

    print(
        f"OmniRoute URL: "
        f"{OMNIROUTE_BASE_URL}"
    )

    print(
        f"Model: "
        f"{OMNIROUTE_MODEL}"
    )

