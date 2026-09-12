import os
import re
from typing import List, Dict, Any

from openai import OpenAI


OMNIROUTE_BASE_URL = "http://localhost:20128/v1"
OMNIROUTE_MODEL = "auto"  # Let Omniroute choose


_client: OpenAI = None


def get_omniroute_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OMNIROUTE_API_KEY")
        if not api_key:
            raise ValueError("OMNIROUTE_API_KEY environment variable not set")
        _client = OpenAI(
            api_key=api_key,
            base_url=OMNIROUTE_BASE_URL,
        )
    return _client


def build_prompt(query: str, top_docs: List[Dict[str, Any]]) -> str:
    doc_blocks = []
    for i, doc in enumerate(top_docs):
        doc_id = doc.get("doc_id", f"unknown_{i}")
        content = doc.get("content", "").strip()
        # Truncate very long content to keep prompt reasonable
        if len(content) > 3000:
            content = content[:3000] + "... [truncated]"
        doc_blocks.append(f'[Doc: {doc_id}] "{content}"')

    docs_text = "\n\n".join(doc_blocks)

    prompt = f"""You are a support assistant. Using ONLY the documents below, answer the customer's question and cite the document ID for every fact you state.
If the documents don't contain the answer, say so honestly instead of guessing.

Documents:
{docs_text}

Question: {query}

Answer:"""

    return prompt


def generate_answer(query: str, top_docs: List[Dict[str, Any]]) -> str:
    prompt = build_prompt(query, top_docs)
    client = get_omniroute_client()

    try:
        response = client.chat.completions.create(
            model=OMNIROUTE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1,
        )

        answer = response.choices[0].message.content
        # Sanitize for Windows console
        answer = answer.encode('ascii', 'replace').decode('ascii')
        return answer

    except Exception as e:
        error_msg = f"[ERROR] Omniroute generation failed: {type(e).__name__}: {e}"
        print(error_msg)
        return error_msg


CITATION_PATTERN = re.compile(r'\[Doc:\s*([^\]]+)\]')


def validate_citations(answer: str, valid_doc_ids: List[str]) -> Dict[str, Any]:
    found_citations = CITATION_PATTERN.findall(answer)
    valid_set = set(valid_doc_ids)
    
    invalid_citations = [c for c in found_citations if c not in valid_set]
    valid_citations = [c for c in found_citations if c in valid_set]
    
    return {
        "total_citations_found": len(found_citations),
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "has_valid_citation": len(valid_citations) > 0,
        "all_valid": len(invalid_citations) == 0 and len(found_citations) > 0,
    }


if __name__ == "__main__":
    print("Generation module ready. Run tests/test_full_generation.py")