"""
CatalogIQ Generation Evaluation (Step 7)
Measures LLM output quality without modifying generation.py.
Metrics: citation validity, completeness, answer relevance, faithfulness, 
no-answer accuracy, unsupported-claim rate.
Cross-references with retrieval failure_analysis.csv.
"""
import os
import sys
import json
import re
import time
import csv
from typing import List, Dict, Any, Optional, Tuple, Set

# Add src to path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from search import hybrid_search, get_document, RRF_TOP_K, COLLECTION_NAME
from generation import generate_answer, validate_citations, build_prompt, CITATION_PATTERN
from evaluation import load_mappings, extract_skus_from_results, ALL_VALID_SKUS

# Load queries
with open("evaluation/queries.jsonl", 'r') as f:
    queries = [json.loads(line) for line in f]

# Load retrieval failure analysis
failure_by_query = {}
try:
    with open("evaluation/results/failure_analysis.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            failure_by_query[row["query"]] = row["failure_reason"]
except:
    pass

# Load relaxed failure analysis
relaxed_by_query = {}
try:
    with open("evaluation/results/relaxed_failure_analysis.json", 'r') as f:
        relaxed = json.load(f)
        for r in relaxed:
            if r["query"] not in relaxed_by_query:
                relaxed_by_query[r["query"]] = []
            relaxed_by_query[r["query"]].append(r)
except:
    pass

def get_retrieval_context(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Run full retrieval pipeline to get top_k docs for generation."""
    results = hybrid_search(query, top_k=top_k)
    docs = []
    for r in results[:top_k]:
        doc = get_document(r["doc_id"])
        if doc:
            doc["rrf_score"] = r["rrf_score"]
            doc["dense_rank"] = r.get("dense_rank")
            doc["bm25_rank"] = r.get("bm25_rank")
            docs.append(doc)
    return docs

def extract_cited_doc_ids(answer: str) -> List[str]:
    """Extract all [Doc: ...] citations from answer."""
    return CITATION_PATTERN.findall(answer)

def check_no_answer_appropriateness(query: str, answer: str, relevant_skus: List[str]) -> Dict[str, Any]:
    """Check if model correctly declines when it should."""
    answer_lower = answer.lower()
    decline_phrases = [
        "does not provide", "doesn't provide", "not provide", "not available",
        "not in the document", "not in the documentation", "no information",
        "cannot answer", "unable to answer", "don't know", "do not know",
        "insufficient information", "not mentioned", "not found"
    ]
    is_decline = any(phrase in answer_lower for phrase in decline_phrases)
    
    # Should decline if irrelevant query or retrieval failed
    should_decline = len(relevant_skus) == 0 or query in failure_by_query
    
    return {
        "is_decline": is_decline,
        "should_decline": should_decline,
        "correct_no_answer": is_decline == should_decline,
        "decline_type": "correct" if is_decline == should_decline else 
                       ("false_decline" if is_decline and not should_decline else "missed_decline")
    }

def check_unsupported_claims(answer: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Heuristic: count factual claims without citations."""
    # Split into sentences
    sentences = re.split(r'[.!?]+', answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    cited = extract_cited_doc_ids(answer)
    claims_with_citation = len(cited) > 0
    
    # Count sentences that look like factual claims but have no citation
    factual_indicators = [
        "is", "has", "supports", "includes", "features", "specifies",
        "requires", "uses", "contains", "provides", "offers", "rated",
        "capacity", "weight", "size", "dimension", "voltage", "torque",
        "dpi", "impedance", "warranty", "angle", "speed", "range"
    ]
    
    unsupported = 0
    total_claims = 0
    for sent in sentences:
        sent_lower = sent.lower()
        # Skip if it's a citation-only sentence
        if "doc:" in sent_lower:
            continue
        # Check if it makes a factual claim about a product/spec
        if any(ind in sent_lower for ind in factual_indicators):
            total_claims += 1
            # Check if this sentence has a citation nearby (heuristic)
            # Simple: if total citations < total claim sentences, some are unsupported
    
    # Approximation: unsupported = max(0, total_claims - len(cited))
    unsupported = max(0, total_claims - len(cited))
    
    return {
        "total_sentences": len(sentences),
        "total_citations": len(cited),
        "factual_claims_est": total_claims,
        "unsupported_claims_est": unsupported,
        "citation_coverage": len(cited) / max(1, total_claims) if total_claims > 0 else 1.0
    }

def check_faithfulness(answer: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check if answer hallucinates info not in retrieved docs."""
    # Get all content from retrieved docs
    doc_content = " ".join([d.get("content", "") for d in docs])
    doc_content_lower = doc_content.lower()
    
    # Extract key factual statements from answer
    answer_sentences = re.split(r'[.!?]+', answer)
    hallucinated = 0
    supported = 0
    
    for sent in answer_sentences:
        sent = sent.strip()
        if len(sent) < 15 or "doc:" in sent.lower():
            continue
        # Check if key nouns/values appear in doc content
        words = set(re.findall(r'\b[A-Z0-9-]{3,}\b', sent))
        if words:
            matches = sum(1 for w in words if w.lower() in doc_content_lower)
            if matches == 0 and len(words) > 0:
                hallucinated += 1
            else:
                supported += 1
    
    return {
        "supported_statements": supported,
        "potentially_hallucinated": hallucinated,
        "faithfulness_score": supported / max(1, supported + hallucinated)
    }

def check_answer_relevance(query: str, answer: str) -> Dict[str, Any]:
    """Check if answer addresses the query."""
    query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
    answer_words = set(re.findall(r'\b\w{3,}\b', answer.lower()))
    
    # Remove stop words
    stop = {"the", "and", "for", "with", "what", "how", "does", "is", "are", "not", "you", "your"}
    query_words = query_words - stop
    answer_words = answer_words - stop
    
    overlap = query_words & answer_words
    relevance = len(overlap) / max(1, len(query_words))
    
    return {
        "query_keywords": len(query_words),
        "answer_keywords": len(answer_words),
        "overlap": len(overlap),
        "relevance_score": round(relevance, 3)
    }

# Run evaluation
print("=" * 80)
print("GENERATION EVALUATION (Step 7)")
print("=" * 80)

results = []
gen_errors = 0

for idx, qentry in enumerate(queries, 1):
    query = qentry["query"]
    relevant = qentry.get("relevant_skus", [])
    qtype = qentry.get("query_type", "unknown")
    
    print(f"\n[{idx}/{len(queries)}] {query[:70]}...", flush=True)
    
    # Retrieve context
    docs = get_retrieval_context(query, top_k=5)
    valid_doc_ids = [d.get("doc_id") for d in docs]
    
    # Generate answer
    start = time.perf_counter()
    try:
        answer = generate_answer(query, docs)
        gen_time = (time.perf_counter() - start) * 1000
    except Exception as e:
        gen_errors += 1
        answer = f"[ERROR] {e}"
        gen_time = 0
    
    # Metrics
    citation_info = validate_citations(answer, valid_doc_ids)
    no_answer = check_no_answer_appropriateness(query, answer, relevant)
    unsupported = check_unsupported_claims(answer, docs)
    faithfulness = check_faithfulness(answer, docs)
    relevance = check_answer_relevance(query, answer)
    
    # Retrieval quality for this query
    retrieved_skus = extract_skus_from_results([d.get("doc_id") for d in docs])
    retrieval_hit = len(set(relevant) & set(retrieved_skus)) > 0
    
    result = {
        "query_idx": idx,
        "query": query,
        "query_type": qtype,
        "relevant_skus": ";".join(relevant),
        "retrieved_skus": ";".join(retrieved_skus),
        "retrieval_hit": retrieval_hit,
        "retrieval_failure_reason": failure_by_query.get(query, "none"),
        "answer": answer,
        "generation_time_ms": round(gen_time, 1),
        "answer_length": len(answer),
        
        # Citation metrics
        "citations_found": citation_info["total_citations_found"],
        "valid_citations": citation_info["has_valid_citation"],
        "all_citations_valid": citation_info["all_valid"],
        "invalid_citations": ";".join(citation_info["invalid_citations"]),
        
        # Quality metrics
        "no_answer_correct": no_answer["correct_no_answer"],
        "no_answer_type": no_answer["decline_type"],
        "unsupported_claims_est": unsupported["unsupported_claims_est"],
        "citation_coverage": round(unsupported["citation_coverage"], 3),
        "faithfulness_score": round(faithfulness["faithfulness_score"], 3),
        "relevance_score": relevance["relevance_score"],
    }
    results.append(result)
    
    # Progress
    if idx % 10 == 0:
        print(f"  Progress: {idx}/{len(queries)}")

# Save results
csv_path = "evaluation/results/generation_evaluation.csv"
json_path = "evaluation/results/generation_evaluation.json"

with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

# Summary statistics
print("\n" + "=" * 80)
print("GENERATION EVALUATION SUMMARY")
print("=" * 80)

total = len(results)
irrelevant = sum(1 for r in results if r["query_type"] == "irrelevant")
relevant_qs = [r for r in results if r["query_type"] != "irrelevant"]

print(f"\nTotal queries: {total} (Relevant: {len(relevant_qs)}, Irrelevant: {irrelevant})")
print(f"Generation errors: {gen_errors}")

# Citation validity
valid_citations = sum(1 for r in relevant_qs if r["valid_citations"])
all_valid = sum(1 for r in relevant_qs if r["all_citations_valid"])
print(f"\nCitation Metrics (relevant queries):")
print(f"  Has valid citations: {valid_citations}/{len(relevant_qs)} ({valid_citations/len(relevant_qs)*100:.1f}%)")
print(f"  All citations valid: {all_valid}/{len(relevant_qs)} ({all_valid/len(relevant_qs)*100:.1f}%)")
print(f"  Mean citations per answer: {sum(r['citations_found'] for r in relevant_qs)/len(relevant_qs):.1f}")

# No-answer accuracy
correct_declines = sum(1 for r in results if r["no_answer_correct"])
print(f"\nNo-Answer Accuracy:")
print(f"  Correct (decline when should, answer when should): {correct_declines}/{total} ({correct_declines/total*100:.1f}%)")
print(f"  False declines (should answer but declined): {sum(1 for r in relevant_qs if r['no_answer_type']=='false_decline')}")
print(f"  Missed declines (should decline but answered): {sum(1 for r in results if r['query_type']=='irrelevant' and r['no_answer_type']=='missed_decline')}")

# Faithfulness
faith_scores = [r["faithfulness_score"] for r in relevant_qs]
print(f"\nFaithfulness (relevant queries):")
print(f"  Mean: {sum(faith_scores)/len(faith_scores):.3f}")
print(f"  Min: {min(faith_scores):.3f}")
print(f"  Potentially hallucinated statements: {sum(r['potentially_hallucinated'] for r in relevant_qs)}")

# Relevance
rel_scores = [r["relevance_score"] for r in relevant_qs]
print(f"\nAnswer Relevance (relevant queries):")
print(f"  Mean: {sum(rel_scores)/len(rel_scores):.3f}")
print(f"  Min: {min(rel_scores):.3f}")

# Unsupported claims
unsup = sum(r["unsupported_claims_est"] for r in relevant_qs)
print(f"\nUnsupported Claims (est., relevant queries): {unsup} total")
print(f"  Mean per query: {unsup/len(relevant_qs):.1f}")

# Cross-reference with retrieval failures
print("\n" + "=" * 80)
print("CROSS-REFERENCE: Generation Quality vs Retrieval Failures")
print("=" * 80)

for r in relevant_qs:
    if r["retrieval_failure_reason"] != "none" or not r["retrieval_hit"]:
        print(f"  Query: {r['query'][:60]}")
        print(f"    Retrieval: hit={r['retrieval_hit']}, failure={r['retrieval_failure_reason']}")
        print(f"    Gen: faithfulness={r['faithfulness_score']}, relevance={r['relevance_score']}, citations={r['citations_found']}")
        print()

print(f"\nResults saved to:\n  - {csv_path}\n  - {json_path}")