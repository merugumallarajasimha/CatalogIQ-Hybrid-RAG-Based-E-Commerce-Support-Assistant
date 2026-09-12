import sys
import os
import csv
import time
import argparse
from typing import Dict, List, Any, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.search import sparse_search, dense_search, hybrid_search, get_document
from src.reranker import rerank
from src.generation import generate_answer, validate_citations, build_prompt


def check_keywords(answer: str, expected_keywords: str, cited_doc_ids: List[str]) -> Tuple[bool, str]:
    if not expected_keywords.strip():
        return False, "No expected keywords provided"
    
    keywords = [k.strip().lower() for k in expected_keywords.split(",")]
    answer_lower = answer.lower()
    
    matched = []
    missing = []
    for kw in keywords:
        if kw in answer_lower:
            matched.append(kw)
        else:
            missing.append(kw)
    
    if missing:
        return False, f"Missing keywords: {', '.join(missing)}"
    return True, f"All keywords found: {', '.join(matched)}"


def run_single_eval(
    question: str,
    expected_doc_id: str,
    expected_keywords: str,
    question_type: str,
    verbose: bool = False
) -> Dict[str, Any]:
    
    result = {
        "question": question,
        "expected_doc_id": expected_doc_id,
        "expected_keywords": expected_keywords,
        "question_type": question_type,
        "sparse_pass": False,
        "dense_pass": False,
        "rerank_pass": False,
        "generation_pass": False,
        "sparse_details": "",
        "dense_details": "",
        "rerank_details": "",
        "generation_details": "",
        "answer": "",
        "cited_doc_ids": [],
    }
    
    try:
        sparse_results = sparse_search(question, top_k=20)
        sparse_ids = [doc_id for doc_id, _ in sparse_results]
        if expected_doc_id != "NONE" and expected_doc_id in sparse_ids:
            result["sparse_pass"] = True
            result["sparse_details"] = f"Found at rank {sparse_ids.index(expected_doc_id) + 1}"
        else:
            result["sparse_details"] = f"Not in top 20 (expected: {expected_doc_id})"
    except Exception as e:
        result["sparse_details"] = f"Error: {e}"
    
    try:
        dense_results = dense_search(question, top_k=20)
        dense_ids = [doc_id for doc_id, _ in dense_results]
        if expected_doc_id != "NONE" and expected_doc_id in dense_ids:
            result["dense_pass"] = True
            result["dense_details"] = f"Found at rank {dense_ids.index(expected_doc_id) + 1}"
        else:
            result["dense_details"] = f"Not in top 20 (expected: {expected_doc_id})"
    except Exception as e:
        result["dense_details"] = f"Error: {e}"
    
    try:
        candidates = hybrid_search(question, top_k=20)
        candidates = [dict(c) for c in candidates]
        for c in candidates:
            doc = get_document(c["doc_id"])
            if doc:
                c["content"] = doc.get("content", "")
                c["sku"] = doc.get("sku", "unknown")
                c["part_numbers"] = doc.get("part_numbers", [])
                c["product_name"] = doc.get("product_name", "unknown")
        
        reranked = rerank(question, candidates, top_n=5)
        rerank_ids = [d["doc_id"] for d in reranked]
        if expected_doc_id != "NONE" and expected_doc_id in rerank_ids:
            result["rerank_pass"] = True
            result["rerank_details"] = f"Found at rank {rerank_ids.index(expected_doc_id) + 1}"
        else:
            result["rerank_details"] = f"Not in top 5 (expected: {expected_doc_id})"
    except Exception as e:
        result["rerank_details"] = f"Error: {e}"
        reranked = []
    
    try:
        answer = generate_answer(question, reranked)
        result["answer"] = answer
        
        valid_doc_ids = [d["doc_id"] for d in reranked]
        citation_result = validate_citations(answer, valid_doc_ids)
        result["cited_doc_ids"] = citation_result["valid_citations"]
        
        if expected_doc_id == "NONE":
            gen_pass = citation_result["has_valid_citation"] == False or "don't know" in answer.lower() or "cannot answer" in answer.lower()
            result["generation_pass"] = gen_pass
            result["generation_details"] = "Correctly refused to answer" if gen_pass else "Incorrectly attempted answer"
        else:
            keyword_pass, kw_detail = check_keywords(answer, expected_keywords, citation_result["valid_citations"])
            cite_correct = expected_doc_id in citation_result["valid_citations"]
            result["generation_pass"] = keyword_pass and cite_correct
            result["generation_details"] = f"Keywords: {'PASS' if keyword_pass else 'FAIL'} ({kw_detail}); Citation: {'PASS' if cite_correct else 'FAIL'}"
            
            if not result["generation_pass"] and verbose:
                print(f"\n  [VERBOSE] Question: {question}")
                print(f"  [VERBOSE] Expected doc: {expected_doc_id}")
                print(f"  [VERBOSE] Expected keywords: {expected_keywords}")
                print(f"  [VERBOSE] Cited docs: {citation_result['valid_citations']}")
                print(f"  [VERBOSE] Answer:\n  {answer[:500]}...")
    except Exception as e:
        result["generation_details"] = f"Error: {e}"
    
    return result


def run_eval(csv_path: str, verbose: bool = False) -> List[Dict[str, Any]]:
    results = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        questions = list(reader)
    
    print(f"Running evaluation on {len(questions)} questions...")
    print("=" * 80)
    
    for i, row in enumerate(questions):
        q = row["question"]
        exp_doc = row["expected_doc_id"]
        exp_kw = row["expected_keywords"]
        q_type = row["question_type"]
        
        print(f"\n[{i+1}/{len(questions)}] [{q_type}] {q[:60]}...")
        
        result = run_single_eval(q, exp_doc, exp_kw, q_type, verbose)
        results.append(result)
        
        print(f"  Sparse:  {'PASS' if result['sparse_pass'] else 'FAIL'} - {result['sparse_details']}")
        print(f"  Dense:   {'PASS' if result['dense_pass'] else 'FAIL'} - {result['dense_details']}")
        print(f"  Rerank:  {'PASS' if result['rerank_pass'] else 'FAIL'} - {result['rerank_details']}")
        print(f"  Gen:     {'PASS' if result['generation_pass'] else 'FAIL'} - {result['generation_details']}")
    
    return results


def print_summary(results: List[Dict[str, Any]]):
    total = len(results)
    if total == 0:
        print("No results to summarize")
        return
    
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    stages = ["sparse_pass", "dense_pass", "rerank_pass", "generation_pass"]
    stage_names = ["Sparse Search", "Dense Search", "Rerank", "Generation"]
    
    print(f"\n{'Overall':<20} {'Total':>8} {'Pass':>8} {'Fail':>8} {'Accuracy':>10}")
    print("-" * 54)
    
    for stage, name in zip(stages, stage_names):
        passed = sum(1 for r in results if r[stage])
        failed = total - passed
        acc = (passed / total * 100) if total > 0 else 0
        print(f"{name:<20} {total:>8} {passed:>8} {failed:>8} {acc:>9.1f}%")
    
    print(f"\n{'By Question Type':<20} {'Total':>8} {'Pass':>8} {'Fail':>8} {'Accuracy':>10}")
    print("-" * 54)
    
    types = ["exact_sku", "fuzzy", "edge_typo", "edge_nomatch"]
    for q_type in types:
        type_results = [r for r in results if r["question_type"] == q_type]
        if not type_results:
            continue
        type_total = len(type_results)
        type_passed = sum(1 for r in type_results if r["generation_pass"])
        type_acc = (type_passed / type_total * 100) if type_total > 0 else 0
        print(f"{q_type:<20} {type_total:>8} {type_passed:>8} {type_total-type_passed:>8} {type_acc:>9.1f}%")
    
    print("=" * 80)


def save_results(results: List[Dict[str, Any]], output_path: str):
    fieldnames = [
        "question", "expected_doc_id", "expected_keywords", "question_type",
        "sparse_pass", "dense_pass", "rerank_pass", "generation_pass",
        "sparse_details", "dense_details", "rerank_details", "generation_details",
        "answer", "cited_doc_ids"
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            row["cited_doc_ids"] = "; ".join(r.get("cited_doc_ids", []))
            writer.writerow(row)
    
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on hybrid RAG pipeline")
    parser.add_argument("--csv", default="test_questions.csv", help="Path to test questions CSV")
    parser.add_argument("--output", default="eval_results.csv", help="Output path for results")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output for generation failures")
    args = parser.parse_args()
    
    results = run_eval(args.csv, verbose=args.verbose)
    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()