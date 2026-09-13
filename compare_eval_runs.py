import csv
import sys
from typing import Dict, List


def load_results(csv_path: str) -> List[Dict[str, Any]]:
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def compare_runs(old_path: str, new_path: str):
    old_results = load_results(old_path)
    new_results = load_results(new_path)
    
    old_by_q = {r["question"]: r for r in old_results}
    new_by_q = {r["question"]: r for r in new_results}
    
    all_questions = set(old_by_q.keys()) | set(new_by_q.keys())
    
    stages = ["sparse_pass", "dense_pass", "rerank_pass", "generation_pass"]
    
    print("=" * 80)
    print(f"COMPARISON: {old_path} -> {new_path}")
    print("=" * 80)
    
    changes = 0
    for q in sorted(all_questions):
        if q not in old_by_q:
            print(f"\n[NEW] {q}")
            changes += 1
            continue
        if q not in new_by_q:
            print(f"\n[REMOVED] {q}")
            changes += 1
            continue
        
        old = old_by_q[q]
        new = new_by_q[q]
        
        question_changes = []
        for stage in stages:
            old_val = old.get(stage, "False").lower() == "true"
            new_val = new.get(stage, "False").lower() == "true"
            if old_val != new_val:
                direction = "PASS->FAIL" if old_val and not new_val else "FAIL->PASS"
                question_changes.append(f"{stage}: {direction}")
        
        if question_changes:
            print(f"\n[CHANGED] {q[:60]}...")
            for c in question_changes:
                print(f"  {c}")
            changes += 1
    
    if changes == 0:
        print("\nNo changes detected between runs.")
    
    print(f"\nTotal questions with changes: {changes}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_eval_runs.py <old_results.csv> <new_results.csv>")
        sys.exit(1)
    
    compare_runs(sys.argv[1], sys.argv[2])