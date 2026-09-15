import json, csv, os, re

test_csv = os.path.join('test_questions.csv')
output_jsonl = os.path.join('evaluation', 'queries.jsonl')
os.makedirs('evaluation', exist_ok=True)

queries = []
with open(test_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        question = row.get('question', '').strip().strip('"')
        expected_doc = row.get('expected_doc_id', '').strip().strip('"')
        question_type = row.get('question_type', '').strip().strip('"')

        if not question or question == 'NONE':
            continue

        relevant_skus = []

        if expected_doc and expected_doc != 'NONE':
            sku_match = re.search(r'([A-Z]{2,}-[A-Z0-9-]+)', expected_doc)
            if sku_match:
                relevant_skus.append(sku_match.group(1))

        if question_type == 'exact_sku':
            patterns = [
                r'([A-Z]{2,}-[A-Z0-9-]+)',
                r'(B0[A-Z0-9]{6,10})',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, question)
                for m in matches:
                    if m not in relevant_skus and len(m) >= 5:
                        relevant_skus.append(m)

        relevant_skus = list(dict.fromkeys(relevant_skus))[:3]

        if relevant_skus:
            queries.append({
                "query": question,
                "relevant_skus": relevant_skus,
                "_flagged": question_type != 'exact_sku',
            })

with open(output_jsonl, 'w', encoding='utf-8') as f:
    for q in queries:
        f.write(json.dumps(q, ensure_ascii=False) + '\n')

print(f"Wrote {len(queries)} queries to {output_jsonl}")
for q in queries[:10]:
    flagged = " [FLAGGED]" if q.get("_flagged") else ""
    print(f"  {q['query'][:60]}... -> {q['relevant_skus']}{flagged}")
