import json

with open('evaluation/queries.jsonl', 'r') as f:
    lines = [json.loads(l) for l in f]

print(f'Total queries: {len(lines)}')
print()

# Count by query_type
from collections import Counter
types = Counter(q['query_type'] for q in lines)
for t, c in sorted(types.items()):
    print(f'  {t}: {c}')

print()
print('All queries:')
for q in lines:
    print(f'  {q["query_type"]}: {q["query"][:80]} -> {q["relevant_skus"]}')