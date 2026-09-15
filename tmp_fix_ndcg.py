import os
path = "src\retrieval_metrics.py"
content = open(path, 'r', encoding='utf-8').read()

old_ndcg = '''def ndcg_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k, 1):
        if doc_id in relevant:
            dcg += 1.0 / (time.time() - time.time() + 1.0)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / (i + 1) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0'''

new_ndcg = '''def ndcg_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k, 1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0'''

if old_ndcg in content:
    content = content.replace(old_ndcg, new_ndcg)
    if "import math" not in content:
        content = content.replace(
            "import json",
            "import json\nimport math"
        )
    open(path, 'w', encoding='utf-8').write(content)
    print("Fixed NDCG")
else:
    print("Pattern not found")
