import csv
import sys
sys.path.insert(0, 'src')
from search import hybrid_search, get_document
from reranker import rerank
from generation import generate_answer, validate_citations

with open('test_questions.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)[:5]

for i, row in enumerate(rows):
    q = row['question']
    exp_doc = row['expected_doc_id']
    exp_kw = row['expected_keywords']
    
    print('[%d/5] %s' % (i+1, q))
    
    candidates = hybrid_search(q, top_k=20)
    for c in candidates:
        doc = get_document(c['doc_id'])
        if doc:
            c['content'] = doc.get('content', '')
            c['sku'] = doc.get('sku', 'unknown')
            c['part_numbers'] = doc.get('part_numbers', [])
            c['product_name'] = doc.get('product_name', 'unknown')
    
    reranked = rerank(q, candidates, top_n=5)
    answer = generate_answer(q, reranked)
    
    valid_doc_ids = [d['doc_id'] for d in reranked]
    citation_result = validate_citations(answer, valid_doc_ids)
    
    print('  Answer: %s...' % answer[:150])
    print('  Citations: %s' % citation_result['valid_citations'])
    print()