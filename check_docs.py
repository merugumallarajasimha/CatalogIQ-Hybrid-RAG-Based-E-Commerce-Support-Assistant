import sys
sys.path.insert(0, 'src')
from search import get_document

for doc_id in ['CHAIR-ERG-X99-0', 'CHAIR-ERG-X99-1', 'CHAIR-ERG-X99-5']:
    doc = get_document(doc_id)
    print(f'=== {doc_id} ===')
    print(f'SKU: {doc.get("sku")}')
    print(f'Content: {doc.get("content", "")[:500]}')
    print()