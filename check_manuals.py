import json

with open('data/product_manuals.json') as f:
    data = json.load(f)
for d in data:
    print(f'SKU: {d["sku"]} | Product: {d["product_name"]} | Category: {d["category"]} | DocType: {d["doc_type"]}')
    print(f'  Content preview: {d["content"][:300]}...')
    print()