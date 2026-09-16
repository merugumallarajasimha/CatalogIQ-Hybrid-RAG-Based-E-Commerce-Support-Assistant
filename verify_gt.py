import json

# Load queries
with open('evaluation/queries.jsonl', 'r') as f:
    queries = [json.loads(line) for line in f]

# Load normalized records to get product names
with open('data/normalized_records.json', 'r') as f:
    records = json.load(f)

# Build SKU -> product_name mapping
sku_to_name = {}
for r in records:
    sku = r.get('sku')
    name = r.get('product_name')
    if sku and name and sku not in sku_to_name:
        sku_to_name[sku] = name

# Print verification report
print('=' * 120)
print('VERIFICATION REPORT: Ground Truth Dataset (evaluation/queries.jsonl)')
print('=' * 120)
print(f'{"#":<4} {"Query":<60} {"Expected SKU(s)":<35} {"Product Name(s)":<40} {"Status":<15}')
print('-' * 120)

all_verified = True
for i, q in enumerate(queries, 1):
    query = q['query'][:58]
    skus = q['relevant_skus']
    sku_str = ', '.join(skus) if skus else '(none)'
    
    names = []
    for sku in skus:
        names.append(sku_to_name.get(sku, 'UNKNOWN: ' + sku))
    name_str = '; '.join(names)[:38]
    
    # Check if all SKUs exist in our data
    verified = all(sku in sku_to_name for sku in skus) if skus else True
    status = 'VERIFIED' if verified else 'MISSING SKU'
    if not verified:
        all_verified = False
    
    print(f'{i:<4} {query:<60} {sku_str:<35} {name_str:<40} {status:<15}')

print('-' * 120)
print(f'Total: {len(queries)} queries | All verified: {all_verified}')
print('=' * 120)