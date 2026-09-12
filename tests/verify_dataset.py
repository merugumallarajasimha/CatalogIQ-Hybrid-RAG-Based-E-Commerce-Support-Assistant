import json

# Check product_manuals.json
with open('dataset/product_manuals.json') as f:
    manuals = json.load(f)
print('=== product_manuals.json ===')
print(f'Records: {len(manuals)}')
for m in manuals:
    print(f'  SKU: {m["sku"]} | Product: {m["product_name"]} | Category: {m["category"]} | Doc Type: {m["doc_type"]} | Content length: {len(m["content"])} chars')

print()

# Check parts_catalog.json
with open('dataset/parts_catalog.json') as f:
    parts = json.load(f)
print('=== parts_catalog.json ===')
print(f'Records: {len(parts)}')
for p in parts[:5]:
    print(f'  Part: {p["part_number"]} | Desc: {p["description"][:60]}... | Compatible: {p["compatible_skus"]}')
print('  ... and', len(parts)-5, 'more')

print()

# Check troubleshooting_tickets.json
with open('dataset/troubleshooting_tickets.json') as f:
    tickets = json.load(f)
print('=== troubleshooting_tickets.json ===')
print(f'Records: {len(tickets)}')
for t in tickets[:5]:
    print(f'  Ticket: {t["ticket_id"]} | SKU: {t["sku"]} | Issue: {t["issue_type"]} | Parts: {t["part_numbers"]}')
print('  ... and', len(tickets)-5, 'more')