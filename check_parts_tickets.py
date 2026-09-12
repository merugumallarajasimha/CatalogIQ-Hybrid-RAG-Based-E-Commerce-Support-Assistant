import json

with open('data/parts_catalog.json') as f:
    data = json.load(f)
for d in data:
    print(f'Part: {d["part_number"]} | SKU: {d["compatible_skus"]} | Desc: {d["description"][:100]}')

print("\n\n--- Tickets ---")
with open('data/troubleshooting_tickets.json') as f:
    data = json.load(f)
for d in data:
    print(f'Ticket: {d["ticket_id"]} | SKU: {d["sku"]} | Issue: {d["issue_type"]} | Parts: {d["part_numbers"]}')