"""
Build canonical product-document format from real data sources.
Only includes fields that actually exist in the source data.
"""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def build_canonical_from_manual(record):
    """Build canonical doc from product_manuals.json record."""
    lines = []
    lines.append(f"Product Name: {record.get('product_name', '')}")
    lines.append(f"SKU: {record.get('sku', '')}")
    lines.append(f"Category: {record.get('category', '')}")
    lines.append(f"Document Type: {record.get('doc_type', '')}")
    lines.append(f"Title: {record.get('title', '')}")
    lines.append(f"Version: {record.get('version', '')}")
    lines.append(f"Last Updated: {record.get('last_updated', '')}")
    lines.append("")
    lines.append("Description:")
    lines.append(record.get('content', ''))
    return "\n".join(lines)


def build_canonical_from_parts(record):
    """Build canonical doc from parts_catalog.json record."""
    lines = []
    lines.append(f"Product Name: {record.get('description', '')}")
    lines.append(f"Part Number: {record.get('part_number', '')}")
    lines.append(f"Category: {record.get('category', '')}")
    lines.append(f"Material: {record.get('material', '')}")
    lines.append(f"Weight (kg): {record.get('weight_kg', '')}")
    lines.append(f"Stock Status: {record.get('stock_status', '')}")
    lines.append(f"Unit Cost (USD): {record.get('unit_cost_usd', '')}")
    lines.append("")
    lines.append("Description:")
    lines.append(record.get('description', ''))
    lines.append("")
    lines.append("Replacement Procedure:")
    lines.append(record.get('replacement_procedure', ''))
    if record.get('related_parts'):
        lines.append("")
        lines.append("Related Parts:")
        for p in record['related_parts']:
            lines.append(f"- {p}")
    if record.get('compatible_skus'):
        lines.append("")
        lines.append("Compatible SKUs:")
        for s in record['compatible_skus']:
            lines.append(f"- {s}")
    return "\n".join(lines)


def build_canonical_from_ticket(record):
    """Build canonical doc from troubleshooting_tickets.json record."""
    lines = []
    lines.append(f"Product Name: {record.get('product_name', '')}")
    lines.append(f"SKU: {record.get('sku', '')}")
    lines.append(f"Ticket ID: {record.get('ticket_id', '')}")
    lines.append(f"Category: {record.get('category', '')}")
    lines.append(f"Issue Type: {record.get('issue_type', '')}")
    lines.append(f"Status: {record.get('status', '')}")
    lines.append(f"Created Date: {record.get('created_date', '')}")
    if record.get('resolved_date'):
        lines.append(f"Resolved Date: {record.get('resolved_date', '')}")
    if record.get('agent_id'):
        lines.append(f"Agent ID: {record.get('agent_id', '')}")
    if record.get('part_numbers'):
        lines.append("")
        lines.append("Part Numbers:")
        for p in record['part_numbers']:
            lines.append(f"- {p}")
    lines.append("")
    lines.append("Customer Description:")
    lines.append(record.get('customer_description', ''))
    lines.append("")
    lines.append("Agent Resolution:")
    lines.append(record.get('agent_resolution', ''))
    if record.get('resolution_steps'):
        lines.append("")
        lines.append("Resolution Steps:")
        for i, step in enumerate(record['resolution_steps'], 1):
            lines.append(f"{i}. {step}")
    if record.get('parts_used'):
        lines.append("")
        lines.append("Parts Used:")
        for p in record['parts_used']:
            lines.append(f"- {p}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Load data
    with open(os.path.join(DATA_DIR, "product_manuals.json"), encoding="utf-8") as f:
        manuals = json.load(f)
    with open(os.path.join(DATA_DIR, "parts_catalog.json"), encoding="utf-8") as f:
        parts = json.load(f)
    with open(os.path.join(DATA_DIR, "troubleshooting_tickets.json"), encoding="utf-8") as f:
        tickets = json.load(f)

    print("=" * 80)
    print("CANONICAL FORMAT EXAMPLE 1: From product_manuals.json")
    print("=" * 80)
    print(build_canonical_from_manual(manuals[0]))

    print("\n" + "=" * 80)
    print("CANONICAL FORMAT EXAMPLE 2: From parts_catalog.json")
    print("=" * 80)
    print(build_canonical_from_parts(parts[0]))

    print("\n" + "=" * 80)
    print("CANONICAL FORMAT EXAMPLE 3: From troubleshooting_tickets.json")
    print("=" * 80)
    print(build_canonical_from_ticket(tickets[0]))