"""
Chunk-Quality Validator
Checks every chunk for quality issues and reports failures with examples.
"""
import re
import json
import os
from typing import List, Dict, Any


MIN_CHARS = 20
MAX_CHARS = 3000

REQUIRED_FIELDS = [
    'doc_id', 'parent_product_id', 'chunk_id', 'sku', 'product_name',
    'category', 'chunk_type', 'chunk_index', 'total_chunks', 'content'
]

VALID_CHUNK_TYPES = {
    'product_identity', 'product_description', 'product_features',
    'product_specifications', 'product_variants', 'product_identifiers',
    'troubleshooting'
}


def validate_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run all validation checks on a list of chunks."""
    report = {
        'total_chunks': len(chunks),
        'checks': {},
    }

    # Check 1: Empty content
    empty = [c for c in chunks if not c.get('content', '').strip()]
    report['checks']['empty_content'] = {
        'count': len(empty),
        'examples': empty[:3]
    }

    # Check 2: Too short (below MIN_CHARS)
    too_short = [c for c in chunks if 0 < len(c.get('content', '')) < MIN_CHARS]
    report['checks']['too_short'] = {
        'count': len(too_short),
        'examples': too_short[:3]
    }

    # Check 3: Too long (above MAX_CHARS)
    too_long = [c for c in chunks if len(c.get('content', '')) > MAX_CHARS]
    report['checks']['too_long'] = {
        'count': len(too_long),
        'examples': too_long[:3]
    }

    # Check 4: Missing product_name
    missing_name = [c for c in chunks if not c.get('product_name', '').strip()]
    report['checks']['missing_product_name'] = {
        'count': len(missing_name),
        'examples': missing_name[:3]
    }

    # Check 5: Missing or invalid chunk_type
    missing_type = [c for c in chunks if c.get('chunk_type') not in VALID_CHUNK_TYPES]
    report['checks']['missing_chunk_type'] = {
        'count': len(missing_type),
        'examples': missing_type[:3]
    }

    # Check 6: Broken/truncated sentences
    # A chunk is broken if it ends mid-word without legitimate terminator.
    # Skip chunks ending with apostrophes (Dell'Orto, d') or other legitimate endings.
    broken = []
    for c in chunks:
        content = c.get('content', '').strip()
        if not content:
            continue
        # Skip if content contains apostrophe near end (contractions like Dell'Orto)
        if "'" in content[-5:]:
            continue
        last_char = content[-1]
        if last_char in '.,!?;:\'"\\])}0123456789':
            continue
        words = content.split()
        if words:
            last_word = words[-1]
            if len(last_word) <= 3 and last_word.isalpha():
                broken.append(c)
                if len(broken) >= 3:
                    break
    report['checks']['broken_sentences'] = {
        'count': len(broken),
        'examples': broken[:3]
    }

    # Check 7: Duplicate chunks (only flag if from DIFFERENT products)
    seen_content = {}
    duplicates = []
    for c in chunks:
        content_key = c.get('content', '').strip().lower()
        parent = c.get('parent_product_id', '')
        if content_key and content_key in seen_content:
            prev = seen_content[content_key]
            if prev.get('parent_product_id', '') != parent:
                duplicates.append(c)
        else:
            seen_content[content_key] = c
    report['checks']['duplicates'] = {
        'count': len(duplicates),
        'examples': duplicates[:3]
    }

    # Check 8: Missing required metadata fields
    missing_meta_count = 0
    missing_meta_examples = []
    for c in chunks:
        missing = [f for f in REQUIRED_FIELDS if f not in c or c[f] is None]
        if missing:
            missing_meta_count += 1
            if len(missing_meta_examples) < 3:
                missing_meta_examples.append({**c, '_missing_fields': missing})
    report['checks']['missing_metadata'] = {
        'count': missing_meta_count,
        'examples': missing_meta_examples
    }

    report['total_failures'] = sum(v['count'] for v in report['checks'].values())
    report['passed'] = report['total_failures'] == 0

    return report


def print_report(report: Dict[str, Any]):
    """Print a human-readable validation report."""
    print("=" * 80)
    print("CHUNK VALIDATION REPORT")
    print("=" * 80)
    print(f"Total chunks: {report['total_chunks']}")
    print(f"Total failures: {report['total_failures']}")
    print(f"Passed: {report['passed']}")
    print()

    for check_name, check_data in report['checks'].items():
        print(f"--- {check_name} ---")
        print(f"  Failed: {check_data['count']}")
        for i, example in enumerate(check_data['examples'], 1):
            print(f"  Example {i}:")
            print(f"    doc_id: {example.get('doc_id', 'N/A')}")
            print(f"    chunk_type: {example.get('chunk_type', 'N/A')}")
            content = example.get('content', '')
            print(f"    content (first 150 chars): {content[:150]}")
            if '_missing_fields' in example:
                print(f"    missing fields: {example['_missing_fields']}")
        print()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from field_chunker import build_chunks_from_record

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    with open(os.path.join(data_dir, 'product_manuals.json'), encoding='utf-8') as f:
        manuals = json.load(f)
    with open(os.path.join(data_dir, 'parts_catalog.json'), encoding='utf-8') as f:
        parts = json.load(f)
    with open(os.path.join(data_dir, 'troubleshooting_tickets.json'), encoding='utf-8') as f:
        tickets = json.load(f)

    all_chunks = []
    for m in manuals:
        chunks = build_chunks_from_record(m, 'product_manuals.json', m.get('sku', 'unknown'))
        all_chunks.extend(chunks)
    for p in parts:
        chunks = build_chunks_from_record(p, 'parts_catalog.json', p.get('part_number', 'unknown'))
        all_chunks.extend(chunks)
    for t in tickets:
        chunks = build_chunks_from_record(t, 'troubleshooting_tickets.json', t.get('ticket_id', 'unknown'))
        all_chunks.extend(chunks)

    print(f"Built {len(all_chunks)} chunks from {len(manuals) + len(parts) + len(tickets)} records")

    report = validate_chunks(all_chunks)
    print_report(report)