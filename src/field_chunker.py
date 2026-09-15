"""
Field-aware semantic chunker with logical sections.
Splits canonical documents into named sections (product_identity, product_description,
product_features, product_specifications, product_variants, product_identifiers, troubleshooting).
Preserves metadata on every chunk. Enforces hard rules against harmful splits.
"""
import re
import json
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple


# ============================================================
# SECTION DEFINITIONS
# ============================================================

SECTION_HEADERS = {
    'product_identity': ['Product Name:', 'SKU:', 'Part Number:', 'Category:', 'Document Type:',
                          'Title:', 'Version:', 'Last Updated:', 'Material:', 'Weight (kg):',
                          'Stock Status:', 'Unit Cost (USD):', 'Ticket ID:', 'Issue Type:',
                          'Status:', 'Created Date:', 'Resolved Date:', 'Agent ID:'],
    'product_description': ['Description:'],
    'product_features': ['Features:'],
    'product_specifications': ['Specifications:'],
    'product_variants': ['Compatible SKUs:', 'Related Parts:'],
    'product_identifiers': ['Part Numbers:'],
    'troubleshooting': ['Customer Description:', 'Agent Resolution:', 'Resolution Steps:', 'Parts Used:'],
}


def identify_section(line: str) -> Optional[str]:
    """Identify which logical section a line belongs to."""
    for section, headers in SECTION_HEADERS.items():
        for header in headers:
            if line.strip().startswith(header):
                return section
    return None


def is_header_line(line: str) -> bool:
    """Check if a line is a section header."""
    for headers in SECTION_HEADERS.values():
        for header in headers:
            if line.strip().startswith(header):
                return True
    return False


def split_canonical_into_sections(canonical_text: str) -> List[Tuple[str, str]]:
    """
    Split canonical document text into (section_name, section_content) tuples.
    All product_identity fields stay in ONE section.
    """
    lines = canonical_text.split('\n')
    sections = []
    current_section = 'product_identity'
    current_lines = []

    for line in lines:
        section = identify_section(line)
        if section and section != current_section and current_lines:
            # Save previous section
            content = '\n'.join(current_lines).strip()
            if content:
                sections.append((current_section, content))
            current_section = section
            current_lines = []
        elif section:
            current_section = section
            current_lines = []
        current_lines.append(line)

    # Save last section
    content = '\n'.join(current_lines).strip()
    if content:
        sections.append((current_section, content))

    return sections


def merge_short_standalone_fields(sections: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Merge very short standalone fields into the product_identity chunk.
    A field like just "Color: Black" (under 20 chars) should be part of identity,
    not its own isolated chunk.
    
    Also merges empty description sections (just "Description:" with no content)
    into the identity chunk.
    """
    if not sections:
        return sections

    merged = []
    identity_lines = []

    for section_name, section_content in sections:
        if section_name == 'product_identity':
            identity_lines.append(section_content)
        else:
            # Check if section content (after removing the header) is empty or very short
            content_without_header = section_content
            for header in SECTION_HEADERS.get(section_name, []):
                if content_without_header.startswith(header):
                    content_without_header = content_without_header[len(header):].strip()
                    break
            
            # If section has no real content (just the header) or is very short,
            # merge into identity instead of creating a separate chunk
            if len(content_without_header) < 20:
                identity_lines.append(section_content)
            else:
                # Flush accumulated identity content first
                if identity_lines:
                    merged.append(('product_identity', '\n'.join(identity_lines)))
                    identity_lines = []
                merged.append((section_name, section_content))

    # Flush remaining identity content
    if identity_lines:
        merged.append(('product_identity', '\n'.join(identity_lines)))

    return merged if merged else sections


def split_section_naturally(section_name: str, section_content: str,
                             max_chunk_chars: int = 2000) -> List[str]:
    """
    Split a section into chunks at natural boundaries.
    Never splits mid-spec, mid-feature, or mid-sentence.
    """
    if len(section_content) <= max_chunk_chars:
        return [section_content]

    chunks = []
    lines = section_content.split('\n')
    current_chunk = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > max_chunk_chars and current_chunk:
            chunks.append('\n'.join(current_chunk).strip())
            current_chunk = []
            current_len = 0
        current_chunk.append(line)
        current_len += line_len

    if current_chunk:
        chunks.append('\n'.join(current_chunk).strip())

    return chunks if chunks else [section_content]


def build_chunks_from_record(record: Dict[str, Any], source_file: str,
                             chunk_id_prefix: str = '') -> List[Dict[str, Any]]:
    """
    Build field-aware chunks from a normalized record.
    Returns list of chunk dicts with full metadata.
    """
    from canonical_builder import build_canonical_from_manual, build_canonical_from_parts, build_canonical_from_ticket

    # Determine source type and build canonical text
    if source_file == 'product_manuals.json':
        canonical_text = build_canonical_from_manual(record)
        source_type = 'manual'
    elif source_file == 'parts_catalog.json':
        canonical_text = build_canonical_from_parts(record)
        source_type = 'parts'
    elif source_file == 'troubleshooting_tickets.json':
        canonical_text = build_canonical_from_ticket(record)
        source_type = 'ticket'
    else:
        canonical_text = str(record.get('content', record.get('text', '')))
        source_type = 'unknown'

    # Split into sections
    sections = split_canonical_into_sections(canonical_text)

    # Merge short standalone fields into identity
    sections = merge_short_standalone_fields(sections)

    # Build chunks
    chunks = []
    chunk_index = 0

    for section_name, section_content in sections:
        # Split section if too long
        sub_chunks = split_section_naturally(section_name, section_content)

        for sub_chunk in sub_chunks:
            chunk = {
                'doc_id': f"{chunk_id_prefix}-{chunk_index}",
                'parent_product_id': chunk_id_prefix,
                'chunk_id': f"{chunk_id_prefix}-{chunk_index}",
                'sku': record.get('sku', record.get('part_number', record.get('product_name', 'unknown'))),
                'product_name': record.get('product_name', record.get('description', record.get('product_name', 'unknown'))),
                'category': record.get('category', 'unknown'),
                'brand': record.get('brand', 'unknown'),
                'chunk_type': section_name,
                'chunk_index': chunk_index,
                'total_chunks': 0,
                'content': sub_chunk,
                'source_file': source_file,
                'source_type': source_type,
            }
            chunks.append(chunk)
            chunk_index += 1

    # Update total_chunks
    total = len(chunks)
    for c in chunks:
        c['total_chunks'] = total

    return chunks


if __name__ == '__main__':
    # Quick test
    test_record = {
        'sku': 'TEST-SKU-001',
        'product_name': 'Test Product',
        'category': 'Electronics',
        'content': 'This is a test description.\n\nDescription:\nThis is the product description.',
        'source_file': 'product_manuals.json',
    }
    chunks = build_chunks_from_record(test_record, 'product_manuals.json', 'TEST-SKU-001')
    for c in chunks:
        print(f"Chunk {c['chunk_index']}: type={c['chunk_type']}, content={c['content'][:80]}...")