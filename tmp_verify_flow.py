content = open('src/search.py', 'r', encoding='utf-8').read()

# Verify key changes
checks = [
    ('DENSE_TOP_K', 'individual search uses DENSE_TOP_K'),
    ('sparse_search(\n        query,\n        DENSE_TOP_K', 'sparse uses DENSE_TOP_K'),
    ('dense_search(\n        query,\n        DENSE_TOP_K', 'dense uses DENSE_TOP_K'),
    ('fused[:RRF_TOP_K]', 'returns RRF_TOP_K'),
    ('top_k: int = RRF_TOP_K', 'default param is RRF_TOP_K'),
]
for pattern, desc in checks:
    found = pattern in content
    print(f'{desc}: {\"OK\" if found else \"MISSING\"} ({pattern[:50]}...)')
