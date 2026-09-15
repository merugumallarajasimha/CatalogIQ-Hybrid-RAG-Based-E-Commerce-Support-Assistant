import os
path = "src/search.py"
content = open(path, 'r', encoding='utf-8').read()
content = content.replace(
    'key=lambda x: x["combined_score"]',
    'key=lambda x: x["rrf_score"]'
)
if 'x["rrf_score"]' in content:
    open(path, 'w', encoding='utf-8').write(content)
    print("Fixed sort key")
else:
    print("Already fixed or not found")
