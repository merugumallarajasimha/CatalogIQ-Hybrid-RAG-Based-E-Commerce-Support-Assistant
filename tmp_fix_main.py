import os
path = "src/main.py"
content = open(path, 'r', encoding='utf-8').read()

# Fix: Change top_k=5 to top_k=15 for hybrid_search (get RRF top 15)
old = """        candidates = hybrid_search(
            question,
            top_k=5
        )"""
new = """        candidates = hybrid_search(
            question
        )"""
content = content.replace(old, new)

# Fix: Change top_n=5 to top_n=5 for rerank (stays 5) - already correct
# But we need to update the comment to reflect 15 -> 5

open(path, 'w', encoding='utf-8').write(content)
print("Fixed main.py")
