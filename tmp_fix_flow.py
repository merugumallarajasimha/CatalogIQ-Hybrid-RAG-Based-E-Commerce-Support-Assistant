import os
path = "src/search.py"
content = open(path, 'r', encoding='utf-8').read()

# Fix 1: individual searches inside hybrid_search should use DENSE_TOP_K, not top_k
old_sparse = """    sparse_results = sparse_search(
        query,
        top_k
    )"""
new_sparse = """    sparse_results = sparse_search(
        query,
        DENSE_TOP_K
    )"""
content = content.replace(old_sparse, new_sparse)

# Fix 2: dense search should use DENSE_TOP_K
old_dense = """    dense_results = dense_search(
        query,
        top_k
    )"""
new_dense = """    dense_results = dense_search(
        query,
        DENSE_TOP_K
    )"""
content = content.replace(old_dense, new_dense)

# Fix 3: BM25 search should use BM25_TOP_K
old_bm25 = "bm25_results_raw = bm25.search(query, top_k=BM25_TOP_K)"
new_bm25 = "bm25_results_raw = bm25.search(query, top_k=BM25_TOP_K)"
# This is already correct

# Fix 4: Return RRF_TOP_K instead of top_k for hybrid_search
old_return = "return fused[:top_k]"
new_return = "return fused[:RRF_TOP_K]"
content = content.replace(old_return, new_return)

# Fix 5: hybrid_search default should return RRF_TOP_K (15) not FINAL_TOP_K (5)
old_default = "top_k: int = FINAL_TOP_K,"
new_default = "top_k: int = RRF_TOP_K,"
content = content.replace(old_default, new_default)

open(path, 'w', encoding='utf-8').write(content)
print("Fixed search.py pipeline flow")
