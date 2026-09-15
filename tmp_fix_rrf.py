import os
path = "src/search.py"
content = open(path, 'r', encoding='utf-8').read()

old_fused = '''        fused.append({
            "doc_id": doc_id,
            "combined_score": score,
            "rank_in_sparse":
                sparse_ranks.get(doc_id),
            "rank_in_dense":
                dense_ranks.get(doc_id),
            "rank_in_bm25":
                bm25_results.index(doc_id) + 1
                if doc_id in bm25_results
                else None,
        })'''

new_fused = '''        retrieval_sources = []
        if doc_id in sparse_ranks:
            retrieval_sources.append("sparse")
        if doc_id in dense_ranks:
            retrieval_sources.append("dense")
        if doc_id in bm25_results:
            retrieval_sources.append("bm25")

        fused.append({
            "doc_id": doc_id,
            "rrf_score": score,
            "dense_rank": dense_ranks.get(doc_id),
            "bm25_rank": (
                bm25_results.index(doc_id) + 1
                if doc_id in bm25_results
                else None
            ),
            "retrieval_sources": retrieval_sources,
        })'''

if old_fused in content:
    content = content.replace(old_fused, new_fused)
    open(path, 'w', encoding='utf-8').write(content)
    print("Updated RRF fusion evidence")
else:
    print("Pattern not found")
