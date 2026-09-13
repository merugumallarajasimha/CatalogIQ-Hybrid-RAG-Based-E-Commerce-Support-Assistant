import sys
sys.path.insert(0, 'src')
from search import sparse_search, dense_search, hybrid_search, get_document

q = 'What is the torque spec for screw S-4012-SCRW?'
print('Testing sparse search...')
sparse = sparse_search(q, top_k=20)
print('Sparse results:', len(sparse))
for doc_id, rank in sparse[:5]:
    print('  Rank {}: {}'.format(rank, doc_id))

print()
print('Testing dense search...')
dense = dense_search(q, top_k=20)
print('Dense results:', len(dense))
for doc_id, rank in dense[:5]:
    print('  Rank {}: {}'.format(rank, doc_id))

print()
print('Testing hybrid search...')
hybrid = hybrid_search(q, top_k=20)
print('Hybrid results:', len(hybrid))
for r in hybrid[:5]:
    print('  Score {:.4f}: {} (sparse={}, dense={})'.format(
        r["combined_score"], r["doc_id"], r["rank_in_sparse"], r["rank_in_dense"]))