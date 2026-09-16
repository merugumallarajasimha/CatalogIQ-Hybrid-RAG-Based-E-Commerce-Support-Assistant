# CatalogIQ — Evolution Comparison

This document compares **earlier project state** vs **current approved configuration**, including model swaps, pipeline changes, and measured results.

**Locked production defaults (approved 2026-09-16):**

| Component | Model / setting |
|-----------|-----------------|
| Embedder (ingest + query) | `sentence-transformers/all-MiniLM-L6-v2` (384d) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | OmniRoute (`OMNIROUTE_MODEL`, typically `auto` / random routing) — **not benchmark-selected** |
| Collection | `support_docs` (Qdrant `@ localhost:6333`) |
| Candidate pool | Dense 20 + BM25 20 → RRF top 15 → rerank top 5 |

---

## 1. End-to-end pipeline

### Earlier

```
User Question
     ↓
Sparse (CRC32 TF hash in Qdrant) + Dense
     ↓
RRF (often top_k=20, internal fetch 40/40)
     ↓
Cross-Encoder → Top 5
     ↓
LLM → Citation Validation
```

### Current

```
User Question
     ↓
Identifier-aware routing (exact / bm25_only / hybrid)
     ↓
Hybrid path: Dense (Qdrant) + BM25Okapi (in-memory) → RRF (k=60)
     ↓
Top 15 fused candidates
     ↓
Cross-Encoder → Top 5
     ↓
OmniRoute LLM → Citation Validation
```

**Important naming note:** Do not claim “BM25 + SPLADE” for the Qdrant sparse channel. Ingest still stores a **hash-based lexical sparse vector** in Qdrant. True **BM25Okapi** (`rank_bm25`) is a separate in-memory retrieval channel in `src/bm25_search.py`. Prefer: *hybrid dense + BM25 lexical retrieval with RRF*.

---

## 2. Latency: cold vs warm vs optimized

| Stage | Problematic cold / early run | After warm-up + smaller pools | Notes |
|-------|------------------------------|-------------------------------|-------|
| Dense / hybrid search | ~16–17 s (model load) | ~100–110 ms warm | First request paid embedder load |
| Reranker | ~10–11 s (model load) / ~2.7 s for 8 pairs | ~330 ms for 5 pairs warm | Candidate count dominates CPU |
| Generation | ~14–23 s | ~15 s (OmniRoute) | LLM remains the E2E bottleneck |
| **Total** | **~45–52 s** (cold) / **~81 s** (bad top_k=8 cold cycle) | **~15.8 s** warm | ~80% reduction vs worst cold cycle |

### Warm API example (historical instrumented run)

| Stage | Time |
|-------|------|
| Sparse (legacy debug) | ~13 ms |
| Dense | ~86 ms |
| RRF | &lt;1 ms |
| Hybrid total | ~100–111 ms |
| Rerank (5 pairs) | ~331 ms |
| Generation | ~15.3 s |
| **Total** | **~15.8 s** |

**Retrieval-only SLA check (&lt;1.2 s E2E retrieval target):** confirmation eval mean for full RRF+reranker ≈ **920 ms** (50 queries) — under 1.2 s mean; p95 can still spike on heavy rerank.

---

## 3. Dataset & ingestion

| Item | Earlier | Current |
|------|---------|---------|
| Source focus | Small manuals / parts / tickets | Same support corpus + optional Amazon shopping parquet (excluded from loader) |
| Shopping parquet | Accidentally processed | Excluded via `EXCLUDED_FILE_PATTERNS` |
| Records after dedup | ~43 (original) / noisy 66 with JSON+CSV dups | Normalized support set used for BM25 + eval; ingest capped |
| Ingest cap | Unbounded / unclear | `MAX_INGEST_RECORDS=1000` (benchmark-friendly on 2-core/8 GB) |
| Categories | Often `"unknown"` | Taxonomy mapped; defaults tracked |
| Part numbers in text | Singular `part_number` missed | Singular + plural both included |
| HTML | Present in text | Stripped |
| Confidence fields | None | `confidence_flags`, `extraction_method`, `has_sku`, `has_category` |

Shopping dataset scale (discovered, not fully ingested):

- Examples: 2,621,288  
- Products: 1,814,924  
- Sources: 130,652  
- Deliberately **not** fully ingested during RAG benchmarking.

---

## 4. Embedding model comparison

| Candidate | Role | Recall@5 | MRR | nDCG@5 | Query latency (mean) | Decision |
|-----------|------|----------|-----|--------|----------------------|----------|
| **all-MiniLM-L6-v2** | **Baseline / locked** | **0.875** | **0.858** | **0.861** | **~36 ms** | **KEEP** |
| BAAI/bge-base-en-v1.5 | Candidate (768d, separate collection `support_docs_bge_base`) | 0.880 | 0.856 | 0.863 | ~143 ms | Rejected — +0.005 R@5 costs ~+107 ms |
| paraphrase-multilingual-MiniLM-L12-v2 | Earlier `.env` / multilingual experiment | Used historically | — | — | — | Superseded by approved MiniLM default; ingest+query must stay matched |

**Rule that was fixed:** ingest and query must use the **same** embedding model and dimension.

---

## 5. Reranker comparison (100 queries, frozen MiniLM RRF top-15)

| Candidate | Recall@5 | MRR | nDCG@5 | Mean latency | Decision |
|-----------|----------|-----|--------|--------------|----------|
| **ms-marco-MiniLM-L-6-v2** | **0.863** | **0.875** | **0.862** | **~1642 ms** | **KEEP (locked)** |
| BAAI/bge-reranker-base | 0.863 | 0.843 | 0.840 | ~3629 ms | Reject — same R@5, worse ranking, ~2.2× slower |
| BAAI/bge-reranker-large | 0.863 | 0.857 | 0.849 | ~8175 ms | Reject — ~5× slower |
| mixedbread-ai/mxbai-rerank-base-v1 | 0.863 | 0.858 | 0.852 | ~39 s | Reject — unusable latency |
| BAAI/bge-reranker-v2-m3 | Earlier production default (pre-benchmark) | — | — | — | Replaced after MiniLM won on accuracy+speed |

**SLA flag:** even the winning reranker’s *standalone* mean in the isolated bench can exceed 1.2 s; warm API path with 5 pairs is ~331 ms. Prefer warm, small-candidate measurements for ops; use ablation means for model selection fairness.

---

## 6. LLM / generation

| Item | Earlier | Current |
|------|---------|---------|
| Client | Ad-hoc / fragile | Cached `OpenAI` client via OmniRoute |
| Config | Hardcoded | `OMNIROUTE_BASE_URL`, `OMNIROUTE_MODEL`, `OMNIROUTE_API_KEY` |
| Response parsing | Broke on list/Gemini-style content | `extract_text_from_content()` handles str/list/dict |
| ASCII scrub | `encode("ascii","replace")` mangled text | Removed; Unicode preserved |
| Prompt | Weaker grounding | Use only retrieved docs; cite `[Doc: id]`; refuse when missing |
| Truncation | Unbounded risk | 2500 chars/doc |
| Temp / tokens | Varied | `TEMPERATURE=0.1`, `MAX_TOKENS=512` |
| Model selection | Assumed fixed | OmniRoute **random/auto routing** → **LLM bench cancelled** (not used for winner pick) |

Smoke notes: `auto` hit HTTP 503 (combo retry limit); direct `groq/openai/gpt-oss-120b` once ~6.8 s — not used for production selection.

---

## 7. Retrieval ablation (confirmation, 50 queries, locked stack)

| Config | Recall@5 | MRR | nDCG@5 | Latency (ms) |
|--------|----------|-----|--------|--------------|
| A. Dense only | 0.983 | 0.990 | 0.983 | 67.1 |
| B. BM25 only | 1.000 | 0.970 | 0.978 | 0.4 |
| C. Dense + BM25 (concat) | 1.000 | 0.990 | 0.994 | 47.9 |
| D. Dense + BM25 + RRF | 1.000 | 0.990 | 0.993 | 33.4 |
| **E. RRF + Reranker (final)** | **0.993** | **1.000** | **0.994** | **919.7** |

Failures (strict Top-5 miss on relevant queries): **0** on this confirmation run.

**Insight pattern across phases:** BM25 dominates identifier-heavy queries; dense helps semantic/troubleshooting; reranker often improves MRR/nDCG but can drop multi-SKU recall — hence routing (`exact` / `bm25_only` / `hybrid`) matters.

---

## 8. What changed in code (high level)

| Area | Change |
|------|--------|
| `load_dataset.py` | Exclude shopping files; category map; part_number singular; dedup; HTML strip; confidence flags |
| `bm25_search.py` | Real BM25Okapi channel (replaces CRC32 as primary lexical scorer for search) |
| `exact_search.py` | Identifier lookup for SKUs / parts / ASINs |
| `search.py` | Routing; Dense+BM25+RRF constants; hybrid diagnostics |
| `reranker.py` | Default → `ms-marco-MiniLM-L-6-v2` (approved) |
| `ingest.py` / `search.py` | Default embedder → `all-MiniLM-L6-v2` (approved, matched) |
| `generation.py` | OmniRoute env config, robust extraction, citations, timing |
| `canonical_builder.py` / `field_chunker.py` / `chunk_validator.py` | Field-aware chunking path |
| `evaluation.py` + metrics + benchmarks | Ablation + embedder/reranker/LLM bench harnesses |
| `main.py` | FastAPI ask path: hybrid → rerank → generate → cite; timing fields |

---

## 9. Performance improvement summary

| Metric | Before (worst / cold) | After (warm / optimized) |
|--------|----------------------|---------------------------|
| Search | ~17 s | ~0.10 s |
| Rerank | ~11 s | ~0.33 s |
| Generation | ~23 s | ~15 s |
| Total | ~52 s (up to ~81 s) | ~16 s |
| Retrieval quality (final E) | Fragile / mismatched models | R@5 0.993, MRR 1.0 @ 50 queries |

Generation still dominates wall-clock; retrieval+rerank is in the sub-second warm band under the locked models.

---

## 10. Artifacts to cite

| Path | Contents |
|------|----------|
| `evaluation/benchmarks/baseline/baseline_dense_only.json` | MiniLM dense bench |
| `evaluation/benchmarks/bge_base_dense_only.json` | BGE-base dense bench |
| `evaluation/benchmarks/rerankers/summary.json` | Reranker bake-off |
| `evaluation/benchmarks/final_baseline/confirmation_summary.json` | Approved config + 50-query confirmation |
| `evaluation/results/evaluation_results.json` | Latest ablation per-query output |
| `PROJECT_SUMMARY_2.0.md` | Full architecture + file map |

---

*Generated to replace outdated narrative in the original `PROJECT_SUMMARY.md`. Prefer this file + `PROJECT_SUMMARY_2.0.md` for resumes, demos, and handoffs.*
