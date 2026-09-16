# CatalogIQ — Project Summary 2.0

Hybrid RAG-based e-commerce support assistant: retrieve product manuals, parts catalogs, and troubleshooting tickets, then generate grounded answers with citations via OmniRoute.

This document supersedes the narrative in `PROJECT_SUMMARY.md` for current architecture, models, and file roles. For before/after metrics and model bake-offs, see `COMPARISON.md`.

---

## 1. Locked models & runtime services

| Layer | Value |
|-------|--------|
| Dense embedder | `sentence-transformers/all-MiniLM-L6-v2` (384-d, cosine) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM gateway | OmniRoute OpenAI-compatible API (`OMNIROUTE_BASE_URL`, default `http://localhost:20128/v1`) |
| LLM model id | `OMNIROUTE_MODEL` (default `auto\fast` / routed — **not fixed by benchmark**) |
| Vector DB | Qdrant `support_docs` @ `QDRANT_URL` (default `http://localhost:6333`) |
| Lexical BM25 | In-memory `rank_bm25.BM25Okapi` over `data/normalized_records.json` |
| Qdrant sparse | Hash / TF-style sparse vectors at ingest (not SPLADE; not BM25Okapi) |

**Retrieval constants (`src/search.py`):** `DENSE_TOP_K=20`, `BM25_TOP_K=20`, `RRF_TOP_K=15`, `FINAL_TOP_K=5`, `RRF_K=60`.

**Generation:** `TEMPERATURE=0.1`, `MAX_TOKENS=512`, per-doc prompt truncate 2500 chars.

**Ingest cap:** `MAX_INGEST_RECORDS` default `1000`.

---

## 2. Architecture

```
                    ┌─────────────────────┐
                    │   Client / curl     │
                    └──────────┬──────────┘
                               │ POST /ask
                               ▼
                    ┌─────────────────────┐
                    │  FastAPI main.py    │
                    │  validate question  │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ search.py       │  │ reranker.py     │  │ generation.py   │
│ route + hybrid  │→ │ CrossEncoder    │→ │ OmniRoute LLM   │
│ Dense+BM25+RRF  │  │ top 15 → top 5  │  │ + citations     │
└────────┬────────┘  └─────────────────┘  └────────┬────────┘
         │                                         │
    ┌────┴────┐                                    ▼
    ▼         ▼                           Answer + sources
┌───────┐ ┌────────┐                      + timing JSON
│Qdrant │ │BM25    │
│dense  │ │Okapi   │
│(+hash │ │in-mem  │
│sparse)│ │index   │
└───────┘ └────────┘

Offline / batch path:

data/*.json|csv
    → load_dataset.py → normalized_records.json
    → chunker / field_chunker / canonical_builder
    → ingest.py → Qdrant support_docs
    → bm25_search builds from normalized_records at query time
```

### Request path (runtime)

1. **Route** (`detect_identifiers` / `route_query`): `exact` | `bm25_only` | `hybrid` (and related).
2. **Retrieve:** exact lookup and/or BM25 and/or dense + RRF fusion.
3. **Rerank:** cross-encoder scores (query, doc) pairs; keep top 5.
4. **Generate:** grounded prompt with `[Doc: id]` instructions via OmniRoute.
5. **Validate citations:** regex check that cited IDs were retrieved.
6. **Return:** answer, sources, citation check, stage timings.

### Offline path (data → index)

1. Load & clean catalogs (`load_dataset.py`).
2. Optional canonical + field-aware chunking.
3. Embed dense + build sparse payload (`ingest.py`).
4. Upsert into Qdrant; BM25 index rebuilt from normalized JSON when search first needs it.

---

## 3. Directory map

| Path | Role |
|------|------|
| `src/` | Application + pipeline + eval/bench scripts |
| `data/` | Source catalogs + `normalized_records.json` |
| `evaluation/` | Queries, ablation results, model benchmarks, chunking comparison |
| `tests/` | Unit / integration smoke tests |
| `storage/` | Local Qdrant persistence (do not hand-edit) |
| `docs/` | PRD / SRS / TDD / plans (original product docs) |
| `COMPARISON.md` | Evolution + bake-off results |
| `PROJECT_SUMMARY.md` | Legacy summary (prefer this 2.0 file) |
| `requirements.txt` | Python dependencies |

---

## 4. Core runtime modules (`src/`)

| File | Responsibility | Integrates with |
|------|----------------|-----------------|
| `main.py` | FastAPI app, `/ask`, env validation, orchestration | `search`, `reranker`, `generation` |
| `search.py` | Embedder, Qdrant client, BM25 hook, routing, hybrid RRF, `get_document` | `exact_search`, `bm25_search`, Qdrant, SentenceTransformer |
| `reranker.py` | Lazy-loaded CrossEncoder; top-N rerank | Called from `main` / evaluation |
| `generation.py` | OmniRoute client, prompt, truncate, extract text, citations, timing | OpenAI SDK → OmniRoute |
| `ingest.py` | Create collection, dense embed, hash sparse, payload, batch upsert | Qdrant, `normalized_records.json` |
| `load_dataset.py` | Load JSON/CSV, exclude shopping parquet noise, dedup, HTML strip, confidence flags | Writes `data/normalized_records.json` |
| `bm25_search.py` | `BM25Search` (`BM25Okapi`, k1=1.5, b=0.75) | Built from normalized records; used by `search.py` |
| `exact_search.py` | Identifier presence + Qdrant exact lookup helpers | `extract_ids`, Qdrant |
| `extract_ids.py` | Regex SKUs, part numbers, ASINs | Exact search, enrichment, routing |
| `query_router.py` | Query classification helpers (category routing experiments) | Eval / experiments |
| `metadata_filter.py` | Metadata-aware filter apply/reject logic | Search experiments |
| `chunker.py` | Sentence / length-based chunking (strategy B) | Ingest / comparison |
| `canonical_builder.py` | Canonical product-document text from source types | Field chunker |
| `field_chunker.py` | Field-aware semantic sections (identity, specs, troubleshooting, …) | Chunk validator |
| `chunk_validator.py` | Quality checks (empty, length, metadata, duplicates, …) | Chunking pipeline |
| `llm_client.py` | Alternate / shared LLM client helpers | Generation experiments |

---

## 5. Evaluation & benchmarking modules

| File | Responsibility |
|------|----------------|
| `evaluation.py` | Ablation runner (dense / BM25 / concat / RRF / +reranker); writes `evaluation/results/*` |
| `retrieval_metrics.py` | Recall@k, Precision@k, MRR, nDCG@k, hit rate, latency helpers |
| `embedding_benchmark.py` | Ingest/evaluate alternate dense models into side collections |
| `reranker_benchmark.py` | Freeze RRF candidates; compare cross-encoders |
| `llm_benchmark.py` | Frozen retrieval context → score generation (cancelled for production selection) |
| `generation_evaluation.py` | Generation-quality helpers |
| `verbose_test.py` | Verbose stage debugging |

Root helpers (not package modules): `fast_ablation.py`, `compare_eval_runs.py`, `enrich_parts.py`, `test_pipeline.py`, `verify_gt.py`, etc. — experimental / one-off scripts around the same eval stack.

---

## 6. Evaluation data layout

| Path | Purpose |
|------|---------|
| `evaluation/queries.jsonl` | Ground-truth queries (SKU, parts, troubleshooting, adversarial, …) |
| `evaluation/results/` | Latest ablation CSV/JSON + failure analysis |
| `evaluation/benchmarks/baseline/` | MiniLM dense metrics |
| `evaluation/benchmarks/bge_base_*` | BGE-base ingest + dense metrics |
| `evaluation/benchmarks/rerankers/` | Reranker bake-off + frozen RRF caches |
| `evaluation/benchmarks/llms_smoke/` | Incomplete LLM smoke (auto 503 / single-shot) |
| `evaluation/benchmarks/final_baseline/` | Approved config confirmation (50 queries) |
| `evaluation/chunking_comparison/` | Strategy A/B/C chunk stats (Step 8) |

---

## 7. Tests (`tests/`)

| File | What it checks |
|------|----------------|
| `test_individual_search.py` | Sparse/dense-style individual search |
| `test_hybrid_search.py` | Hybrid / RRF ordering |
| `test_full_rerank_pipeline.py` | Hybrid → rerank timings/order |
| `test_full_generation.py` | Full RAG + citations |
| `test_sku_rerank.py` | Exact SKU through hybrid + rerank |
| `test_reranker_alone.py` | Cross-encoder ranking sanity |
| `test_omniroute_connection.py` | OmniRoute connectivity |
| `verify_dataset.py` | Dataset structure sanity |

---

## 8. Data files (`data/`)

| File | Role |
|------|------|
| `product_manuals.json` / `.csv` | Manual-style support docs |
| `parts_catalog.json` / `.csv` | Parts, materials, procedures |
| `troubleshooting_tickets.json` / `.csv` | Tickets / resolutions |
| `normalized_records.json` | Loader output used by ingest + BM25 |
| `shopping_queries_dataset_*` | Large Amazon-style set — **excluded** from loader by pattern |

---

## 9. How modules integrate (call graph)

```
main.ask
  ├─ search.hybrid_search / routed path
  │    ├─ exact_search (if identifiers)
  │    ├─ bm25_search.BM25Search
  │    ├─ SentenceTransformer.encode → Qdrant dense
  │    └─ RRF fuse → top_k candidates
  ├─ search.get_document (payload enrichment)
  ├─ reranker.rerank
  ├─ generation.generate_answer
  └─ generation.validate_citations

ingest.main
  ├─ load normalized_records.json
  ├─ SentenceTransformer.encode (dense)
  ├─ compute_sparse_vector (hash TF)
  └─ Qdrant upsert support_docs

evaluation.run_evaluation
  ├─ same retrieval stages as ablation configs
  ├─ retrieval_metrics.compute_metrics
  └─ write evaluation/results/*
```

---

## 10. Confirmed performance snapshot (approved stack)

**50-query confirmation** (`evaluation/benchmarks/final_baseline/confirmation_summary.json`):

| Config | R@5 | MRR | nDCG@5 | Latency |
|--------|-----|-----|--------|---------|
| Dense only | 0.983 | 0.990 | 0.983 | 67 ms |
| BM25 only | 1.000 | 0.970 | 0.978 | 0.4 ms |
| Dense+BM25+RRF | 1.000 | 0.990 | 0.993 | 33 ms |
| **RRF + reranker** | **0.993** | **1.000** | **0.994** | **920 ms** |

Warm API (historical instrumented): retrieval ~100 ms, rerank ~331 ms, generation ~15 s → total ~16 s. Generation dominates E2E; retrieval+rerank meets sub-1.2 s **mean** retrieval SLA on the confirmation set.

---

## 11. Design decisions (short)

1. **Matched embedder** on ingest and query (approved MiniLM).
2. **BM25Okapi** as primary lexical channel for identifier-heavy e-commerce text.
3. **RRF** to fuse dense + BM25; avoid naive concat as primary fusion.
4. **Small rerank pool (5–15)** — largest practical CPU win on this machine.
5. **Identifier routing** — avoid paying dense+rerank when exact/BM25 is enough.
6. **OmniRoute stays configurable** — no LLM “winner” baked into code after cancelled LLM bake-off.
7. **Honest sparse wording** — Qdrant sparse ≠ BM25 ≠ SPLADE.

---

## 12. How to run (typical)

```text
1. Start Qdrant (localhost:6333)
2. Start OmniRoute (localhost:20128) + set OMNIROUTE_API_KEY in src/.env
3. (Optional) python src/load_dataset.py
4. (Optional) python src/ingest.py          # rebuild support_docs
5. uvicorn: python -m src.main  (or project’s usual uvicorn entry)
6. Eval:   python src/evaluation.py --queries evaluation/queries.jsonl
```

Ensure `EMBEDDING_MODEL` / `RERANKER_MODEL` in `.env` (if set) match the locked defaults above, or rely on code defaults.

---

## 13. Related docs

| Doc | Use |
|-----|-----|
| `COMPARISON.md` | Before/after + model bake-off tables |
| `PROJECT_SUMMARY.md` | Legacy (outdated model/latency narrative) |
| `docs/*` | Original PRD / SRS / TDD / implementation & evaluation plans |
| `evaluation/benchmarks/final_baseline/confirmation_summary.json` | Machine-readable locked config + confirmation metrics |

---

*CatalogIQ Hybrid RAG — Project Summary 2.0 (aligned with approved MiniLM + ms-marco baselines, 2026-09-16).*
