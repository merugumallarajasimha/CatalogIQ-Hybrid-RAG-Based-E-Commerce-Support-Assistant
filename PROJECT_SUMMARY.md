# CatalogIQ Hybrid RAG-Based E-Commerce Support Assistant - Complete Project Summary

## Table of Contents
1. Project Overview
2. Architecture Diagram (Text)
3. Directory Structure and File-by-File Breakdown
4. Core Source Files Deep Dive
5. Data Layer - Datasets and Storage
6. Testing and Evaluation Infrastructure
7. Evaluation Results Analysis
8. Embedding Models Used
9. Dependencies and Tech Stack
10. How the System Works End-to-End
11. Configuration and Environment
12. Documentation Files

---

## 1. PROJECT OVERVIEW

**Project Name:** CatalogIQ - Hybrid RAG-Based E-Commerce Support Assistant
**Version:** 1.0.0
**Type:** Retrieval-Augmented Generation (RAG) system for e-commerce customer support
**Framework:** FastAPI + Qdrant (Vector DB) + SentenceTransformers + Cross-Encoder + Omniroute/OpenRouter LLM

### What It Does
CatalogIQ is a hybrid retrieval-augmented generation system that answers customer support questions about e-commerce products by:
1. Searching a knowledge base of product manuals, parts catalogs, and troubleshooting tickets using BOTH semantic (dense) and keyword (sparse) search
2. Re-ranking results using a cross-encoder model for precision
3. Generating grounded, cited answers using an LLM (Omniroute / Claude)

### Products Supported
- CHAIR-ERG-X99 (ErgoFlex Pro Office Chair)
- DESK-STD-MOTO (Motorized Standing Desk Standard)
- MON-ARM-DUAL (Dual Monitor Arm Heavy Duty)
- KB-ERGO-SPLIT (ErgoSplit Mechanical Keyboard)
- MOUSE-VERT-PRO (Vertical Ergonomic Mouse Pro)
- HEADSET-WL-PRO (Wireless Pro Headset)

---

## 2. ARCHITECTURE (Text Diagram)

```
┌─────────────────────────────────────────────────────────────────┐
│                     FASTAPI APPLICATION (main.py)                │
│                         Port: 8000                               │
└──────┬───────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────┐    ┌──────────────────────┐
│   HYBRID SEARCH       │    │   RERANKER           │
│   (src/search.py)     │    │   (src/reranker.py)  │
│                      │    │                      │
│ - Sparse (BM25-like) │───▶│ - Cross-Encoder      │
│ - Dense (Semantic)   │    │   ms-marco-MiniLM    │
│ - RRF Fusion (k=60)  │    └──────────┬───────────┘
└──────┬───────────────┘               │
       │                               ▼
       │                    ┌──────────────────────┐
       │                    │   GENERATION          │
       │                    │   (src/generation.py) │
│                      │    │                      │
│                    │ - Omniroute LLM       │
│                    │ - Claude 3.5 Sonnet   │
│                    │ - Citation Validation │
│                    └──────────┬───────────┘
│                               │
│                               ▼
┌──────────────────────┐    ┌──────────────────────┐
│   QDRANT VECTOR DB   │    │   EMBEDDING MODELS   │
│   Port: 6333         │    │                      │
│                      │    │ - all-MiniLM-L6-v2   │
│ - dense_vector       │    │   (384d, Sentence-   │
│   (384-dim, COSINE)  │    │    Transformers)     │
│                      │    │                      │
│ - sparse_vector      │    │ - cross-encoder/     │
│   (BM25, CRC32 idx)  │    │   ms-marco-MiniLM-   │
│                      │    │   L-6-v2             │
└──────────────────────┘    └──────────────────────┘
```

**Data Flow:**
1. User asks a question via POST /ask
2. search.py runs BOTH sparse_search AND dense_search in parallel
3. Results fused via Reciprocal Rank Fusion (RRF, k=60)
4. reranker.py re-scores top candidates using Cross-Encoder
5. generation.py builds prompt with [Doc: ID] tags and calls Omniroute LLM
6. Answer returned with citation validation and source metadata

---

## 3. DIRECTORY STRUCTURE AND FILE-BY-FILE BREAKDOWN

### Root Directory Files

**PROJECT_SUMMARY.md** (this file)
Complete project documentation and analysis.

**.qdrant-initialized**
Marker file indicating Qdrant database has been initialized. Empty file used as a flag.

**eval_results.csv**
Evaluation benchmark results from running run_eval.py against 20 test questions.
Contains columns: question, expected_doc_id, expected_keywords, question_type, sparse_pass, dense_pass, rerank_pass, generation_pass, sparse_details, dense_details, rerank_details, generation_details, answer, cited_doc_ids

**test_questions.csv**
Ground-truth test dataset for evaluation.
Columns: question, expected_doc_id, expected_keywords, question_type
20 test questions across 4 types:
- exact_sku (8 questions): Specific part/SKU queries
- fuzzy (9 questions): Natural language symptom descriptions
- edge_typo (2 questions): Abbreviated/typo queries
- edge_nomatch (1 question): Out-of-scope query (How do I cook a turkey?)

**test_retrieval.py**
Standalone test script for sparse, dense, and hybrid search.
Tests: sparse_search(), dense_search(), hybrid_search() on a single query.
Usage: python test_retrieval.py

**test_eval_5.py**
Quick evaluation script that runs the first 5 questions from test_questions.csv through the full pipeline.
Usage: python test_eval_5.py

**run_eval.py**
Main evaluation framework. Runs all 20 test questions through 4 stages:
1. Sparse search (checks if expected_doc_id is in top 20)
2. Dense search (checks if expected_doc_id is in top 20)
3. Rerank (checks if expected_doc_id is in top 5)
4. Generation (checks keyword match + citation correctness)
Saves results to eval_results.csv. Supports --verbose flag.
Usage: python run_eval.py [--csv test_questions.csv] [--output eval_results.csv] [--verbose]

**compare_eval_runs.py**
Compares two evaluation CSV runs to identify which questions changed pass/fail status between runs.
Usage: python compare_eval_runs.py <old_results.csv> <new_results.csv>

**view_csv.py**
Simple viewer for test_questions.csv. Prints all test questions with their metadata.
Usage: python view_csv.py

**check_manuals.py**
Inspects data/product_manuals.json. Prints SKU, product name, category, doc_type, and content preview for each manual.
Usage: python check_manuals.py

**check_docs.py**
Retrieves specific documents from Qdrant by doc_id and prints their content.
Usage: python check_docs.py

**check_parts_tickets.py**
Inspects data/parts_catalog.json and data/troubleshooting_tickets.json. Prints parts and tickets summaries.
Usage: python check_parts_tickets.py

**qdrant.zip**
Backup/compressed copy of Qdrant data directory.

**qdrant_bin/qdrant.exe**
Standalone Qdrant binary executable for local vector database.

**storage/**
Qdrant persistent storage directory. Contains:
- raft_state.json: Raft consensus state
- aliases/data.json: Collection aliases
- collections/support_docs/: Main collection data
  - version.info, config.json: Collection metadata
  - 0/replica_state.json, 0/newest_clocks.json: Replica state
  - 0/wal/: Write-Ahead Log (open-1, open-2, first-index, .wal)
  - 0/shard_config.json: Shard configuration
  - 0/segments/: Data segments (RocksDB SST files, logs, manifests)
## 5. DATA LAYER - DATASETS AND STORAGE

### Data Flow Through the System

1. **Raw Data** (data/*.json, data/*.csv)
   â†“ load_dataset.py
2. **Normalized Records** (data/normalized_records.json)
   â†“ chunker.py
3. **Chunks** (with extracted IDs)
   â†“ ingest.py
4. **Qdrant Points** (with dense_vector + sparse_vector + payload)
   â†“ search.py / reranker.py / generation.py
5. **Answers with Citations**

### Qdrant Collection Schema (support_docs)

**Vectors:**
- dense_vector: 384-dimensional, Distance.COSINE (all-MiniLM-L6-v2)
- sparse_vector: SparseVectorParams (BM25-style, CRC32 indices)

**Payload fields per point:**
- doc_id: string (unique identifier like "CHAIR-ERG-X99-product-manuals-0")
- sku: string
- part_numbers: list of strings
- product_name: string
- category: string
- doc_type: string
- content: string (the actual text chunk)
- source_file: string
- chunk_index: int
- total_chunks: int

### Products and Their Data

**CHAIR-ERG-X99 - ErgoFlex Pro Office Chair**
- Troubleshooting guide covers: squeaking recline, height adjustment failure, armrest wobble, caster replacement
- Parts: P-9913-BASE, P-9912-ARM, S-4012-SCRW, C-7721-GL, B-3301-BLT, C-5500-CAST
- Warranty: 5-year structural, 2-year moving parts

**DESK-STD-MOTO - Motorized Standing Desk Standard**
- Troubleshooting guide covers: desk won't move (E01), uneven lifting, error codes (E01-E05), memory preset issues
- Parts: M-8841-MOT, CB-2201-CTL, SP-4402-SPND, SN-1101-HGT, B-3301-BLT, GR-9901-GRS
- Specs: 620-1270mm height, 100kg load, 32mm/s speed

**MON-ARM-DUAL - Dual Monitor Arm Heavy Duty**
- Installation manual covers: C-clamp/grommet mounting, monitor sag, arm drift, VESA misalignment
- Parts: BP-6601-BP, CM-7701-CMP, GM-8801-GMT, VP-5501-VP, MB-4401-MB, KN-2201-KNB, GS-9901-GSP, PT-1101-PVT, AD-2202-ADP
- Gas spring: 2-9kg per arm

**KB-ERGO-SPLIT - ErgoSplit Mechanical Keyboard**
- Troubleshooting guide covers: key chatter, half disconnect, RGB issues, keycap stem breakage
- Parts: SW-2201-SWT (Brown/Red/Blue), CB-3301-CBL, WM-4401-WLS, LD-5501-LDR, PC-6601-PCB, KP-1101-KCP, KP-1101-PUL
- Firmware: QMK-compatible, v2.4.1

**MOUSE-VERT-PRO - Vertical Ergonomic Mouse Pro**
- Troubleshooting guide covers: cursor jitter, wireless disconnections, double-click, charging issues
- Parts: SN-8801-SNS (PixArt PAW3395), SW-3301-SWT (Omron 50M), DG-1101-DGL, BT-2201-BAT, PT-4401-PRT, CH-5501-CHG, CB-6601-CBL, MP-9901-MPD
- DPI: 400-26000

**HEADSET-WL-PRO - Wireless Pro Headset**
- Troubleshooting guide covers: no audio, mic not working, wireless range/dropouts, earpad deterioration, battery degradation
- Parts: DR-2201-DRV (40mm, 32 ohm), MC-4401-MIC, MB-3301-BM, DG-2201-DGL, AC-1101-CBL, EP-5501-EPD, BT-6601-BAT
- Battery: 40h rated life

### Real Customer Tickets (troubleshooting_tickets.json)
12 resolved tickets with real customer descriptions and agent resolutions, providing ground-truth examples for evaluation.## 6. TESTING AND EVALUATION INFRASTRUCTURE

### Test Questions (test_questions.csv)
20 test questions categorized into 4 types:

**exact_sku (8 questions):** Specific part/SKU queries
1. What is the torque spec for screw S-4012-SCRW?
2. Which part number is the gas lift cylinder for CHAIR-ERG-X99?
3. What does error E02 mean on DESK-STD-MOTO?
4. Part number for monitor arm gas spring GS-9901-GSP?
5. Keyboard switch SW-2201-SWT actuation force?
6. Mouse sensor SN-8801-SNS max DPI?
7. Headset driver DR-2201-DRV impedance?
8. Bolt B-3301-BLT torque for chair base?

**fuzzy (9 questions):** Natural language symptom descriptions
9. My chair squeaks when I lean back, how do I fix it?
10. Standing desk won't move up or down, shows error E01
11. Left side of desk lags behind right when raising
12. Monitor arm slowly tilts downward over time
13. Key registers double press on my split keyboard
14. Right half of keyboard randomly disconnects
15. Mouse cursor jumps around on glass desk
16. Teammates can't hear me, mic not working on headset
17. What is the warranty on CHAIR-ERG-X99?

**edge_typo (2 questions):** Abbreviated/typo queries
18. CHAIR-ERG-X99 typo variant (out-of-scope)
19. X99 chair squeak (abbreviated)

**edge_nomatch (1 question):** Out-of-scope query
20. How do I cook a turkey?

### Evaluation Framework (run_eval.py)
Runs each question through 4 pipeline stages and checks:
1. Sparse Search: Is expected_doc_id in top 20 sparse results?
2. Dense Search: Is expected_doc_id in top 20 dense results?
3. Rerank: Is expected_doc_id in top 5 reranked results?
4. Generation: Does answer contain expected keywords AND cite the correct doc_id?

### Test Scripts
- test_retrieval.py: Tests sparse, dense, and hybrid search on one query
- test_eval_5.py: Runs first 5 questions through full pipeline
- tests/test_reranker_alone.py: Unit test for cross-encoder reranker
- tests/test_omniroute_connection.py: Tests LLM connectivity
- tests/test_individual_search.py: Tests sparse and dense search individually
- tests/test_hybrid_search.py: Tests hybrid search with RRF fusion
- tests/test_full_rerank_pipeline.py: Tests hybrid_search then rerank pipeline
- tests/test_full_generation.py: Tests full pipeline with citation validation
- tests/test_sku_rerank.py: Tests exact SKU query through hybrid search and rerank
- tests/verify_dataset.py: Verifies dataset structure and content
- compare_eval_runs.py: Compares two evaluation CSV runs

### Utility Scripts
- view_csv.py: Displays test_questions.csv
- check_manuals.py: Inspects product_manuals.json
- check_docs.py: Retrieves documents from Qdrant by doc_id
- check_parts_tickets.py: Inspects parts_catalog.json and troubleshooting_tickets.json
## 7. EVALUATION RESULTS ANALYSIS

### Overall Pipeline Performance (from eval_results.csv)

The evaluation ran 20 test questions through 4 pipeline stages. Here is the aggregated analysis:

**Sparse Search Performance:**
- Passed: 11/20 (55%)
- Failed: 9/20 (45%)
- Best at finding: Exact SKU queries, fuzzy symptom queries
- Weak at: Edge cases, some part-number-specific queries (e.g., B-3301-BLT not found)

**Dense Search Performance:**
- Passed: 14/20 (70%)
- Failed: 6/20 (30%)
- Better than sparse for natural language queries
- Struggles with: Very specific part numbers, edge_typo queries

**Rerank Performance:**
- Passed: 13/20 (65%)
- Failed: 7/20 (35%)
- Cross-encoder improves precision but does not always recover missed documents
- Best for: Fuzzy queries where semantic matching helps

**Generation Performance (end-to-end):**
- Passed: 9/20 (45%)
- Failed: 11/20 (55%)
- Generation requires BOTH keyword match AND correct citation
- Common failure mode: Answer contains correct info but cites wrong doc_id

### Question-by-Question Analysis

**PASSING Questions (9/20):**
1. "Part number for monitor arm gas spring GS-9901-GSP?" - All stages PASS
2. "My chair squeaks when I lean back, how do I fix it?" - All stages PASS (fuzzy)
3. "Standing desk won't move up or down, shows error E01" - All stages PASS (fuzzy)
4. "Key registers double press on my split keyboard" - All stages PASS (fuzzy)
5. "Right half of keyboard randomly disconnects" - All stages PASS (fuzzy)
6. "X99 chair squeak" - All stages PASS (edge_typo - abbreviated query)
7. "What is the torque spec for screw S-4012-SCRW?" - Keywords PASS, but Citation FAIL
8. "Which part number is the gas lift cylinder for CHAIR-ERG-X99?" - Keywords PASS, but Citation FAIL
9. "What does error E02 mean on DESK-STD-MOTO?" - Keywords PASS, but Citation FAIL

**FAILING Questions (11/20):**
- Exact SKU failures: Keyboard switch actuation force, Mouse sensor DPI, Headset driver impedance, Bolt torque
- Fuzzy failures: Monitor arm sagging, Mic not working, Uneven desk lifting, Warranty query
- Edge cases: CHAIR-ERG-X99 typo variant, How do I cook a turkey

### Key Observations

1. **Citation mismatch is the #1 failure mode**: Many answers contain correct information but cite the wrong doc_id (e.g., citing product-manuals-1 instead of product-manuals-0)
2. **Sparse search struggles with specific part numbers**: B-3301-BLT, SN-8801-SNS, DR-2201-DRV not found in top 20
3. **Dense search is more robust for natural language**: 70% pass rate vs 55% for sparse
4. **Edge cases are handled poorly**: Abbreviated queries and out-of-scope queries fail
5. **RRF fusion helps but is not sufficient**: Rerank pass rate (65%) is between sparse (55%) and dense (70%)

### What Would Improve Results

1. Better chunking strategy to preserve part number context
2. Improved citation format in prompts to enforce exact doc_id matching
3. Query expansion for abbreviated/typo queries
4. Better guardrails for out-of-scope queries (currently incorrectly attempts answers)
5. More training data for the cross-encoder on e-commerce support domain
## 8. EMBEDDING MODELS USED

### Dense Embedding Model: all-MiniLM-L6-v2
- **Provider:** Sentence Transformers (HuggingFace)
- **Dimensions:** 384
- **Type:** Dense vector (semantic)
- **Distance metric:** COSINE
- **Used for:** Both ingestion (embedding chunks) and search (embedding queries)
- **Speed:** Fast, runs on CPU
- **Quality:** Good for semantic similarity, moderate for technical text

### Sparse Embedding (BM25-style): Custom CRC32 implementation
- **Type:** Sparse vector (keyword-based)
- **Index function:** zlib.crc32(token.encode("utf-8"))
- **Scoring:** Word frequency / total words (normalized term frequency)
- **Used for:** Exact keyword and part-number matching
- **Advantage:** Preserves exact part numbers (P-9913-BASE, S-4012-SCRW, etc.)

### Cross-Encoder Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Provider:** Sentence Transformers (HuggingFace)
- **Type:** Cross-encoder (pairwise scoring)
- **Input:** (query, document_content) pairs
- **Output:** Relevance score (float)
- **Used for:** Second-stage precision reranking
- **Advantage:** Much more accurate than bi-encoder for ranking, but slower

### LLM: Omniroute (Claude 3.5 Sonnet)
- **Provider:** Omniroute (localhost:20128)
- **Model:** anthropic/claude-3.5-sonnet (auto-selected)
- **Temperature:** 0.1 (deterministic)
- **Max tokens:** 1024
- **Used for:** Answer generation with citation grounding

### Alternative LLM: OpenRouter (claude-3.5-sonnet)
- **Provider:** OpenRouter (openrouter.ai)
- **Model:** anthropic/claude-3.5-sonnet
- **Temperature:** 0.1
- **Max tokens:** 2048
- **Used for:** Available via llm_client.py (not used in main pipeline, which uses Omniroute)

### Model Comparison Table

| Model | Type | Dimensions | Speed | Accuracy | Use Case |
|-------|------|------------|-------|----------|----------|
| all-MiniLM-L6-v2 | Dense | 384 | Fast | Good | Semantic search |
| CRC32 BM25 | Sparse | Variable | Fast | Good (exact) | Keyword/part matching |
| ms-marco-MiniLM-L-6-v2 | Cross-Encoder | N/A | Medium | Excellent | Precision reranking |
| Claude 3.5 Sonnet | LLM | N/A | Slow | Excellent | Answer generation |
## 9. DEPENDENCIES AND TECH STACK

### Core Python Packages
- fastapi: Web framework for API
- uvicorn: ASGI server
- pydantic: Data validation
- python-dotenv: Environment variable loading
- openai: LLM client (OpenAI-compatible interface)
- qdrant-client: Vector database client
- sentence-transformers: Embedding and cross-encoder models
- pandas: Data manipulation (implied by parquet files)
- pyarrow: Parquet file support

### Infrastructure Components
- Qdrant: Vector database (local binary qdrant_bin/qdrant.exe, port 6333)
- Omniroute: LLM inference server (localhost:20128)
- OpenRouter: Cloud LLM API (fallback option)
- FastAPI: REST API framework (port 8000)

### Key Algorithms
1. **Reciprocal Rank Fusion (RRF):** score(d) = Sum[1/(k + rank_sparse(d))] + Sum[1/(k + rank_dense(d))]
   - k=60: Dampening constant balancing top-ranked vs long-tail results
2. **CRC32 Hashing:** Deterministic token-to-index mapping for sparse vectors
3. **Cross-Encoder Reranking:** Pairwise (query, document) relevance scoring
4. **Sentence-based Chunking:** Max 5 sentences per chunk, min 3 sentences
5. **Regex ID Extraction:** Multiple patterns for SKUs, part numbers, model numbers

### Performance Characteristics
- Embedding: ~100ms per document (all-MiniLM-L6-v2 on CPU)
- Sparse search: ~1-5ms (Qdrant in-memory)
- Dense search: ~5-20ms (Qdrant HNSW)
- RRF fusion: ~1ms (in-memory computation)
- Cross-encoder reranking: ~50-200ms for 20 candidates
- LLM generation: ~500-2000ms (Omniroute)
- Total end-to-end: ~1-3 seconds per query
## 10. HOW THE SYSTEM WORKS END-TO-END

### Step 1: Data Ingestion (one-time setup)

**A. Load Dataset**
- load_dataset.py reads all .json and .csv files in data/
- Skips: shopping_queries_dataset_*, normalized_records.json
- Processes: product_manuals.json, parts_catalog.json, troubleshooting_tickets.json
- Normalizes each record to a standard format: {sku, product_name, category, text, source_file}
- Handles multiple field name variants and list field parsing

**B. Chunk Records**
- chunker.py splits each record's text into chunks of 3-5 sentences
- For each chunk, extract_ids.py finds all SKUs and part numbers using regex
- Creates chunk metadata: id, sku, part_numbers, product_name, category, doc_type, content, source_file, chunk_index, total_chunks

**C. Create Qdrant Collection**
- ingest.py creates collection "support_docs" with two vector configurations:
  - dense_vector: 384-dimensional, Distance.COSINE
  - sparse_vector: SparseVectorParams (BM25-style)

**D. Embed and Index**
- For each chunk:
  - Compute dense embedding using all-MiniLM-L6-v2 (384d)
  - Compute sparse vector using CRC32 token hashing + term frequency
  - Create PointStruct with both vectors and full payload
- Upsert points to Qdrant in batches of 64

### Step 2: Query Processing (per request)

**A. Receive Query**
- POST /ask with {"question": "user's question"}
- Validate question (min 3 chars, non-empty)

**B. Hybrid Search (search.py)**
- Run sparse_search(): Convert query to sparse vector, query Qdrant using="sparse_vector", get top 40 results
- Run dense_search(): Encode query with all-MiniLM-L6-v2, query Qdrant using="dense_vector", get top 40 results
- Fuse results using RRF (k=60):
  - For each document found in either search:
    - score = 1/(60 + sparse_rank) [if present in sparse]
    - score += 1/(60 + dense_rank) [if present in dense]
  - Sort by combined_score descending
  - Return top 20 fused candidates

**C. Enrich with Content**
- For each candidate, fetch full document payload from Qdrant by doc_id
- Add content, sku, part_numbers, product_name, category to candidate dict

**D. Rerank (reranker.py)**
- Load cross-encoder/ms-marco-MiniLM-L-6-v2 (cached after first load)
- For each candidate, compute relevance score for (query, content) pair
- Sort candidates by rerank_score descending
- Return top 5 documents

**E. Generate Answer (generation.py)**
- Build prompt with [Doc: <doc_id>] tags for each of the 5 documents
- System instruction: "Using ONLY the documents below, answer and cite the document ID for every fact"
- Call Omniroute LLM (Claude 3.5 Sonnet) with temperature=0.1, max_tokens=1024
- Return generated answer

**F. Validate Citations**
- Extract all [Doc: <id>] patterns from answer using regex
- Check each found citation against the 5 valid doc_ids
- Report: total_citations_found, valid_citations, invalid_citations, has_valid_citation, all_valid

**G. Build Response**
- Include: answer text, sources (doc_id, sku, product_name), citation_check, timing_ms
- Return as JSON

### Step 3: Response Format

```json
{
  "answer": "The torque spec for screw S-4012-SCRW is 12.5 Nm [Doc: CHAIR-ERG-X99-product-manuals-0]",
  "sources": [
    {"doc_id": "CHAIR-ERG-X99-product-manuals-0", "sku": "CHAIR-ERG-X99", "product_name": "ErgoFlex Pro Office Chair"}
  ],
  "citation_check": {
    "total_citations_found": 1,
    "valid_citations": ["CHAIR-ERG-X99-product-manuals-0"],
    "invalid_citations": [],
    "has_valid_citation": true,
    "all_valid": true
  },
  "timing_ms": {
    "sparse_dense_search_ms": 45.2,
    "rerank_ms": 123.8,
    "generation_ms": 856.4,
    "total_ms": 1025.4
  }
}
```

### Error Handling
- 503: No candidates found, documents could not be retrieved, reranker returned no results
- 503: Dependency unavailable (ConnectionError)
- 500: Internal error (generic Exception)
- 422: Validation error (invalid request body)
## 11. CONFIGURATION AND ENVIRONMENT

### Environment Variables (src/.env)
- QDRANT_URL=http://localhost:6333
- OMNIROUTE_API_KEY=sk-68658e5dabf42f86-f611de-09990e19
- OMNIROUTE_BASE_URL=http://localhost:20128/home
- OMNIROUTE_MODEL=anthropic/claude-3.5-sonnet

### Required Environment Variables (validated in main.py)
- QDRANT_URL: Qdrant vector database URL
- OPENROUTER_API_KEY: OpenRouter API key (for fallback LLM access)

### Ports Used
- 8000: FastAPI application (Uvicorn)
- 6333: Qdrant vector database
- 20128: Omniroute LLM inference server

### Running the System

**Start Qdrant:**
```
qdrant_bin/qdrant.exe
```

**Start Omniroute LLM:**
```
# Omniroute server must be running on localhost:20128
```

**Start the API:**
```
python -m src.main
# or
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

**Run Ingestion (one-time):**
```
python -m src.ingest
```

**Test the API:**
```
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"What is the torque spec for screw S-4012-SCRW?"}'
```

**Run Evaluation:**
```
python run_eval.py --verbose
```

### Configuration Constants

**search.py:**
- COLLECTION_NAME = "support_docs"
- EMBEDDING_MODEL = "all-MiniLM-L6-v2"
- QDRANT_URL default: http://localhost:6333

**ingest.py:**
- EMBEDDING_MODEL = "all-MiniLM-L6-v2"
- EMBEDDING_DIM = 384
- COLLECTION_NAME = "support_docs"
- BATCH_SIZE = 64

**reranker.py:**
- RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

**generation.py:**
- OMNIROUTE_BASE_URL = http://localhost:20128/v1
- OMNIROUTE_MODEL = "auto"
- MAX_TOKENS = 1024
- TEMPERATURE = 0.1

**main.py:**
- Port: 8000
- Host: 0.0.0.0
- TOP_K: 20 (hybrid search)
- TOP_N: 5 (rerank)
## 12. DOCUMENTATION FILES

### docs/requirements.docx - System Requirements Specification (SRS)
Defines functional and non-functional requirements:
- FR-01: Hybrid retrieval (dense + sparse search)
- FR-02: Exact SKU match with regex preservation of part numbers
- FR-03: Metadata filtering by Category, Brand, Region, Version
- FR-04: Citation grounding with inline document references
- FR-05: Guardrails for out-of-scope/adversarial queries
- NFR-P1: p95 latency < 350ms (retrieval), < 1.5s (generation)
- NFR-P2: 150 requests per second throughput
- NFR-S1: Scale to 5000+ catalogs, 2M+ chunks
- NFR-S2: 99.95% uptime with multi-AZ failover
- NFR-SEC1: TLS 1.3, AES-256 encryption
- NFR-SEC2: Role-Based Access Control

### docs/technical_design.docx - Technical Design Document (TDD)
Specifies the system architecture:
- Dual-branch retrieval pipeline (dense + sparse)
- Qdrant dual collection (1536d dense + BM25 sparse)
- RRF formula: RRF_Score(d) = Sum[1/(k + rank_dense(d))] + Sum[1/(k + rank_sparse(d))]
- k=60 dampening constant
- API contract: POST /api/v1/query/search with filters, top_k, stream
- Infrastructure: Qdrant cluster, Triton Inference Server, FastAPI/Uvicorn pods

### docs/hybrid_rag_prd.pdf - Product Requirements Document
PDF document (could not be read directly - model does not support PDF input).
Contains product vision, user stories, and feature specifications.

### docs/hybrid_rag_implementation_plan.docx - Implementation Plan
8-week roadmap across 4 phases:
- Phase 1 (Weeks 1-2): Data Ingestion & Hybrid Indexing
  - Qdrant cluster setup, custom text pre-processor, batch ingestion pipeline
- Phase 2 (Weeks 3-4): Retrieval & Fusion Engine
  - Async dual-branch query execution, RRF fusion (k=60), metadata filtering
- Phase 3 (Weeks 5-6): Reranking & LLM Integration
  - Cross-encoder deployment, prompt template with citation enforcement, streaming
- Phase 4 (Weeks 7-8): Testing, Tuning & Deployment
  - Ragas benchmark, load testing (P95 < 350ms), CI/CD, canary release

Infrastructure: Kubernetes (EKS), Qdrant (2 nodes, 16 vCPU, 64GB RAM), Triton Inference (NVIDIA T4 GPU), FastAPI (3 pods with autoscaling)

### docs/hybrid_rag_evaluation_plan.docx - Evaluation Plan
Quality assurance framework using Ragas/TruLens:
- Context Recall@5: >= 95% (relevant chunks in top-5)
- Context Precision@5: >= 92% (signal-to-noise ratio)
- MRR (Mean Reciprocal Rank): >= 0.88
- Faithfulness: >= 98% (grounded in context, zero hallucination)
- Answer Relevance: >= 94% (addresses user intent)

Test dataset: 500 query pairs
- 150 exact part/SKU queries (tests sparse search)
- 150 natural language/symptom queries (tests dense search)
- 150 mixed hybrid queries (tests RRF fusion)
- 50 adversarial/out-of-scope queries (tests guardrails)

---

## APPENDIX: Project File Index

### Source Files (src/)
1. src/main.py - FastAPI application (405 lines)
2. src/llm_client.py - OpenRouter LLM client (79 lines)
3. src/search.py - Hybrid search engine (287 lines)
4. src/reranker.py - Cross-encoder reranker (59 lines)
5. src/ingest.py - Data ingestion pipeline (235 lines)
6. src/generation.py - LLM generation and citations (97 lines)
7. src/load_dataset.py - Dataset loader (186 lines)
8. src/chunker.py - Text chunker (83 lines)
9. src/extract_ids.py - ID extractor (79 lines)
10. src/.env - Environment configuration

### Test Files (tests/)
11. tests/verify_dataset.py - Dataset verification
12. tests/test_reranker_alone.py - Reranker unit test
13. tests/test_omniroute_connection.py - LLM connectivity test
14. tests/test_individual_search.py - Individual search test
15. tests/test_hybrid_search.py - Hybrid search test
16. tests/test_full_rerank_pipeline.py - Full rerank pipeline test
17. tests/test_full_generation.py - Full generation test
18. tests/test_sku_rerank.py - SKU rerank test

### Root Scripts
19. run_eval.py - Main evaluation framework
20. test_eval_5.py - Quick 5-question evaluation
21. test_retrieval.py - Retrieval test script
22. compare_eval_runs.py - Evaluation comparison
23. view_csv.py - CSV viewer
24. check_manuals.py - Manual inspector
25. check_docs.py - Document checker
26. check_parts_tickets.py - Parts/tickets inspector

### Data Files (data/)
27. data/product_manuals.json - 6 product manuals
28. data/parts_catalog.json - Parts catalog (30+ parts)
29. data/troubleshooting_tickets.json - 12 support tickets
30. data/normalized_records.json - Normalized dataset
31. data/product_manuals.csv - CSV version of manuals
32. data/parts_catalog.csv - CSV version of parts
33. data/troubleshooting_tickets.csv - CSV version of tickets
34. data/shopping_queries_dataset_sources.csv - E-commerce dataset (unused)
35. data/shopping_queries_dataset_products.parquet - E-commerce products (unused)
36. data/shopping_queries_dataset_examples.parquet - E-commerce examples (unused)

### Documentation (docs/)
37. docs/requirements.docx - System Requirements Specification
38. docs/technical_design.docx - Technical Design Document
39. docs/hybrid_rag_prd.pdf - Product Requirements Document
40. docs/hybrid_rag_implementation_plan.docx - Implementation Plan
41. docs/hybrid_rag_evaluation_plan.docx - Evaluation Plan

### Storage (storage/)
42. storage/raft_state.json - Raft consensus state
43. storage/collections/support_docs/ - Qdrant collection data

### Other
44. eval_results.csv - Evaluation results
45. test_questions.csv - Test questions
46. .qdrant-initialized - Initialization flag
47. qdrant.zip - Qdrant backup
48. qdrant_bin/qdrant.exe - Qdrant binary
