# CatalogIQ - Hybrid RAG-Based E-Commerce Support Assistant

A high-performance Retrieval-Augmented Generation (RAG) backend and interactive dashboard designed for complex technical e-commerce catalogs. CatalogIQ resolves fuzzy product queries, SKU lookups, and part-number specifications using an identifier-aware hybrid retrieval pipeline.

---

## System Architecture

## System Architecture

<p align="center">
  <img src="assets/architecture.jpg" alt="CatalogIQ System Architecture" width="100%">
</p>

## Overview

E-commerce technical catalogs present unique challenges for traditional RAG systems:
* **Fuzzy Queries vs. Exact Identifiers:** Dense vector embeddings excel at broad semantic intent (e.g., *"ergonomic chair for back pain"*) but fail on exact part numbers or alphanumeric identifiers (e.g., `S-4012-SCRW` or `C-7721-GL`).
* **Hybrid Retrieval:** CatalogIQ routes incoming queries dynamically to direct vector/payload lookups, sparse BM25 indexing, or hybrid Reciprocal Rank Fusion (RRF) depending on the presence of detected catalog identifiers.
* **Fail-Safe Generation:** Implements strict source citation enforcement and an instant dynamic context parser to guarantee system availability even under upstream model gateway failures.

---

## Key Features

* **Identifier-Aware Query Routing:** Automatically detects SKUs and part numbers to bypass or supplement semantic search with precision exact matching.
* **Hybrid Search with RRF:** Merges Dense vector search (`sentence-transformers/all-MiniLM-L6-v2`) and BM25 sparse lexical indexing using Reciprocal Rank Fusion ($k=60$).
* **Cross-Encoder Re-Ranking:** Reranks candidate contexts using `cross-encoder/ms-marco-MiniLM-L-6-v2` to select the top 5 relevant sources.
* **Strict Source Citation Enforcement:** Validates `[doc_id]` alignment in LLM outputs to prevent hallucinations and ungrounded claims.
* **Instant Dynamic Fallback:** Features a zero-latency fallback engine that extracts structured product specs if the LLM gateway experiences upstream errors or timeouts.
* **Scope Guarding:** Rejects out-of-scope non-catalog requests before triggering vector retrieval or LLM generation.
* **Interactive Streamlit UI:** Features a single-page interactive UI for real-time querying, retrieval route visibility, and execution latency breakdowns.

---

## Tech Stack

| Component | Technology / Model |
| :--- | :--- |
| **API Framework** | FastAPI, Uvicorn |
| **Frontend UI** | Streamlit |
| **Vector Store** | Qdrant Client |
| **Dense Embedder** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Sparse Lexical Search** | `rank-bm25` |
| **Re-Ranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM Orchestration** | OpenAI Python SDK (connecting via OmniRoute Gateway) |
| **Language & Environment** | Python 3.10+, `python-dotenv` |

---

## Installation

### 1. Prerequisites
* Python 3.10 or higher
* Qdrant instance running on `localhost:6333`

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/CatalogIQ.git
cd CatalogIQ
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root with the following variables:
```env
QDRANT_URL=http://localhost:6333
OMNIROUTE_API_KEY=your_omniroute_api_key
OMNIROUTE_BASE_URL=http://localhost:20128/v1
OMNIROUTE_MODEL=auto/fast
```

### 5. Ingest Catalog Data
Run the ingestion script to populate the vector database:
```bash
python src/ingest.py
```

### 6. Run the Application

**API Server (FastAPI):**
```bash
python src/main.py
```
Access the API docs at `http://localhost:8000/docs`

**Streamlit UI:**
```bash
streamlit run streamlit_app.py
```
Access the UI at `http://localhost:8501`

---

## Project Structure
```
CatalogIQ/
??? streamlit_app.py          # Streamlit frontend
??? requirements.txt          # Python dependencies
??? .env                      # Environment variables (not tracked)
??? .gitignore               # Git ignore rules
??? .gitattributes           # Git LFS and attribute config
??? src/
?   ??? main.py              # FastAPI entry point
?   ??? catalog_service.py   # Core RAG pipeline
?   ??? search.py            # Hybrid search (dense + sparse + RRF)
?   ??? reranker.py          # Cross-encoder reranking
?   ??? query_router.py      # Query classification & scope guard
?   ??? ingest.py            # Data ingestion pipeline
?   ??? chunker.py           # Document chunking
?   ??? llm_client.py        # LLM client with fallback
?   ??? ...                  # Other modules
??? tests/                   # Unit & integration tests
??? evaluation/              # Evaluation scripts
```

---

## Usage Examples

### Product Overview
```
"Tell me about the 11 Degrees Core Pull Over Hoodie"
```

### Exact SKU / Part Lookup
```
"What is the torque spec for screw S-4012-SCRW?"
```

### Troubleshooting
```
"What does error E02 mean on DESK-STD-MOTO?"
"Standing desk won't move up or down, shows error E01"
```

### Comparisons
```
"Compare CHAIR-ERG-X99 and DESK-STD-MOTO"
```

### Warranty & Support
```
"What is the warranty on CHAIR-ERG-X99?"
```

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/ask` | Submit a catalog question |
| `GET` | `/health` | Health check |

### Request Format (`/ask`)
```json
{
  "question": "What is the torque spec for screw S-4012-SCRW?"
}
```

### Response Format
```json
{
  "answer": "The torque specification for S-4012-SCRW is...",
  "sources": [
    {"doc_id": "...", "sku": "S-4012-SCRW", "product_name": "Screw Assembly"}
  ],
  "citation_check": {
    "total_citations_found": 1,
    "valid_citations": ["..."],
    "invalid_citations": [],
    "has_valid_citation": true,
    "all_valid": true
  },
  "timing_ms": {
    "sparse_dense_search_ms": 45.2,
    "rerank_ms": 120.5,
    "generation_ms": 850.3,
    "total_ms": 1016.0
  },
  "out_of_scope": false,
  "scope_reason": null
}
```

---

## Evaluation

Run the evaluation suite:
```bash
# Full evaluation
python run_eval.py

# Ablation studies
python run_ablation.py

# Individual test modules
pytest tests/
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.
