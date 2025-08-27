# Vexia Search

Production‑ready Retrieval Augmented Generation (RAG) stack for multi‑document technical PDF search and conversational answering. Designed for low cold‑start latency and deployment on Vercel serverless.

---

## ✨ Key Features

* Lightweight ingestion (PyMuPDF + pdfminer fallback) – no heavy OCR/`unstructured` dependency.
* Heading/table heuristic aware chunking with token budgeting + overlap.
* Near‑duplicate suppression using MinHash (with pure‑Python fallback).
* Hybrid retrieval: pgvector similarity + BM25 (`tsvector`) + RRF fusion.
* Query rewriting (original + 2 reformulations) → fused retrieval set.
* LLM semantic rerank of top candidates → context compaction (token cap).
* Deterministic citation format `[file pX–Y]` + abstains with `UNKNOWN`.
* Batched OpenAI embeddings (`text-embedding-3-small` by default) with retry/backoff.
* Idempotent upserts keyed by `(user_id, source_file, checksum)`.
* Serverless‑friendly: uses `/tmp`, async I/O, minimal dependencies, small wheels.
* Benchmark + basic unit tests (chunking, RRF, rerank edge case).

---

## 🏗 Architecture Overview

Component | Responsibility
--------- | --------------
FastAPI (`src/rag.py`) | Endpoints `/deploy` (async ingestion trigger), `/chat` (hybrid retrieval + answer)
Ingestion (`src/ingestion/pdf_parser.py`) | Block extraction → heuristics → chunking → dedupe
Embedding (`src/ingestion/embed.py`) | Batch OpenAI embedding via direct HTTP (no langchain)
Storage | Supabase Postgres with `pgvector` + `tsvector` generated column
Retrieval | RPCs: `match_documents` (vector), `bm25_match` (BM25) + app‑side RRF fusion
Rerank | Single LLM call ordering top ~20 chunks
Answering | Context compaction → grounded answer with citations or `UNKNOWN`

Token budgeting ensures context <= `MAX_TOKENS_CONTEXT` (default 8000).

---

## 🗃 Data Model (documents table)

```
user_id text
content text
content_type text               -- text | table_heuristic
source_file text
page_range int4range
section_title text
checksum text
embedding vector(1536)
ts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
```

Indexes / constraints (see migration):
* IVFFLAT on embedding
* GIN on ts
* UNIQUE (user_id, source_file, checksum)

RPCs:
* `match_documents(query_embedding, match_count, p_user_id, similarity_threshold)`
* `bm25_match(p_query, p_user_id, p_limit)`

---

## 📦 Dependencies (runtime highlights)

Category | Libs
-------- | ----
Framework | fastapi, uvicorn
Async I/O / HTTP | aiofiles, httpx
ML / RAG | tiktoken, pymupdf, pdfminer.six, datasketch (optional)
Embeddings | OpenAI REST API (direct)
DB | supabase (Python client), pgvector extension
Resilience | tenacity
Config | pydantic-settings, python-dotenv

Removed heavy libs: `unstructured`, `langchain-openai`.

---

## 🚀 Getting Started

### 1. Clone & Environment
```
git clone https://github.com/AbhiramVSA/VexiaSearch.git
cd VexiaSearch
python -m venv .venv
./.venv/Scripts/activate  # (Windows) or source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `.env`
See `.env.example` – do NOT quote values.
```
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_KEY=<anon-or-service-role>
EMBED_MODEL=text-embedding-3-small
MAX_TOKENS_CONTEXT=8000
```

### 3. Apply Migration
Open Supabase SQL Editor → run `migrations/20250828_rag_hybrid.sql`.

### 4. Run API
```
uvicorn src.rag:app --host 0.0.0.0 --port 8000
```
Docs available at `/docs`.

---

## 📥 Ingestion (`POST /deploy`)
Multipart form fields:
* `user_id` (text)
* `files` (one or more PDFs)

Processing (background): extract → chunk (900 token target / 120 overlap) → dedupe → embed (batched) → upsert.

Response (202 Accepted) returns a session id + counts once background finishes (logs show detail).

---

## 💬 Chat (`POST /chat`)
Request JSON:
```
{ "message": "Your question", "user_id": "u123", "k": 6 }
```
Pipeline: rewrite queries → parallel vector + BM25 → RRF fuse → rerank → compact → answer.
Response JSON fields: `answer`, `citations`, `used_chunks`, `elapsed_ms`.
If insufficient evidence → `answer: "UNKNOWN"`.

---

## 🔍 Retrieval Details
* RRF (k=60) stabilizes rank fusion.
* Fusion uses IDs from vector & BM25 lists across rewrites.
* Rerank prompt orders top ~20; fallback returns initial order if LLM fails.
* Context compaction dedupes by hash & token budget.

---

## 🧪 Testing
```
pytest -q
```
Tests cover:
* Chunking + deduping basics.
* RRF combiner behavior.
* Rerank empty input edge case.

Note: Network LLM calls are minimal; extend with mocks for CI hardening if desired.

---

## 📊 Benchmark
```
python scripts/bench.py
```
Uploads sample PDFs (place under `./pdfs`) then executes a few queries, printing latency.

Target (guideline on modest hardware):
* Ingest 100 pages < 30s (excluding OpenAI network variance)
* Retrieval + rerank p50 < 800ms (generation excluded)

---

## 🛡 Security & Secrets
* Never commit real API keys – rotate immediately if exposed.
* Use service role key only in secure server contexts (not browser).
* All environment variables loaded via `pydantic-settings` + `.env`.

---

## 🔄 Future Roadmap
Item | Status
---- | ------
Result caching (query hash) | Planned
SSE streaming responses | Planned
Adaptive weighting (learned fusion) | Planned
Metrics / tracing middleware | Planned
Answer quality eval harness | Planned

---

## 🤝 Contributing
1. Fork & branch (`feature/<name>`)
2. Add/adjust tests for changes
3. Run lint/tests locally
4. Submit PR with concise description + perf notes

Issue templates: please include reproduction steps & expected vs actual retrieval output.

---

## 📄 License
MIT – see `LICENSE`.

---

## 🙌 Acknowledgements
* OpenAI for embeddings & LLM APIs
* Supabase Team for a clean developer DX
* PyMuPDF & pdfminer authors for fast and fallback parsing

---

For questions or improvements, open an issue or start a discussion.

---

## RAG Stack (Updated)

The backend was refactored to a lightweight, Vercel-friendly RAG pipeline:

Ingestion:
* PyMuPDF parsing with heading + table heuristics (no unstructured).
* Token-aware chunking (tiktoken) with overlap.
* MinHash near-duplicate removal.
* Batched OpenAI embeddings (text-embedding-3-small).

Storage:
* Supabase Postgres `documents` table with pgvector + tsvector hybrid indexes.

Retrieval:
* Query rewriting (original + 2 alts).
* Parallel vector similarity + BM25 (RPCs `match_documents` + `bm25_match`).
* RRF fusion, LLM rerank (top 20), context compaction to token budget.
* Answers cite `[file pX–Y]` or return `UNKNOWN` if insufficient evidence.

Run backend locally:
```
pip install -r requirements.txt
uvicorn src.rag:app --host 0.0.0.0 --port 8000
```

Apply migration SQL in `migrations/20250828_rag_hybrid.sql` via Supabase SQL editor.

Benchmark:
```
python scripts/bench.py
```

Environment template: see `.env.example`.
