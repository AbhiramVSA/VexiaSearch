# Vexia Search

Vexia Search is a React client prototype for document upload and RAG chat. The tracked frontend lives in [`front-end/`](front-end/).

## Run the frontend

```bash
cd front-end
npm ci
npm run dev
```

The Vite client runs at the local URL printed by the command. Its RAG helper uses `http://127.0.0.1:8000/deploy` and `/chat` by default. This checkout does not include the API service that those routes require.

The upload screen currently simulates upload progress in the browser. It does not confirm that a PDF reached a RAG backend.

## Current repository scope

- `front-end/` contains the Vite, React, and Supabase client code.
- The repository does not contain the documented FastAPI entry point `src/rag.py` or its migration SQL.
- Root `vercel.json` still points to `src/rag.py`, which is absent from this checkout.
- The root `.env.example` lists backend settings. The frontend RAG helper uses fixed local URLs in `front-end/src/services/ragService.ts`.
- `requirements.txt` and `pyproject.toml` describe different Python dependency sets.
- No root `LICENSE` file is present.

## Backend design notes

The earlier README described a FastAPI service with PDF ingestion, Supabase Postgres, `pgvector`, full-text search, reciprocal-rank fusion, and LLM reranking. The notes below record that intended design. They do not describe code included in this checkout.

The documented API used `POST /deploy` to accept a `user_id` and one or more PDF files. It used `POST /chat` with a message, user ID, and result count. The described answer format used citations such as `[file pX–Y]` or returned `UNKNOWN` when it found insufficient evidence.

The documented `documents` table included these fields:

```text
user_id, content, content_type, source_file, page_range,
section_title, checksum, embedding, ts
```

The design described vector and BM25 retrieval, rank fusion, reranking, and a bounded context. Its implementation, database migration, and tests are not present in the tracked tree.
