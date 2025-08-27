import asyncio, os, uuid, hashlib, time, aiofiles, shutil
from typing import Any, Dict, List, Optional
from datetime import datetime
from .load_api import Settings
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .agents import OpenAIAgentInit
from .ingestion.pdf_parser import extract_blocks, blocks_to_chunks, dedupe_chunks
from .ingestion.embed import embed_texts
from .supabase_vector import Client, create_client, SupabaseVectorStore
import httpx
import tiktoken

# Retrieval constants
MAX_REWRITE = 3
TOP_K_VECTOR = 12
TOP_K_BM25 = 12
RERANK_TOP = 20
CONTEXT_MAX_TOKENS = int(os.getenv("MAX_TOKENS_CONTEXT", "8000"))
enc = tiktoken.get_encoding("cl100k_base")

TEMP_FILE_DIR = "/tmp"
os.makedirs(TEMP_FILE_DIR, exist_ok=True)

settings = Settings()
agent_create = OpenAIAgentInit(api_key=settings.OPENAI_API_KEY)
try:
    supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
except Exception as e:
    print(f"Supabase init error: {e}")
    supabase = None
vector_store = SupabaseVectorStore(supabase)

# FastApi and Middleware
app = FastAPI(
    title="Vexia-Search API Endpoint",
    description="Endpoint to upload and process PDF documents for RAG system and process agentic chatting.",
)

# Need to work on deployment, not setup yet
origins = [
    "http://localhost:8080",
    "http://localhost:5173", 
    "http://localhost:3000", 
    "https://vexia-search-ui.vercel.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

async def process_and_store_document(file_path: str, user_id: str):
    file_name = os.path.basename(file_path)
    print(f"[ingest] {file_name} start")
    blocks = extract_blocks(file_path)
    chunks = blocks_to_chunks(blocks)
    deduped = dedupe_chunks(chunks)
    print(f"[ingest] {file_name} blocks={len(blocks)} chunks={len(chunks)} deduped={len(deduped)}")
    texts = [c.text for c in deduped]
    embeddings = await embed_texts(texts)
    docs = []
    for c, emb in zip(deduped, embeddings):
        docs.append({
            "id": c.id,
            "content": c.text,
            "content_type": c.content_type,
            "source_file": file_name,
            "user_id": user_id,
            "page_range": f"[{c.page_start},{c.page_end}]",
            "section_title": c.section_title,
            "checksum": c.checksum,
            "embedding": emb,
        })
    if docs:
        await vector_store.add_documents(docs)
    return {"file": file_name, "blocks": len(blocks), "chunks": len(chunks), "deduped": len(deduped), "inserted": len(docs)}

# A wrapper to run the processing for all files and clean up afterward.
async def run_processing_in_background(file_paths: List[str], temp_dir: str, user_id: str):
    try:
        tasks = [process_and_store_document(fp, user_id) for fp in file_paths]
        results = await asyncio.gather(*tasks)
        print("[ingest] summary", results)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# Endpoints

class ChatRequest(BaseModel):
    message: str
    user_id: str
    k: int | None = 6

@app.get("/ping")
def ping_root():
    return {"message": "RAG Processing API is running. POST files to /deploy to start."}

# Accepts multiple PDF files, saves them, and triggers a background task 
# to process them and add their content to the vector store.
@app.post("/deploy", status_code=202) 
async def deploy_documents(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    files: List[UploadFile] = File(..., description="A list of PDF files to process.")
    ):

    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    session_id = str(uuid.uuid4())
    session_dir = os.path.join(TEMP_FILE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    saved_file_paths: List[str] = []
    try:
        for file in files:
            if file.content_type != 'application/pdf':
                raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}. Only PDF files are accepted.")
            file_path = os.path.join(session_dir, file.filename)
            try:
                async with aiofiles.open(file_path, 'wb') as out_file:
                    content = await file.read()
                    await out_file.write(content)
                saved_file_paths.append(file_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error saving file {file.filename}: {e}")
        background_tasks.add_task(run_processing_in_background, saved_file_paths, session_dir, user_id)
        return {"success": True, "message": f"Processing started for {len(files)} documents in the background.", "document_count": len(files), "session_id": session_id}
    except Exception:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise


# Answers a user's question based on documents they have previously uploaded.
def rrf(rank_lists: List[List[str]], k: int = 60):
    scores: Dict[str, float] = {}
    for ranks in rank_lists:
        for i, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + i + 1)
    return sorted(scores, key=scores.get, reverse=True)

async def query_rewrites(query: str) -> List[str]:
    agent = agent_create.create_chat_agent(
        sysprompt="Generate concise alternative phrasings of the user's query for better document retrieval.",
        model_name="gpt-4o-mini"
    )
    prompt = f"Provide 2 alternative phrasings for: {query}\nReturn each on new line."
    try:
        res = await agent.run(prompt)
        alts = [q.strip('- ').strip() for q in res.output.splitlines() if q.strip()]
        return [query] + alts[:2]
    except Exception:
        return [query]

async def vector_search(query: str, user_id: str, k: int) -> List[Dict[str, Any]]:
    emb = await vector_store.create_embedding(query)
    if not emb:
        return []
    return await vector_store.similarity_search(emb, user_id=user_id, match_count=k)

async def bm25_search(query: str, user_id: str, k: int) -> List[Dict[str, Any]]:
    if not supabase:
        return []
    # simple text search via tsvector index
    try:
        resp = await asyncio.to_thread(
            supabase.rpc("bm25_match", {"p_query": query, "p_user_id": user_id, "p_limit": k}).execute
        )
        return resp.data or []
    except Exception:
        return []

def compact_context(docs: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
    out = []
    tok_total = 0
    seen_hash = set()
    for d in docs:
        text = d.get("content", "")
        h = hashlib.sha1(text.encode()).hexdigest()[:12]
        if h in seen_hash:
            continue
        toks = len(enc.encode(text))
        if tok_total + toks > max_tokens:
            break
        tok_total += toks
        seen_hash.add(h)
        out.append(d)
    return out

async def rerank(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not docs:
        return []
    agent = agent_create.create_chat_agent(
        sysprompt="You rerank passages for relevance and faithfulness.",
        model_name="gpt-4o-mini"
    )
    sample = docs[:RERANK_TOP]
    passage_text = "\n".join(f"ID={i}\n{text['content'][:800]}" for i, text in enumerate(sample))
    prompt = f"Query: {query}\nPassages:\n{passage_text}\nReturn JSON list of top ids in order."
    try:
        res = await agent.run(prompt)
        import json
        ids = json.loads(res.output)
        ordered = [sample[i] for i in ids if 0 <= i < len(sample)]
        return ordered
    except Exception:
        return sample

@app.post("/chat")
async def chat_with_documents(request: ChatRequest):
    t0 = time.time()
    rewrites = await query_rewrites(request.message)
    # parallel searches
    vector_tasks = [vector_search(q, request.user_id, TOP_K_VECTOR) for q in rewrites]
    bm25_tasks = [bm25_search(q, request.user_id, TOP_K_BM25) for q in rewrites]
    vector_results = await asyncio.gather(*vector_tasks)
    bm25_results = await asyncio.gather(*bm25_tasks)
    # flatten & ranking lists for RRF
    vector_rank_lists = [[d['id'] for d in res] for res in vector_results]
    bm25_rank_lists = [[d['id'] for d in res] for res in bm25_results]
    fused_ids = rrf(vector_rank_lists + bm25_rank_lists)
    id_to_doc: Dict[str, Dict[str, Any]] = {}
    for res in vector_results + bm25_results:
        for d in res:
            id_to_doc[d['id']] = d
    fused_docs = [id_to_doc[i] for i in fused_ids if i in id_to_doc][: request.k]
    reranked = await rerank(request.message, fused_docs)
    compacted = compact_context(reranked, CONTEXT_MAX_TOKENS)
    if not compacted:
        return {"answer": "UNKNOWN", "citations": [], "used_chunks": []}
    context_blocks = []
    citations = []
    for d in compacted:
        context_blocks.append(f"[{d.get('source_file')} p{d.get('page_range','')}]: {d.get('content')}")
        citations.append({
            "id": d.get('id'),
            "file": d.get('source_file'),
            "pages": d.get('page_range'),
            "similarity": d.get('similarity')
        })
    context_text = "\n---\n".join(context_blocks)
    agent = agent_create.create_chat_agent(
        sysprompt="Answer with citations in format [file pX–Y]. If insufficient evidence reply UNKNOWN.",
        model_name="gpt-4o-mini"
    )
    prompt = f"Context:\n{context_text}\n\nQuestion: {request.message}\nAnswer:"
    try:
        resp = await agent.run(prompt)
        return {"answer": resp.output, "citations": citations, "used_chunks": citations, "elapsed_ms": int((time.time()-t0)*1000)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
