-- Schema adjustments for documents table
ALTER TABLE documents ADD COLUMN IF NOT EXISTS page_range int4range;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS section_title text;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS checksum text;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_type text;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- Vector index (adjust lists based on dataset size)
CREATE INDEX IF NOT EXISTS documents_embedding_ivfflat ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- Full text index
CREATE INDEX IF NOT EXISTS documents_ts_idx ON documents USING GIN (ts);
-- Uniqueness for idempotent upserts
CREATE UNIQUE INDEX IF NOT EXISTS documents_uid_src_ck ON documents (user_id, source_file, checksum);

-- RPC: vector similarity (expects existing 'match_documents' maybe recreate safer)
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1536),
    match_count int DEFAULT 5,
    p_user_id text,
    similarity_threshold float DEFAULT 0.1
) RETURNS TABLE(id uuid, content text, source_file text, similarity float, page_range int4range, section_title text) LANGUAGE sql STABLE AS $$
    SELECT d.id, d.content, d.source_file,
           1 - (d.embedding <=> query_embedding) AS similarity,
           d.page_range, d.section_title
    FROM documents d
    WHERE d.user_id = p_user_id
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- RPC: BM25 style search using ts_rank_cd
CREATE OR REPLACE FUNCTION bm25_match(p_query text, p_user_id text, p_limit int DEFAULT 10)
RETURNS TABLE(id uuid, content text, source_file text, rank float, page_range int4range, section_title text) LANGUAGE sql STABLE AS $$
    SELECT d.id, d.content, d.source_file,
           ts_rank_cd(d.ts, query) AS rank,
           d.page_range, d.section_title
    FROM documents d, plainto_tsquery('english', p_query) query
    WHERE d.user_id = p_user_id AND d.ts @@ query
    ORDER BY rank DESC
    LIMIT p_limit;
$$;
