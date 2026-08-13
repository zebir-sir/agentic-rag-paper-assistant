CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    pdf_blob BYTEA,
    pdf_media_type TEXT NOT NULL DEFAULT 'application/pdf',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents (created_at DESC);

ALTER TABLE documents ADD COLUMN IF NOT EXISTS pdf_blob BYTEA;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS pdf_media_type TEXT NOT NULL DEFAULT 'application/pdf';

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1024),
    chunk_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks (document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_content_trgm ON chunks USING GIN (content gin_trgm_ops);

-- 原始科研证据与语义检索卡分层存储：正文/伪代码仍在 chunks，图表保留完整载体。
CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    artifact_type TEXT NOT NULL CHECK (artifact_type IN ('figure', 'table')),
    caption TEXT NOT NULL DEFAULT '',
    page_number INTEGER,
    section_path TEXT,
    context_before TEXT NOT NULL DEFAULT '',
    context_after TEXT NOT NULL DEFAULT '',
    raw_content TEXT NOT NULL DEFAULT '',
    structured_data JSONB NOT NULL DEFAULT '{}',
    retrieval_text TEXT NOT NULL DEFAULT '',
    image_blob BYTEA,
    image_media_type TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_artifacts_document_id ON artifacts (document_id, artifact_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_structured_data ON artifacts USING GIN (structured_data);
CREATE INDEX IF NOT EXISTS idx_artifacts_retrieval_trgm ON artifacts USING GIN (retrieval_text gin_trgm_ops);

CREATE TABLE IF NOT EXISTS document_translations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_language TEXT NOT NULL CHECK (target_language IN ('zh', 'en')),
    source_sha256 TEXT NOT NULL,
    content_markdown TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document_id, target_language, source_sha256)
);

CREATE TABLE IF NOT EXISTS document_annotations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER,
    page_x DOUBLE PRECISION NOT NULL DEFAULT 0.5 CHECK (page_x >= 0 AND page_x <= 1),
    page_y DOUBLE PRECISION NOT NULL DEFAULT 0.5 CHECK (page_y >= 0 AND page_y <= 1),
    quote TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL,
    color TEXT NOT NULL DEFAULT 'yellow',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE document_annotations ADD COLUMN IF NOT EXISTS page_x DOUBLE PRECISION NOT NULL DEFAULT 0.5;
ALTER TABLE document_annotations ADD COLUMN IF NOT EXISTS page_y DOUBLE PRECISION NOT NULL DEFAULT 0.5;

CREATE INDEX IF NOT EXISTS idx_document_translations_document ON document_translations (document_id, target_language);
CREATE INDEX IF NOT EXISTS idx_document_annotations_document ON document_annotations (document_id, created_at DESC);

-- Selection translation keeps paper-wide terminology separate from short, reusable reader translations.
CREATE TABLE IF NOT EXISTS document_translation_profiles (
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_language TEXT NOT NULL CHECK (target_language IN ('zh', 'en')),
    source_sha256 TEXT NOT NULL,
    profile_json JSONB NOT NULL DEFAULT '{}',
    model TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (document_id, target_language, source_sha256)
);

CREATE TABLE IF NOT EXISTS document_selection_translations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_language TEXT NOT NULL CHECK (target_language IN ('zh', 'en')),
    source_sha256 TEXT NOT NULL,
    selection_sha256 TEXT NOT NULL,
    source_text TEXT NOT NULL,
    translated_text TEXT NOT NULL,
    context_before TEXT NOT NULL DEFAULT '',
    context_after TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document_id, target_language, source_sha256, selection_sha256)
);

CREATE INDEX IF NOT EXISTS idx_selection_translations_document ON document_selection_translations (document_id, target_language, source_sha256);

-- Paper-level graph index. It is derived from the original paper chunks and
-- remains separate from evidence retrieval so graph links never masquerade as citations.
CREATE TABLE IF NOT EXISTS paper_graph_nodes (
    document_id UUID PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    abbreviation TEXT NOT NULL,
    profile_text TEXT NOT NULL DEFAULT '',
    embedding vector(1024),
    embedding_language TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}',
    graph_version INTEGER NOT NULL DEFAULT 1,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_paper_graph_nodes_embedding ON paper_graph_nodes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1);
CREATE INDEX IF NOT EXISTS idx_paper_graph_nodes_language ON paper_graph_nodes (embedding_language);

CREATE TABLE IF NOT EXISTS paper_graph_edges (
    source_document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL CHECK (relation_type IN ('semantic_similarity', 'cites', 'method_lineage')),
    score DOUBLE PRECISION NOT NULL CHECK (score >= 0 AND score <= 1),
    evidence JSONB NOT NULL DEFAULT '{}',
    graph_version INTEGER NOT NULL DEFAULT 1,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_document_id, target_document_id, relation_type),
    CHECK (source_document_id <> target_document_id)
);

-- Existing installations created the initial semantic-only check constraint.
-- Upgrade it in place so the graph remains a durable relation index.
ALTER TABLE paper_graph_edges DROP CONSTRAINT IF EXISTS paper_graph_edges_relation_type_check;
ALTER TABLE paper_graph_edges ADD CONSTRAINT paper_graph_edges_relation_type_check
    CHECK (relation_type IN ('semantic_similarity', 'cites', 'method_lineage'));

CREATE INDEX IF NOT EXISTS idx_paper_graph_edges_source ON paper_graph_edges (source_document_id, score DESC);
CREATE INDEX IF NOT EXISTS idx_paper_graph_edges_target ON paper_graph_edges (target_document_id, score DESC);
CREATE INDEX IF NOT EXISTS idx_paper_graph_edges_relation ON paper_graph_edges (relation_type, score DESC);

-- Cached Chinese research cards for the graph. Full-document translations stay
-- in document_translations; this table is intentionally compact and UI-focused.
CREATE TABLE IF NOT EXISTS paper_graph_localizations (
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_language TEXT NOT NULL CHECK (target_language IN ('zh')),
    source_sha256 TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    model TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'ready', 'failed')),
    content JSONB NOT NULL DEFAULT '{}',
    validation JSONB NOT NULL DEFAULT '{}',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (document_id, target_language, source_sha256, schema_version)
);

CREATE INDEX IF NOT EXISTS idx_paper_graph_localizations_status ON paper_graph_localizations (status, updated_at);

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions (expires_at);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages (session_id, created_at);

-- Structured, versioned session memory. The immutable messages table remains the source log.
CREATE TABLE IF NOT EXISTS session_memory_snapshots (
    session_id UUID PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    version INTEGER NOT NULL DEFAULT 0,
    covered_message_count INTEGER NOT NULL DEFAULT 0,
    summary JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_session_memory_snapshots_updated ON session_memory_snapshots (updated_at DESC);

CREATE TABLE IF NOT EXISTS ingestion_tasks (
    task_id TEXT PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    file_path TEXT NOT NULL,
    filename TEXT NOT NULL DEFAULT '',
    fast BOOLEAN NOT NULL DEFAULT FALSE,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'processing', 'paused', 'done', 'failed', 'paused_quota', 'deleted')),
    queue_order BIGINT NOT NULL DEFAULT 0,
    progress_percent INTEGER NOT NULL DEFAULT 0 CHECK (progress_percent BETWEEN 0 AND 100),
    progress_stage TEXT NOT NULL DEFAULT '等待入库',
    error_message TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE
);

ALTER TABLE ingestion_tasks ADD COLUMN IF NOT EXISTS fast BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE ingestion_tasks ADD COLUMN IF NOT EXISTS filename TEXT NOT NULL DEFAULT '';
ALTER TABLE ingestion_tasks ADD COLUMN IF NOT EXISTS queue_order BIGINT NOT NULL DEFAULT 0;
ALTER TABLE ingestion_tasks ADD COLUMN IF NOT EXISTS progress_percent INTEGER NOT NULL DEFAULT 0;
ALTER TABLE ingestion_tasks ADD COLUMN IF NOT EXISTS progress_stage TEXT NOT NULL DEFAULT '等待入库';
ALTER TABLE ingestion_tasks ALTER COLUMN fast SET DEFAULT FALSE;
ALTER TABLE ingestion_tasks DROP CONSTRAINT IF EXISTS ingestion_tasks_status_check;
ALTER TABLE ingestion_tasks ADD CONSTRAINT ingestion_tasks_status_check
    CHECK (status IN ('queued', 'processing', 'paused', 'done', 'failed', 'paused_quota', 'deleted'));

CREATE INDEX IF NOT EXISTS idx_ingestion_tasks_status ON ingestion_tasks (status);
CREATE INDEX IF NOT EXISTS idx_ingestion_tasks_created_at ON ingestion_tasks (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ingestion_tasks_queue_order ON ingestion_tasks (queue_order, created_at);


CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(1024),
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    similarity FLOAT,
    metadata JSONB,
    document_title TEXT,
    document_source TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id AS chunk_id,
        c.document_id,
        c.content,
        (1 - (c.embedding <=> query_embedding))::double precision AS similarity,
        c.metadata,
        d.title AS document_title,
        d.source AS document_source
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding vector(1024),
    query_text TEXT,
    match_count INT DEFAULT 10,
    text_weight FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    combined_score FLOAT,
    vector_similarity FLOAT,
    text_similarity FLOAT,
    metadata JSONB,
    document_title TEXT,
    document_source TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT 
            c.id AS chunk_id,
            c.document_id,
            c.content,
            (1 - (c.embedding <=> query_embedding))::double precision AS vector_sim,
            c.metadata,
            d.title AS doc_title,
            d.source AS doc_source
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.embedding IS NOT NULL
    ),
    text_results AS (
        SELECT 
            c.id AS chunk_id,
            c.document_id,
            c.content,
            ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', query_text))::double precision AS text_sim,
            c.metadata,
            d.title AS doc_title,
            d.source AS doc_source
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
    )
    SELECT 
        COALESCE(v.chunk_id, t.chunk_id) AS chunk_id,
        COALESCE(v.document_id, t.document_id) AS document_id,
        COALESCE(v.content, t.content) AS content,
        (
            COALESCE(v.vector_sim, 0)::double precision * (1 - text_weight) +
            COALESCE(t.text_sim, 0)::double precision * text_weight
        ) AS combined_score,
        COALESCE(v.vector_sim, 0)::double precision AS vector_similarity,
        COALESCE(t.text_sim, 0)::double precision AS text_similarity,
        COALESCE(v.metadata, t.metadata) AS metadata,
        COALESCE(v.doc_title, t.doc_title) AS document_title,
        COALESCE(v.doc_source, t.doc_source) AS document_source
    FROM vector_results v
    FULL OUTER JOIN text_results t ON v.chunk_id = t.chunk_id
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$;

CREATE OR REPLACE FUNCTION get_document_chunks(doc_id UUID)
RETURNS TABLE (
    chunk_id UUID,
    content TEXT,
    chunk_index INTEGER,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        id AS chunk_id,
        chunks.content,
        chunks.chunk_index,
        chunks.metadata
    FROM chunks
    WHERE document_id = doc_id
    ORDER BY chunk_index;
END;
$$;

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_sessions_updated_at ON sessions;
CREATE TRIGGER update_sessions_updated_at BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_ingestion_tasks_updated_at ON ingestion_tasks;
CREATE TRIGGER update_ingestion_tasks_updated_at BEFORE UPDATE ON ingestion_tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE VIEW document_summaries AS
SELECT 
    d.id,
    d.title,
    d.source,
    d.created_at,
    d.updated_at,
    d.metadata,
    COUNT(c.id) AS chunk_count,
    AVG(c.token_count) AS avg_tokens_per_chunk,
    SUM(c.token_count) AS total_tokens
FROM documents d
LEFT JOIN chunks c ON d.id = c.document_id
GROUP BY d.id, d.title, d.source, d.created_at, d.updated_at, d.metadata;
