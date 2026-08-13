"""Deterministic paper graph materialization and read APIs.

The graph is a paper-selection index. Final answers must still be grounded in
chunks and artifacts returned by the normal retrieval tools.
"""
import json
from typing import Any, Dict, List, Optional

from .db_utils import db_pool
from .graph_abbreviation import build_title_abbreviation
from .graph_localization_schema import LOCALIZATION_SCHEMA_VERSION, graph_localization_hash
from .graph_relation_runtime import GraphCandidate, extract_evidence_backed_relations
from .graph_schema import PaperGraphResponse

GRAPH_NEIGHBOR_LIMIT = 8
GRAPH_MIN_SIMILARITY = 0.62


async def _refresh_evidence_backed_relations(conn: Any) -> None:
    """Rebuild small, explicit paper-to-paper relations from persisted source text."""
    candidate_rows = await conn.fetch(
        "SELECT d.id::text AS document_id, d.title, n.abbreviation FROM documents d JOIN paper_graph_nodes n ON n.document_id=d.id"
    )
    candidates = [GraphCandidate(document_id=str(row["document_id"]), title=str(row["title"]), abbreviation=str(row["abbreviation"] or "")) for row in candidate_rows]
    chunk_rows = await conn.fetch(
        """SELECT c.id::text AS id, c.document_id::text AS document_id, c.content, c.metadata
           FROM chunks c
           WHERE COALESCE(c.metadata->>'content_type', 'pdf') <> 'artifact'
             AND (LOWER(COALESCE(c.metadata->>'section_title', '')) LIKE '%reference%'
                  OR LOWER(COALESCE(c.metadata->>'section_path_text', '')) LIKE '%reference%'
                  OR LOWER(COALESCE(c.metadata->>'section_title', '')) LIKE '%method%'
                  OR LOWER(COALESCE(c.metadata->>'section_path_text', '')) LIKE '%method%'
                  OR COALESCE(c.metadata->>'section_title', '') LIKE '%参考文献%'
                  OR COALESCE(c.metadata->>'section_path_text', '') LIKE '%参考文献%'
                  OR COALESCE(c.metadata->>'section_title', '') LIKE '%方法%'
                  OR COALESCE(c.metadata->>'section_path_text', '') LIKE '%方法%')"""
    )
    await conn.execute("DELETE FROM paper_graph_edges WHERE relation_type IN ('cites', 'method_lineage')")
    for row in chunk_rows:
        source_document_id = str(row["document_id"])
        chunk = {"id": str(row["id"]), "content": str(row["content"] or ""), "metadata": json.loads(row["metadata"] or "{}")}
        for relation in extract_evidence_backed_relations(source_document_id, [chunk], candidates):
            await conn.execute(
                """INSERT INTO paper_graph_edges(source_document_id,target_document_id,relation_type,score,evidence,graph_version)
                   VALUES($1::uuid,$2::uuid,$3,$4,$5::jsonb,1)
                   ON CONFLICT (source_document_id,target_document_id,relation_type) DO UPDATE
                   SET score=EXCLUDED.score,evidence=EXCLUDED.evidence,
                       graph_version=paper_graph_edges.graph_version+1,updated_at=CURRENT_TIMESTAMP""",
                source_document_id, relation["target_document_id"], relation["relation_type"], relation["score"],
                json.dumps(relation["evidence"], ensure_ascii=False),
            )


async def refresh_paper_graph(document_id: str) -> None:
    """Refresh one graph node, semantic edges, and evidence-backed cross-paper edges."""
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            document = await conn.fetchrow(
                "SELECT id::text, title, source, metadata FROM documents WHERE id=$1::uuid", document_id
            )
            if not document:
                return
            language = str(json.loads(document["metadata"] or "{}").get("document_language") or "")
            profile = await conn.fetchrow(
                """
                SELECT avg(embedding) AS embedding,
                       string_agg(left(content, 900), E'\n' ORDER BY chunk_index) AS profile_text
                FROM chunks
                WHERE document_id=$1::uuid AND embedding IS NOT NULL
                  AND COALESCE(metadata->>'content_type', 'pdf') <> 'artifact'
                """,
                document_id,
            )
            if not profile or profile["embedding"] is None:
                return
            raw_embedding = profile["embedding"]
            embedding_value = raw_embedding if isinstance(raw_embedding, str) else "[" + ",".join(map(str, raw_embedding)) + "]"
            abbreviation = build_title_abbreviation(str(document["title"]))
            metadata = {
                "title_abbreviation_source": "deterministic_title_rule",
                "relation_policy": "paper_embedding_cosine",
            }
            await conn.execute(
                """
                INSERT INTO paper_graph_nodes(document_id, abbreviation, profile_text, embedding, embedding_language, metadata, graph_version)
                VALUES($1::uuid,$2,$3,$4::vector,$5,$6::jsonb,1)
                ON CONFLICT (document_id) DO UPDATE SET abbreviation=EXCLUDED.abbreviation,
                    profile_text=EXCLUDED.profile_text, embedding=EXCLUDED.embedding,
                    embedding_language=EXCLUDED.embedding_language, metadata=EXCLUDED.metadata,
                    graph_version=paper_graph_nodes.graph_version+1, updated_at=CURRENT_TIMESTAMP
                """,
                document_id, abbreviation, str(profile["profile_text"] or "")[:12000], embedding_value, language,
                json.dumps(metadata, ensure_ascii=False),
            )
            source_hash = graph_localization_hash(str(document["title"]), str(profile["profile_text"] or "")[:12000])
            await conn.execute(
                """
                INSERT INTO paper_graph_localizations(document_id,target_language,source_sha256,schema_version,status)
                VALUES($1::uuid,'zh',$2,$3,'pending')
                ON CONFLICT (document_id,target_language,source_sha256,schema_version) DO NOTHING
                """, document_id, source_hash, LOCALIZATION_SCHEMA_VERSION,
            )
            await conn.execute(
                """DELETE FROM paper_graph_edges WHERE relation_type='semantic_similarity'
                   AND (source_document_id=$1::uuid OR target_document_id=$1::uuid)""", document_id
            )
            neighbors = await conn.fetch(
                """
                SELECT document_id::text, (1 - (embedding <=> (SELECT embedding FROM paper_graph_nodes WHERE document_id=$1::uuid)))::float8 AS score
                FROM paper_graph_nodes
                WHERE document_id <> $1::uuid AND embedding IS NOT NULL
                  AND embedding_language = (SELECT embedding_language FROM paper_graph_nodes WHERE document_id=$1::uuid)
                ORDER BY embedding <=> (SELECT embedding FROM paper_graph_nodes WHERE document_id=$1::uuid)
                LIMIT $2
                """,
                document_id, GRAPH_NEIGHBOR_LIMIT,
            )
            for neighbor in neighbors:
                score = float(neighbor["score"])
                if score < GRAPH_MIN_SIMILARITY:
                    continue
                left, right = sorted((str(document_id), str(neighbor["document_id"])))
                await conn.execute(
                    """
                    INSERT INTO paper_graph_edges(source_document_id,target_document_id,relation_type,score,evidence,graph_version)
                    VALUES($1::uuid,$2::uuid,'semantic_similarity',$3,$4::jsonb,1)
                    ON CONFLICT (source_document_id,target_document_id,relation_type) DO UPDATE
                    SET score=EXCLUDED.score,evidence=EXCLUDED.evidence,
                        graph_version=paper_graph_edges.graph_version+1,updated_at=CURRENT_TIMESTAMP
                    """,
                    left, right, score,
                    json.dumps({"kind": "paper_embedding_cosine", "score": round(score, 4), "explanation": "论文正文语义相近；最终结论仍需回到原文证据核验。"}, ensure_ascii=False),
                )
            await _refresh_evidence_backed_relations(conn)


async def get_paper_graph() -> PaperGraphResponse:
    async with db_pool.acquire() as conn:
        node_rows = await conn.fetch(
            """
            SELECT n.document_id::text, d.title, d.source, n.abbreviation, n.metadata,
                   n.graph_version, n.updated_at, COUNT(c.id)::int AS chunk_count,
                   l.status AS localization_status, l.content AS localization_content
            FROM paper_graph_nodes n JOIN documents d ON d.id=n.document_id
            LEFT JOIN LATERAL (
                SELECT status, content FROM paper_graph_localizations
                WHERE document_id=n.document_id AND target_language='zh'
                  AND schema_version=$1
                ORDER BY updated_at DESC LIMIT 1
            ) l ON TRUE
            LEFT JOIN chunks c ON c.document_id=d.id
            GROUP BY n.document_id,d.title,d.source,d.created_at,n.abbreviation,n.metadata,n.graph_version,n.updated_at,l.status,l.content
            ORDER BY d.created_at DESC
            """, LOCALIZATION_SCHEMA_VERSION,
        )
        edge_rows = await conn.fetch(
            "SELECT source_document_id::text,target_document_id::text,relation_type,score,evidence FROM paper_graph_edges"
        )
    nodes = [{"document_id": row["document_id"], "title": row["title"], "source": row["source"], "abbreviation": row["abbreviation"], "chunk_count": row["chunk_count"], "metadata": json.loads(row["metadata"] or "{}"), "title_zh": str((json.loads(row["localization_content"] or "{}")).get("title_zh") or ""), "localization_status": str(row["localization_status"] or "pending"), "research_card": json.loads(row["localization_content"] or "{}")} for row in node_rows]
    edges = [{"source_document_id": row["source_document_id"], "target_document_id": row["target_document_id"], "relation_type": row["relation_type"], "score": float(row["score"]), "evidence": json.loads(row["evidence"] or "{}")} for row in edge_rows]
    version = max((int(row["graph_version"]) for row in node_rows), default=0)
    updated_at = max((row["updated_at"].isoformat() for row in node_rows if row["updated_at"]), default=None)
    return PaperGraphResponse(version=version, nodes=nodes, edges=edges, updated_at=updated_at)


async def ensure_paper_graph() -> None:
    """Materialize nodes missing from an existing knowledge base on first graph read."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.id::text AS document_id
            FROM documents d
            LEFT JOIN paper_graph_nodes n ON n.document_id=d.id
            WHERE n.document_id IS NULL
            ORDER BY d.created_at
            """
        )
    for row in rows:
        await refresh_paper_graph(str(row["document_id"]))


async def get_graph_neighbor_document_ids(
    document_ids: List[str],
    limit: int = 12,
    relation_types: Optional[List[str]] = None,
    direction: str = "both",
) -> List[str]:
    """Return relation-filtered graph neighbors for retrieval candidate expansion."""
    normalized = list(dict.fromkeys(value for value in document_ids if value))
    if not normalized:
        return []
    async with db_pool.acquire() as conn:
        allowed_relations = list(dict.fromkeys(relation_types or ["semantic_similarity", "method_lineage", "cites"]))
        if direction == "outgoing":
            query = """SELECT target_document_id::text AS document_id FROM paper_graph_edges
                       WHERE source_document_id = ANY($1::uuid[]) AND relation_type = ANY($2::text[])
                       ORDER BY score DESC LIMIT $3"""
        elif direction == "incoming":
            query = """SELECT source_document_id::text AS document_id FROM paper_graph_edges
                       WHERE target_document_id = ANY($1::uuid[]) AND relation_type = ANY($2::text[])
                       ORDER BY score DESC LIMIT $3"""
        else:
            query = """SELECT CASE WHEN source_document_id = ANY($1::uuid[]) THEN target_document_id::text ELSE source_document_id::text END AS document_id
                       FROM paper_graph_edges
                       WHERE (source_document_id = ANY($1::uuid[]) OR target_document_id = ANY($1::uuid[]))
                         AND relation_type = ANY($2::text[])
                       ORDER BY score DESC LIMIT $3"""
        rows = await conn.fetch(query, normalized, allowed_relations, limit)
    return [str(row["document_id"]) for row in rows if str(row["document_id"]) not in normalized]
