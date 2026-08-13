"""Audit ingestion, source display, translation cache, and graph API contracts."""
from __future__ import annotations
import argparse, asyncio, json, time
from pathlib import Path
from agent.db_utils import close_database, initialize_database, db_pool
from agent.selection_translation_runtime import translate_selection
from common.encoding_utils import write_json_utf8, write_text_utf8

async def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--output-dir', default='evals/results/overnight_20260813'); args=parser.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    await initialize_database()
    try:
        async with db_pool.acquire() as conn:
            docs=await conn.fetch("""
                SELECT d.id::text, d.title
                FROM documents d JOIN chunks c ON c.document_id=d.id
                WHERE coalesce(c.metadata->>'content_type','pdf') <> 'artifact'
                  AND length(c.content)>160
                GROUP BY d.id, d.title
                ORDER BY CASE WHEN d.title ILIKE '%multi-agent rrt%' THEN 0 ELSE 1 END, d.title
                LIMIT 1
            """)
            stats=await conn.fetchrow("""SELECT count(DISTINCT document_id) docs, count(*) chunks,
                count(*) FILTER (WHERE metadata ? 'section_path_text') sectioned,
                count(*) FILTER (WHERE metadata ? 'section_start_line' AND metadata ? 'section_end_line') line_addressable,
                count(*) FILTER (WHERE metadata->>'content_type'='artifact') artifacts,
                count(*) FILTER (WHERE metadata->>'artifact_type'='table') tables,
                count(*) FILTER (WHERE metadata->>'artifact_type'='figure') figures,
                count(*) FILTER (WHERE metadata->>'artifact_type'='algorithm') algorithms
                FROM chunks""")
            sample=await conn.fetchrow("SELECT content FROM chunks WHERE document_id=$1::uuid AND coalesce(metadata->>'content_type','pdf') <> 'artifact' AND length(content)>160 ORDER BY chunk_index LIMIT 1", str(docs[0]['id']))
            graph=await conn.fetchrow("SELECT count(*) nodes, count(*) FILTER (WHERE l.status='ready') localized FROM paper_graph_nodes n LEFT JOIN LATERAL (SELECT status FROM paper_graph_localizations l WHERE l.document_id=n.document_id AND l.target_language='zh' ORDER BY updated_at DESC LIMIT 1) l ON TRUE")
        selection=str(sample['content'])[:240]
        started=time.perf_counter(); first=await translate_selection(str(docs[0]['id']),'zh',selection); first_seconds=time.perf_counter()-started
        started=time.perf_counter(); second=await translate_selection(str(docs[0]['id']),'zh',selection); second_seconds=time.perf_counter()-started
        translated=str(second.get('translated_text') or '')
        report={'ingestion':dict(stats),'translation':{'document_id':str(docs[0]['id']),'document_title':str(docs[0]['title']),'selection_length':len(selection),'first_cached':bool(first.get('cached')),'second_cached':bool(second.get('cached')),'profile_cached_after_first':bool(second.get('profile_cached')),'translation_nonempty':bool(translated.strip()),'rrt_token_preserved':('RRT' not in selection.upper()) or ('RRT' in translated.upper()),'first_seconds':round(first_seconds,3),'second_seconds':round(second_seconds,3)},'graph':dict(graph)}
    finally: await close_database()
    write_json_utf8(out/'feature_contract_eval.json',report,indent=2)
    write_text_utf8(out/'feature_contract_eval.md','# Feature Contract Evaluation\n\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': asyncio.run(main())
