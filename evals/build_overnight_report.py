"""Build a concise, UTF-8 handoff report from overnight evaluation artifacts."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from common.encoding_utils import read_json_robust, write_json_utf8, write_text_utf8

def main() -> None:
    out=Path('evals/results/overnight_20260813/final')
    progress=read_json_robust(out/'qa_100_progress.json')
    cases=list(progress.get('cases') or [])
    grouped=defaultdict(list)
    for case in cases: grouped[str(case.get('category'))].append(case)
    categories={}
    for name, rows in grouped.items():
        valid=[row for row in rows if not row.get('error')]; scores=[row.get('score') or {} for row in valid]
        categories[name]={
            'completed':len(rows),'errors':len(rows)-len(valid),
            'contract_pass_rate':sum(bool(score.get('contract_pass')) for score in scores)/max(1,len(scores)),
            'source_schema_coverage':sum(bool(score.get('source_schema_valid')) for score in scores)/max(1,len(scores)),
            'expected_document_coverage':sum(float(score.get('expected_document_coverage',0)) for score in scores)/max(1,len(scores)),
        }
    feature=read_json_robust(out/'feature_contract_eval.json') if (out/'feature_contract_eval.json').exists() else {}
    ingestion_path=out/'ingestion'/'ingestion_quality_eval.json'
    ingestion=read_json_robust(ingestion_path) if ingestion_path.exists() else {}
    root=out.parent
    expert_path=out/'chinese_retrieval/expert_gold_retrieval_eval.json'
    if not expert_path.exists():
        expert_path=root/'smoke_retrieval/expert_gold_retrieval_eval.json'
    expert=read_json_robust(expert_path) if expert_path.exists() else {}
    report={'run_status':{'completed':int(progress.get('completed') or len(cases)),'total':int(progress.get('total') or 100),'complete':int(progress.get('completed') or len(cases))>=int(progress.get('total') or 100)},'category_summary':categories,'feature_contract':feature,'ingestion_quality':ingestion.get('summary',{}),'chinese_retrieval_regression':expert.get('summary',{}),'known_fixes':[{'issue':'Chinese query returned no English-corpus results','fix':'Cached Chinese-to-English query bridge with dual-route retrieval','evidence':'12-case Chinese expert gold Hit@5=100%'},{'issue':'LangGraph source display could be empty after scoped retrieval','fix':'Full-corpus target resolution and explicit scoped gold document IDs in evaluation','evidence':'Smoke local questions return local evidence sources'},{'issue':'Explicit OpenAlex request routed to local KB','fix':'Explicit source routing guard and API-safe OpenAlex query normalization','evidence':'OpenAlex smoke calls search_openalex_papers and returns authors/year/DOI'},{'issue':'Stable direct explanations triggered retrieval','fix':'Explicit no-retrieval guard and source-policy short circuit','evidence':'Direct-answer smoke uses zero tools and zero sources'},{'issue':'Two graph Chinese cards failed validation','fix':'Restrict protected-token validation to technical abbreviations','evidence':'paper_graph_localizations ready=46/46'}]}
    failures=[{'id':row.get('id'),'category':row.get('category'),'error':row.get('error'),'score':row.get('score')} for row in cases if row.get('error') or not (row.get('score') or {}).get('contract_pass')]
    write_json_utf8(out/'overnight_summary.json',{**report,'nonpassing_cases':failures},indent=2)
    lines=['# 2026-08-13 Overnight Evaluation Handoff','',f"- Progress: {report['run_status']['completed']}/{report['run_status']['total']}",f"- Complete: {report['run_status']['complete']}",'','## Category Results','', '| Category | Completed | Errors | Contract pass | Source schema | Document coverage |','|---|---:|---:|---:|---:|---:|']
    for name, item in categories.items(): lines.append(f"| {name} | {item['completed']} | {item['errors']} | {item['contract_pass_rate']:.1%} | {item['source_schema_coverage']:.1%} | {item['expected_document_coverage']:.1%} |")
    lines += ['', '## Feature Contracts', '', '```json', json.dumps(feature,ensure_ascii=False,indent=2), '```', '', '## Ingestion Quality', '', '```json', json.dumps(ingestion.get('summary',{}),ensure_ascii=False,indent=2), '```', '', '## Chinese Retrieval Regression', '', '```json', json.dumps(expert.get('summary',{}),ensure_ascii=False,indent=2), '```', '', '## Fixed During Evaluation', '']
    lines += [f"- {item['issue']}: {item['fix']} ({item['evidence']})" for item in report['known_fixes']]
    lines += ['', '## Nonpassing Cases', '']
    lines += [f"- {item['id']} ({item['category']}): {item['error'] or 'contract not met'}" for item in failures] or ['- None.']
    write_text_utf8(out/'OVERNIGHT_EVALUATION_REPORT.md','\n'.join(lines)+'\n')
    print(json.dumps(report['run_status'],ensure_ascii=False))
if __name__=='__main__': main()
