"""Reproducible overnight audit for the current local paper corpus."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from urllib.request import Request, urlopen
from common.encoding_utils import write_json_utf8, write_text_utf8
from evals.run_corpus_readiness_eval import request_json, percentile

TEMPLATES = [
    "What problem and method does the paper {title} present?",
    "Summarize the main contribution of {title} from the local library.",
    "Which algorithmic idea is described by {title}?",
]

def request_get_json(url: str) -> dict:
    with urlopen(Request(url, method="GET"), timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--api-url', default='http://localhost:8888'); p.add_argument('--output-dir', default='evals/results/overnight_20260813'); args=p.parse_args()
    api=args.api_url.rstrip('/'); out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    docs=request_get_json(api+'/documents?limit=100').get('documents', [])
    cases=[]
    for index in range(100):
        doc=docs[index % len(docs)]; title=str(doc['title']); question=TEMPLATES[index % len(TEMPLATES)].format(title=title)
        started=time.perf_counter(); response=request_json(api+'/search/hybrid', {'query': question, 'limit': 5}); latency=(time.perf_counter()-started)*1000
        rows=response.get('results') or []; ids=[]
        for row in rows:
            if row.get('document_id') not in ids: ids.append(row.get('document_id'))
        rank=(ids.index(doc['id'])+1) if doc['id'] in ids else None
        cases.append({'id':f'qa_{index+1:03d}','question':question,'gold_document_id':doc['id'],'gold_title':title,'rank':rank,'hit':rank is not None,'source_schema_valid':all(all(k in r for k in ('document_id','document_title','chunk_id','content','metadata')) for r in rows),'latency_ms':round(latency,1)})
    graph=request_get_json(api+'/paper-graph')
    nodes=graph.get('nodes') or []; edges=graph.get('edges') or []; node_ids={str(n.get('document_id') or n.get('id')) for n in nodes}
    graph_audit={'nodes':len(nodes),'edges':len(edges),'node_document_coverage':len(node_ids)/max(1,len(docs)),'edge_endpoints_valid':all(str(e.get('source_document_id') or e.get('source')) in node_ids and str(e.get('target_document_id') or e.get('target')) in node_ids for e in edges),'abbreviation_coverage':sum(bool(str(n.get('short_title') or n.get('abbreviation') or '').strip()) for n in nodes)/max(1,len(nodes))}
    summary={'cases':100,'document_hit_at_1':sum(c['rank']==1 for c in cases)/100,'document_hit_at_5':sum(c['hit'] for c in cases)/100,'mrr_at_5':sum(1/c['rank'] for c in cases if c['rank'])/100,'source_schema_coverage':sum(c['source_schema_valid'] for c in cases)/100,'latency_ms_p50':percentile([c['latency_ms'] for c in cases],.5),'latency_ms_p90':percentile([c['latency_ms'] for c in cases],.9)}
    report={'protocol':{'type':'deterministic document-grounded QA retrieval and source-contract evaluation','answer_boundary':'This 100-case suite measures whether the answer chain receives the correct paper evidence; generated-answer faithfulness is separately smoke-tested with the full LangGraph chain.'},'summary':summary,'knowledge_graph':graph_audit,'cases':cases}
    write_json_utf8(out/'qa_retrieval_source_100.json',report,indent=2)
    lines=['# Overnight System Evaluation','', '## 100-case Retrieval and Source Contract','', *(f'- {k}: {v}' for k,v in summary.items()), '', '## Knowledge Graph Contract','', *(f'- {k}: {v}' for k,v in graph_audit.items())]
    write_text_utf8(out/'qa_retrieval_source_100.md','\n'.join(lines)+'\n')
    print(json.dumps({'summary':summary,'knowledge_graph':graph_audit},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
