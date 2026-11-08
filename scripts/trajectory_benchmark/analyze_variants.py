"""Separate variant generation, verification and human/export readiness."""
import json,collections
from pathlib import Path
OUT=Path('/root/autodl-tmp/Glasseek/reports/trajectory-stratified-qwen38-20261107')
def main():
    rows=[]
    for directory in sorted((OUT/'variant_probes').iterdir()):
        if not directory.is_dir():continue
        artifacts=[json.loads(p.read_text()) for p in (directory/'variants').glob('*.json')]
        summary=json.loads((directory/'variant_summary.json').read_text()) if (directory/'variant_summary.json').exists() else {}
        calls=[json.loads(l) for l in (directory/'calls.jsonl').open()] if (directory/'calls.jsonl').exists() else []
        kinds=collections.Counter(a['sample']['lineage']['operation'] for a in artifacts)
        row={'source':directory.name,'selected_parents':1,'precondition_holds':summary.get('precondition_holds',0),
             'rewrite_generated':kinds['rewrite'],'rewrite_scored':sum(a['sample']['lineage']['operation']=='rewrite' and a['result']['status']=='scored' for a in artifacts),
             'rewrite_candidates':sum(a['sample']['lineage']['operation']=='rewrite' and a['result'].get('offline_quality_candidate',False) for a in artifacts),
             'negative_generated_and_reviewed':kinds['counterfactual'],
             'negative_review_verdicts':dict(collections.Counter(a['result'].get('review',{}).get('verdict') for a in artifacts if a['sample']['lineage']['operation']=='counterfactual')),
             'generation_or_review_failures':summary.get('failed',0),'calls':len(calls),'failed_calls':sum(c['status']=='failed' for c in calls),
             'errors':dict(collections.Counter(c.get('error') for c in calls if c['status']=='failed'))}
        rows.append(row)
    result={'sources':rows,'selected_parents':len(rows),'precondition_holds':sum(r['precondition_holds'] for r in rows),'rewrite_generated':sum(r['rewrite_generated'] for r in rows),'rewrite_candidates':sum(r['rewrite_candidates'] for r in rows),'verified_negative_candidates':sum(r['negative_review_verdicts'].get('contradicted',0) for r in rows),'calls':sum(r['calls'] for r in rows),'training_ready':0,'preference_pairs_ready':0,'human_approvals':0}
    (OUT/'variant_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
if __name__=='__main__':main()
