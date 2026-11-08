"""Aggregate observable pipeline outcomes; never label unresolved data as bad quality."""
import json,collections,datetime,statistics,hashlib
from pathlib import Path
OUT=Path('/root/autodl-tmp/Glasseek/reports/trajectory-stratified-qwen38-20261107')
def readlines(p):return [json.loads(l) for l in p.open()] if p.exists() else []
def main():
    manifest=readlines(OUT/'manifest.jsonl');calls=readlines(OUT/'calls.jsonl')
    for extra in sorted((OUT/'variant_probes').glob('*/calls.jsonl')):calls.extend(readlines(extra))
    artifacts=[json.loads(p.read_text()) for p in (OUT/'results').glob('*.json')]
    result={x['sample']['id']:x['result'] for x in artifacts};original_ids={x['id'] for x in manifest}
    original_calls=readlines(OUT/'calls.jsonl')
    sources=[]
    for source in dict.fromkeys(r['source'] for r in manifest):
        rows=[r for r in manifest if r['source']==source];ids={r['id'] for r in rows};rr=[v for k,v in result.items() if k in ids]
        row={'source':source,'selected':len(rows),'normalized':sum(r['normalization_status']=='passed' for r in rows),
             'eligible':sum(r['eligible'] for r in rows),'completed':len(rr),'scored':sum(r['status']=='scored' for r in rr),
             'candidates':sum(r.get('offline_quality_candidate',False) for r in rr),
             'pre_llm_holds':dict(collections.Counter(r['hold_category'] for r in rows if not r['eligible'])),
             'scored_blockers_nonexclusive':dict(collections.Counter(b for r in rr for b in r.get('blockers',[]))),
             'annotation_failures':dict(collections.Counter(r['error'] for r in rr if r['status']=='failed'))}
        row['stages']={stage:len({c['id'] for c in original_calls if c['id'] in ids and c['stage']==stage and c['status']=='success'}) for stage in ['inventory','audit','review','behaviors']}
        sources.append(row)
    stages=[]
    for stage in dict.fromkeys(c['stage'] for c in calls):
        cs=[c for c in calls if c['stage']==stage];oc=[c for c in original_calls if c['stage']==stage]
        stages.append({'stage':stage,'calls_including_variants':len(cs),'original_samples_attempted':len({c['id'] for c in oc}),
                       'original_samples_validated':len({c['id'] for c in oc if c['status']=='success'}),
                       'call_failures_including_retries':sum(c['status']=='failed' for c in cs),'length_finishes':sum(c.get('finish_reason')=='length' for c in cs),
                       'prompt_tokens':sum(c.get('usage',{}).get('prompt_tokens',0) for c in cs),
                       'completion_tokens':sum(c.get('usage',{}).get('completion_tokens',0) for c in cs),
                       'reasoning_tokens':sum(c.get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens',0) or 0 for c in cs),
                       'max_completion_tokens':max(c.get('usage',{}).get('completion_tokens',0) for c in cs),
                       'request_seconds_sum':round(sum(c['seconds'] for c in cs),3),
                       'request_seconds_median':statistics.median(c['seconds'] for c in cs)})
    starts=[datetime.datetime.fromisoformat(c['started_at']) for c in calls];ends=[datetime.datetime.fromisoformat(c['finished_at']) for c in calls]
    totals={'selected':len(manifest),'eligible':sum(r['eligible'] for r in manifest),'scored':sum(r['status']=='scored' for r in result.values()),
            'offline_candidates':sum(r.get('offline_quality_candidate',False) for r in result.values()),'human_review_completed':0,'harness_replayed':0,'training_ready':0,
            'llm_wall_span_seconds':(max(ends)-min(starts)).total_seconds() if calls else None,'calls':len(calls),
            'failed_calls':sum(c['status']=='failed' for c in calls),'prompt_tokens':sum(s['prompt_tokens'] for s in stages),
            'completion_tokens':sum(s['completion_tokens'] for s in stages),'reasoning_tokens':sum(s['reasoning_tokens'] for s in stages),'external_api_cost_usd':0,
            'local_compute_cost':'Not priced; GPU time is not free. Wall span includes queuing and any pauses.'}
    pricing_path=OUT/'original_pricing_cny.json'
    if pricing_path.exists():
        hourly=json.loads(pricing_path.read_text())['gpu_hourly_rate']
        totals['gpu_hourly_rate_cny']=hourly
        totals['gpu_cost_cny_for_llm_wall_span']=totals['llm_wall_span_seconds']/3600*hourly if calls else None
        totals['cached_prompt_tokens']=sum(c.get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens',0) or 0 for c in calls)
        totals['local_compute_cost']='User rate applied once to wall-clock span across original and variant calls. Includes queue/retries/phase gaps; excludes earlier compilation/backend tests and later idle rental time.'
    output={'scope':'Exploratory stratified coverage; raw pass rates are not population estimates. Schema/budget holds are not quality rejection.',
            'totals':totals,'sources':sources,'stages':stages,'call_failure_reasons':dict(collections.Counter(c.get('error') for c in calls if c['status']=='failed'))}
    (OUT/'run_summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');print(json.dumps(totals))
if __name__=='__main__':main()
