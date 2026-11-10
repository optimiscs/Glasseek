"""Compare a frozen paired pilot; distinguish routing, model errors and quality."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime
import json
from pathlib import Path
from client import save,digest
from selection import diverse_order


def outcome(result):
    if result['status']=='routed': return result['decision']
    if result['status']=='annotation_error':return 'annotation_error'
    if result.get('annotation_issues'):return 'annotation_review'
    if not result.get('offline_quality_candidate'):return 'quality_hold'
    if not result.get('automatic_acceptance_passed'):return 'acceptance_hold'
    if not result.get('selection',{}).get('eligible_for_ranking'):return 'segment_review'
    return 'candidate'


def compare(root, baseline):
    artifacts=[json.loads(p.read_text()) for p in root.glob('gpu*/results/*.json')]
    byid={a['sample']['id']:a for a in artifacts}
    inputs={r['id']:r for r in map(json.loads,(root/'clean.jsonl').open())}
    old_inputs={r['id']:r for r in map(json.loads,(baseline/'clean.jsonl').open())}
    if len(byid)!=len(artifacts) or set(byid)!=set(inputs) or inputs!=old_inputs:
        raise ValueError('Pilot must contain exactly the same frozen input records')
    old={r['id']:r for r in map(json.loads,(baseline/'decisions.jsonl').open())}
    dataset=defaultdict(Counter);routes=defaultdict(Counter);pairs=[];results=[];calls=[]
    canonical={r['id']:r for r in map(json.loads,(root/'selected-export.jsonl').open())}
    candidates=[]
    for ident,a in byid.items():
        r=a['result'];status=outcome(r);results.append(r)
        dataset[a['sample']['source']][status]+=1
        gate=r.get('eligibility',{});routes[a['sample']['source']][gate.get('status','gate_error')]+=1
        pairs.append({'id':ident,'source':a['sample']['source'],'old':old[ident], 'new':status,
                      'seconds':r['seconds'],'routing':gate.get('routing'),'reasons':gate.get('reasons',[])})
        if status=='candidate':
            row={**canonical[ident],'supervision':r['supervision'],'training_ready':False,'annotation_policy_hash':r['policy_hash'],
                 'review_flags':canonical[ident].get('review_flags',[])+['human_semantic_supervision_review_required']}
            candidates.append(row)
    for path in root.glob('gpu*/calls.jsonl'):calls.extend(map(json.loads,path.open()))
    starts=[datetime.fromisoformat(c['started_at']) for c in calls];ends=[datetime.fromisoformat(c['finished_at']) for c in calls]
    errors=Counter(r.get('error','').split(':',1)[0] for r in results if r['status']=='annotation_error')
    summary={'input_records':len(inputs),'outcomes':dict(Counter(outcome(r) for r in results)),
             'annotation_scored':sum(r['status']=='scored' for r in results),
             'offline_quality_candidates':sum(r.get('offline_quality_candidate',False) for r in results),
             'by_dataset':dict(dataset),'routing_by_dataset':dict(routes),'annotation_errors_by_stage':dict(errors),
             'requests':len(calls),'request_failures':sum(c['status']!='success' for c in calls),
             'model_request_wall_seconds':(max(ends)-min(starts)).total_seconds() if starts else 0,
             'mean_eligible_seconds':sum(r['seconds'] for r in results if r['status']!='routed')/max(1,sum(r['status']!='routed' for r in results)),
             'usage':{k:sum(c.get('usage',{}).get(k,0) for c in calls) for k in ('prompt_tokens','completion_tokens','total_tokens')},
             'segments':dict(Counter(s['decision'] for r in results for s in r.get('segments',[]))),
             'ranked_candidate_ids':diverse_order(results),'training_ready':0,
             'old_summary':json.loads((baseline/'summary.json').read_text()),
             'note':'Same rows and safety/quality validators; new semantic audit adds calls. Candidates still require human review and real replay.'}
    save(root/'comparison.json',{'summary':summary,'paired':pairs})
    save(root/'summary.json',summary)
    for name,rows in [('annotations',artifacts),('decisions',pairs),('candidates',candidates)]:
        (root/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True)
    args=ap.parse_args();print(json.dumps(compare(args.input,args.baseline),ensure_ascii=False,indent=2))
