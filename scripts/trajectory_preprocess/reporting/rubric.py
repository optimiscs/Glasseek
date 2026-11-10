"""Paired one-call pilot comparison. Different quality scales are never conflated."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import sqlite3

from client import digest,save
from rubric_batch import unpack,percentile
from trajectory_rubric import score
from report import outcome


def compare(root, baseline):
    db=sqlite3.connect(root/'batch.sqlite')
    inputs={ident:unpack(sample) for ident,sample in db.execute('SELECT id,sample FROM inputs')}
    new={ident:json.loads(result) for ident,result in db.execute("SELECT id,result FROM results WHERE state='completed'")}
    old_artifacts=[json.loads(p.read_text()) for p in baseline.glob('gpu*/results/*.json')]
    old={a['sample']['id']:a for a in old_artifacts}
    if len(inputs)!=128 or set(inputs)!=set(new) or set(inputs)!=set(old):
        raise ValueError('Exactly the same 128 records must have completed')
    for ident,item in inputs.items():
        if item!=old[ident]['sample'] or digest(item)!=new[ident]['sample_hash']:
            raise ValueError('Paired samples changed')
    calls=[]
    for p in root.glob('gpu*/calls.jsonl'):
        calls.extend(json.loads(line) for line in p.open())
    call_counts=Counter(c['id'] for c in calls)
    eligible={i for i,r in new.items() if r.get('llm_eligible')}
    checks={
        'same_128_inputs':True,
        'exactly_one_call_per_eligible':set(call_counts)==eligible and all(v==1 for v in call_counts.values()),
        'no_calls_for_routed':not(set(call_counts)&{i for i,r in new.items() if r['status']=='routed'}),
        'local_9b_only':all(c.get('returned_model')=='qwen3.5-9b-local' for c in calls if 'returned_model' in c)
                         and all(c['model']=='qwen3.5-9b-local' for c in calls),
        'no_generation_or_transport_retries':all(c['attempt']==1 and c.get('transport_attempts',1)==1 for c in calls),
        'no_training_release':all(r['training_ready'] is False for r in new.values()),
        'unchanged_rule_routes':all(r.get('eligibility',{}).get('status')==old[i]['result'].get('eligibility',{}).get('status') for i,r in new.items()),
        'program_scores_reproducible':all(all(r[k]==v for k,v in score(r['judgment']).items()) for r in new.values() if 'judgment' in r),
    }
    errors=sum(r['status']=='annotation_error' for r in new.values())
    checks['valid_annotation_rate_at_least_95pct']= bool(eligible) and (len(eligible)-errors)/len(eligible)>=.95
    pairs=[]; transitions=Counter(); source=defaultdict(Counter)
    for ident,r in new.items():
        before=outcome(old[ident]['result'])
        after=r.get('decision') if r['status']=='routed' else r['status']
        transitions[(before,after)]+=1; source[r['source']][after]+=1
        pairs.append({'id':ident,'source':r['source'],'old':before,'new':after,'score':r['score'],
                      'old_average_behavior_score':old[ident]['result'].get('average_score'),
                      'old_seconds':old[ident]['result']['seconds'],'new_seconds':r['seconds'],
                      'failed_gates':r.get('failed_gates',[]),'label_conflicts':r.get('label_conflicts',[])})
    old_calls=[]
    for p in baseline.glob('gpu*/calls.jsonl'):
        old_calls.extend(json.loads(line) for line in p.open())
    def performance(rows):
        starts=[datetime.fromisoformat(c['started_at']) for c in rows]
        ends=[datetime.fromisoformat(c['finished_at']) for c in rows]
        secs=[c['seconds'] for c in rows]
        return {'calls':len(rows),'request_window_seconds':(max(ends)-min(starts)).total_seconds(),
                'request_seconds_p50':percentile(secs,.5),'request_seconds_p95':percentile(secs,.95),
                'usage':{k:sum(c.get('usage',{}).get(k,0) for c in rows) for k in ('prompt_tokens','completion_tokens','total_tokens')}}
    ranks=[json.loads(line) for line in (root/'ranking.jsonl').open()]
    old_candidates={i for i,a in old.items() if outcome(a['result'])=='candidate'}
    new_rankable={r['id'] for r in ranks}
    semantic_path=root/'semantic-validation.json'
    semantic=json.loads(semantic_path.read_text()) if semantic_path.exists() else {'status':'pending'}
    comparison={'checks':checks,'automatic_validation_passed':all(checks.values()),
                'manual_semantic_review':semantic['status'],
                'full_corpus_allowed':all(checks.values()) and semantic['status']=='passed','eligible_records':len(eligible),'annotation_errors':errors,
                'transitions':[{'old':a,'new':b,'count':n} for (a,b),n in sorted(transitions.items())],
                'by_dataset':dict(source),'performance':{'old':performance(old_calls),'new':performance(calls)},
                'selection_overlap':{'old_candidates':len(old_candidates),'new_rankable':len(new_rankable),
                                     'intersection':len(old_candidates&new_rankable),
                                     'new_rankable_not_old_candidate':len(new_rankable-old_candidates),
                                     'old_candidate_not_new_rankable':len(old_candidates-new_rankable)},
                'ranking_ties':{'score_counts':dict(Counter(r['score'] for r in ranks)),
                                'full_score_count':sum(r['score']==100 for r in ranks),
                                'distinct_scores':len({r['score'] for r in ranks}),
                                'tie_break':'ID ascending is deterministic, not a quality distinction'},
                'score_distribution':{'p10':percentile([r['score'] for r in ranks],.1),
                                      'p50':percentile([r['score'] for r in ranks],.5),
                                      'p90':percentile([r['score'] for r in ranks],.9)},
                'top20':ranks[:20],
                'note':'Old score is average behavior reward, new score is weighted whole-trajectory quality. Do not equate scales; rankable is not independently accepted for training.',
                'paired':pairs}
    save(root/'comparison.json',comparison)
    return {k:v for k,v in comparison.items() if k not in ('paired','top20')}


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,required=True)
    args=ap.parse_args()
    print(json.dumps(compare(args.input,args.baseline),ensure_ascii=False,indent=2))
