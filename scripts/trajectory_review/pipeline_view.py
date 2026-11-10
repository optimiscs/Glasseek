"""Read-only live projection of immutable curation artifacts, not a new gate."""
import collections
import hashlib
import json
import os
import threading
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
STAGES=[('safety','安全与协议'),('eligibility','审计条件'),('inventory','要求与主张'),('audit','证据初审'),
        ('review','独立证据复核'),('behaviors','行为检查'),('quality','整轨质量门槛'),('acceptance','独立整轨验收'),
        ('segment_partition','语义片段划分'),('segment_causal','动作当时的合理性'),
        ('segment_contribution','片段贡献与监督'),('human','人工审核'),('replay','真实回放'),('release','训练发布')]
LABELS={'candidate':'待人工审核候选','quality_hold':'质量／证据待核实','annotation_review':'标注待修复',
        'acceptance_hold':'整轨验收待复核','segment_review':'片段待复核','annotation_error':'标注失败',
        'incomplete':'轨迹未完成','manual_review':'提前人工分流','evidence_insufficient':'缺少审计证据',
        'repair_pending':'结构待修复','quarantine':'安全隔离','unsupported_protocol':'协议不兼容',
        'running':'处理中','queued':'排队中'}
LOCK=threading.RLock()
CACHE={}


def run_dir():
    return Path(os.environ.get('GLASSEEK_CURATION_RUN',ROOT/'data/processed/qwen35-9b-curation128-20261109')).resolve()


def read(path, default=None):
    if not path.is_file():return default
    stamp=(path.stat().st_mtime_ns,path.stat().st_size)
    old=CACHE.get(str(path))
    if old and old[0]==stamp:return old[1]
    value=json.loads(path.read_text())
    CACHE[str(path)]=(stamp,value)
    return value


def lines(path):
    if not path.is_file():return []
    rows=[]
    for line in path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:continue  # the live writer may have an unfinished final line
    return rows


def percentile(values,q):
    if not values:return None
    v=sorted(values);x=(len(v)-1)*q;i=int(x)
    return round(v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i),2)


def disposition(r, active=False):
    if not r:return 'running' if active else 'queued'
    if r['status']=='routed':return r['decision']
    if r['status'] in ('annotation_error','failed'):return 'annotation_error'
    if r.get('annotation_issues'):return 'annotation_review'
    if not r.get('offline_quality_candidate'):return 'quality_hold'
    if not r.get('automatic_acceptance_passed'):return 'acceptance_hold'
    if not r.get('selection',{}).get('eligible_for_ranking'):return 'segment_review'
    return 'candidate'


def states(item,r,calls,active_stage=None):
    out={k:'not_reached' for k,_ in STAGES}
    bystage=collections.defaultdict(list)
    for c in calls:bystage[c['stage']].append(c)
    for stage,entries in bystage.items():
        if stage not in out:continue
        successes={c['id'] for c in entries if c['status']=='success'}
        expected=len(item['steps']) if stage=='segment_causal' else 1
        out[stage]='done' if len(successes)>=expected else 'running'
    if active_stage in out and out[active_stage]=='not_reached':out[active_stage]='running'
    if not r:
        if calls or active_stage:
            # The audit has started, but final per-record rule receipts have
            # not been written yet. Do not invent a pass or say "not reached".
            out['safety']=out['eligibility']='done'
        return out
    gate=r.get('eligibility',{});safety=gate.get('safety',{})
    if safety:out['safety']='pass' if safety.get('state')=='pass_rules' and safety.get('coverage_complete') else 'hold'
    out['eligibility']='pass' if gate.get('status')=='eligible' else 'hold'
    if r.get('status')=='annotation_error':
        stage=r.get('error','').split(':',1)[0]
        if stage in out:out[stage]='error'
    if r.get('inventory'):out['inventory']='pass'
    if r.get('audit'):
        out['review']='pass' if all(x['status']=='supported' for x in r['audit']['items']) else 'hold'
    if r.get('raw_behaviors'):
        out['behaviors']='hold' if r.get('negative_behavior_issues') or r.get('annotation_issues') else 'pass'
        unresolved=any(r.get(k) for k in ('quality_issues','evidence_gaps','annotation_issues','negative_behavior_issues'))
        out['quality']='hold' if unresolved else 'pass'
    if r.get('acceptance'):out['acceptance']='pass' if r.get('automatic_acceptance_passed') else 'hold'
    if r.get('segments'):
        out['segment_partition']='pass'
        verdicts=[a['verdict'] for s in r['segments'] for a in s.get('causal_assessments',{}).values()]
        out['segment_causal']='hold' if 'uncertain' in verdicts else 'done'
        out['segment_contribution']='hold' if any(s['decision']=='review' for s in r['segments']) else 'pass'
    # Candidate status is never a human signature, execution receipt or release.
    out['human']='pending' if r.get('human_review')=='pending' else 'not_reached'
    out['replay']='pending' if disposition(r)=='candidate' else 'not_reached'
    out['release']='pass' if r.get('training_ready') is True else 'not_reached'
    return out


def snapshot():
    with LOCK:
        root=run_dir()
        if not (root/'clean.jsonl').is_file():raise FileNotFoundError('尚未配置清洗批次')
        items=lines(root/'clean.jsonl');ids={s['id'] for s in items}
        artifacts={}
        for p in root.glob('gpu*/results/*.json'):
            a=read(p)
            if a and a['sample']['id'] in ids:artifacts[a['sample']['id']]=a
        calls=[]
        for p in root.glob('gpu*/calls.jsonl'):calls.extend(lines(p))
        grouped=collections.defaultdict(list)
        for c in calls:grouped[c['id'].split(':step:',1)[0]].append(c)
        # First-attempt prompts exist before any response; report activity, not an invented pass.
        hashes={hashlib.sha256(json.dumps(i,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:24]:i for i in ids}
        active={}
        for p in root.glob('gpu*/prompts/*/*.json'):
            ident=hashes.get(p.parent.name)
            if ident and (ident not in active or p.stat().st_mtime>active[ident][0]):active[ident]=(p.stat().st_mtime,p.stem)
        rows=[]
        for item in items:
            ident=item['id'];r=artifacts.get(ident,{}).get('result',{});cs=grouped[ident]
            stage=active.get(ident,(0,None))[1];status=disposition(r,bool(cs or stage))
            rows.append({'id':ident,'source':item['source'],'question':item['question'],
                         'outcome':status,'outcome_label':LABELS.get(status,status),'steps':len(item['steps']),
                         'seconds':r.get('seconds'),'states':states(item,r,cs,stage),
                         'training_ready':r.get('training_ready') is True,
                         'conflicts':sum(x['status']=='conflict' for x in r.get('audit',{}).get('items',[])),
                         'verification_merges':len(r.get('merged_behavior_annotations',[]))})
        stages=[]
        for key,label in STAGES:
            cs=[c for c in calls if c['stage']==key];ok=[c['seconds'] for c in cs if c['status']=='success']
            stages.append({'key':key,'label':label,'counts':dict(collections.Counter(r['states'][key] for r in rows)),
                           'requests':len(cs),'failed_attempts':sum(c['status']!='success' for c in cs),
                           'p50':percentile([c['seconds'] for c in cs],.5),'p95':percentile([c['seconds'] for c in cs],.95),
                           'success_p95':percentile(ok,.95)})
        metrics=lines(root/'vllm-metrics.jsonl')
        summary={'total':len(rows),'completed':len(artifacts),'candidate':sum(r['outcome']=='candidate' for r in rows),
                 'training_ready':sum(r['training_ready'] for r in rows),'outcomes':dict(collections.Counter(r['outcome'] for r in rows))}
        return {'run':root.name,'updated_at':time.time(),'summary':summary,'stages':stages,'rows':rows,
                'run_metadata':{k:v for k,v in (read(root/'run.json',{}) or {}).items() if k in ('started_at','finished_at','producer_per_gpu','producer_total','vllm_max_num_seqs','wall_seconds')},
                'metrics':metrics[-1] if metrics else None,'labels':LABELS}


def detail(ident):
    with LOCK:
        root=run_dir();items={r['id']:r for r in lines(root/'clean.jsonl')}
        if ident not in items:raise KeyError(ident)
        key=hashlib.sha256(json.dumps(ident,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:24]
        artifact=next((read(p) for p in root.glob('gpu*/results/'+key+'.json')),None)
        item=items[ident];r=artifact['result'] if artifact else {}
        if r.get('eligibility',{}).get('status')=='quarantine':
            return {'id':ident,'quarantined':True,'message':'安全隔离记录不在此看板展示原文，请使用受控人工审核流程。'}
        audit=[]
        targets={x['id']:x['text'] for k in ('constraints','claims') for x in r.get('inventory',{}).get(k,[])}
        for x in r.get('audit',{}).get('items',[]):
            review=x.get('review',{})
            audit.append({'id':x['id'],'text':targets.get(x['id'],''),'status':x['status'],'reason':x.get('reason'),
                          'model_verdict':review.get('model_verdict'),'model_reason':review.get('model_reason'),
                          'decision_origin':review.get('decision_origin'),
                          'evidence':[{'span_id':e.get('span_id'),'step_id':e['step_id'],'quote':e['quote']} for e in x['evidence']]})
        calls=[]
        for p in root.glob('gpu*/calls.jsonl'):
            for c in lines(p):
                if c['id'].split(':step:',1)[0]==ident:
                    calls.append({k:c.get(k) for k in ('stage','id','attempt','status','seconds','usage','error','finish_reason')})
        return {'id':ident,'source':item['source'],'question':item['question'],'outcome':disposition(r,bool(calls)),
                'eligibility':r.get('eligibility'),'inventory':r.get('inventory'),'audit':audit,'acceptance':r.get('acceptance'),
                'segments':r.get('segments',[]),'supervision':r.get('supervision'),'error':r.get('error'),
                'issues':{k:r.get(k,[]) for k in ('quality_issues','evidence_gaps','annotation_issues','negative_behavior_issues')},
                'verification_merges':[{k:m.get(k) for k in ('step_id','tag','target_id','extra_score')} for m in r.get('merged_behavior_annotations',[])],
                'human_review':r.get('human_review','pending'),'training_ready':r.get('training_ready',False),'calls':calls,
                'steps':[{k:s[k] for k in ('step_id','assistant','calls','observation')} for s in item['steps']]}
