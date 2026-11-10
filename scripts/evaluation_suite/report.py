"""Native benchmark metrics and cumulative inference accounting (never estimates)."""
import collections
import csv
import json
import os
from pathlib import Path
import re
import sqlite3
from suite import ROOT,dump,jsonl,read


def load(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def case_usage(row):
    profiles={r.get('profile') for r in row.get('retry_attempts',[]) if isinstance(r,dict)}
    profiles.add(row.get('profile'));profiles.discard(None);profiles.discard('')
    total=collections.Counter();tools=collections.Counter();missing=[];seen=set()
    for profile in sorted(profiles):
        path=Path(os.environ.get('HERMES_HOME',ROOT/'Evaluation/data/runtime/hermes'))/'profiles'/profile/'state.db'
        if not path.is_file():missing.append(profile);continue
        with sqlite3.connect(f'file:{path}?mode=ro',uri=True,timeout=1) as db:
            for sid,i,o,c,w,n in db.execute('SELECT id,input_tokens,output_tokens,cache_read_tokens,cache_write_tokens,api_call_count FROM sessions'):
                if (str(path),sid) in seen:continue
                seen.add((str(path),sid));total.update(input_tokens=(i or 0)+(c or 0)+(w or 0),output_tokens=o or 0,cached_input_tokens=c or 0,model_calls=n or 0)
            for role,content,calls in db.execute('SELECT role,content,tool_calls FROM messages WHERE COALESCE(inherited_from_previous_session,0)=0'):
                if role=='assistant':
                    for call in json.loads(calls or '[]'):tools[call.get('function',{}).get('name',call.get('name','unknown'))]+=1
                if role=='tool' and 'LIVE_CTX_MAIN.txt: edit applied' in (content or ''):
                    total['context_edits']+=1
                    m=re.search(r'context ~(\d+)->(\d+) tokens',content)
                    if m and abs(int(m[1])-int(m[2]))>=max(32,int(m[1])*.01):total['context_edits_substantial']+=1
    if not seen:
        sessions={h['session_id']:h for h in row.get('history',[]) if h.get('session_id') and 'input_tokens' in h}
        for h in sessions.values():total.update(input_tokens=(h.get('input_tokens') or 0)+(h.get('cache_read_tokens') or 0)+(h.get('cache_write_tokens') or 0),output_tokens=h.get('output_tokens') or 0,cached_input_tokens=h.get('cache_read_tokens') or 0,model_calls=h.get('api_call_count') or 0)
        if not sessions:return {'usage_status':'missing','input_tokens':None,'output_tokens':None,'cached_input_tokens':None,'context_edits':None,'tool_calls':row.get('tool_calls',{})}
    # CLM semantic web calls are inside bash; keep them separate from physical bash calls.
    web=collections.Counter();work=Path(row['workspace_dir']) if row.get('workspace_dir') else None
    if work:
        for p in work.parent.glob('attempt_*/official-clm/workspace/.clm/web-calls.json'):
            web.update(load(p,{}) or {})
    for key in ('web_search','web_extract'):
        tools[key]=web[key] if web else (row.get('tool_calls') or {}).get(key,tools[key])
    # Research reviews deliberately bypass the main conversation transcript.
    # Count their measured usage separately, then include it once in totals.
    review_records = []
    if work:
        paths = set(work.parent.glob('attempt_*/research-result-a*.json'))
        paths.update(work.glob('research-result-a*.json'))
        for path in sorted(paths):
            review_records.extend((load(path, {}) or {}).get('reviews', []))
    if not review_records:
        review_records = (row.get('research_submission') or {}).get('reviews', [])
    review_usage_missing = 0
    for event in review_records:
        total['model_calls'] += 1
        usage = event.get('usage')
        if not isinstance(usage, dict):
            review_usage_missing += 1
            continue
        total['input_tokens'] += usage.get('prompt_tokens') or 0
        total['output_tokens'] += usage.get('completion_tokens') or 0
        total['cached_input_tokens'] += (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
    measured={k:total[k] for k in ['input_tokens','output_tokens','cached_input_tokens','model_calls','context_edits','context_edits_substantial']}
    if not seen:measured.update(context_edits=None,context_edits_substantial=None)
    return {**measured,
        'usage_status':'complete' if seen and not missing and not review_usage_missing else 'partial','missing_profiles':missing,'tool_calls':dict(tools),
        'research_review_calls': len(review_records), 'research_review_usage_missing': review_usage_missing,
        'token_scope':'native session counters plus measured research review usage across attempts; input includes cache once; output includes reasoning and final'}


def native_metrics(folder,name):
    folder=Path(folder);metrics={};scored=None
    if name in ('browsecomp','hle','gaia'):
        p=folder/('author/scores.jsonl' if name=='gaia' else 'scores.jsonl')
        rows=list(read(p)) if p.exists() else []
        rows=[r for r in rows if isinstance(r.get('correct'),bool)]
        scored=len(rows);metrics={'correct':sum(r['correct'] for r in rows),'accuracy_scored':sum(r['correct'] for r in rows)/len(rows) if rows else None}
    elif name=='browsecomp_plus':
        receipts=[load(p,{}) for p in folder.glob('*.author/receipt.json')]
        valid=[r for r in receipts if r.get('status')=='scored' and isinstance(r.get('correct'),bool)]
        scored=len(valid);metrics={'correct':sum(r['correct'] for r in valid),
            'accuracy_scored':sum(r['correct'] for r in valid)/scored if scored else None,
            'calibration_status':'undefined_below_100' if scored<100 else 'see_author_outputs'}
        recall=[load(r['judge_result_path'],{}).get('retrieval',{}).get('recall') for r in valid]
        recall=[v for v in recall if isinstance(v,(int,float))]
        metrics['evidence_recall']=sum(recall)/len(recall) if recall else None
    elif name=='widesearch':
        p=folder/'native_metrics.jsonl';rows=list(read(p)) if p.exists() else []
        rows=[r for r in rows if not str(r.get('msg','')).startswith('evaluator error:')];scored=len(rows)
        for k in ['score','precision_by_row','recall_by_row','f1_by_row','precision_by_item','recall_by_item','f1_by_item']:
            values=[r[k] for r in rows if isinstance(r.get(k),(int,float))];metrics[k]=sum(values)/len(values) if values else None
    elif name=='drbench':
        p=folder/'author/race/raw_results.jsonl';rows=list(read(p)) if p.exists() else []
        scores=[r['overall_score'] for r in rows if isinstance(r.get('overall_score'),(int,float))];metrics['RACE']=sum(scores)/len(scores) if scores else None
        fact=folder/'author/fact/fact_result.txt'
        if fact.exists():
            metrics['FACT']={k:float(v) for k,v in re.findall(r'^([a-z_]+):\s*([0-9.]+)',fact.read_text(),re.M)}
        status=load(folder/'author/fact/fact_result.status.json',{})
        if status:metrics['FACT_status']=status
        scored=len(scores) if isinstance(metrics.get('FACT',{}).get('valid_rate'),(int,float)) else 0
    elif name=='liveresearchbench':
        for p in folder.rglob('summary_*.json'):metrics.update(load(p,{}).get('overall_results',{}))
        scored=min((v.get('count',v.get('total',0)) for k,v in metrics.items() if k!='depth'),default=0)
    elif name=='mind2web2':
        value=load(folder/'author/glasseek/summary.json',{})
        import sys
        sys.path.insert(0,str(ROOT/'Evaluation'))
        from benchmark_scores import task_scores
        native=task_scores(folder,name,[{'id':key} for key in value.get('tasks',{})])
        valid=[r for r in native.values() if r['status']=='scored']
        values=[{m['key']:m['value'] for m in r['metrics']} for r in valid]
        scored=len(valid)
        metrics={'avg_score':sum(v['score'] for v in values)/len(values) if values else None,
                 'success_rate':sum(v['success'] for v in values)/len(values) if values else None,'num_tasks':scored}
    return metrics,scored


def write_report(args,states):
    root=args.output_root/args.run_id;cases=[];rows=[]
    for name,state in states.items():
        pred=args.output_root/'predictions'/(args.run_id+'_'+name)/('suite_'+name)
        group=[]
        for p in sorted(pred.glob('conv/*/*.json')):
            row=load(p,{})
            try:usage=case_usage(row)
            except (sqlite3.Error,OSError,ValueError,TypeError) as exc:usage={'usage_status':'error','usage_error':type(exc).__name__,'tool_calls':{}}
            elapsed=max(row.get('wall_seconds') or row.get('duration_seconds') or 0,sum(a.get('duration_seconds') or 0 for a in row.get('retry_attempts',[])))
            if row.get('workspace_dir'):elapsed+=(load(Path(row['workspace_dir'])/'continuation.json',{}) or {}).get('previous_wall_seconds',0)
            item={'benchmark':name,'question_id':row.get('question_id'),'status':row.get('status'),
                'duration_seconds':elapsed,'prediction_path':str(p),**usage}
            cases.append(item);group.append(item)
        score=Path(state['score_dir']) if state.get('score_dir') else None
        metrics,scored=native_metrics(score,name) if score else ({},0)
        if not score:
            task_states=[load(p,{}) for p in (root/'trajectory-scores'/name).glob('*/state.json')]
            latest={s['opaque_id']:s for s in task_states if s.get('opaque_id')}
            task_metrics={};scored=0
            for key,s in latest.items():
                if s.get('score_dir'):
                    value,count=native_metrics(s['score_dir'],name)
                    task_metrics[key]=value;scored+=count or 0
            if task_metrics:
                metrics={'per_task':task_metrics}
                if name in ('browsecomp','hle','gaia','browsecomp_plus'):
                    correct=sum(v.get('correct',0) for v in task_metrics.values())
                    metrics.update(correct=correct,accuracy_scored=correct/scored if scored else None)
        if 'correct' in metrics:
            metrics['accuracy_assigned']=metrics['correct']/state['assigned'] if scored==state['assigned'] else None
            metrics['verified_correct_over_assigned']=metrics['correct']/state['assigned']
        usages=list(read(score/'api_usage.jsonl')) if score and (score/'api_usage.jsonl').exists() else []
        judged_tokens={'judge_input_tokens':sum(r['usage']['prompt_tokens'] for r in usages if r.get('usage')) if usages else (0 if name=='gaia' and score else None),
            'judge_output_tokens':sum(r['usage']['completion_tokens'] for r in usages if r.get('usage')) if usages else (0 if name=='gaia' and score else None)}
        totals={k:sum(r.get(k) or 0 for r in group) if any(r.get(k) is not None for r in group) else None for k in ['input_tokens','output_tokens','cached_input_tokens','context_edits','context_edits_substantial']}
        tools=collections.Counter()
        for r in group:tools.update(r['tool_calls'])
        rows.append({'benchmark':name,'assigned':state['assigned'],'completed':len(group),'successful':sum(r['status']=='success' for r in group),
            'scored':scored,'state':state['state'],'reason':state.get('reason'),'native_metrics':metrics,
            'inference_wall_seconds':state.get('infer_seconds'),'judge_wall_seconds':state.get('score_seconds'),
            'case_wall_seconds_sum':sum(r.get('duration_seconds') or 0 for r in group),
            **totals,**judged_tokens,'usage_complete_cases':sum(r.get('usage_status')=='complete' for r in group),'tool_calls':dict(tools)})
    dump(root/'status.json',states);dump(root/'report.json',{'benchmarks':rows,'cases':cases,
        'notes':['native metrics retain original scales; no blended accuracy','input includes cache once; output includes reasoning and final','bash is physical calls; web_search/web_extract are semantic subcalls, do not add these to bash','missing measurements remain null; judge errors are unscored']})
    jsonl(root/'cases.jsonl',cases)
    with (root/'summary.csv').open('w') as f:
        fields=['benchmark','assigned','completed','successful','scored','state','reason','inference_wall_seconds','judge_wall_seconds','case_wall_seconds_sum','input_tokens','output_tokens','cached_input_tokens','context_edits','context_edits_substantial','judge_input_tokens','judge_output_tokens','usage_complete_cases','tool_calls','native_metrics']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else v for k,v in row.items()})


def publish_scores(args,name,score,env,opaque_ids=None):
    """Reuse viewer's native score sidecars; multidimensional scores remain multidimensional."""
    import subprocess
    pred=args.output_root/'predictions'/(args.run_id+'_'+name)/('suite_'+name)
    refs={r['id']:r for r in read(args.suite/'private/references.jsonl') if r['benchmark']==name}
    if opaque_ids is not None:refs={k:r for k,r in refs.items() if r['opaque_id'] in opaque_ids}
    paths=score/('author/scores.jsonl' if name=='gaia' else 'scores.jsonl')
    grades={r.get('id',r.get('task_id')):r for r in read(paths)} if paths.exists() else {}
    import sys
    sys.path.insert(0,str(ROOT/'Evaluation'))
    from benchmark_scores import task_scores
    native=task_scores(score,name,list(refs.values()))
    for r in refs.values():
        for p in pred.glob('conv/*/*.json'):
            row=load(p,{})
            if row.get('question_id')!=r['opaque_id']:continue
            grade=grades.get(r['id'],{});evaluation=native[str(r['id'])];correct=evaluation['correct']
            dump(pred/'judge_official'/r['opaque_id']/(row['run_id']+'.json'),{'question_id':r['opaque_id'],'run_id':row['run_id'],
                'status':'judged' if isinstance(correct,bool) else 'judge_failed' if evaluation['status']=='error' else 'native_metrics',
                'score':int(correct) if isinstance(correct,bool) else None,'evaluation':evaluation,
                'prediction':row.get('model_response'),'judge_response':grade or evaluation['details'],'protocol':name,'native_scores_path':str(score)})
    viewer_env={**env,'GLASSEEK_VIEWER_OUTPUT_DIR':str(pred),'GLASSEEK_VIEWER_DATA_PATH':str(args.suite/'inference'/(name+'.jsonl')),'GLASSEEK_VIEWER_LIMIT':str(len(refs))}
    with (pred/'viewer.log').open('a') as log:
        subprocess.run([env['EVAL_PYTHON'],str(ROOT/'Evaluation/services/viewer_publish.py'),'--run',args.run_id+'_'+name,
            '--dataset','suite_'+name,'--model',env.get('GLASSEEK_VIEWER_MODEL','glasseek-base-9b'),'--label',f'{name} · 原生评分','--once'],env=viewer_env,stdout=log,stderr=subprocess.STDOUT,check=False)
