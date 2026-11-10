"""Read-only phase accounting against the archived pre-resume native databases."""
import collections,datetime,json,pathlib,sqlite3,sys,time,urllib.request

ROOT=pathlib.Path('/root/autodl-tmp/Glasseek')
OUT=pathlib.Path(__file__).resolve().parent
AUDIT=pathlib.Path((OUT/'resume-control-dir.txt').read_text())
sys.path.insert(0,'/root/autodl-tmp/Quest/web/backend/src')
from quest_viewer.api.tool_health import classify,payload,shell_result
from quest_viewer.api.tool_operations import operation_details

def read(path,default=None):
    try:return json.loads(pathlib.Path(path).read_text())
    except (OSError,ValueError):return default

def native(path):
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
        db.execute('BEGIN')
        values=db.execute('SELECT SUM(input_tokens),SUM(output_tokens),SUM(cache_read_tokens),'
                          'SUM(cache_write_tokens),SUM(api_call_count) FROM sessions').fetchone()
        uncached,output,cached,writes,calls=[v or 0 for v in values]
        messages=db.execute('SELECT id,role,tool_calls,tool_call_id,content FROM messages '
                           'WHERE COALESCE(inherited_from_previous_session,0)=0 ORDER BY id').fetchall()
        timings=dict(db.execute('SELECT phase,SUM(duration_seconds) FROM message_timings GROUP BY phase'))
    results={cid:content for _,role,_,cid,content in messages if role=='tool'}
    seen=set();operations=collections.Counter();outcomes=collections.Counter();edits=0
    for _,role,serialized,_,_ in messages:
        if role!='assistant':continue
        for call in json.loads(serialized or '[]'):
            cid=call.get('id')
            if cid in seen:continue
            seen.add(cid)
            fn=call['function'];args=fn.get('arguments',{})
            if isinstance(args,str):
                try:args=json.loads(args)
                except ValueError:args={}
            step={'tool':{'name':fn['name'],'code':args.get('command','')},'rawToolOutput':results.get(cid,'')}
            stdout,_=shell_result(step,step['rawToolOutput'])
            operations[operation_details(step,payload(stdout))['tool']]+=1
            outcomes[classify(step)['status']]+=1
            if 'LIVE_CTX_MAIN.txt: edit applied' in step['rawToolOutput']:edits+=1
    return dict(input_tokens=uncached+cached+writes,uncached_tokens=uncached,output_tokens=output,
        cached_tokens=cached,model_calls=calls,operations=dict(operations),tool_outcomes=dict(outcomes),
        model_seconds=timings.get('model',0),tool_seconds=timings.get('tool',0),observed_applied_edits=edits)

def aggregate(rows):
    keys=['input_tokens','uncached_tokens','output_tokens','cached_tokens','model_calls',
          'model_seconds','tool_seconds','observed_applied_edits']
    result={k:sum(r[k] for r in rows) for k in keys}
    for key in ['operations','tool_outcomes']:
        counter=collections.Counter()
        for row in rows:counter.update(row[key])
        result[key]=dict(counter)
    result['cache_hit_fraction']=result['cached_tokens']/result['input_tokens'] if result['input_tokens'] else None
    result['mean_input_per_response']=result['input_tokens']/result['model_calls'] if result['model_calls'] else None
    return result

def difference(after,before):
    result={k:after[k]-before[k] for k in ['input_tokens','uncached_tokens','output_tokens','cached_tokens',
          'model_calls','model_seconds','tool_seconds','observed_applied_edits']}
    for key in ['operations','tool_outcomes']:
        result[key]={k:after[key].get(k,0)-before[key].get(k,0) for k in after[key].keys()|before[key].keys()}
    return result

def interval(rows,start,end):
    selected=[r for r in rows if start<=r['timestamp']<=end and 'prefix_cache_queries_total' in r]
    first,last=selected[0],selected[-1];seconds=last['timestamp']-first['timestamp']
    keys=['prompt_tokens_total','generation_tokens_total','prefix_cache_queries_total',
          'prefix_cache_hits_total','num_preemptions_total']
    delta={k:last[k]-first[k] for k in keys}
    return dict(start=datetime.datetime.fromtimestamp(first['timestamp'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        end=datetime.datetime.fromtimestamp(last['timestamp'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        seconds=seconds,samples=len(selected),input_tokens_per_second=delta['prompt_tokens_total']/seconds,
        output_tokens_per_second=delta['generation_tokens_total']/seconds,
        cache_hit_fraction=delta['prefix_cache_hits_total']/delta['prefix_cache_queries_total'] if delta['prefix_cache_queries_total'] else None,
        avg_kv_usage=sum(r['kv_cache_usage_perc'] for r in selected)/len(selected),
        peak_kv_usage=max(r['kv_cache_usage_perc'] for r in selected),
        avg_running=sum(r['num_requests_running'] for r in selected)/len(selected),
        avg_waiting=sum(r['num_requests_waiting'] for r in selected)/len(selected),
        preemptions=delta['num_preemptions_total'],counter_delta=delta)

state=read(OUT/'status.json');run=state['runs']['clm']['run'];folder=ROOT/'Evaluation/output/preds'/run/state['dataset']
plan=read(AUDIT/'resume-plan.json');selected={r['question_id']:r for r in plan['resuming']}
inventory=read(AUDIT/'inventory.json');before=[];after=[];resumed=[];rows=[]
for old in inventory:
    q=old['q'];work=pathlib.Path(old['workspace']);cfg=read(work/'official-clm-request.json')
    db=ROOT/'Evaluation/data/runtime/hermes/profiles'/cfg['session_id'].removeprefix('clm-')/'state.db'
    current=native(db)
    prior=native(pathlib.Path(selected[q]['archive'])/'state.db') if q in selected else current
    before.append(prior);after.append(current)
    if q in selected:resumed.append(difference(current,prior))
    result=read(ROOT/old['path'],{})
    judge=read(folder/'judge_official'/q/(q+'__r1.json'),{})
    sidecar=read(work/'official-clm-result.json',{})
    web=read(work/'official-clm/workspace/.clm/web-calls.json',{})
    previous_web=read(pathlib.Path(selected[q]['archive'])/'workspace/official-clm/workspace/.clm/web-calls.json',{}) if q in selected else web
    budget=read(work/'official-clm/agent/tool-round-budget.json',{})
    row=dict(question_id=q,resumed=q in selected,status=result.get('status','running'),
        score=judge.get('score') if judge.get('status')=='judged' else None,judge_status=judge.get('status'),
        stop_reason=sidecar.get('stop_reason'),tool_rounds_used=budget.get('used'),
        tool_rounds_remaining=max(0,120-budget['used']) if 'used' in budget else None,
        cumulative=current,c5_increment=difference(current,prior) if q in selected else None,
        web_calls=web,c5_web_increment={k:web.get(k,0)-previous_web.get(k,0) for k in web.keys()|previous_web.keys()})
    rows.append(row)
metric_rows=[json.loads(l) for l in (OUT/'vllm_metrics.jsonl').read_text().splitlines() if l.strip()]
now=time.time()
raw=urllib.request.urlopen('http://127.0.0.1:19109/metrics',timeout=10).read().decode()
import re
live={'timestamp':now}
for line in raw.splitlines():
    m=re.match(r'vllm:(\w+)(?:\{[^}]*\})?\s+(\S+)',line)
    if m and m[1] in metric_rows[0]:live[m[1]]=live.get(m[1],0)+float(m[2])
metric_rows.append(live)
paused=datetime.datetime.strptime(AUDIT.name.removeprefix('resume-c5-'),'%Y%m%d-%H%M%S').replace(tzinfo=datetime.timezone.utc).timestamp()
c5=state['resumed_at'];start=state['started_at']
windows={'c12_full':interval(metric_rows,start,paused),'c12_last10min':interval(metric_rows,paused-600,paused),
         'c5_full':interval(metric_rows,c5,now),'c5_last10min':interval(metric_rows,now-600,now)}
scored=[r for r in rows if r['score'] is not None];new_scored=[r for r in scored if r['resumed']]
result=dict(at_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    progress=dict(total=34,submitted=sum(r['status']=='success' for r in rows),scored=len(scored),
        correct=sum(r['score']==1 for r in scored),resumed_total=16,resumed_scored=len(new_scored),
        resumed_correct=sum(r['score']==1 for r in new_scored),
        remaining=[{k:r[k] for k in ('question_id','status','tool_rounds_used','tool_rounds_remaining')} for r in rows if r['status']!='success']),
    c12_native=aggregate(before),c5_native_increment=aggregate(resumed),cumulative_native=aggregate(after),
    c5_web_increment={k:sum(r['c5_web_increment'].get(k,0) for r in rows) for k in ('web_search','web_extract')},
    engine_windows=windows,rows=rows,
    caveats=['Engine input throughput includes cache hits; not pure uncached prefill throughput.',
        'Native token totals cover completed API responses; engine counters also include in-flight work.',
        'c5 resumed harder unfinished tasks, added a 120-round cumulative tool budget and restarted the server; this is an observational comparison.',
        'Engine-wide window rates include model/tool idle periods, and are not single-request decode speeds.',
        'Bash operation counts are one per original call, including composite commands; bridge web calls are counted separately.'])
destination=OUT/'progress-c5-latest.json';destination.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('rows','caveats')},ensure_ascii=False,indent=2))
