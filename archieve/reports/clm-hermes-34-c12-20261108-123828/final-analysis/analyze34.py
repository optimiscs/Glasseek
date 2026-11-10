"""Audit all paired questions from complete native history, not visible tails."""
import bisect,collections,csv,datetime,hashlib,json,math,pathlib,re,sqlite3,statistics,sys

ROOT=pathlib.Path('/root/autodl-tmp/Glasseek')
OUT=pathlib.Path(__file__).resolve().parent
RUN=OUT.parent
OUT.mkdir(exist_ok=True)
sys.path[:0]=[str(ROOT/'scripts/trajectory_benchmark'),'/root/autodl-tmp/Quest/web/backend/src']
from compare_clm_to_hermes import recorded_tools
from quest_viewer.api.tool_operations import operation_details
from quest_viewer.api.tool_health import classify,payload,shell_result

def read(path,default=None):
    try:return json.loads(pathlib.Path(path).read_text())
    except (OSError,ValueError):return default

def evidence_text(value):
    if isinstance(value,dict):
        return '\n'.join(str(v) if k in ('title','snippet','content','text') and isinstance(v,str)
            else evidence_text(v) if isinstance(v,(dict,list)) and k not in ('searchParameters','relatedSearches','peopleAlsoAsk') else ''
            for k,v in value.items())
    if isinstance(value,list):return '\n'.join(evidence_text(v) for v in value)
    return ''

def database(path,aliases=()):
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        sessions=[dict(r) for r in db.execute('SELECT * FROM sessions')]
        messages=[dict(r) for r in db.execute('SELECT * FROM messages WHERE COALESCE(inherited_from_previous_session,0)=0 ORDER BY timestamp,id')]
        requests=[]
        for r in db.execute('SELECT id,recorded_at,context_json FROM request_contexts ORDER BY recorded_at,id'):
            meta=json.loads(r['context_json']);used=meta.get('usedTokens')
            if isinstance(used,(int,float)):requests.append(dict(id=r['id'],at=r['recorded_at'],tokens=used))
        model_times={r['message_id']:dict(r) for r in db.execute("SELECT * FROM message_timings WHERE phase='model'")}
        full_tools=recorded_tools(db)
        all_input=sum((s.get('input_tokens') or 0)+(s.get('cache_read_tokens') or 0)+(s.get('cache_write_tokens') or 0) for s in sessions)
        cached=sum(s.get('cache_read_tokens') or 0 for s in sessions)
        outputs=sum(s.get('output_tokens') or 0 for s in sessions)
        call_count=sum(s.get('api_call_count') or 0 for s in sessions)
        call_by_id={}
        for m in messages:
            if m['role']=='assistant':
                for call in json.loads(m.get('tool_calls') or '[]'):call_by_id[call.get('id')]=(m,call)
        evidence=[];edits=[];failures=[];operations=collections.Counter();queries=collections.Counter();seen=set();no_change=0;rejected=0
        for m in messages:
            if m['role']!='tool':continue
            cid=m.get('tool_call_id')
            if not cid or cid in seen or cid not in call_by_id:continue
            seen.add(cid);assistant,call=call_by_id[cid]
            fn=call['function'];args=fn.get('arguments',{})
            if isinstance(args,str):
                try:args=json.loads(args)
                except ValueError:args={}
            command=args.get('command','')
            supplied_queries=args.get('query',args.get('queries',args.get('q',[])))
            if isinstance(supplied_queries,str):supplied_queries=[supplied_queries]
            supplied_queries=[{'query':v if isinstance(v,str) else v.get('query','')} for v in supplied_queries] if isinstance(supplied_queries,list) else []
            step={'tool':{'name':fn['name'],'code':command,'queries':supplied_queries},'rawToolOutput':m.get('content') or ''}
            stdout,_=shell_result(step,step['rawToolOutput']);obj=payload(stdout)
            details=operation_details(step,obj);operation=details['tool'];operations[operation]+=1
            if operation=='web_search':
                queries.update(re.sub(r'\s+',' ',q).strip().casefold() for q in details['queries'])
            result=classify(step)
            if result['status']=='failed':failures.append(dict(message_id=m['id'],tool_call_id=cid,operation=operation,error=result['error'],command=command))
            text=evidence_text(obj)
            hits=[a for a in aliases if a and re.search(r'(?<!\w)'+re.escape(a)+r'(?!\w)',text,re.I)]
            if hits:evidence.append(dict(message_id=m['id'],at=m['timestamp'],aliases=hits,text=text[:16000]))
            content=m.get('content') or ''
            no_change+=bool(re.search(r'\[LIVE_CTX_MAIN\.txt: NO change',content))
            rejected+=bool(re.search(r'\[LIVE_CTX_MAIN\.txt:[^\n]*(?:reject|REJECT)',content))
            match=re.search(r'\[LIVE_CTX_MAIN\.txt: edit applied[^\n]*context ~(\d+)->(\d+)',content)
            if not match:continue
            old,new=map(int,match.groups());at=m['timestamp']
            previous=[r for r in requests if r['at']<=at]
            following=next((r for r in requests if r['at']>at),None)
            prior=previous[-1] if previous else None
            after_messages=json.loads(db.execute('SELECT messages_json FROM request_contexts WHERE id=?',(following['id'],)).fetchone()[0]) if following else []
            after_text=json.dumps(after_messages,ensure_ascii=False)
            edit=dict(message_id=m['id'],tool_call_id=cid,at=at,command=command,
                estimated_before=old,estimated_after=new,estimated_reduction=1-new/old if old else 0,
                nontrivial=abs(old-new)>=max(32,old*.01),
                actual_input_before=prior['tokens'] if prior else None,
                actual_input_after=following['tokens'] if following else None,
                completed_calls_before=len(previous),input_spent_before=sum(r['tokens'] for r in previous),
                input_fraction_spent_before=sum(r['tokens'] for r in previous)/all_input if all_input else None,
                reference_in_tools_before=any(e['at']<=at for e in evidence),
                reference_aliases_retained=[a for a in aliases if a and re.search(r'(?<!\w)'+re.escape(a)+r'(?!\w)',after_text,re.I)],
                next_context_excerpt=after_text[:18000])
            edits.append(edit)
        compression_starts=sum(s.get('end_reason')=='compression' for s in sessions)
    return dict(db_path=str(path),input_tokens=all_input,cached_tokens=cached,uncached_tokens=all_input-cached,
        output_tokens=outputs,model_calls=call_count,cache_hit_fraction=cached/all_input if all_input else None,
        native_session_transitions=compression_starts,operations=dict(operations),full_tools=full_tools,
        context_applied=len(edits),context_nontrivial=sum(e['nontrivial'] for e in edits),
        context_no_change=no_change,context_rejected=rejected,edits=edits,failures=failures,
        repeated_exact_searches=sum(v-1 for v in queries.values()),top_repeated_queries=queries.most_common(6),
        reference_evidence=evidence,recorded_request_count=len(requests),
        response_call_count=len(model_times))

def summarize(rows,kind):
    selected=[r[kind] for r in rows]
    result={k:sum(r[k] for r in selected) for k in ['input_tokens','cached_tokens','uncached_tokens','output_tokens','model_calls','context_applied','context_nontrivial','context_no_change','context_rejected']}
    result.update(n=len(rows),correct=sum(r[kind+'_score']==1 for r in rows),
        native_session_transitions=sum(r['native_session_transitions'] for r in selected),
        operations=dict(sum((collections.Counter(r['operations']) for r in selected),collections.Counter())),
        failed_tools=sum(len(r['failures']) for r in selected))
    result['accuracy']=result['correct']/len(rows) if rows else None
    result['cache_hit_fraction']=result['cached_tokens']/result['input_tokens'] if result['input_tokens'] else None
    return result

def paired(rows,raw_key='raw'):
    wins=[r['question_id'] for r in rows if r['clm_score']==1 and r[raw_key+'_score']!=1]
    losses=[r['question_id'] for r in rows if r['clm_score']!=1 and r[raw_key+'_score']==1]
    both=[r['question_id'] for r in rows if r['clm_score']==1 and r[raw_key+'_score']==1]
    neither=[r['question_id'] for r in rows if r['clm_score']!=1 and r[raw_key+'_score']!=1]
    n=len(wins)+len(losses);k=min(len(wins),len(losses))
    return dict(wins=wins,losses=losses,both_correct=both,both_incorrect=neither,
        mcnemar_exact_p=min(1,2*sum(math.comb(n,i) for i in range(k+1))/2**n) if n else 1)

source=read(ROOT/'reports/url-modes-34-20261107/live-v3-full-c8/final-analysis/report.json')
raw_by_id={r['question_id']:r for r in source['rows'] if r['mode']=='raw'}
folder=ROOT/'Evaluation/output/preds/clm_hermes_34_c12_20261108_123828/browsecomp_rl9b_non_timeout_34'
repeat_folder=ROOT/'Evaluation/output/preds/url_34_raw_20261107_221522/browsecomp_rl9b_non_timeout_34'
repeat_by_id={r['question_id']:r for p in (repeat_folder/'conv').glob('*/*.json') if (r:=read(p)) and r['status']=='success'}
rows=[]
alias_overrides={'row0031':['August 13','13 August','August 29','29 August'],
    'row0084':['Trefor Jenkins','Benjamin Pogrund'],'row0098':['Officina Profumo','Santa Maria Novella'],
    'row0114':['Singh Is Bliing','Singh is Bling'],'row0130':['5TB','5 TB'],
    'row0170':['30 July 1980','July 30, 1980','30/07/1980'],
    'row0171':['Robert Pfaff','Robert Henzi'],'row0175':['Spain'],
    'row0239':['Constanza Guzmán','Constanza Guzman'],'row0265':['C29','nonacosane'],
    'row0296':['Sunset in Biafra','Sunset at Biafra']}
for path in sorted((folder/'conv').glob('*/*.json')):
    case=read(path);q=case['question_id'];raw=raw_by_id[q]
    assert hashlib.sha256(case['question'].encode()).digest()==hashlib.sha256(raw['question'].encode()).digest()
    judge=read(folder/'judge_official'/q/(case['run_id']+'.json'))
    assert judge['status']=='judged'
    aliases=alias_overrides.get(q,[raw['reference_answer'].strip().rstrip('.')])
    work=pathlib.Path(case['workspace_dir']);cfg=read(work/'official-clm-request.json')
    clm_db=ROOT/'Evaluation/data/runtime/hermes/profiles'/cfg['session_id'].removeprefix('clm-')/'state.db'
    native_clm=database(clm_db,aliases);native_raw=database(pathlib.Path(raw['db_path']),aliases)
    row=dict(question_id=q,question=case['question'],reference_answer=raw['reference_answer'],
        clm_score=judge['score'],raw_score=raw['judge']['score'],clm_answer=case['model_response'],raw_answer=raw['final_answer'],
        judge_reason=judge.get('judge_reason'),raw_retrieval_error_count=raw['retrieval_error_count'],
        raw=native_raw,clm=native_clm,source_prediction=str(path),source_raw_prediction=raw['conversation_path'])
    rep=repeat_by_id.get(q)
    if rep:
        j=read(repeat_folder/'judge_official'/q/(rep['run_id']+'.json'))
        if j and j['status']=='judged':
            row['repeat_score']=j['score'];row['repeat']=database(ROOT/'Evaluation/data/runtime/hermes/profiles'/rep['profile']/'state.db',aliases)
    row['review']=read(OUT/'review_annotations.json',{}).get(q,{})
    rows.append(row)

assert len(rows)==34
edits=[dict(question_id=r['question_id'],score=r['clm_score'],**e) for r in rows for e in r['clm']['edits']]
substantial=[e for e in edits if e['nontrivial']]
first_edits=[next(e for e in r['clm']['edits'] if e['nontrivial']) for r in rows if r['clm']['context_nontrivial']]
edited=[r for r in rows if r['clm']['context_nontrivial']];not_edited=[r for r in rows if not r['clm']['context_nontrivial']]
repeat_rows=[r for r in rows if 'repeat' in r];clean=[r for r in rows if not r['raw_retrieval_error_count']]
compression=dict(applied=len(edits),nontrivial=len(substantial),edited_tasks=len(edited),
    edited_correct=sum(r['clm_score']==1 for r in edited),not_edited_tasks=len(not_edited),not_edited_correct=sum(r['clm_score']==1 for r in not_edited),
    median_local_estimated_reduction=statistics.median(e['estimated_reduction'] for e in substantial),
    median_actual_next_input_reduction=statistics.median(1-e['actual_input_after']/e['actual_input_before'] for e in substantial if e['actual_input_before'] and e['actual_input_after']),
    median_first_edit_call=statistics.median(e['completed_calls_before'] for e in first_edits),
    median_input_fraction_spent_before_first_edit=statistics.median(e['input_fraction_spent_before'] for e in first_edits),
    edits_with_reference_seen_but_not_retained=[{'q':e['question_id'],'message_id':e['message_id']} for e in substantial if e['reference_in_tools_before'] and not e['reference_aliases_retained']])
report=dict(at_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    full34=dict(raw=summarize(rows,'raw'),clm=summarize(rows,'clm'),paired=paired(rows)),
    raw_repeat_matched=dict(raw=summarize(repeat_rows,'repeat'),clm=summarize(repeat_rows,'clm'),paired=paired(repeat_rows,'repeat')),
    raw_first_no_retrieval_errors=dict(raw=summarize(clean,'raw'),clm=summarize(clean,'clm'),paired=paired(clean)),
    compression=compression,rows=rows,
    passed_clm=summarize([r for r in rows if r['clm_score']==1],'clm'),
    failed_clm=summarize([r for r in rows if r['clm_score']!=1],'clm'),
    manual_failure_reasons=dict(collections.Counter(r['review'].get('reason') for r in rows if r['clm_score']!=1)),
    token_scope=['All native sessions, compression parents, completed model responses and resumed segments are included exactly once.',
        'Input includes cached reads; uncached input is input minus cached reads. Output includes provider-reported reasoning.',
        'Raw auxiliary summary calls do not persist usage in the historical compressor/auxiliary client; their exact token usage cannot be reconstructed. Raw end-to-end totals are lower bounds.',
        'Provider work from aborted/in-flight calls without a recorded usage response is not fully attributable per task. Judges/retrievers are excluded.',
        'CLM editing uses its main model loop and is included; native session transitions also include checkpoint resume and are not CLM edit counts.'],
    comparison_limits=['Same 34 questions and model; raw first run had retrieval outages affecting 11 questions.',
        'Raw allowed parallel native web tools and traditional summarization; CLM uses one bash action per response.',
        'CLM includes original concurrency12/3600s work and concurrency5 continuation with 120 cumulative tool rounds.',
        'Edited tasks are selected by difficulty and context growth; edited versus unedited accuracy is not a causal compression experiment.'])
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
(OUT/'compression-events.json').write_text(json.dumps(edits,ensure_ascii=False,indent=2))
with (OUT/'paired34.csv').open('w',newline='',encoding='utf-8-sig') as f:
    fields=['question_id','raw_score','clm_score','repeat_score','reference_answer','raw_input','clm_input','raw_uncached','clm_uncached','raw_output','clm_output','raw_model_calls','clm_model_calls','raw_compressions','clm_context_edits','clm_tool_failures','raw_retrieval_errors','diagnosis','finding','raw_answer','clm_answer']
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for r in rows:
        writer.writerow(dict(question_id=r['question_id'],raw_score=r['raw_score'],clm_score=r['clm_score'],repeat_score=r.get('repeat_score'),reference_answer=r['reference_answer'],
            **{f'{kind}_{field}':r[kind][key] for kind in ['raw','clm'] for field,key in [('input','input_tokens'),('uncached','uncached_tokens'),('output','output_tokens'),('model_calls','model_calls')]},
            raw_compressions=r['raw']['native_session_transitions'],clm_context_edits=r['clm']['context_nontrivial'],clm_tool_failures=len(r['clm']['failures']),raw_retrieval_errors=r['raw_retrieval_error_count'],
            diagnosis=r['review'].get('reason'),finding=r['review'].get('finding'),raw_answer=r['raw_answer'],clm_answer=r['clm_answer']))
(OUT/'passed-trajectories.json').write_text(json.dumps([dict(question_id=r['question_id'],score=r['clm_score'],
    source_prediction=r['source_prediction'],db_path=r['clm']['db_path'],input_tokens=r['clm']['input_tokens'],
    output_tokens=r['clm']['output_tokens'],context_edits=r['clm']['context_nontrivial'],review=r['review'],
    viewer='http://localhost:60007/?dataset=clm_hermes_34_c12_20261108_123828#'+r['question_id'])
    for r in rows if r['clm_score']==1],ensure_ascii=False,indent=2))
(OUT/'failed-trajectories.json').write_text(json.dumps([dict(question_id=r['question_id'],source_prediction=r['source_prediction'],
    reference=r['reference_answer'],answer=r['clm_answer'],judge_reason=r['judge_reason'],review=r['review'],
    context_edits=r['clm']['context_nontrivial'],input_tokens=r['clm']['input_tokens'],output_tokens=r['clm']['output_tokens'])
    for r in rows if r['clm_score']!=1],ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('rows','token_scope','comparison_limits')},ensure_ascii=False,indent=2))
