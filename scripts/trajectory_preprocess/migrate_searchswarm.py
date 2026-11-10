"""Offline SearchSwarm migration. No model calls, dataset code execution or live agents.

Writes reversible review artifacts, not training-approved data. Original files remain
canonical archives. Missing execution metadata is omitted rather than invented.
"""
import argparse,ast,collections,copy,hashlib,json,re,sqlite3,time
from pathlib import Path
from typing import Optional
import pyarrow.parquet as pq
from client import digest
from ingest import normalize,harness_plan,semantic_hash
from glasseek_contract import build_contract,materialize,validate_args
from tool_xml import CALL,THINK,normalize_text,serialize
ROOT=Path(__file__).resolve().parents[2]
HEADER=re.compile(r'^A sub-agent dispatched for goal "(.*?)" returned the following report:\s*\n',re.M)


def response_parts(text):
    match=re.fullmatch(r'\s*<tool_response>(.*?)</tool_response>(.*)',text,re.S)
    if match:return match.group(1).strip(),match.group(2)
    raise ValueError('response_wrapper_not_supported')


def unwrap(text):
    return response_parts(text)[0]


def parse_reports(text,tasks):
    body=unwrap(text);matches=list(HEADER.finditer(body))
    labels=[t['goal'] for t in tasks]
    if len(set(labels))!=len(labels):raise ValueError('duplicate_goal_labels')
    if len(matches)!=len(tasks) or body[:matches[0].start()].strip():raise ValueError('report_boundary_mismatch')
    found={}
    for i,m in enumerate(matches):
        end=matches[i+1].start() if i+1<len(matches) else len(body)
        report=body[m.end():end].strip()
        if i+1<len(matches) and report.endswith('\n---'):report=report[:-4].rstrip()
        label=m.group(1)
        if label not in labels or label in found or not report:raise ValueError('report_label_or_content_mismatch')
        found[label]=report
    return [{'task_index':i,'summary':found[t['goal']]} for i,t in enumerate(tasks)]


def target_tasks(call):
    args=call['arguments']
    if set(args)!={'prompts'} or not isinstance(args['prompts'],list) or not args['prompts']:
        raise ValueError('unsupported_delegation_arguments')
    tasks=[]
    for task in args['prompts']:
        if not isinstance(task,dict) or set(task)!={'goal','prompt'} or not all(isinstance(task[k],str) and task[k].strip() for k in ('goal','prompt')):
            raise ValueError('invalid_delegated_task')
        tasks.append({'goal':task['goal'],'context':task['prompt']})
    return tasks


def child_prompt_builder():
    # Compile only this trusted, pure repository function; never dataset code.
    path=ROOT/'Evaluation_backend/tools/delegate_tool.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_build_child_system_prompt')
    module=ast.Module(body=[node],type_ignores=[]);space={'Optional':Optional}
    exec(compile(module,str(path),'exec'),space)
    return space['_build_child_system_prompt']


def convert_parent(row,row_index,source_path,contract):
    messages=copy.deepcopy(row['messages']);events=[];errors=[]
    bundle=digest(['SearchSwarm',str(source_path),row_index])[:24]
    for mi,m in enumerate(messages):
        if m['role']!='assistant':continue
        try:text,calls=normalize_text(m['content'])
        except (ValueError,TypeError) as e:errors.append('assistant_parse_error:'+type(e).__name__);continue
        positions=[p for p,c in enumerate(calls) if c['name']=='call_sub_agent']
        if not positions:continue
        # Mixed calls share an untyped response: require an explicit adapter.
        if len(calls)!=1:errors.append('mixed_delegation_calls');continue
        try:
            tasks=target_tasks(calls[0])
            if mi+1>=len(messages) or messages[mi+1]['role'] not in ('tool','user'):raise ValueError('missing_delegation_response')
            original_response=messages[mi+1]['content']
            reports=parse_reports(original_response,tasks)
            mapped={'name':'delegate_task','arguments':{'tasks':tasks}}
            protected=[(x.start(),x.end()) for x in THINK.finditer(text)]
            matches=[x for x in CALL.finditer(text) if not any(a<=x.start()<b for a,b in protected)]
            if len(matches)!=1:raise ValueError('call_count_mismatch')
            x=matches[0];m['content']=text[:x.start()]+serialize(mapped)+text[x.end():]
            assert normalize_text(m['content'],[mapped])[1]==[mapped]
            messages[mi+1]['content']='<tool_response>\n'+json.dumps({'results':reports},ensure_ascii=False)+'\n</tool_response>'+response_parts(original_response)[1]
            events.append({'call_id':f'{bundle}:call:{mi}','source_message_index':mi,'tasks':tasks,'reports':reports,
                           'original_response_sha256':digest(original_response),'original_response':original_response,
                           'original_call':calls[0],'mapped_call':mapped,'runtime_issues':validate_args('delegate_task',mapped['arguments'],contract)})
        except (ValueError,TypeError,KeyError) as e:errors.append('delegate_adapter:'+str(e))
    return bundle,messages,events,sorted(set(errors))


def final_report(child):
    msgs=child.get('messages') or []
    if not msgs or msgs[-1].get('role')!='assistant':return None
    try:text,calls=normalize_text(msgs[-1].get('content',''))
    except (ValueError,TypeError,KeyError):return None
    if calls:return None
    text=THINK.sub('',text).strip()
    report=re.fullmatch(r'<report>\s*(.*?)\s*</report>',text,re.S)
    return report.group(1).strip() if report else text


def associate(children,events,bundle):
    byprompt=collections.defaultdict(list)
    for ci,c in enumerate(children):byprompt[c.get('question')].append(ci)
    edges=[];child_edges=collections.defaultdict(list)
    for event in events:
        for task,report in zip(event['tasks'],event['reports']):
            candidates=byprompt[task['context']];method='exact_prompt'
            if len(candidates)>1:
                candidates=[ci for ci in candidates if final_report(children[ci])==report['summary'].strip()];method='exact_prompt_and_report'
            ci=candidates[0] if len(candidates)==1 else None
            edge={'call_id':event['call_id'],'task_index':report['task_index'],'goal':task['goal'],'context':task['context'],
                  'parent_id':bundle+':parent','child_id':f'{bundle}:child:{ci}' if ci is not None else None,
                  'child_index':ci,'match':method if ci is not None else 'missing_or_ambiguous',
                  'report_matches_child_final':bool(ci is not None and final_report(children[ci])==report['summary'].strip())}
            edges.append(edge)
            if ci is not None:child_edges[ci].append(edge)
    return edges,child_edges


def normalize_target(messages,question,source,row_index,source_path,ident,bundle,role,contract,extra_errors=None):
    item=normalize({'question':question,'messages':messages},source,str(source_path),row_index)
    item['id']=ident;item['bundle_id']=bundle;item['agent_role']=role
    item['split_group_id']=digest(re.sub(r'\s+',' ',question).strip().casefold())
    from migration_rules import memberships
    item['migration_rules']=memberships(item)
    item['harness']=harness_plan(item,{'search_name':'web_search','extract_name':'web_extract','extract_max_urls':5,'contract':contract})
    errors=set(item['program_errors']+item['harness']['errors']+(extra_errors or []))
    if role=='child':
        for c in item['harness']['calls']:
            if c['name'] in contract['delegation']['blocked_child_tools']:errors.add('child_tool_blocked:'+c['name'])
    # Do not partially materialize an incompatible episode.
    if not errors:
        item['source_content_hash']=item['content_hash'];materialize(item);item['content_hash']=semantic_hash(item)
    item['migration_errors']=sorted(errors);item['training_ready']=False;item['harness_replayed']=False
    item['review_flags']=sorted(set(item['harness']['flags']+['source_system_prompt_reconciliation_required','factual_review_pending']))
    return item


def review_messages(item):
    output=[]
    for step in item['steps']:
        if step['step_id']==1:
            for m in item.get('instruction_messages',[]):
                if m['before_step']==1:output.append({'role':m['role'],'content':m['content'], 'source_message_index':m.get('source_message_index')})
            output.append({'role':'user','content':item['question'], 'source_message_index':item.get('question_source_message_index', -1)})
        else:
            for m in item.get('instruction_messages',[]):
                if m['before_step']==step['step_id']:output.append({'role':m['role'],'content':m['content'], 'source_message_index':m.get('source_message_index')})
        for m in item.get('context_messages',[]):
            if m['before_step']==step['step_id']:output.append({'role':m['role'],'content':m['content'], 'source_message_index':m.get('source_message_index')})
        from call_protocol import render_step
        output.extend(render_step(step))
    if all(type(m.get('source_message_index')) is int for m in output):
        output.sort(key=lambda m:m['source_message_index'])
    return output


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--limit',type=int)
    ap.add_argument('--sample-manifest',type=Path)
    ap.add_argument('--compress-archive',action='store_true',help='Losslessly gzip parent/child archives and candidates')
    args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise SystemExit('Output already contains a migration; use a new directory')
    source=ROOT/'data/SearchSwarm-SFT/train.parquet';contract=build_contract();pf=pq.ParquetFile(source)
    selection=json.loads(args.sample_manifest.read_text()) if args.sample_manifest else None
    selected=set(selection['indices_by_path'][str(source)]) if selection else None
    builder=child_prompt_builder();started=time.monotonic();counts=collections.Counter();reasons=collections.defaultdict(collections.Counter)
    code_files=[Path(__file__),*[Path(__file__).with_name(n) for n in ('ingest.py','message_adapter.py','tool_xml.py','glasseek_contract.py','client.py','runtime_metadata.py','call_protocol.py','extract_batching.py','web_batching.py','tool_prompt.py','migration_rules.py')],ROOT/'Shared/glasseek_shared/extract_protocol.py',ROOT/'Shared/glasseek_shared/historical_tools.py',ROOT/'Shared/glasseek_shared/search_protocol.py',ROOT/'Shared/glasseek_shared/web_return_audit.py',ROOT/'Shared/glasseek_shared/hermes/qwen_xml_protocol.py']
    stat=source.stat();manifest={'version':'searchswarm-glasseek-migration-9-search-return-audit','extract_return_policy':'historical-web-call-batch-v3','source':str(source),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'selection':selection,
        'expected_bundles':pf.metadata.num_rows,'limit':args.limit,'contract':contract,'annotation_char_budget_applied':False,
        'notes':['Migration compatibility is separate from annotation budget and semantic quality.','Unknown historical status/api_calls/duration are omitted.','Message loss roles are specified; token-level masks must be verified with the final training template.'],
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');(out/'frozen_source').mkdir()
    for path in code_files:(out/'frozen_source'/path.name).write_bytes(path.read_bytes())
    import gzip
    files={n:(gzip.open(out/(n+'.jsonl.gz'),'wt',encoding='utf-8',compresslevel=1)
              if args.compress_archive and n in ('parents','children','parent_candidates','child_candidates') else (out/(n+'.jsonl')).open('w'))
           for n in ('parents','children','parent_candidates','child_candidates','held','links','delegation_events','duplicates')}
    db=sqlite3.connect(out/'exact_dedup.sqlite');db.execute('CREATE TABLE seen (hash TEXT PRIMARY KEY,id TEXT)')
    def emit(name,value):files[name].write(json.dumps(value,ensure_ascii=False)+'\n')
    def save(item):
        kind=item['agent_role'];counts[kind+'_normalized']+=1
        item['message_loss_roles']='assistant_only; system/user/tool context has zero loss'
        item['serialized_chars']=len(json.dumps(item,ensure_ascii=False))
        from call_protocol import model_visible_chars
        item['model_visible_chars']=model_visible_chars(item)
        item['over_legacy_250k_char_budget']=item['serialized_chars']>250000
        emit('parents' if kind=='parent' else 'children',item)
        if item['migration_errors']:
            counts[kind+'_held']+=1;reasons[kind].update(item['migration_errors']);emit('held',{'id':item['id'],'agent_role':kind,'reasons':item['migration_errors'],'bundle_id':item['bundle_id'],'migration_rules':item['migration_rules']});return
        hit=db.execute('SELECT id FROM seen WHERE hash=?',(item['content_hash'],)).fetchone()
        if hit:
            counts[kind+'_exact_duplicates']+=1;emit('duplicates',{'id':item['id'],'duplicate_of':hit[0],'agent_role':kind,'migration_rules':item['migration_rules']});return
        db.execute('INSERT INTO seen VALUES (?,?)',(item['content_hash'],item['id']))
        counts[kind+'_candidates']+=1
        item['messages']=review_messages(item);item['message_loss_mask']=[int(m['role']=='assistant') for m in item['messages']]
        assert len(item['messages'])==len(item['message_loss_mask'])
        emit(kind+'_candidates',item)
    try:
        for ri,batch in enumerate(pf.iter_batches(batch_size=1,use_threads=False)):
            if args.limit is not None and ri>=args.limit:break
            if selected is not None and ri not in selected:continue
            row=batch.to_pylist()[0];counts['bundles']+=1
            bundle,msgs,events,errors=convert_parent(row,ri,source,contract)
            children=row.get('subagents') or [];counts['children_read']+=len(children)
            edges,child_edges=associate(children,events,bundle)
            group=digest(re.sub(r'\s+',' ',row['question']).strip().casefold())
            for edge in edges:
                emit('links',{**edge,'split_group_id':group});counts['delegated_tasks']+=1
                counts['linked_tasks' if edge['child_id'] else 'unresolved_tasks']+=1
                if edge['child_id'] and not edge['report_matches_child_final']:counts['report_final_mismatch']+=1
            for event in events:emit('delegation_events',{'bundle_id':bundle,**event})
            counts['delegate_calls_converted']+=len(events)
            try:
                item=normalize_target(msgs,row['question'],'SearchSwarm',ri,source,bundle+':parent',bundle,'parent',contract,errors)
                item['reference_answer']=row.get('answer','');item['split_group_id']=group
                item['child_ids']=[f'{bundle}:child:{ci}' for ci in range(len(children))]
                item['delegation_links_complete']=all(e['child_id'] for e in edges)
                item['review_flags']+=['historical_execution_metadata_unavailable']
                if any(not e['child_id'] for e in edges):item['review_flags'].append('child_archive_incomplete_or_ambiguous')
                save(item)
            except (ValueError,TypeError,KeyError) as e:
                counts['parent_held']+=1;reasons['parent']['normalization:'+str(e)]+=1;emit('held',{'id':bundle+':parent','agent_role':'parent','reasons':['normalization:'+str(e)]})
            for ci,child in enumerate(children):
                ident=f'{bundle}:child:{ci}'
                try:
                    links=child_edges.get(ci,[]);briefings={(e['goal'],e['context']) for e in links}
                    cm=copy.deepcopy(child['messages']);source_instructions=[m for m in cm if m['role'] in ('system','developer')]
                    extra=[]
                    if len(briefings)==1:
                        goal,context=next(iter(briefings))
                        # Match the target child-visible placement: context in system, goal as user.
                        body=[m for m in cm if m['role'] not in ('system','developer')]
                        if not body or body[0]['role']!='user' or body[0]['content']!=context:raise ValueError('child_initial_prompt_mismatch')
                        body[0]={'role':'user','content':goal}
                        cm=[{'role':'system','content':builder(goal,context,max_spawn_depth=contract['delegation']['max_spawn_depth'],child_depth=1,timeout_seconds=900)},*body]
                        question=goal
                    else:question=child['question'];extra.append('child_briefing_missing_or_ambiguous')
                    item=normalize_target(cm,question,'SearchSwarm',ri,source,ident,bundle,'child',contract,extra)
                    item['split_group_id']=group;item['parent_id']=bundle+':parent';item['child_index']=ci
                    item['source_instruction_messages']=source_instructions;item['source_child_question']=child['question']
                    item['source_child_sha256']=digest(child);item['parent_call_refs']=[{'call_id':e['call_id'],'task_index':e['task_index']} for e in links]
                    save(item)
                except (ValueError,TypeError,KeyError) as e:
                    counts['child_held']+=1;reasons['child']['normalization:'+str(e)]+=1;emit('held',{'id':ident,'agent_role':'child','reasons':['normalization:'+str(e)]})
            if counts['bundles']%100==0:
                db.commit();print(json.dumps({'counts':counts,'elapsed_seconds':time.monotonic()-started}),flush=True)
        expected=len([i for i in selected if args.limit is None or i<args.limit]) if selected is not None else (min(args.limit,pf.metadata.num_rows) if args.limit is not None else pf.metadata.num_rows)
        assert counts['bundles']==expected
        assert counts['parent_candidates']+counts['parent_held']+counts['parent_exact_duplicates']==counts['bundles']
        assert counts['child_candidates']+counts['child_held']+counts['child_exact_duplicates']==counts['children_read']
        db.commit()
    finally:
        db.close()
        for f in files.values():f.close()
    summary={'status':'completed','counts':dict(counts),'held_reasons_overlapping':dict(reasons),'elapsed_seconds':time.monotonic()-started,
             'training_ready':False,'llm_calls':0,'runtime_replayed':False,'char_budget':'not applied in migration; over_legacy_250k_char_budget is a diagnostic only',
             'artifact_scope':'Compatibility candidates with role-level loss plan; system/evidence semantics, live replay, quality and final token masks still require review.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
