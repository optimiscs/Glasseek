"""Four real paired examples, with no invented Glasseek child transcripts."""
import argparse,ast,collections,copy,html,json,sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'trajectory_preprocess'))
from migrate_searchswarm import review_messages,unwrap,response_parts
from glasseek_contract import validate_args
from tool_xml import normalize_text,serialize
from tool_xml import THINK
from migration_render import target_messages
from runtime_metadata import clean_response, VERSION as METADATA_POLICY
ROOT=Path(__file__).resolve().parents[2]

def rows(path):
    for line in path.open():yield json.loads(line)

def parse_result(observation):
    body=unwrap(observation)
    # The original Glasseek corpus sometimes stores escaped envelope newlines.
    while body.startswith('\\n'):body=body[2:].lstrip()
    while body.endswith('\\n'):body=body[:-2].rstrip()
    return json.loads(body)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--migration',type=Path,required=True);args=ap.parse_args()
    out=args.migration/'comparison';out.mkdir(exist_ok=True)
    contract=json.loads((args.migration/'manifest.json').read_text())['contract']
    tree=ast.parse((ROOT/'Evaluation_backend/tools/delegate_tool.py').read_text())
    strip_keys=ast.literal_eval(next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_MAIN_AGENT_STRIP_KEYS' for t in n.targets)))
    child_cache={}
    for child in rows(args.migration/'child_candidates.jsonl'):
        child_cache[child['id']]=child
        if len(child_cache)>=1000:break
    edges=collections.defaultdict(list)
    for e in rows(args.migration/'links.jsonl'):
        if e['child_id'] in child_cache and e['report_matches_child_final']:edges[e['parent_id']].append(e)
    swarm=[]
    for item in rows(args.migration/'parent_candidates.jsonl'):
        if not item['delegation_links_complete'] or not edges[item['id']]:continue
        edge=edges[item['id']][0];mi=int(edge['call_id'].rsplit(':',1)[1])
        step=next(s for s in item['steps'] if s['source_message_index']==mi)
        call=next(c for c in step['calls'] if c['name']=='delegate_task')
        swarm.append((item,step,call,edge,child_cache[edge['child_id']]))
        if len(swarm)==4:break
    assert len(swarm)==4
    chosen_calls={entry[3]['call_id'] for entry in swarm};batch_edges=collections.defaultdict(list)
    for e in rows(args.migration/'links.jsonl'):
        if e['call_id'] in chosen_calls:batch_edges[e['call_id']].append(e)
    needed={e['child_id'] for group in batch_edges.values() for e in group if e['child_id']};batch_children={}
    for child in rows(args.migration/'children.jsonl'):
        if child['id'] in needed:
            child['messages']=review_messages(child);batch_children[child['id']]=child
        if len(batch_children)==len(needed):break
    assert set(batch_children)==needed
    glasseek=[]
    for line in (ROOT/'data/processed/trajectory-glasseek-v5/clean.jsonl').open():
        if '"source": "Glasseek"' not in line[:300]:continue
        item=json.loads(line)
        for step in item['steps']:
            if len(step['calls'])!=1 or step['calls'][0]['name']!='delegate_task':continue
            try:result=parse_result(step['observation'])
            except (ValueError,TypeError):continue
            if not result.get('results'):continue
            glasseek.append((item,step,step['calls'][0]));break
    # Prefer the same supported batch mode/cardinality, without altering a source call.
    pool=glasseek;glasseek=[]
    for _,_,scall,_,_ in swarm:
        n=len(scall['arguments']['tasks'])
        choices=[g for g in pool if len(g[2]['arguments'].get('tasks') or [])==n]
        picked=choices[0] if choices else pool[0]
        glasseek.append(picked);pool.remove(picked)
    assert len(glasseek)==4
    pairs=[];probe=[];htmlparts=['<!doctype html><meta charset="utf-8"><title>Glasseek / SearchSwarm 迁移结构对比</title>',
        '<style>body{font:16px/1.6 sans-serif;max-width:1300px;margin:32px auto;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f7f9;padding:16px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;text-align:left}details{margin:12px 0}</style>',
        '<h1>四组真实样例的结构对比</h1><p>均为离线兼容候选，尚未通过重放或质量终审。SearchSwarm 未知运行元数据没有伪造；Glasseek SFT 原文件未提供独立子代理对话。</p>']
    for index,((ss,sstep,scall,edge,child),(gs,gstep,gcall)) in enumerate(zip(swarm,glasseek),1):
        source_gr=parse_result(gstep['observation']);gs=copy.deepcopy(gs)
        for step in gs['steps']:
            if len(step['calls'])==1 and step['calls'][0]['name']=='delegate_task':
                result=parse_result(step['observation'])
                for entry in result['results']:
                    for key in strip_keys:entry.pop(key,None)
                step['observation']='<tool_response>\n'+json.dumps(result,ensure_ascii=False)+'\n</tool_response>'+response_parts(step['observation'])[1]
                step['observation'],_=clean_response(step['observation'])
        gstep=next(s for s in gs['steps'] if s['step_id']==gstep['step_id'])
        sr=parse_result(sstep['observation']);gr=parse_result(gstep['observation'])
        for call in (scall,gcall):
            assert not validate_args('delegate_task',call['arguments'],contract)
            assert normalize_text(serialize(call),[call])[1]==[call]
        for result in (sr,gr):
            assert all(isinstance(r.get('task_index'),int) and isinstance(r.get('summary'),str) for r in result['results'])
        sdata={'id':ss['id'],'row_index':ss['provenance']['row_index'],'question':ss['question'],'call':scall,'call_xml':serialize(scall),'response':sr,
               'parent_messages':ss['messages'],'child_link':edge,'child_messages':target_messages(child)[0],
               'child_final_transforms':target_messages(child)[1],
               'child_id':child['id'],'parent_review_flags':ss['review_flags']}
        sdata['children']=[{'task_index':e['task_index'],'id':e['child_id'],'migration_errors':batch_children[e['child_id']]['migration_errors'],
                           'messages':target_messages(batch_children[e['child_id']])[0]} for e in batch_edges[edge['call_id']]]
        assert THINK.sub('',sdata['child_messages'][-1]['content']).strip()==sr['results'][edge['task_index']]['summary'].strip()
        gdata={'id':gs['id'],'row_index':gs['provenance']['row_index'],'question':gs['question'],'call':gcall,'call_xml':serialize(gcall),'response':gr,
               'parent_messages':review_messages(gs),'source_response':source_gr,'current_runtime_projection_removed_keys':list(strip_keys),
               'training_metadata_policy':METADATA_POLICY,
               'child_messages':None,'child_trace_availability':'not included in Glasseek SFT source file'}
        pair={'pair':index,'Glasseek':gdata,'SearchSwarm':sdata,'checks':{
            'delegate_schema_valid_both':True,'xml_roundtrip_both':True,'return_results_task_index_summary_both':True,
            'loss_policy':'assistant-only for each independent agent; reports are context with zero loss in parent',
            'same_response_metadata':set(gr)==set(sr) and all(set(x)==set(y) for x,y in zip(gr['results'],sr['results'])) and len(gr['results'])==len(sr['results']),
            'source_system_prompts_identical':False,'searchswarm_child_report_matches_parent':edge['report_matches_child_final'],
            'full_structures_identical':False,'training_ready':False}}
        (out/f'pair_{index}.json').write_text(json.dumps(pair,ensure_ascii=False,indent=2)+'\n');pairs.append(pair)
        htmlparts.append(f'<h2>第 {index} 组</h2><table><tr><th>Glasseek 原生父轨迹</th><th>SearchSwarm 迁移父轨迹</th></tr><tr><td>row {gdata["row_index"]}: {html.escape(gs["question"])}</td><td>row {sdata["row_index"]}: {html.escape(ss["question"])}</td></tr></table>')
        for name,data in [('Glasseek',gdata),('SearchSwarm',sdata)]:
            for label,value in [('调用 XML',data['call_xml']),('返回结果',json.dumps(data['response'],ensure_ascii=False,indent=2)),('父轨迹（完整）',json.dumps(data['parent_messages'],ensure_ascii=False,indent=2)),('独立子轨迹',json.dumps(data['child_messages'],ensure_ascii=False,indent=2))]:
                htmlparts.append('<details><summary>'+name+' '+label+'</summary><pre>'+html.escape(value)+'</pre></details>')
            if 'source_response' in data:
                htmlparts.append('<details><summary>Glasseek 原始返回（含旧诊断字段）</summary><pre>'+html.escape(json.dumps(data['source_response'],ensure_ascii=False,indent=2))+'</pre></details>')
            if 'children' in data:
                htmlparts.append('<details><summary>SearchSwarm 本次调用的全部子轨迹（含暂缓原因）</summary><pre>'+html.escape(json.dumps(data['children'],ensure_ascii=False,indent=2))+'</pre></details>')
        examples=[('Glasseek',gs,gdata['parent_messages'],'parent'),('SearchSwarm',ss,ss['messages'],'parent')]
        examples += [('SearchSwarm',batch_children[c['id']],c['messages'],'child') for c in sdata['children'] if not c['migration_errors']]
        for name,item,msgs,role in examples:
            allowed=contract['schemas'] if role=='parent' else {k:v for k,v in contract['schemas'].items() if k not in contract['delegation']['blocked_child_tools']}
            probe.append({'id':item['id'],'pair':index,'source':name,'agent_role':role,'messages':json.dumps(msgs,ensure_ascii=False),
                          'tools':json.dumps([{'type':'function','function':v} for v in allowed.values()],ensure_ascii=False),'enable_thinking':True,'training_ready':False})
    pq.write_table(pa.Table.from_pylist(probe),out/'mask_probe.parquet')
    (out/'comparison.html').write_text('\n'.join(htmlparts))
    compact=[{'pair':p['pair'],'glasseek_row':p['Glasseek']['row_index'],'searchswarm_row':p['SearchSwarm']['row_index'],
              'glasseek_question':p['Glasseek']['question'],'searchswarm_question':p['SearchSwarm']['question'],
              'glasseek_tasks':len(p['Glasseek']['call']['arguments'].get('tasks') or [p['Glasseek']['call']['arguments']]),
              'searchswarm_tasks':len(p['SearchSwarm']['call']['arguments']['tasks']),'checks':p['checks']} for p in pairs]
    for row,pair in zip(compact,pairs):
        row['searchswarm_child_candidates']=sum(not c['migration_errors'] for c in pair['SearchSwarm']['children'])
        row['searchswarm_child_holds']=[{'task_index':c['task_index'],'errors':c['migration_errors']} for c in pair['SearchSwarm']['children'] if c['migration_errors']]
    (out/'summary.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2)+'\n');print(json.dumps(compact,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
