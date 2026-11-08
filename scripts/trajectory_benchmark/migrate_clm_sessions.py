"""One-time import of legacy official artifacts into native Hermes records.

Never imports a live attempt. Historical durations retain NULL wall-clock spans.
Original files are backed up; answer text/status and scoring identity stay intact.
"""
import argparse,json,sys,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'Evaluation_backend'),str(ROOT/'Evaluation')]
from agent.session_recorder import AgentSessionRecorder
from hermes_state import SessionDB

def rebind_judgment(path, old, new, backup):
    assert {k:v for k,v in old.items() if k!='history'}=={k:v for k,v in new.items() if k!='history'}
    digest=lambda value:hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    prior=path.parents[2]/'judge_official'/new['question_id']/(new['run_id']+'.json')
    if prior.exists():
        value=json.loads(prior.read_text())
        if value.get('pred_fingerprint')==digest(old):
            shutil.copy2(prior,backup/'judge.json')
            value['pred_fingerprint']=digest(new)
            temp=prior.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(prior)
    return {'old_fingerprint':digest(old),'new_fingerprint':digest(new),'scoring_inputs_unchanged':True}

def migrate(path):
    case=json.loads(path.read_text())
    if case.get('status') not in ('success','failed','policy_violation'):return {'path':str(path),'status':'not_terminal'}
    work=Path(case['workspace_dir']);art=work/'official-clm/agent'
    if not (art/'trajectory.json').exists():return {'path':str(path),'status':'no_artifact'}
    profile=ROOT/'Evaluation/data/runtime/hermes/profiles'/case['profile']
    sid=case.get('session_id') or 'clm-'+case['profile']
    probe=SessionDB(profile/'state.db');already=probe.get_session(sid);probe.close()
    if already:return {'path':str(path),'status':'already_native'}
    backup=work/'legacy-clm-backup';backup.mkdir(exist_ok=True)
    shutil.copy2(path,backup/'prediction.json')
    request=json.loads((work/'official-clm-request.json').read_text())
    snapshot=Path(request['snapshot'])
    if snapshot.exists():shutil.copy2(snapshot,backup/snapshot.name)
    frames=[]
    for p in sorted((art/'context_snapshots').glob('*.json')):
        try:frames.append(json.loads(p.read_text()).get('messages',[]))
        except ValueError:pass
    final=json.loads((art/'trajectory.json').read_text());frames.append(final)
    actions=[];results={};seen=set()
    for frame in frames:
        for m in frame:
            calls=m.get('tool_calls') or []
            if m.get('role')=='assistant' and calls:
                cid=calls[0].get('id')
                if cid and cid not in seen:actions.append(m);seen.add(cid)
            elif m.get('role')=='tool' and m.get('tool_call_id'):results[m['tool_call_id']]=m
    timings=json.loads((art/'timing.json').read_text()) if (art/'timing.json').exists() else []
    measured=[t for t in timings if 'cmd' in t and 'llm_s' in t and 'bash_s' in t]
    def command(m):
        a=m['tool_calls'][0]['function'].get('arguments') or '{}'
        return (json.loads(a) if isinstance(a,str) else a).get('command','')
    # Match complete ordered sequences, or uniquely occurring command prefixes.
    ordered=len(actions)==len(measured) and all(command(m)[:200]==t['cmd'] for m,t in zip(actions,measured))
    recorder=AgentSessionRecorder(profile,snapshot,sid,case['model'],case.get('model_base_url',''),str(work))
    recorder.before_query(final[:2],record_request=False);durations=0
    for m in actions:
        mid=recorder.append(m)
        matches=[t for t in measured if t['cmd']==command(m)[:200]]
        timing=measured[actions.index(m)] if ordered else matches[0] if len(matches)==1 and sum(command(a)[:200]==command(m)[:200] for a in actions)==1 else None
        if timing:recorder.db.record_message_timing(mid,'model',timing['llm_s'],provenance='legacy_duration');durations+=1
        call=m['tool_calls'][0];cid=call['id']
        if cid in results:
            tool=dict(results[cid],tool_name=call['function']['name']);tid=recorder.append(tool)
            if timing:recorder.db.record_message_timing(tid,'tool',timing['bash_s'],provenance='legacy_duration')
    usage=json.loads((art/'usage.json').read_text()) if (art/'usage.json').exists() else {}
    cached=max(usage.get('cached_tokens',0),usage.get('cache_read_input_tokens',0));write=usage.get('cache_creation_input_tokens',0)
    recorder.db.update_token_counts(sid,input_tokens=max(0,usage.get('prompt_tokens',0)-cached-write),output_tokens=usage.get('completion_tokens',0),cache_read_tokens=cached,cache_write_tokens=write,api_call_count=usage.get('n_lm_calls',0),absolute=True)
    start=request.get('started_at');oldsidecar=work/'official-clm-result.json';old=json.loads(oldsidecar.read_text()) if oldsidecar.exists() else {}
    end=old.get('session',{}).get('session_end')
    if start is not None:
        recorder.db._execute_write(lambda db:db.execute('UPDATE sessions SET started_at=?, ended_at=?,end_reason=? WHERE id=?',(start,end,'legacy_import',sid)))
    from glasseek_shared.context.official import activate
    activate()
    from clm_harness.utils.tool_schemas import BASH_TOOL
    for request_path in sorted((art/'context_snapshots').glob('*.json'),reverse=True):
        request_data=json.loads(request_path.read_text())
        if request_data.get('kind')=='agent':
            recorder.db.record_request_context(sid,request_data['messages'],[BASH_TOOL],capacity_tokens=request.get('context_length'),provenance='saved_request')
            break
    session=recorder.publish();recorder.close()
    if end is not None:
        from datetime import datetime
        session['last_updated']=datetime.fromtimestamp(end).isoformat()
        temp=snapshot.with_suffix('.tmp');temp.write_text(json.dumps(session));temp.replace(snapshot)
    old_case=json.loads(json.dumps(case))
    case['history']=[session];case['session_id']=sid
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(case,ensure_ascii=False,indent=2));temp.replace(path)
    association=rebind_judgment(path,old_case,case,backup)
    return {**association,'path':str(path),'status':'migrated','actions':len(actions),'duration_pairs':durations,'absolute_timestamps':'unavailable','backup':str(backup)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--dataset',required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    rows=[migrate(path) for path in (ROOT/'Evaluation/output/preds'/a.run/a.dataset/'conv').rglob('*.json')]
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(rows,indent=2));print(json.dumps({s:sum(x['status']==s for x in rows) for s in {x['status'] for x in rows}}))
if __name__=='__main__':main()
