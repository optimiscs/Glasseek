"""Replay exactly the viewer's 34 historical tool-call sets, without live tools."""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'Shared'))
from glasseek_shared.hermes.url_registry import URLRegistry,rewrite_assistant,annotate_schemas,MODES,VERSION


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    baseline=out/'baseline-traces.json';viewer=json.loads(baseline.read_text());archive=ROOT/'Evaluation/data/runtime/archive/checkpoint_reeval/remote_snapshot/sessions'
    from tokenizers import Tokenizer
    tokenizer_path=ROOT.parent/'models/Qwen3.5-9B/tokenizer.json';tok=Tokenizer.from_file(str(tokenizer_path))
    def nt(text):return len(tok.encode(text or '',add_special_tokens=False).ids)
    report={'protocol':VERSION,'kind':'historical deterministic replay; CPU timing is not model inference latency','cases':len(viewer['cases']),
            'viewer_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),'tokenizer_sha256':hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            'token_metric':'Unique original assistant content/reasoning and tool observations; tool arguments canonically JSON serialized in every mode. Excludes repeated prompt history and schema framing. Live runs measure actual billed/model tokens.',
            'modes':{},'by_case':[]};totals={mode:collections.Counter() for mode in MODES}
    raw_identity=True
    audit=(out/'call_equivalence.jsonl').open('w')
    for case in viewer['cases']:
        wanted={s['toolCallId'] for s in case['steps'] if s.get('toolCallId')}
        files=list(archive.glob('glasseek_base9b*_'+case['id']+'__r1*.sqlite'));assert len(files)==1
        db=sqlite3.connect('file:'+str(files[0].resolve())+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
        sessions={r['id']:dict(r) for r in db.execute('select id,source,parent_session_id,started_at from sessions')}
        rows=[dict(r) for r in db.execute('select * from messages order by timestamp,id')];db.close()
        original_digest=hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        for mode in MODES:
            stats=collections.Counter();registries={};seen_tools=set();seen_actions=set();checked_ids=set();timeline=[];before=time.perf_counter();transform_seconds=0
            for r in rows:
                sid=r['session_id']
                if sid not in registries:
                    parent=sessions[sid]['parent_session_id']
                    state=registries[parent].snapshot() if parent in registries and sessions[sid]['source']=='cli' else None
                    registries[sid]=URLRegistry(mode,state)
                reg=registries[sid];role=r['role'];start=time.perf_counter()
                if role=='assistant':
                    calls=json.loads(r['tool_calls'] or '[]');ids=tuple(c.get('id') for c in calls)
                    if calls and not any(i in wanted for i in ids):continue
                    if calls and ids in seen_actions:continue
                    if r.get('inherited_from_previous_session'):continue
                    seen_actions.add(ids) if calls else None
                    reg.begin_turn()
                    content=(r.get('reasoning_content') or r.get('reasoning') or '')+'\n'+(r.get('content') or '')
                    converted=rewrite_assistant(content,reg)
                    argument_text=[]
                    for c in calls:
                        if c.get('id') not in wanted or c.get('id') in checked_ids:continue
                        checked_ids.add(c['id'])
                        fn=c.get('function',c);a=fn.get('arguments') or {};a=json.loads(a) if isinstance(a,str) else a
                        encoded=reg.encode_args(fn['name'],a)
                        decoded=reg.resolve_args(fn['name'],encoded)
                        expected=reg.resolve_args(fn['name'],a)
                        same=decoded==expected
                        if not same:raise AssertionError((case['id'],mode,c['id']))
                        stats['calls_checked']+=1;stats['calls_target_equivalent']+=same
                        argument_text.append(json.dumps(encoded,ensure_ascii=False,separators=(',',':')))
                        audit.write(json.dumps({'case':case['id'],'mode':mode,'call_id':c['id'],'name':fn['name'],'equivalent':same,
                                                'known_ref_count_before_action':len(reg.visible)},ensure_ascii=False)+'\n')
                    transform_seconds+=time.perf_counter()-start
                    stats['assistant_text_tokens']+=nt(converted);stats['tool_argument_tokens']+=sum(nt(x) for x in argument_text)
                    if mode=='raw':raw_identity &= converted==content
                    timeline.append({'role':role,'content':converted,'arguments':argument_text})
                elif role=='tool' and r['tool_call_id'] in wanted and r['tool_call_id'] not in seen_tools:
                    seen_tools.add(r['tool_call_id']);name=r.get('tool_name') or ''
                    converted=reg.observe(name,r.get('content') or '')
                    transform_seconds+=time.perf_counter()-start
                    stats['tool_output_tokens']+=nt(converted);stats['observations']+=1
                    if mode=='raw':raw_identity &= converted==(r.get('content') or '')
                    timeline.append({'role':role,'name':name,'call_id':r['tool_call_id'],'content':converted})
            missing=wanted-seen_tools
            assert all(not s.get('rawToolOutput') for s in case['steps'] if s.get('toolCallId') in missing),(case['id'],missing)
            assert checked_ids==wanted,(case['id'],'missing actions',wanted-checked_ids)
            stats['pending_observations']=len(missing)
            stats['transform_seconds']=transform_seconds;stats['tokenize_and_replay_seconds']=time.perf_counter()-before
            stats['total_unique_tokens']=stats['assistant_text_tokens']+stats['tool_argument_tokens']+stats['tool_output_tokens']
            totals[mode].update(stats);report['by_case'].append({'case':case['id'],'mode':mode,**stats})
            directory=out/'replay'/mode;directory.mkdir(parents=True,exist_ok=True)
            (directory/(case['id']+'.json')).write_text(json.dumps({'case':case['id'],'messages':timeline,'registries':{k:v.snapshot() for k,v in registries.items()}},ensure_ascii=False))
        assert hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest()==original_digest
        print('REPLAYED',case['id'],len(wanted),flush=True)
    audit.close();report['raw_identity']=raw_identity;assert raw_identity
    for mode,stats in totals.items():
        report['modes'][mode]=dict(stats)
        report['modes'][mode]['unique_token_saving_percent']=100*(1-stats['total_unique_tokens']/totals['raw']['total_unique_tokens'])
        report['modes'][mode]['tool_input_saving_percent']=100*(1-stats['tool_output_tokens']/totals['raw']['tool_output_tokens'])
        report['modes'][mode]['target_equivalence_percent']=100*stats['calls_target_equivalent']/stats['calls_checked']
    (out/'replay_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['modes'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
