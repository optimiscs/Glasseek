"""Internal official CLM worker launched by the existing Hermes evaluation client."""
import argparse,asyncio,json,sys,signal,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'Shared'))
from glasseek_shared.context.official.run import run

async def main(args):
    cfg=json.loads(args.config.read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    options=SimpleNamespace(question=cfg['question'],output=out,model=cfg['model'],base_url=cfg['base_url'],
        environment='local',host='44',image='',max_steps=cfg['max_steps'],context_budget=cfg['context_budget'],
        max_tokens=cfg.get('max_tokens'),temperature=cfg.get('temperature'),resume_from=str(out) if (out/'agent/messages.json').exists() else None)
    sys.path.insert(0,str(ROOT/'Evaluation_backend'))
    from agent.session_recorder import AgentSessionRecorder
    import os
    profile=args.profile or cfg['session_id'].removeprefix('clm-')
    home=Path(os.environ.get('HERMES_HOME',ROOT/'Evaluation/data/runtime/hermes'))
    from glasseek_shared.context.official import activate
    activate()
    from clm_harness.utils.tool_schemas import BASH_TOOL
    recorder=AgentSessionRecorder(home/'profiles'/profile,Path(cfg['snapshot']),cfg['session_id'],cfg['model'],cfg['base_url'],str(out/'workspace'),tools=[BASH_TOOL],capacity_tokens=cfg.get('context_length'))
    if cfg.get('max_tool_rounds') is not None:
        options.max_tool_rounds=cfg['max_tool_rounds']
        # Full native history survives compaction; visible checkpoint messages do not.
        import sqlite3
        from glasseek_shared.context.official.tool_rounds import recorded_tool_rounds
        with sqlite3.connect(f'file:{home/"profiles"/profile/"state.db"}?mode=ro',uri=True) as connection:
            options.tool_rounds_used=recorded_tool_rounds(connection)
    async def query(agent,model,messages):
        recorder.before_query(messages)
        start=time.time()
        response=await type(agent)._query_with_retry(agent,model,messages)
        recorder.model_response(response,start,time.time())
        return response
    try:
        result=await run(options,query_hook=query,recorder=recorder)
        session=recorder.finish()
        if cfg.get('max_tool_rounds') is not None:
            import sqlite3,re
            segment=result['usage']
            (out/'agent/usage.segment.json').write_text(json.dumps(segment,indent=2))
            with sqlite3.connect(f'file:{home/"profiles"/profile/"state.db"}?mode=ro',uri=True) as connection:
                values=connection.execute('SELECT SUM(input_tokens),SUM(output_tokens),SUM(cache_read_tokens),'
                    'SUM(cache_write_tokens),SUM(api_call_count) FROM sessions').fetchone()
                inputs,outputs,cached,writes,calls=[v or 0 for v in values]
                notes=[row[0] for row in connection.execute("SELECT content FROM messages WHERE role='tool' "
                    "AND COALESCE(inherited_from_previous_session,0)=0 AND content LIKE '%LIVE_CTX_MAIN.txt: edit applied%'")]
            edits=0
            for note in notes:
                match=re.search(r'context ~(\d+)->(\d+) tokens',note)
                if match and abs(int(match[1])-int(match[2]))>=max(32,int(match[1])*.01):edits+=1
            result['usage']={**segment,'prompt_tokens':inputs+cached+writes,'completion_tokens':outputs,
                'total_tokens':inputs+cached+writes+outputs,'cached_tokens':cached,
                'cache_read_input_tokens':cached,'cache_creation_input_tokens':writes,'n_lm_calls':calls,
                'n_bash':segment['tool_rounds_used'],'n_ctx_syncs':len(notes),'n_ctx_syncs_real':edits,
                'usage_scope':'cumulative native history across resumed segments',
                'context_edit_count_source':'native tool observations; segment counters retained separately'}
            (out/'agent/usage.json').write_text(json.dumps(result['usage'],indent=2))
        args.result.write_text(json.dumps({'session':session,'answer':result['answer'],'completion_protocol':result['protocol'],
            'stop_reason':result['stop_reason'],'web_calls':result['web_calls'],'usage':result['usage']}))
        print('session_id: '+session['session_id'])
    except BaseException:
        recorder.finish('interrupted')
        raise
    finally:recorder.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--profile')
    asyncio.run(main(p.parse_args()))
