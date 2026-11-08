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
    async def query(agent,model,messages):
        recorder.before_query(messages)
        start=time.time()
        response=await type(agent)._query_with_retry(agent,model,messages)
        recorder.model_response(response,start,time.time())
        return response
    try:
        result=await run(options,query_hook=query,recorder=recorder)
        session=recorder.finish()
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
