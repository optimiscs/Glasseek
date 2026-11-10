"""Official CLM context mechanics with Hermes answers and Glasseek tools."""
import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from . import activate,ROOT,COMMIT
from .environment import DockerEnvironment

async def run(args, query_hook=None, tools=None, recorder=None):
    activate()
    from .completion import PROTOCOL
    if args.resume_from:
        marker=Path(args.resume_from)/'completion-protocol.json'
        if not marker.exists() or json.loads(marker.read_text()).get('protocol')!=PROTOCOL:
            raise ValueError('Cannot resume an old CLM file/echo checkpoint with the Hermes completion protocol; start a new run')
    from .agent import create_agent, release_agent
    import yaml
    upstream=ROOT/'third_party/context-language-models'
    cfg=yaml.safe_load((upstream/'clm/clm_harness/configs/bcp.yaml').read_text())
    from glasseek_shared.context.config import config as context_config
    kwargs=dict(cfg['agent_kwargs'])
    overrides=context_config().official
    allowed={'context_budget_reserve_tokens','nudge_ratios','persistent_nudge_ratio','persistent_nudge_obs_mult','max_num_retry_on_limit','retry_edit_margin_tokens','finalize_nudge_turns','lm_call_cap','enable_thinking','checkpoint_interval_s'}
    if set(overrides)-allowed:raise ValueError('Unsupported official CLM config keys: '+str(sorted(set(overrides)-allowed)))
    kwargs.update(overrides)
    kwargs.update(max_steps=args.max_steps,context_budget_tokens=args.context_budget,
        cost_metric='usd',emit_ctx_trajectory=False)
    kwargs.setdefault('checkpoint_interval_s',30)
    kwargs.pop('flops_model_key',None)
    for name in ('max_tokens','temperature','top_p'):
        value=getattr(args,name,None)
        if value is not None:kwargs[name]=value
    from .local import LocalEnvironment
    env=(LocalEnvironment(args.output/'workspace') if args.environment=='local'
         else DockerEnvironment(args.host,args.image))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'completion-protocol.json').write_text(json.dumps({'protocol':PROTOCOL}))
    (args.output/'agent').mkdir(parents=True,exist_ok=True)
    # ATIF-CTX is not a live recording format in Glasseek. Preserve old files
    # for audit; native Hermes records and standard message checkpoints win.
    old_ctx=args.output/'agent/trajectory.ctx.json'
    if recorder is not None and old_ctx.exists():
        import shutil,hashlib
        archive=args.output/'legacy-recording';archive.mkdir(exist_ok=True)
        shutil.move(str(old_ctx),str(archive/('trajectory-'+hashlib.sha256(old_ctx.read_bytes()).hexdigest()[:12]+'.ctx.json')))
    agent=create_agent(workspace=env._workdir,logs_dir=args.output/'agent',model_name='openai/'+args.model,api_base=args.base_url,
        allow_edit_growth='shrink',resume_from=args.resume_from,**kwargs)
    tool_limit = getattr(args, 'max_tool_rounds', None)
    if tool_limit is not None:
        from .tool_rounds import ToolRoundBudget
        agent.glasseek_tool_round_budget = ToolRoundBudget(args.output/'agent/tool-round-budget.json',
            tool_limit, getattr(args, 'tool_rounds_used', 0))
        agent.lm_call_cap = 0
    # Context instructions are upstream; action/final-answer wording uses Hermes.
    bridge=None
    if query_hook is not None:
        import types
        agent._query_with_retry=types.MethodType(query_hook,agent)
    if recorder is not None:
        original_step=agent._ctx.step
        async def recorded_step(command,messages,**kwargs):
            import time
            start=time.time()
            result=await original_step(command,messages,**kwargs)
            recorder.tool_result(kwargs.get('pending'),result,start,time.time())
            return result
        agent._ctx.step=recorded_step
    helper_template='''#!/usr/bin/env python3
import sys,json,urllib.request,urllib.error
mode=sys.argv[1]
call={'name':'web_search','arguments':{'query':' '.join(sys.argv[2:])}} if mode=='search' else {'name':'web_extract','arguments':{'urls':sys.argv[2:]}}
if 'GATEWAY'.startswith('unix:'):
    import http.client,socket
    class UnixHTTP(http.client.HTTPConnection):
        def connect(self):
            self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            self.sock.connect('GATEWAY'[5:])
    connection=UnixHTTP('localhost',timeout=180)
    connection.request('POST','/session/SESSION',body=json.dumps(call),headers={'Content-Type':'application/json'})
    response=connection.getresponse();payload=json.load(response)
    print(payload['output']);raise SystemExit(0 if response.status==200 else 1)
r=urllib.request.Request('GATEWAY/session/SESSION',data=json.dumps(call).encode(),headers={'Content-Type':'application/json'})
try:
    with urllib.request.urlopen(r,timeout=180) as f:payload=json.load(f)
except urllib.error.HTTPError as error:
    with error:payload=json.load(error)
    print(payload.get('output') or json.dumps(payload,ensure_ascii=False))
    raise SystemExit(1)
print(payload['output'])
'''
    await env.start()
    try:
        await agent.setup(env)
        if args.environment=='local':
            from .web import WebBridge
            bridge=WebBridge(env._workdir,tools)
        helper=helper_template.replace('SESSION',env.name).replace('GATEWAY',bridge.url if bridge else 'http://127.0.0.1:39187')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'glasseek-web';p.write_text(helper);await env.upload_file(str(p),str(Path(env._workdir)/'glasseek-web'))
        import shlex
        helper_path=str(Path(env._workdir)/'glasseek-web')
        await env.exec('chmod +x '+shlex.quote(helper_path))
        brief=args.question+'\n\nAvailable web commands: '+helper_path+' search QUERY; '+helper_path+' extract URL_OR_ID [URL_OR_ID ...]. Return the final answer directly in an assistant message without tool calls, following the answer format requested above.'
        import os
        if os.environ.get('GLASSEEK_FIXED_CORPUS_CONFIG'):
            brief+='\n\nRetrieval uses a fixed document corpus. Search returns original URLs and document IDs. Read only those corpus documents with the commands above; there is no live-web fallback. Cite evidence as [docid] using document IDs already returned by your tools. Bash network sockets are disabled; the supplied web command uses the task-local Unix bridge.'
        context=SimpleNamespace()
        await agent.run(brief,env,context)
        transcript=json.loads((args.output/'agent/trajectory.json').read_text())
        if not await agent._ckpt.save(env,transcript):raise RuntimeError('CLM checkpoint save failed')
        await env.download_dir(env._workdir,str(args.output/'artifacts/workspace-final'))
        completion=dict(agent.glasseek_completion)
        if completion['stop_reason']!='answer':
            if agent.hit_lm_call_cap:completion['stop_reason']='lm_call_limit'
            elif agent._budget.finalized:completion['stop_reason']='context_limit'
            elif getattr(agent, 'glasseek_tool_round_budget', None) is not None and agent.glasseek_tool_round_budget.exhausted:
                completion['stop_reason']='tool_round_limit'
            elif agent.n_lm_calls>=args.max_steps:completion['stop_reason']='turn_limit'
        (args.output/'provenance.json').write_text(json.dumps({'official_commit':COMMIT,'system_prompt':'official context instructions with Hermes completion wording and per-task paths','completion_protocol':PROTOCOL,'paths':agent.glasseek_paths,'prompt':agent.glasseek_prompt,'environment':args.environment,'task_addition':'Glasseek web command usage and native assistant answer','model':args.model},indent=2))
        usage=json.loads((args.output/'agent/usage.json').read_text())
        if tool_limit is not None:
            usage.update(tool_round_limit=tool_limit, tool_rounds_used=agent.glasseek_tool_round_budget.used)
            (args.output/'agent/usage.json').write_text(json.dumps(usage,indent=2))
        return {'messages':transcript,**completion,
                'web_calls':dict(bridge.counts) if bridge else {},'usage':usage}
    finally:
        try:
            if bridge is not None:bridge.close()
            await env.stop()
        finally:release_agent(agent)

def main():
    p=argparse.ArgumentParser();p.add_argument('--question',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base-url',default='http://127.0.0.1:19109/v1');p.add_argument('--model',default='glasseek-base-9b')
    p.add_argument('--environment',choices=['local','docker'],default='local')
    p.add_argument('--host',default='44');p.add_argument('--image',default='709_model:final')
    p.add_argument('--max-steps',type=int,default=8);p.add_argument('--context-budget',type=int,default=28672)
    p.add_argument('--resume-from');args=p.parse_args();asyncio.run(run(args))
if __name__=='__main__':main()
