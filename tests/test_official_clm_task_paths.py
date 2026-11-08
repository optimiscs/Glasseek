"""Run upstream agents concurrently using deterministic LM responses, no model/API."""
import asyncio
import json
import sys
import types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Shared'))
from glasseek_shared.context.official import activate
activate()
from glasseek_shared.context.official.agent import create_agent,release_agent
from glasseek_shared.context.official.local import LocalEnvironment
from clm_harness.utils.resume import Checkpointer,ResumeState
from litellm import ModelResponse


def test_two_agents_have_distinct_prompt_mirror_shell_and_restored_paths(tmp_path):
    async def exercise(label):
        work=tmp_path/label/'workspace';logs=tmp_path/label/'agent';logs.mkdir(parents=True)
        env=LocalEnvironment(work)
        agent=create_agent(workspace=work,logs_dir=logs,model_name='openai/test',max_steps=4,
            context_budget_tokens=10000,context_budget_reserve_tokens=512,emit_ctx_trajectory=False)
        calls=[];mirror=agent.glasseek_paths['context_file']
        async def query(self,model,messages):
            calls.append(messages[0]['content'])
            assert mirror in messages[0]['content']
            assert '/tmp/.live_ctx' not in messages[0]['content']
            assert mirror in self._budget.compaction_hint
            n=len(calls)
            if n==1:command=f'mkdir -p sub; cd sub; export CLM_TASK={label}; printf {label} > marker'
            elif n==2:
                command="python3 - <<'END'\nfrom pathlib import Path\np=Path("+repr(mirror)+");s=p.read_text();p.write_text(s.split('\\n',1)[0]+'\\nResearch notes retained.\\n')\nEND"
            else:
                return ModelResponse(model='openai/test',choices=[{'index':0,'message':{'role':'assistant','content':'Research complete.'},'finish_reason':'stop'}],usage={'prompt_tokens':1,'completion_tokens':1,'total_tokens':2})
            return ModelResponse(model='openai/test',choices=[{'index':0,'message':{'role':'assistant','content':'THOUGHT: execute task','tool_calls':[{'id':f'{label}-{n}','type':'function','function':{'name':'bash','arguments':json.dumps({'command':command})}}]},'finish_reason':'tool_calls'}],usage={'prompt_tokens':1,'completion_tokens':1,'total_tokens':2})
        agent._query_with_retry=types.MethodType(query,agent)
        try:
            await env.start();await agent.setup(env)
            await agent.run('Keep the task marker and submit.',env,types.SimpleNamespace())
            assert agent._ctx.n_ctx_syncs>=1
            assert (work/'sub/marker').read_text()==label
            result=await env.exec('printf "$CLM_TASK"')
            # Official ContextEnv.wrap owns persistent shell restoration.
            result=await agent._ctx.execute('printf "$CLM_TASK"',env)
            assert result.stdout==label
            transcript=json.loads((logs/'trajectory.json').read_text())
            assert await Checkpointer(logs,1).save(env,transcript)
        finally:await env.stop();release_agent(agent)
        restored=LocalEnvironment(tmp_path/(label+'-restored')/'workspace')
        replacement=create_agent(workspace=restored._workdir,logs_dir=tmp_path/(label+'-restored')/'agent',model_name='openai/test',context_budget_tokens=10000,emit_ctx_trajectory=False)
        try:
            await restored.start();await replacement.setup(restored)
            resume=ResumeState(tmp_path/label);await resume.restore(restored)
            replayed=resume.replay(transcript[:2],budget_tokens=10000)
            assert replayed and (Path(restored._workdir)/'sub/marker').read_text()==label
            result=await replacement._ctx.execute('pwd; printf "$CLM_TASK"',restored)
            assert result.stdout==str(Path(restored._workdir)/'sub')+'\n'+label
            assert str(work) not in (Path(restored._workdir)/'.clm/bash/cwd').read_text()
        finally:await restored.stop();release_agent(replacement)
        return mirror,calls[0]
    async def main():
        a,b=await asyncio.gather(exercise('taskA'),exercise('taskB'))
        assert a[0]!=b[0] and a[0] not in b[1] and b[0] not in a[1]
        import clm_harness.clm_agent.harness as original
        assert original._CTX_FILE=='/tmp/.live_ctx/LIVE_CTX_MAIN.txt'
        assert '/tmp/.live_ctx/LIVE_CTX_MAIN.txt' in original._SYSTEM_TEMPLATE
        for (mirror,prompt) in (a,b):
            from glasseek_shared.context.official.completion import prompt_template
            expected=prompt_template(original._SYSTEM_TEMPLATE).replace('{{context_budget}}','9488 tokens')
            expected=expected.replace('/tmp/.live_ctx/LIVE_CTX_MAIN.txt',mirror).replace(
                '/tmp/.live_ctx',str(Path(mirror).parent))
            assert prompt==expected
            assert 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' not in prompt
    asyncio.run(main())


def test_duplicate_workspace_is_rejected_without_global_lock(tmp_path):
    async def main():
        a,b,c=LocalEnvironment(tmp_path/'same'),LocalEnvironment(tmp_path/'same'),LocalEnvironment(tmp_path/'other')
        try:
            await a.start();await c.start()
            import pytest
            with pytest.raises(RuntimeError,match='workspace is already in use'):await b.start()
        finally:await a.stop();await b.stop();await c.stop()
    asyncio.run(main())
