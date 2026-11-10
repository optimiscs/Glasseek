"""Exercise the real pinned loop and local shell, with deterministic LM output."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'Shared'),str(ROOT/'Evaluation')]
from glasseek_shared.context.official import activate
activate()
from litellm import ModelResponse
from glasseek_shared.context.official.agent import create_agent,release_agent
from glasseek_shared.context.official.local import LocalEnvironment
from glasseek_shared.context.official.completion import PROTOCOL,final_text,patch_loop,prompt_template
from browse_comp_eval.session_reader import extract_final_assistant_message


def response(content='',command=None,finish='stop',name='bash'):
    message={'role':'assistant','content':content}
    if command is not None:
        message['tool_calls']=[{'id':str(id(message)),'type':'function','function':
            {'name':name,'arguments':json.dumps({'command':command})}}]
    return ModelResponse(choices=[{'message':message,'finish_reason':finish}],
        usage={'prompt_tokens':100,'completion_tokens':10,'total_tokens':110})


def exercise(tmp_path, responses, max_steps=8, tool_limit=None, initial_used=0):
    async def run():
        logs=tmp_path/'agent';logs.mkdir(parents=True)
        env=LocalEnvironment(tmp_path/'workspace')
        agent=create_agent(workspace=env._workdir,logs_dir=logs,model_name='openai/test',
            max_steps=max_steps,context_budget_tokens=10000,emit_ctx_trajectory=False)
        if tool_limit is not None:
            from glasseek_shared.context.official.tool_rounds import ToolRoundBudget
            agent.glasseek_tool_round_budget=ToolRoundBudget(logs/'tool-round-budget.json',tool_limit,initial_used)
            agent.lm_call_cap=0
        seen=[]
        async def query(self,model,messages):
            seen.append(json.loads(json.dumps(messages)))
            return responses.pop(0)
        import types
        agent._query_with_retry=types.MethodType(query,agent)
        try:
            await env.start();await agent.setup(env)
            await agent.run('Use tools if needed. Answer directly.',env,SimpleNamespace())
            transcript=json.loads((logs/'trajectory.json').read_text())
            assert await agent._ckpt.save(env,transcript)
            saved=json.loads((logs/'messages.json').read_text())
            assert saved==transcript
            return transcript,agent.glasseek_completion,seen,json.loads((logs/'usage.json').read_text())
        finally:
            await env.stop();release_agent(agent)
    return asyncio.run(run())


def test_direct_final_uses_actual_assistant_and_checkpoints_it(tmp_path):
    transcript,completion,seen,usage=exercise(tmp_path,[response('Exact Answer: READY')])
    assert completion=={'protocol':PROTOCOL,'stop_reason':'answer','answer':'Exact Answer: READY'}
    assert extract_final_assistant_message({'messages':transcript})==('Exact Answer: READY','')
    assert len(transcript)==3 and usage['n_lm_calls']==1 and usage['n_bash']==0
    assert not list(tmp_path.glob('**/answer.txt'))
    assert 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' not in str(seen)


@pytest.mark.parametrize('command',['echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT',
                                    'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; true'])
def test_old_echo_and_answer_file_do_not_end_or_override_answer(tmp_path,command):
    transcript,completion,seen,usage=exercise(tmp_path,[
        response(command='printf STALE > answer.txt',finish='tool_calls'),
        response(command=command,finish='tool_calls'),
        response('CURRENT')])
    assert completion['answer']=='CURRENT' and len(seen)==3 and usage['n_bash']==2
    assert transcript[-1]['content']=='CURRENT'
    assert 'standalone command' not in str(seen)


@pytest.mark.parametrize('first',[
    response(''),response('<think>still thinking</think>'),
    response('incomplete answer',finish='length'),
    response(command='true',finish='tool_calls',name='unknown_tool')])
def test_invalid_or_incomplete_output_does_not_finish(tmp_path,first):
    transcript,completion,seen,usage=exercise(tmp_path,[first,response('READY')])
    assert len(seen)==2 and completion['answer']=='READY'
    if first.choices[0].message.tool_calls:
        assert seen[1][-1]['role']=='tool'
        assert 'Invalid tool call' in seen[1][-1]['content']


def test_tool_only_turn_limit_is_not_an_answer(tmp_path):
    transcript,completion,seen,usage=exercise(tmp_path,[response(command='true',finish='tool_calls')],max_steps=1)
    assert not completion['answer']
    assert extract_final_assistant_message({'messages':transcript})[0]==''
    assert 'submit command' not in seen[0][-1]['content']


def test_cumulative_tool_limit_keeps_one_real_final_answer_turn(tmp_path):
    transcript,completion,seen,usage=exercise(tmp_path,[
        response(command='printf evidence',finish='tool_calls'),response('FINAL')],
        max_steps=1,tool_limit=3,initial_used=2)
    assert completion['answer']=='FINAL' and len(seen)==2
    assert 'cumulative tool-round limit (3/3)' in seen[-1][-1]['content']
    assert json.loads((tmp_path/'agent/tool-round-budget.json').read_text())['used']==3


def test_tool_limit_does_not_execute_action_on_final_turn_or_rewrite_it(tmp_path):
    action=response(command='touch SHOULD_NOT_EXIST',finish='tool_calls')
    transcript,completion,seen,usage=exercise(tmp_path,[action],tool_limit=2,initial_used=2)
    assert not (tmp_path/'workspace/SHOULD_NOT_EXIST').exists()
    assert completion['stop_reason']=='tool_round_limit' and not completion['answer']
    assert transcript[-2]['tool_calls']==action.choices[0].message.model_dump()['tool_calls']
    assert 'not executed' in transcript[-1]['content']
    assert usage['n_bash']==0 and len(seen)==1


def test_pure_context_compaction_also_consumes_tool_round(tmp_path):
    mirror=tmp_path/'workspace/.clm/context/LIVE_CTX_MAIN.txt'
    command=f"python3 -c {__import__('shlex').quote('from pathlib import Path; Path('+repr(str(mirror))+').write_text('+repr('Evidence summary: retain the verified facts.')+')')}"
    transcript,completion,seen,usage=exercise(tmp_path,[
        response(command=command,finish='tool_calls'),response('FINAL')],tool_limit=1)
    assert completion['answer']=='FINAL' and usage['n_bash']==1
    assert 'cumulative tool-round limit (1/1)' in seen[-1][-1]['content']
    assert json.loads((tmp_path/'agent/tool-round-budget.json').read_text())['used']==1


def test_tool_budget_survives_restart_and_deduplicates_native_history(tmp_path):
    import sqlite3
    from glasseek_shared.context.official.tool_rounds import ToolRoundBudget,recorded_tool_rounds
    p=tmp_path/'budget.json';budget=ToolRoundBudget(p,3,1)
    assert budget.reserve()
    restored=ToolRoundBudget(p,3,0)
    assert restored.used==2 and restored.reserve() and not restored.reserve()
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE messages(id INTEGER,role TEXT,tool_calls TEXT,inherited_from_previous_session INTEGER)')
        call=json.dumps([{'id':'one','function':{'name':'bash'}}])
        db.executemany('INSERT INTO messages VALUES(?,?,?,?)',[(1,'assistant',call,0),
            (2,'assistant',call,1),(3,'assistant',call,0),(4,'tool',call,0)])
        assert recorded_tool_rounds(db)==1


def test_context_instructions_preserved_and_budget_classes_are_upstream(tmp_path):
    from clm_harness.clm_agent.harness import _SYSTEM_TEMPLATE
    from clm_harness.utils.budget import BudgetController
    from clm_harness.utils.resume import Checkpointer
    from clm_harness.context_env.env import ContextEnv
    agent=create_agent(workspace=tmp_path/'work',logs_dir=tmp_path/'logs',model_name='openai/test',context_budget_tokens=10000,emit_ctx_trajectory=False)
    try:
        assert type(agent._budget) is BudgetController
        assert type(agent._ckpt) is Checkpointer and type(agent._ctx) is ContextEnv
        section=lambda text:text.split('## Managing your context',1)[1].split('{{finish_instructions}}')[0].strip()
        assert section(prompt_template(_SYSTEM_TEMPLATE))==section(_SYSTEM_TEMPLATE)
        final=agent._budget.final_message(100000)
        assert 'assistant message without tool calls' in final and 'submit command' not in final
        with pytest.raises(RuntimeError,match='branch changed'):patch_loop('unrecognized source')
    finally:release_agent(agent)


def test_budget_decisions_and_rollback_are_identical_to_upstream(tmp_path):
    import copy
    agent=create_agent(workspace=tmp_path/'work',logs_dir=tmp_path/'logs',model_name='openai/test',
        context_budget_tokens=4000,context_budget_reserve_tokens=256,max_num_retry_on_limit=1,
        retry_edit_margin_tokens=512,emit_ctx_trajectory=False)
    try:
        adapted=agent._budget
        baseline=copy.deepcopy(adapted)
        del baseline.final_message  # Restore the class's official wording only.
        messages=[{'role':'system','content':'system'},{'role':'user','content':'task'},
            {'role':'assistant','content':'research'},{'role':'tool','content':'evidence '*9000}]
        for expected in ('rollback','final_turn','stop'):
            left=adapted.evaluate(copy.deepcopy(messages));right=baseline.evaluate(copy.deepcopy(messages))
            assert left.action==right.action==expected
            assert left.token_count==right.token_count and left.log==right.log
            if expected=='final_turn':
                assert 'assistant message without tool calls' in left.messages[0]['content']
        a=copy.deepcopy(messages);b=copy.deepcopy(messages)
        assert adapted.rollback_to_margin(a,protect=2)==baseline.rollback_to_margin(b,protect=2)
        assert a==b
    finally:release_agent(agent)


@pytest.mark.parametrize('content,finish',[
    ('<think>answer in reasoning only</think>','stop'),('partial','length'),
    ('<tool_call>{"name":"bash"}','stop'),('(empty)','stop')])
def test_no_reasoning_or_malformed_action_promoted_to_answer(content,finish):
    assert final_text({'role':'assistant','content':content},finish)==''


def test_old_checkpoint_cannot_silently_change_completion_protocol(tmp_path):
    from glasseek_shared.context.official.run import run
    with pytest.raises(ValueError,match='old CLM file/echo checkpoint'):
        asyncio.run(run(SimpleNamespace(resume_from=str(tmp_path))))


def test_resume_keeps_official_replay_but_uses_hermes_finish_wording(tmp_path):
    previous=tmp_path/'previous';(previous/'agent').mkdir(parents=True)
    prefix=[{'role':'system','content':'system'},{'role':'user','content':'task'}]
    (previous/'agent/messages.json').write_text(json.dumps(prefix+[{'role':'assistant','content':'Recorded evidence.'}]))
    agent=create_agent(workspace=tmp_path/'work',logs_dir=tmp_path/'logs',model_name='openai/test',
        context_budget_tokens=10000,emit_ctx_trajectory=False,resume_from=previous)
    try:
        messages=agent._resume.replay(prefix,budget_tokens=9000)
        assert messages[2]['content']=='Recorded evidence.'
        assert messages[-1]['content'].startswith('SESSION RESUMED.')
        assert 'assistant message without tool calls' in messages[-1]['content']
        assert 'only finish' not in messages[-1]['content'] and '/workspace is exactly' not in messages[-1]['content']
    finally:release_agent(agent)
