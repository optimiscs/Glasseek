import json,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'Evaluation_backend'),str(ROOT/'Evaluation')]
from agent.session_recorder import AgentSessionRecorder
from quest_viewer_checkpoint_adapter import build_steps,enrich_timing,merge_session_chain
from quest_viewer_timing import apply_message_timings
from litellm import ModelResponse


def test_native_recording_uses_hermes_schema_and_exact_message_spans(tmp_path):
    p=tmp_path/'profile';snapshot=tmp_path/'sessions/session_test.json'
    r=AgentSessionRecorder(p,snapshot,'test','test-model')
    start=time.time();prefix=[{'role':'system','content':'system'},{'role':'user','content':'task'}]
    r.before_query(prefix)
    response=ModelResponse(choices=[{'message':{'role':'assistant','content':'thought','tool_calls':[{'id':'c1','type':'function','function':{'name':'bash','arguments':'{"command":"true"}'}}]}}],usage={'prompt_tokens':200,'completion_tokens':20,'prompt_tokens_details':{'cached_tokens':50}})
    r.model_response(response,start,start+.5)
    pending=response.choices[0].message.model_dump()
    obs=SimpleNamespace(result=SimpleNamespace(return_code=0),render=lambda:'output')
    r.tool_result(pending,obs,start+.6,start+.8)
    session=r.finish();assert session['input_tokens']==150 and session['cache_read_tokens']==50
    steps=build_steps([{**m,'_viewer_session_id':r.sid} for m in session['messages']])
    assert apply_message_timings(session,steps,p/'state.db')
    assert steps[0]['model_seconds']==.5 and abs(steps[0]['tool_seconds']-.2)<1e-5
    assert steps[0]['toolCallId']=='c1'
    assert '_official_clm_active' not in session
    # Compacted input gets a native child; old actions are never overwritten.
    parent=r.sid;r.before_query(prefix+[{'role':'user','content':'edited context'}])
    assert r.sid!=parent and r.db.get_session(r.sid)['parent_session_id']==parent
    assert any(m.get('tool_calls') for m in r.db.get_messages_as_conversation(parent))
    r.close()


def test_duration_only_records_never_fabricate_wall_clock_spans(tmp_path):
    r=AgentSessionRecorder(tmp_path/'profile',tmp_path/'snap/session_x.json','x','m')
    r.before_query([{'role':'user','content':'task'}]);mid=r.append({'role':'assistant','content':'answer'})
    r.db.record_message_timing(mid,'model',1.5,provenance='legacy_duration')
    session=r.finish();steps=build_steps([{**m,'_viewer_session_id':r.sid} for m in session['messages']])
    assert not apply_message_timings(session,steps,tmp_path/'profile/state.db')
    assert steps[0]['model_seconds']==1.5 and steps[0]['modelStart'] is None
    r.close()


def test_reopen_native_session_preserves_existing_actions(tmp_path):
    profile=tmp_path/'profile';snapshot=tmp_path/'snap/session_x.json'
    r=AgentSessionRecorder(profile,snapshot,'x','m');prefix=[{'role':'user','content':'task'}]
    r.before_query(prefix);r.append({'role':'assistant','content':'saved'})
    r.finish('interrupted');r.close()
    restored=AgentSessionRecorder(profile,snapshot,'x','m')
    restored.before_query(prefix+[{'role':'assistant','content':'saved'},{'role':'user','content':'resume'}])
    assert restored.sid=='x'
    assert len(restored.db.get_messages_as_conversation('x'))==3
    restored.close()


def test_request_breakdown_preserves_unknowns_and_latest_server_total(tmp_path):
    from quest_viewer_timing import request_context_stats
    r=AgentSessionRecorder(tmp_path/'profile',tmp_path/'s/session_x.json','x','m',tools=[],capacity_tokens=131072)
    r.before_query([{'role':'system','content':'system prompt'},{'role':'user','content':'question'}])
    r.db.complete_request_context(r.request_id,15200)
    session=r.publish();context=request_context_stats(session,tmp_path/'profile/state.db')
    assert context['usedTokens']==15200 and context['capacityTokens']==131072
    assert context['systemTokens']>0 and context['messageTokens']>0 and context['toolTokens']==0
    assert context['breakdownEstimated']
    r.db.record_request_context('x',[{'role':'user','content':'next'}],None,prompt_tokens=100)
    context=request_context_stats(session,tmp_path/'profile/state.db')
    assert context['usedTokens']==100 and 'toolTokens' not in context
    r.close()
