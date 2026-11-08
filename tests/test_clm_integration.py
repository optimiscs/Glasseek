import json
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Evaluation_backend'))
from agent.clm_context import CLMContextEngine

class Fallback:
    threshold_tokens=5000
    compression_count=0
    last_prompt_tokens=0
    def compress(self, messages, *args, **kwargs):return ['fallback']
    def should_compress(self,*args,**kwargs):return False
    def should_compress_preflight(self,*args,**kwargs):return False

def test_engine_keeps_pinned_task_and_complete_latest_tool_action():
    engine=CLMContextEngine(Fallback())
    messages=[{'role':'system','content':'system'},{'role':'user','content':'task'},
        {'role':'tool','content':'source <url>U1</url>'},
        {'role':'assistant','tool_calls':[{'id':'edit','function':{'name':'context_edit'}}],'content':''}]
    receipt=json.loads(engine.handle_tool_call('context_edit',{'base_version':0,'content':'keep <url>U1</url>'},messages=messages))
    assert receipt['accepted']
    messages.append({'role':'tool','tool_call_id':'edit','content':json.dumps(receipt)})
    result=engine.compress(messages)
    assert result[0:2]==messages[0:2] and result[-2:]==messages[-2:]
    assert engine.should_compress() is False

def test_engine_rejects_future_reference_and_falls_back_on_overflow():
    engine=CLMContextEngine(Fallback());messages=[{'role':'user','content':'task'}]
    result=json.loads(engine.handle_tool_call('context_edit',{'base_version':0,'content':'U99'},messages=messages))
    assert not result['accepted'] and engine.pending is None
    engine.pending={'role':'user','content':'x'*100000}
    assert engine.compress(messages)==['fallback']


def test_cli_defaults_and_explicit_configuration_are_shared(monkeypatch):
    import argparse
    from glasseek_shared.context.config import add_arguments,configure,mode
    monkeypatch.delenv('GLASSEEK_CONTEXT_MANAGER',raising=False)
    parser=argparse.ArgumentParser();add_arguments(parser)
    assert parser.parse_args([]).context_manager=='hermes'
    monkeypatch.setenv('GLASSEEK_CONTEXT_MANAGER','hermes')
    configure(parser.parse_args(['--context-manager','clm']))
    assert mode()=='clm'


def test_hermes_schema_has_native_tool_name():
    assert CLMContextEngine(Fallback()).get_tool_schemas()[0]['name']=='context_edit'
