"""Count full native history without duplicating inherited or repeated call IDs."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/trajectory_benchmark'))

spec=importlib.util.spec_from_file_location('clm_baselines',Path(__file__).resolve().parents[1]/'scripts/trajectory_benchmark/compare_clm_to_hermes.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_complete_history_ignores_inherited_rows_and_deduplicates_calls():
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE messages(id INTEGER,role TEXT,tool_calls TEXT,inherited_from_previous_session INTEGER)')
    def call(ident,name):return {'id':ident,'function':{'name':name,'arguments':'{}'}}
    a,b,c=call('a','web_search'),call('b','web_extract'),call('c','web_search')
    rows=[(1,'assistant',[a,b],0),(2,'assistant',[a,b],1),(3,'assistant',[a,c],0),
          (4,'assistant',[{'function':{'name':'execute_code'}},{'function':{'name':'execute_code'}}],None),
          (5,'user',[call('ignored','web_search')],0),(6,'assistant',[c],0)]
    db.executemany('INSERT INTO messages VALUES(?,?,?,?)',[(i,role,json.dumps(calls),inherited) for i,role,calls,inherited in rows])
    assert module.recorded_tools(db)==dict(web_search=2,web_extract=1,assistant_tool_turns=3,multi_call_turns=2,recorded_tool_calls=5)
    db.close()


def test_empty_history_has_zero_calls_and_turns():
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE messages(id INTEGER,role TEXT,tool_calls TEXT,inherited_from_previous_session INTEGER)')
    assert not any(module.recorded_tools(db).values())
    db.close()


def test_timeout_sigterm_is_failure_and_resume_time_is_cumulative(tmp_path,monkeypatch):
    import clm_comparison
    monkeypatch.setattr(clm_comparison,'ROOT',tmp_path)
    work=tmp_path/'workspace';work.mkdir()
    (work/'continuation.json').write_text(json.dumps({'previous_wall_seconds':3600}))
    folder=tmp_path/'Evaluation/output/preds/run/dataset/conv/q000001';folder.mkdir(parents=True)
    case=dict(workspace_dir=str(work),question_id='one',run_id='one__r1',status='failed',
              returncode=-15,timed_out=True,error='hermes timed out after 3600s',wall_seconds=15)
    (folder/'one.json').write_text(json.dumps(case))
    row=clm_comparison.load_cases('run','dataset')[0]
    assert not row['operator_interrupted'] and row['wall_seconds']==3615
    case.update(status='interrupted',timed_out=False,error='evaluation interrupted')
    (folder/'one.json').write_text(json.dumps(case))
    assert clm_comparison.load_cases('run','dataset')[0]['operator_interrupted']
