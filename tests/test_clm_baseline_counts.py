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
