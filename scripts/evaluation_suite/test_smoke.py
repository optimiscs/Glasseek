import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from suite import dump,jsonl,sha
from smoke import prepare
from report import case_usage,native_metrics
import orchestrate


class SmokeTests(unittest.TestCase):
    def test_empty_fact_and_failed_mind_placeholder_are_not_scored(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            jsonl(root/'author/race/raw_results.jsonl',[{'overall_score':0.4}])
            (root/'author/fact').mkdir();(root/'author/fact/fact_result.txt').write_text('')
            metrics,scored=native_metrics(root,'drbench')
            self.assertEqual(scored,0);self.assertEqual(metrics['RACE'],0.4)
            dump(root/'author/glasseek/summary.json',{'tasks':{'task':{'avg_score':0,'pass':False}}})
            dump(root/'author/glasseek/task/summary.json',[{'status':'failed','score':0}])
            metrics,scored=native_metrics(root,'mind2web2')
            self.assertEqual(scored,0);self.assertIsNone(metrics['avg_score'])

    def test_child_is_repeatable_and_parent_is_not_modified(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d)/'release';members=[];refs=[]
            for name in ['a','b']:
                pool=[]
                for i in range(4):
                    opaque=name+str(i);members.append({'benchmark':name,'id':str(i),'opaque_id':opaque,'stratum':str(i%2)})
                    refs.append({'benchmark':name,'id':str(i),'opaque_id':opaque,'answer':'secret'})
                    pool.append({'id':opaque,'question':'question '+str(i),'answer':'','type':'research'})
                jsonl(parent/'inference'/(name+'.jsonl'),pool)
            jsonl(parent/'private/membership.jsonl',members);jsonl(parent/'private/references.jsonl',refs)
            dump(parent/'summary.json',[{'benchmark':name,'selected':4} for name in ['a','b']]);dump(parent/'assets.lock.json',{'files':{}})
            (parent/'prompts').mkdir();(parent/'prompts/default.txt').write_text('answer')
            dump(parent/'manifest.json',{'files':{}});before=sha(parent/'manifest.json')
            child=prepare(parent,2);h=sha(child/'manifest.json');self.assertEqual(h,sha(prepare(parent,2)/'manifest.json'));self.assertEqual(before,sha(parent/'manifest.json'))
            self.assertEqual(len((child/'private/references.jsonl').read_text().splitlines()),4)
            replay=prepare(parent,selected_ids={'a0','a2','b1'})
            self.assertEqual([r['selected'] for r in json.loads((replay/'summary.json').read_text())],[2,1])
            self.assertEqual(sha(replay/'manifest.json'),sha(prepare(parent,selected_ids={'b1','a2','a0'})/'manifest.json'))
            with self.assertRaises(ValueError):prepare(parent,selected_ids={'absent'})

    def test_native_counters_include_cache_once_and_edits(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'HERMES_HOME':d}):
            p=Path(d)/'profiles/p/state.db';p.parent.mkdir(parents=True)
            with sqlite3.connect(p) as db:
                db.execute('CREATE TABLE sessions(id,input_tokens,output_tokens,cache_read_tokens,cache_write_tokens,api_call_count)')
                db.execute('INSERT INTO sessions VALUES ("s",100,50,300,0,1)')
                db.execute('CREATE TABLE messages(role,content,tool_calls,inherited_from_previous_session)')
                db.execute('INSERT INTO messages VALUES ("tool","LIVE_CTX_MAIN.txt: edit applied; context ~1000->900 tokens",NULL,0)')
                db.execute('INSERT INTO messages VALUES ("tool","LIVE_CTX_MAIN.txt: edit applied",NULL,1)')
            usage=case_usage({'profile':'p','retry_attempts':[{'profile':'p'}],'tool_calls':{'web_search':1}})
            self.assertEqual(usage['input_tokens'],400);self.assertEqual(usage['output_tokens'],50)
            self.assertEqual(usage['context_edits'],1);self.assertEqual(usage['context_edits_substantial'],1)

    def test_scheduler_respects_global_five_slots(self):
        active=0;peak=0
        class Process:
            pid=123
            def __init__(self,cmd,**kwargs):
                nonlocal active,peak
                self.n=int(cmd[cmd.index('--workers-per-endpoint')+1]);active+=self.n;peak=max(peak,active);self.polls=0;self.done=False
                if '--num-workers' in cmd:
                    assert int(cmd[cmd.index('--num-workers')+1])==self.n
            def poll(self):
                nonlocal active
                self.polls+=1
                if self.polls<3:return None
                if not self.done:active-=self.n;self.done=True
                return 0
            def terminate(self):pass
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(output_root=Path(d),run_id='r',workers=5,phase='infer',score_schedule='off');(Path(d)/'r').mkdir()
            names=list('abcdefg');plans=[(n,['fake','--workers-per-endpoint','5','--num-workers','5'],{}) for n in names]
            with patch.object(orchestrate.subprocess,'Popen',Process),patch.object(orchestrate,'write_report'),patch.object(orchestrate.time,'sleep'):
                orchestrate.execute(args,plans,[],names,{n:2 for n in names})
            self.assertEqual(peak,5);self.assertEqual(active,0)


if __name__=='__main__':unittest.main()


def test_review_usage_is_added_once_to_native_session_counters(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    dbpath=tmp_path/'profiles/p/state.db';dbpath.parent.mkdir(parents=True)
    with sqlite3.connect(dbpath) as db:
        db.execute('CREATE TABLE sessions(id,input_tokens,output_tokens,cache_read_tokens,cache_write_tokens,api_call_count)')
        db.execute('INSERT INTO sessions VALUES ("s",100,50,300,0,1)')
        db.execute('CREATE TABLE messages(role,content,tool_calls,inherited_from_previous_session)')
    workspace=tmp_path/'work/attempt_1';workspace.mkdir(parents=True)
    record={'reviews':[{'usage':{'prompt_tokens':200,'completion_tokens':20,'prompt_tokens_details':{'cached_tokens':100}}}]}
    dump(workspace/'research-result-a1.json',record)
    usage=case_usage({'profile':'p','workspace_dir':str(workspace),'research_submission':record})
    assert usage['input_tokens']==600 and usage['cached_input_tokens']==400
    assert usage['output_tokens']==70 and usage['model_calls']==2 and usage['research_review_calls']==1
