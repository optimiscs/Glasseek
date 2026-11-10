import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

from client import Client, digest, public_config, save
from ingest import Dedup, harness_plan, normalize, records
import pipeline as p


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.config=json.loads(Path(__file__).with_name('config.example.json').read_text())

    def tearDown(self): self.temp.cleanup()

    def sample(self):
        return normalize({'question':'Which object?', 'answer':'X','messages':[
            {'role':'assistant','content':'<tool_call>{"name":"search","arguments":{"query":["X","Y"]}}</tool_call>'},
            {'role':'tool','content':'Object X was released in 2042.'},
            {'role':'assistant','content':'Object X was released in 2042.'}]},'synthetic','fixture',0)

    def test_stream_array_and_jsonl_and_large_record(self):
        rows=[{'x':'a'*150000},{'x':'b'}]
        for content in (json.dumps(rows),'\n'.join(json.dumps(r) for r in rows)):
            path=self.root/'input.json';path.write_text(content)
            self.assertEqual(list(records(path)),rows)
        for bad in ('[{"x":1}', '[{"x":1} {}]', '[{"x":1},]', '[]  junk'):
            path.write_text(bad)
            with self.assertRaises(ValueError):list(records(path))

    def test_size_cap_and_bad_schema_do_not_silently_truncate(self):
        path=self.root/'large.json';path.write_text(json.dumps([{'x':'a'*200000}]))
        with self.assertRaises(ValueError):list(records(path,max_record_chars=100000))
        with self.assertRaises(ValueError):normalize({'bundle':[{}]},'x','x',0)

    def test_exact_dedup_preserves_distinct_trajectories_for_same_question(self):
        dedup=Dedup(self.root/'d.sqlite',self.config['dedup']);item=self.sample()
        self.assertIsNone(dedup.add(item)['exact_duplicate_of'])
        self.assertEqual(dedup.add(copy.deepcopy(item))['exact_duplicate_of'],item['id'])
        other=copy.deepcopy(item);other['steps'][-1]['assistant']='A different attempt.'
        other=normalize(other,'synthetic','fixture',1);other['id']='different'
        outcome=dedup.add(other)
        self.assertIsNone(outcome['exact_duplicate_of']);self.assertIn(item['id'],outcome['near_duplicates'])
        dedup.close()

    def test_harness_mapping_preserves_source_and_query_batches(self):
        item=self.sample();before=copy.deepcopy(item)
        plan=harness_plan(item,self.config['harness'])
        self.assertEqual(len(plan['calls']),1);self.assertEqual(plan['calls'][0]['name'],'web_search')
        self.assertEqual(plan['calls'][0]['arguments']['query'],['X','Y'])
        self.assertFalse(plan['replayed']);self.assertEqual(before,item)
        item['steps'][0]['calls']=[{'name':'visit','arguments':{'url':['https://example.org/'+str(i) for i in range(7)],'goal':'Find a fact'}}]
        plan=harness_plan(item,self.config['harness'])
        self.assertEqual(plan['calls'],[])
        self.assertIn('extract_url_batch_limit_requires_explicit_split',plan['errors'])
        self.assertIn('goal_conditioned_extraction_requires_replay',plan['flags'])
        self.assertNotIn('extraction_semantics_require_replay',plan['flags'])
        self.assertTrue(all(c['arguments']['goal']=='Find a fact' for c in plan['calls']))

    def test_unsupported_tools_and_invalid_limits_are_quarantined(self):
        item=self.sample();item['steps'][0]['calls']=[{'name':'python','arguments':{}}]
        self.assertIn('unsupported_tool:python',harness_plan(item,self.config['harness'])['errors'])
        item['steps'][0]['calls']=[{'name':'web_search','arguments':{'query':'q','limit':0}}]
        self.assertIn('invalid_search_limit',harness_plan(item,self.config['harness'])['errors'])

    def test_generated_evidence_must_exist_and_counterfactual_is_minimal(self):
        item=self.sample()
        valid={'original_span':'2042','replacement':'2043','evidence':[{'step_id':1,'quote':'Object X was released in 2042.'}]}
        self.assertEqual(p.validate_counterfactual(valid,item),valid)
        with self.assertRaises(ValueError):p.validate_counterfactual({**valid,'original_span':'absent'},item)
        with self.assertRaises(ValueError):p.validate_rewrite({'final_answer':'New','evidence':[{'step_id':1,'quote':'Invented'}]},item)
        with self.assertRaises(ValueError):p.validate_negative_review({'verdict':'contradicted','reason':'memory','evidence':[]},item)

    def test_counterfactual_whitespace_mapping_preserves_raw_offsets(self):
        item=self.sample();item['steps'][-1]['assistant']='**Example\u202fEntity**'
        value={'original_span':'Example Entity','replacement':'Different Entity',
               'evidence':[{'step_id':1,'quote':'Object X was released in 2042.'}]}
        checked=p.validate_counterfactual(value,item)
        location=checked['edit_location'];original=p.policy.final_text(item)
        self.assertEqual(original[:location['start']]+checked['replacement']+original[location['end']:],'**Different Entity**')
        self.assertEqual(location['match'],'unicode_whitespace')
        item['steps'][-1]['assistant']='Example Entity and Example\u202fEntity'
        with self.assertRaises(ValueError):p.validate_counterfactual(value,item)

    def test_actual_human_and_replay_are_required(self):
        sample=self.sample();result={'offline_quality_candidate':True}
        path=self.root/'result.json';save(path,{'sample':sample,'result':result})
        decision=p.review_entry(sample,result,path)
        decisions=self.root/'decisions.json';replay=self.root/'replay.json';save(replay,[])
        save(decisions,[decision]);self.assertEqual(p.finalize(self.root,decisions,replay)['accepted'],0)
        decision.update(decision='accept',reviewer='human-reviewer',reason='Checked source excerpts')
        save(decisions,[decision]);self.assertEqual(p.finalize(self.root,decisions,replay)['accepted'],0)
        decision['sample_hash']='stale';save(decisions,[decision])
        self.assertEqual(p.finalize(self.root,decisions,replay)['accepted'],0)

    def test_synthetic_negatives_never_export_as_positive_sft(self):
        sample=self.sample();sample['lineage']={'operation':'counterfactual'};result={'offline_quality_candidate':True}
        path=self.root/'result.json';save(path,{'sample':sample,'result':result})
        decision=p.review_entry(sample,result,path);decision.update(decision='accept',reviewer='human',reason='negative')
        decisions=self.root/'decisions.json';replay=self.root/'replay.json';save(decisions,[decision]);save(replay,[])
        self.assertEqual(p.finalize(self.root,decisions,replay)['accepted'],0)

    def test_config_redacts_key(self):
        self.config['providers']['default']['key']='test-secret'
        self.assertNotIn('test-secret',json.dumps(public_config(self.config)))
        self.assertEqual(self.config['providers']['default']['key'],'test-secret')

    def test_ambiguous_model_discovery_requires_explicit_selection(self):
        client=Client(self.config,self.root,'policy')
        client.request=lambda *args,**kwargs:{'data':[{'id':'local-a'},{'id':'local-b'}]}
        with self.assertRaises(ValueError):client.endpoint('inventory')

    def test_nested_bundles_are_not_silently_flattened(self):
        with self.assertRaises(ValueError):normalize({'subagents':[{'messages':[]}],'messages':[]},'bundle','fixture',0)

    def test_local_model_auto_discovery_and_stage_routing_cache(self):
        seen=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_GET(self):
                self.send_response(200);self.end_headers();self.wfile.write(b'{"data":[{"id":"local-model"}]}')
            def do_POST(self):
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append(payload)
                self.send_response(200);self.end_headers()
                self.wfile.write(json.dumps({'model':'local-model','choices':[{'finish_reason':'stop','message':{'content':'{"ok": true}'}}]}).encode())
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            # Only these two provider fields change; no code or model-name edits.
            self.config['providers']['default']['url']=f'http://127.0.0.1:{server.server_port}/v1'
            self.config['providers']['default']['key']=''
            client=Client(self.config,self.root,'policy')
            self.assertTrue(client.call('inventory','x','prompt','system',lambda x:x)['ok'])
            self.assertTrue(client.call('inventory','x','prompt','system',lambda x:x)['ok'])
            self.assertEqual(len(seen),1);self.assertEqual(seen[0]['model'],'local-model')
            self.assertNotIn('thinking',seen[0])
            client.call('inventory','x','different prompt','system',lambda x:x)
            self.assertEqual(len(seen),2)
            client2=Client(self.config,self.root,'new-policy')
            client2.call('inventory','x','different prompt','system',lambda x:x)
            self.assertEqual(len(seen),3)  # Policy changes invalidate model decisions, not just parsing.
            self.config['providers']['second']={**self.config['providers']['default'],'model':'local-reviewer'}
            self.config['stages']['review']={'provider':'second','generation':{'max_tokens':32768}}
            client2.call('review','x','review prompt','system',lambda x:x)
            self.assertEqual(seen[-1]['model'],'local-reviewer')
            self.assertEqual(seen[-1]['max_tokens'],32768)
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
