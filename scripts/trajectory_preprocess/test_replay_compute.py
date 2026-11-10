import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from replay_compute import run


class ComputeReplayTests(unittest.TestCase):
    def test_success_is_not_output_agreement_and_cache_is_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input.jsonl'
            def item(ident, observed):
                return {'id':ident,'training_ready':False,'harness_replayed':False,'messages':[
                    {'role':'assistant','call_links':[{'call_id':'c','name':'execute_code','arguments':{'code':'print(2)'}}]},
                    {'role':'user','response_to':'c','content':'<tool_response>\nTool: execute_code\nResult: '+observed+'\n</tool_response>'}]}
            source.write_text('\n'.join(json.dumps(item(i,v)) for i,v in [('a','2'),('b','3')])+'\n')
            with patch('replay_compute.execute',return_value=json.dumps({'status':'success','output':'2\n','cleanup_verified':True})) as execute:
                run(source,root/'replay',1)
                self.assertEqual(execute.call_count,1)
                run(source,root/'resumed',1,reuse=root/'replay')
                self.assertEqual(execute.call_count,1)
                run(source,root/'replay',1)
                self.assertEqual(execute.call_count,1)
            released=[json.loads(l) for l in (root/'replay/recovered.jsonl').open()]
            self.assertEqual([r['id'] for r in released],['a'])
            self.assertFalse(released[0]['harness_replayed'])
            self.assertFalse(released[0]['training_ready'])
            source.write_text(source.read_text().replace('"a"','"changed-id"'))
            with self.assertRaisesRegex(ValueError,'Replay inputs changed'):
                run(source,root/'replay',1)
            with self.assertRaisesRegex(ValueError,'Cached runtime/input mismatch'):
                run(source,root/'invalid-reuse',1,reuse=root/'replay')

    def test_a_mismatch_skips_later_calls_without_claiming_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input.jsonl'
            item={'id':'a','messages':[]}
            for i in (1,2):
                item['messages'].extend([
                    {'call_links':[{'call_id':str(i),'name':'execute_code','arguments':{'code':f'print({i})'}}]},
                    {'response_to':str(i),'content':'<tool_response>different</tool_response>'}])
            source.write_text(json.dumps(item)+'\n')
            with patch('replay_compute.execute',return_value=json.dumps({'status':'success','output':'1','cleanup_verified':True})) as execute:
                run(source,root/'replay',1)
                self.assertEqual(execute.call_count,1)
            summary=json.loads((root/'replay/summary.json').read_text())
            self.assertEqual(summary['held_output'],1)
            self.assertEqual(summary['held_execution'],0)
            self.assertEqual(summary['exact_output_passed'],0)
            self.assertEqual((root/'replay/recovered.jsonl').read_text(),'')
