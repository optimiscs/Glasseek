import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trajectory_review.source_view import blocks,body

class SourceViewTests(unittest.TestCase):
    def test_roles_newlines_and_literal_backslashes(self):
        content='line 1\n"quoted"\t中文 😀\nC:\\new\\test'
        items=blocks(json.dumps({'system':'rules','messages':[{'role':'assistant','content':content},{'role':'user','content':'<tool_response>\npage\n</tool_response>'}]}))
        self.assertEqual([b['role'] for b in items],['system','assistant','user'])
        self.assertEqual(items[1]['content'],content)
        self.assertEqual(body(json.dumps('one\ntwo')),'one\ntwo')
        self.assertEqual(body('C:\\new\\test'),'C:\\new\\test')
    def test_variants_children_and_extra_fields_remain_visible(self):
        raw={'trajectory':[{'role':'tool','content':'result','tool_call_id':'id1'}],
             'subagents':[{'question':'child goal','conversations':[{'from':'human','value':'child task'}]}], 'answer':'reference'}
        items=blocks(json.dumps(raw))
        self.assertTrue(any(b['role']=='human' and b['content']=='child task' for b in items))
        self.assertTrue(any('tool_call_id' in b['content'] for b in items))
        self.assertTrue(any('reference' in b['content'] for b in items))
        self.assertTrue(any(b['path'].startswith('$.subagents[0]') for b in items))
    def test_structured_text_parts_and_stringified_messages(self):
        items=blocks(json.dumps({'messages':json.dumps([{'role':'user','content':[{'type':'text','text':'a\nb'},{'type':'text','text':'c'}]}])}))
        self.assertEqual(items[0]['content'],'a\nb\n\nc')

if __name__=='__main__':unittest.main()
