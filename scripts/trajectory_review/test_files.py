"""Source browsing is restricted to prepared snapshots, never arbitrary paths."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi import HTTPException
from trajectory_review.core import connect
from trajectory_review import api


class SourceFilesTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'test.sqlite'
        db=connect(self.path)
        payload={'question':'prefix '*100+'unique-search-tail','messages':[]}
        db.execute('INSERT INTO cases VALUES(?,?,?,?,?,?,?)',('case1','Example','hash','policy',json.dumps(payload),'{}',json.dumps({'original_id':'original1'})))
        db.execute('CREATE TABLE case_files(case_id TEXT,kind TEXT,metadata TEXT,content TEXT,PRIMARY KEY(case_id,kind))')
        self.content=json.dumps({'messages':['<img src=x onerror=alert(1)>','a'*35000]})
        db.execute('INSERT INTO case_files VALUES(?,?,?,?)',('case1','dataset',json.dumps({'path':'/dataset/example.jsonl','row_index':12}),self.content))
        db.commit();db.close()
        self.mock=patch.object(api,'database',side_effect=lambda:connect(self.path));self.mock.start()

    def tearDown(self):self.mock.stop();self.temp.cleanup()

    def test_search_includes_full_question_and_original_id_without_program_labels(self):
        row=api.cases()['cases'][0]
        self.assertTrue(row['question'].endswith('unique-search-tail'))
        self.assertEqual(row['original_id'],'original1');self.assertNotIn('automatic',row)

    def test_chunking_and_download_preserve_full_source(self):
        a=api.source_file('case1','dataset');b=api.source_file('case1','dataset',a['next_offset'])
        self.assertEqual(a['content']+b['content'],self.content);self.assertIsNone(b['next_offset'])
        r=api.source_file('case1','dataset',download=True)
        self.assertEqual(r.body.decode(),self.content);self.assertIn('attachment',r.headers['content-disposition'])
        self.assertEqual(r.headers['cache-control'],'no-store')

    def test_invalid_paths_and_missing_records_are_rejected(self):
        for ident,kind,offset,status in [('case1','../../etc/passwd',0,400),('missing','dataset',0,404),('case1','dataset',-1,400)]:
            with self.assertRaises(HTTPException) as exc:api.source_file(ident,kind,offset)
            self.assertEqual(exc.exception.status_code,status)

    def test_full_message_search_uses_literal_case_insensitive_unicode_offsets(self):
        text='😀'+'x'*31000+' NEEDLE <tag> needle .*'
        db=connect(self.path)
        db.execute('UPDATE cases SET payload=? WHERE id=?',(json.dumps({'question':'q','messages':[{'content':text}]}),'case1'))
        db.commit();db.close()
        result=api.search_messages('case1',{'query':'needle'})
        self.assertEqual(result['total'],2);self.assertEqual(result['message_count'],1)
        self.assertEqual(result['hits'][0]['start'],31003)
        self.assertEqual(api.search_messages('case1',{'query':'.*'})['total'],1)
        self.assertEqual(api.search_messages('case1',{'query':'no-such-word'})['hits'],[])

    def test_search_validation_and_navigation_limit(self):
        db=connect(self.path)
        db.execute('UPDATE cases SET payload=? WHERE id=?',(json.dumps({'messages':[{'content':'a'*2100}]}),'case1'))
        db.commit();db.close()
        result=api.search_messages('case1',{'query':'A'})
        self.assertEqual(result['total'],2100);self.assertEqual(len(result['hits']),2000);self.assertTrue(result['limited'])
        for value in ('', 'x'*257, None):
            with self.assertRaises(HTTPException):api.search_messages('case1',{'query':value})

    def test_source_search_covers_unloaded_content_and_both_versions(self):
        db=connect(self.path)
        db.execute('UPDATE case_files SET content=?',('😀'+'x'*29998+'aBcD'+'y'*10000+' abcd .*',))
        db.execute('INSERT INTO case_files VALUES(?,?,?,?)',('case1','candidate','{}','other version'))
        db.commit();db.close()
        result=api.search_source_file('case1','dataset',{'query':'ABCD'})
        self.assertEqual(result['total'],2)
        self.assertEqual(result['hits'][0],{'start':30000,'end':30004})
        self.assertEqual(api.search_source_file('case1','dataset',{'query':'.*'})['total'],1)
        self.assertEqual(api.search_source_file('case1','candidate',{'query':'abcd'})['total'],0)
        self.assertEqual(api.search_source_file('case1','candidate',{'query':'VERSION'})['total'],1)

    def test_source_search_rejects_invalid_access_and_limits_hits(self):
        for ident,kind,query,status in [('case1','../../etc/passwd','a',400),('missing','dataset','a',404),('case1','dataset','',422)]:
            with self.assertRaises(HTTPException) as exc:api.search_source_file(ident,kind,{'query':query})
            self.assertEqual(exc.exception.status_code,status)
        result=api.search_source_file('case1','dataset',{'query':'a'})
        self.assertGreater(result['total'],2000);self.assertEqual(len(result['hits']),2000);self.assertTrue(result['limited'])

    def test_readable_search_chunks_and_original_json_stay_consistent(self):
        content='😀'+'x'*31000+' actual\nline <tool_response>'
        original=json.dumps({'messages':[{'role':'user','content':content}]})
        db=connect(self.path);db.execute('UPDATE case_files SET content=?',(original,));db.commit();db.close()
        self.assertEqual(api.readable_source('case1','dataset')['blocks'][0]['role'],'user')
        first=api.readable_block('case1','dataset',0)
        second=api.readable_block('case1','dataset',0,first['next_offset'])
        self.assertEqual(first['content']+second['content'],content)
        result=api.search_source_file('case1','dataset',{'query':'actual\nline','view':'messages'})
        self.assertEqual(result['total'],1);self.assertEqual(result['hits'][0]['block_index'],0)
        self.assertEqual(api.source_file('case1','dataset',download=True).body.decode(),original)


if __name__=='__main__':unittest.main()
