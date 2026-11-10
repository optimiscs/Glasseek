import ast
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'Shared'),str(ROOT/'Evaluation/services'),str(ROOT/'RL_training')]
from corpus import Corpus


class FixedCorpusTests(unittest.TestCase):
    def test_search_fetch_and_logged_docids_never_use_the_live_web(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);index=p/'index.sqlite'
            with sqlite3.connect(index) as db:
                db.execute('CREATE VIRTUAL TABLE documents USING fts5(docid UNINDEXED,url UNINDEXED,text)')
                db.execute('CREATE TABLE lookup(docid,url,rowid)')
                db.execute('INSERT INTO documents VALUES (?,?,?)',('d1','https://offline.test/page','Coastal ecology research institute.'))
                db.execute('INSERT INTO lookup VALUES (?,?,?)',('d1','https://offline.test/page',1))
            corpus=Corpus({'index':str(index),'docid_events':str(p/'events.jsonl')})
            hits=corpus.search('coastal ecology');self.assertEqual(hits[0]['docid'],'d1')
            self.assertEqual(corpus.fetch(hits[0]['link'])[0][0],'d1')
            with self.assertRaises(ValueError):corpus.fetch('https://not-in-the-corpus.test')
            corpus.record({'opaque_id':'t1','namespace':'run'},'web_search',['d1'])
            self.assertEqual(json.loads((p/'events.jsonl').read_text())['retrieved_docids'],['d1'])

    def test_bash_network_denied_but_unix_web_bridge_and_file_edits_work(self):
        from glasseek_shared.context.official.web import WebBridge
        from glasseek_shared.hermes.url_registry import URLRegistry
        class Tools:
            url_mode='raw';url_registry=URLRegistry('raw')
            def begin_turn(self):pass
            async def execute(self,call):return 'fixed-corpus-evidence'
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'GLASSEEK_FIXED_CORPUS_CONFIG':'offline-test'}):
            bridge=WebBridge(d,Tools())
            try:
                tree=ast.parse((ROOT/'Shared/glasseek_shared/context/official/run.py').read_text())
                node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='helper_template' for t in n.targets))
                helper=Path(d)/'web.py';helper.write_text(ast.literal_eval(node.value).replace('GATEWAY',bridge.url).replace('SESSION','test'))
                sandbox=ROOT/'Shared/glasseek_shared/fixed_corpus_sandbox.py'
                result=subprocess.run([sys.executable,str(sandbox),sys.executable,str(helper),'search','ecology'],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr);self.assertIn('fixed-corpus-evidence',result.stdout)
                script='import socket,pathlib;pathlib.Path("note").write_text("context edit");socket.socket(socket.AF_INET,socket.SOCK_STREAM)'
                result=subprocess.run([sys.executable,str(sandbox),sys.executable,'-c',script],cwd=d,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0);self.assertIn('PermissionError',result.stderr)
                self.assertEqual((Path(d)/'note').read_text(),'context edit')
            finally:bridge.close()


if __name__=='__main__':unittest.main()
