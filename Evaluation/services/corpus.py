"""Build and serve the pinned BC+ corpus with an explicit SQLite FTS5 BM25 retriever."""
import argparse
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
from urllib.parse import urlsplit
import urllib.request

REVISION='b27b02bc3e45511b8b82a13e6f90ce761df726f6'
EXPECTED_ROWS=100195


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def build(source, destination):
    import pyarrow.parquet as pq
    source=Path(source);destination=Path(destination)
    manifest=destination.with_suffix('.manifest.json')
    if destination.exists():
        value=json.loads(manifest.read_text())
        if value['revision']!=REVISION or value['rows']!=EXPECTED_ROWS:raise ValueError('corpus index metadata mismatch')
        return value
    files=sorted(source.glob('data/*.parquet'))
    if len(files)!=7:raise ValueError('complete pinned corpus requires all 7 parquet shards')
    acquisition=json.loads((source/'download.manifest.json').read_text())
    if acquisition['revision']!=REVISION:raise ValueError('corpus acquisition revision mismatch')
    for path in files:
        if sha(path)!=acquisition['files'][str(path.relative_to(source))]['sha256']:raise ValueError('corpus shard hash mismatch')
    destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_suffix('.building.sqlite')
    if temporary.exists():temporary.unlink()
    with sqlite3.connect(temporary) as db:
        db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF');db.execute('PRAGMA cache_size=-262144')
        db.execute("CREATE VIRTUAL TABLE documents USING fts5(docid UNINDEXED,url UNINDEXED,text,tokenize='porter unicode61')")
        db.execute('CREATE TABLE lookup(docid TEXT PRIMARY KEY,url TEXT,rowid INTEGER)')
        db.execute('CREATE INDEX lookup_url ON lookup(url)');count=0
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256,columns=['docid','url','text']):
                for row in batch.to_pylist():
                    cur=db.execute('INSERT INTO documents(docid,url,text) VALUES (?,?,?)',(row['docid'],row['url'],row['text']))
                    db.execute('INSERT INTO lookup VALUES (?,?,?)',(row['docid'],row['url'],cur.lastrowid));count+=1
                db.commit()
            print(json.dumps({'stage':'index','file':path.name,'rows':count}),flush=True)
        if count!=EXPECTED_ROWS:raise ValueError(f'incomplete corpus: {count} rows')
        db.execute("INSERT INTO documents(documents) VALUES ('optimize')")
    value={'revision':REVISION,'rows':count,'retriever':'SQLite FTS5 porter/unicode61 BM25, OR query terms',
           'source_files':{p.name:sha(p) for p in files},'index_sha256':sha(temporary)}
    temporary.replace(destination);manifest.write_text(json.dumps(value,indent=2));return value


class Corpus:
    def __init__(self,config):self.config=config
    def connection(self):return sqlite3.connect(f'file:{self.config["index"]}?mode=ro',uri=True,timeout=30)
    def search(self,query,limit=10):
        tokens=re.findall(r'\w+',str(query),flags=re.UNICODE)[:64]
        if not tokens:return []
        expression=' OR '.join('"'+word.replace('"','""')+'"' for word in tokens)
        with self.connection() as db:
            rows=db.execute("SELECT docid,url,snippet(documents,2,'','',' … ',64),bm25(documents) FROM documents WHERE documents MATCH ? ORDER BY bm25(documents) LIMIT ?",(expression,max(1,min(100,int(limit))))).fetchall()
        return [{'docid':d,'link':u,'title':u,'snippet':text,'score':-score} for d,u,text,score in rows]
    def fetch(self,url):
        with self.connection() as db:
            rows=db.execute('SELECT d.docid,d.text FROM lookup l JOIN documents d ON d.rowid=l.rowid WHERE l.url=? ORDER BY l.docid',(url,)).fetchall()
        if not rows:raise ValueError('URL is outside the fixed corpus; live-web fallback is disabled')
        return rows
    def record(self,body,name,docids,count=1):
        if not body.get('opaque_id') or not body.get('namespace'):raise ValueError('missing per-task retrieval identity')
        path=Path(self.config['docid_events']);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            f.write(json.dumps({'opaque_id':body['opaque_id'],'namespace':body['namespace'],
                'retrieved_docids':docids,'tool_call_counts':{name:count},'corpus_revision':REVISION})+'\n')


def serve(config):
    manifest=json.loads(Path(config['index']).with_suffix('.manifest.json').read_text())
    if manifest['revision']!=config['corpus_revision'] or manifest['rows']!=EXPECTED_ROWS:raise ValueError('corpus provenance mismatch')
    store=Corpus(config);address=urlsplit(config['fetch_endpoint'])
    events=Path(config['docid_events']);events.parent.mkdir(parents=True,exist_ok=True);events.touch(exist_ok=True)
    class Handler(BaseHTTPRequestHandler):
        def send(self,value,code=200):
            data=json.dumps(value,ensure_ascii=False).encode();self.send_response(code)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            self.send({'status':'ok','corpus_revision':REVISION,'rows':manifest['rows'],'live_web_disabled':True} if self.path=='/health' else {'error':'not found'},200 if self.path=='/health' else 404)
        def do_POST(self):
            try:
                body=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
                if self.path=='/search':
                    store.record(body,'web_search',[])
                    results=store.search(body['q'],body.get('num',10));store.record(body,'web_search',[r['docid'] for r in results],0)
                    self.send({'success':True,'organic':results})
                elif self.path=='/fetch':
                    store.record(body,'web_extract',[])
                    rows=store.fetch(body['url']);store.record(body,'web_extract',[r[0] for r in rows],0)
                    self.send({'success':True,'data':'\n\n'.join(f'[docid:{d}]\n{text}' for d,text in rows)})
                else:self.send({'success':False,'error':'not found'},404)
            except (ValueError,KeyError) as exc:self.send({'success':False,'error':str(exc)},400)
            except sqlite3.Error:self.send({'success':False,'error':'fixed corpus database failure; inspect service log'},503)
        def log_message(self,*args):pass
    ThreadingHTTPServer((address.hostname,address.port),Handler).serve_forever()


def ensure(path,python=None):
    path=Path(path);config=json.loads(path.read_text());address=config['fetch_endpoint'].rstrip('/')
    if not config.get('managed_service'):return
    if not Path(config['index']).is_file():raise ValueError('fixed corpus index missing; build it before starting inference')
    lock=Path(config['index']).with_suffix('.service.lock')
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        def health():
            try:
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(address+'/health',timeout=1) as response:value=json.load(response)
                if value.get('corpus_revision')!=config['corpus_revision']:raise ValueError('port is occupied by another service')
                return True
            except OSError:return False
        if health():return
        with lock.with_suffix('.log').open('a') as log:
            proc=subprocess.Popen([python or sys.executable,str(Path(__file__).resolve()),'serve','--config',str(path)],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(50):
            if health():return
            if proc.poll() is not None:break
            time.sleep(.1)
        raise RuntimeError('fixed corpus service did not start; inspect '+str(lock.with_suffix('.log')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','serve'])
    p.add_argument('--source',type=Path);p.add_argument('--index',type=Path);p.add_argument('--config',type=Path)
    a=p.parse_args()
    if a.action=='build':print(json.dumps(build(a.source,a.index)),flush=True)
    else:serve(json.loads(a.config.read_text()))
