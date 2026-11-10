"""Per-task bridge to existing Glasseek tools; no separate service to launch."""
import asyncio
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import sys
import os
import urllib.request
import threading
import tempfile
from socketserver import ThreadingMixIn,UnixStreamServer
from . import ROOT
from glasseek_shared.hermes.tool_errors import ToolFailure,exception_failure,http_json

class WebBridge:
    def __init__(self, workspace, tools=None):
        sys.path.insert(0,str(ROOT/'RL_training'))
        from glasseek_training.web_tools import WebTools
        from glasseek_shared.hermes.url_registry import URLRegistry
        self.tools=tools or WebTools();self.path=Path(workspace)/'.clm/web-state.json'
        # Official CLM uses the existing admitted fetch service, not direct Jina.
        defaults=json.loads((ROOT/'Evaluation/config/tool_services.json').read_text())
        fetch=os.environ.get('GLASSEEK_FETCH_URL',f"http://{defaults['host']}:{defaults['fetch_port']}").rstrip('/')
        if fetch and hasattr(self.tools,'_execute') and not os.environ.get('GLASSEEK_FIXED_CORPUS_CONFIG'):
            original_execute=self.tools._execute
            def execute(call):
                if call['name']!='web_extract':return original_execute(call)
                args=call['arguments']
                try:
                    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    request=urllib.request.Request(fetch+'/fetch',data=json.dumps({'url':args['url'],'extractMode':'markdown'}).encode(),headers={'Content-Type':'application/json'})
                    payload=http_json(opener,request,self.tools.timeout)
                    if isinstance(payload,ToolFailure):return payload
                    if payload.get('success') is not True:return ToolFailure('Fetch service did not confirm success.','fetch_invalid_response')
                    content=payload.get('data','')
                    if not isinstance(content,str) or not content.strip():return ToolFailure('Fetch service returned no page content.','fetch_empty_response')
                    goal=args.get('goal') or os.environ.get('BROWSE_COMP_QUESTION','')
                    selector=os.environ.get('GLASSEEK_QUEST_RETRIEVAL_URL',f"http://{defaults['host']}:{defaults['bridge_port']}").rstrip('/')
                    if goal and selector:
                        request=urllib.request.Request(selector+'/select',data=json.dumps({'text':content,'goal':goal}).encode(),headers={'Content-Type':'application/json'})
                        selection=http_json(opener,request,self.tools.timeout)
                        if isinstance(selection,ToolFailure):return selection
                        content=selection.get('content')
                        if not isinstance(content,str) or not content.strip():return ToolFailure('Evidence selection returned no content.','selection_empty_response')
                    return content[:self.tools.max_chars]
                except (OSError,ValueError,KeyError) as exc:return exception_failure(exc)
            self.tools._execute=execute
        self.counts_path=Path(workspace)/'.clm/web-calls.json'
        self.counts=json.loads(self.counts_path.read_text()) if self.counts_path.exists() else {}
        self.lock=threading.Lock()
        if self.path.exists():self.tools.url_registry=URLRegistry(self.tools.url_mode,json.loads(self.path.read_text()))
        bridge=self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    call=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    if call.get('name') not in ('web_search','web_extract'):raise ValueError('unsupported tool')
                    with bridge.lock:
                        name=call['name'];bridge.counts[name]=bridge.counts.get(name,0)+1
                        bridge.path.parent.mkdir(parents=True,exist_ok=True)
                        try:
                            bridge.tools.begin_turn();output=asyncio.run(bridge.tools.execute(call))
                        finally:
                            counts_temp=bridge.counts_path.with_suffix('.tmp')
                            counts_temp.write_text(json.dumps(bridge.counts));counts_temp.replace(bridge.counts_path)
                        temp=bridge.path.with_suffix('.tmp');temp.write_text(json.dumps(bridge.tools.url_registry.snapshot()));temp.replace(bridge.path)
                    body=json.dumps({'output':output}).encode();self.send_response(200)
                except Exception as e:
                    failure=exception_failure(e)
                    code=504 if failure.payload['error_code']=='tool_timeout' else 503 if failure.payload['retryable'] else 400
                    body=json.dumps({'output':str(failure),**failure.payload}).encode();self.send_response(code)
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            def log_message(self,*args):pass
        self.socket_directory=None
        if os.environ.get('GLASSEEK_FIXED_CORPUS_CONFIG'):
            class UnixServer(ThreadingMixIn,UnixStreamServer):daemon_threads=True
            self.socket_directory=tempfile.TemporaryDirectory(prefix='glasseek-corpus-')
            path=str(Path(self.socket_directory.name)/'web.sock')
            self.server=UnixServer(path,Handler);self.url='unix:'+path
        else:
            self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
            self.url=f'http://127.0.0.1:{self.server.server_port}'
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join()
        if self.socket_directory:self.socket_directory.cleanup()
