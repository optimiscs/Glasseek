"""Read-only live view of Hermes profiles; bind to loopback by default."""
import html
import json
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path('/root/autodl-tmp/Glasseek/Evaluation/data/runtime/hermes/profiles')
PREFIX = 'glasseek_base9b_rl9b_subset_'
PAGE = '''<!doctype html><meta charset="utf-8"><title>BrowseComp 实时轨迹</title>
<style>body{font:15px system-ui;margin:1.5rem;color:#222}main{display:flex;gap:2rem}nav{width:24%;max-height:85vh;overflow:auto}article{flex:1;max-height:85vh;overflow:auto;white-space:pre-wrap}.msg{padding:1em;border-bottom:1px solid #ddd}button{display:block;margin:.4em 0}small{color:#666}</style>
<h2>Qwen3.5-9B 基座 · 实时轨迹</h2><small>每 10 秒刷新；可能包含敏感题目和工具输出</small>
<main><nav id="cases"></nav><article id="trace">请选择题目</article></main>
<script>
let chosen=''; async function update(){let r=await fetch('api/cases');let a=await r.json();let n=document.getElementById('cases');n.replaceChildren();for(let x of a){let b=document.createElement('button');b.textContent=x.name+' · '+x.messages+' 条消息';b.onclick=()=>{chosen=x.name;update()};n.appendChild(b)}if(chosen){let r=await fetch('api/case?profile='+encodeURIComponent(chosen));let a=await r.json(), el=document.getElementById('trace');el.replaceChildren();for(let m of a){let d=document.createElement('div');d.className='msg';d.textContent='【'+m.role+(m.tool_name?' / '+m.tool_name:'')+'】\\n'+(m.content||'')+(m.tool_calls?'\\n调用: '+m.tool_calls:'');el.appendChild(d)}}} update();setInterval(update,10000);
</script>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlsplit(self.path)
        if url.path in ('/', '/index.html'):
            self.reply(PAGE.encode(), 'text/html; charset=utf-8')
        elif url.path == '/api/cases':
            items = []
            for db in ROOT.glob(PREFIX + '*/state.db'):
                try:
                    with sqlite3.connect(f'{db.as_uri()}?mode=ro', uri=True, timeout=1) as con:
                        count = con.execute('SELECT COUNT(*) FROM messages').fetchone()[0]
                    items.append({'name': db.parent.name, 'messages': count})
                except sqlite3.Error:
                    continue
            self.reply(json.dumps(sorted(items, key=lambda x: x['name'])).encode(), 'application/json')
        elif url.path == '/api/case':
            name = parse_qs(url.query).get('profile', [''])[0]
            if not name.startswith(PREFIX) or Path(name).name != name:
                self.send_error(400); return
            db = ROOT / name / 'state.db'
            try:
                with sqlite3.connect(f'{db.as_uri()}?mode=ro', uri=True, timeout=2) as con:
                    rows = con.execute('SELECT role,content,tool_name,tool_calls FROM messages ORDER BY id').fetchall()
                result = [dict(zip(('role','content','tool_name','tool_calls'), row)) for row in rows]
                self.reply(json.dumps(result, ensure_ascii=False, default=str).encode(), 'application/json; charset=utf-8')
            except sqlite3.Error:
                self.send_error(404)
        else:
            self.send_error(404)

    def reply(self, data, content_type):
        self.send_response(200); self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store'); self.send_header('Content-Length', str(len(data)))
        self.end_headers(); self.wfile.write(data)


if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', 8765), Handler).serve_forever()
