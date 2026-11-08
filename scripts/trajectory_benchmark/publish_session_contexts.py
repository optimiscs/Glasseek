"""Refresh context metadata for every viewer session from canonical Hermes DBs."""
import json,sqlite3,sys,hashlib
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2];QUEST=ROOT.parent/'Quest';sys.path.insert(0,str(QUEST/'web/backend/src'))
from quest_viewer.config import Settings
from quest_viewer.db.session import create_session_factory
from sqlalchemy import text
contexts={};ambiguous=set()
for path in (ROOT/'Evaluation/data/runtime/hermes/profiles').glob('*/state.db'):
 try:
  with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as db:
   rows=db.execute('SELECT session_id,context_json FROM request_contexts ORDER BY id').fetchall()
  latest=dict(rows)
  for sid,value in latest.items():
   obj=json.loads(value)
   if sid in contexts and contexts[sid]!=obj:ambiguous.add(sid)
   else:contexts[sid]=obj
 except (sqlite3.Error,OSError,ValueError):pass
for sid in ambiguous:contexts.pop(sid,None)
engine,factory=create_session_factory(Settings());updated=0;scanned=0;runs=set()
with factory() as reader:
 query=text('SELECT run_id,trace_id,body::text AS payload FROM trace_payloads ORDER BY run_id,trace_id')
 for row in reader.execute(query,execution_options={'stream_results':True,'yield_per':10}):
  scanned+=1
  body=json.loads(row.payload)
  ids=[x.get('sourceSessionId') for x in body.get('steps',[])]
  selected=next((contexts[sid] for sid in reversed(ids) if sid in contexts),None)
  previous=(body.get('sessionStats') or {}).get('context') or {}
  if not selected or not (any(previous.get(k)!=v for k,v in selected.items()) or any(k in previous and k not in selected for k in ('systemTokens','toolTokens','messageTokens'))):continue
  target=body.setdefault('sessionStats',{}).setdefault('context',{})
  for key in ('systemTokens','toolTokens','messageTokens'):
   if key not in selected:target.pop(key,None)
  target.update(selected)
  # JSON permits escaped NULs in legacy traces; casting the whole body to JSONB
  # would reject them. Preserve their exact JSON representation instead.
  encoded=json.dumps(body,ensure_ascii=True,separators=(',',':'))
  revision=hashlib.sha256(encoded.encode()).hexdigest()
  with factory.begin() as db:
   db.execute(text('SELECT pg_advisory_xact_lock(hashtext(:run))'),{'run':row.run_id})
   db.execute(text('UPDATE trace_payloads SET body=CAST(:body AS json) WHERE run_id=:run AND trace_id=:trace'),{'body':encoded,'run':row.run_id,'trace':row.trace_id})
   db.execute(text("UPDATE traces SET revision=CAST(:revision AS text),summary=jsonb_set(summary::jsonb,'{revision}',to_jsonb(CAST(:revision AS text))) WHERE run_id=:run AND id=:trace"),{'revision':revision,'run':row.run_id,'trace':row.trace_id})
   db.execute(text('UPDATE runs SET revision=:revision WHERE id=:run'),{'revision':revision,'run':row.run_id})
  updated+=1;runs.add(row.run_id)
  if updated%100==0:print('updated',updated,'scanned',scanned,flush=True)
engine.dispose()
result={'native_sessions_available':len(contexts),'ambiguous_sessions_skipped':len(ambiguous),'viewer_runs_updated':len(runs),'viewer_traces_updated':updated,'scanned':scanned}
(ROOT/'reports/clm-hermes-schema-20261108/viewer-context-refresh.json').write_text(json.dumps(result,indent=2));print(result)
