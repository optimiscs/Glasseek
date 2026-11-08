"""Backfill shared Hermes context accounting from saved requests where available."""
import sys,json,sqlite3,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'Evaluation_backend'),str(ROOT/'Shared')]
from hermes_state import SessionDB
from glasseek_shared.context.official import activate
activate()
from clm_harness.utils.tool_schemas import BASH_TOOL

report=[];official={};dumps={}
for dump in (ROOT/'Evaluation/data/runtime/hermes/sessions').rglob('request_dump*.json'):
    try:
        data=json.loads(dump.read_text());sid=data.get('session_id');request=data.get('request',{});body=request.get('body',request)
        if isinstance(body,str):body=json.loads(body)
        if sid and isinstance(body.get('messages'),list):
            if sid not in dumps or dump.stat().st_mtime>dumps[sid][0]:dumps[sid]=(dump.stat().st_mtime,body)
    except (OSError,ValueError,TypeError,AttributeError):pass
for path in (ROOT/'Evaluation/output/preds').glob('clm*/**/official-clm-request.json'):
 try:
  cfg=json.loads(path.read_text());official[cfg['session_id'].removeprefix('clm-')]=(path.parent/'official-clm/agent',cfg)
 except (OSError,ValueError,KeyError):pass
for dbpath in (ROOT/'Evaluation/data/runtime/hermes/profiles').glob('*/state.db'):
 try:
  with sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True) as db:
   db.row_factory=sqlite3.Row;sessions=[dict(r) for r in db.execute('SELECT * FROM sessions')]
   present={r[0] for r in db.execute('SELECT DISTINCT session_id FROM request_contexts')} if db.execute("SELECT 1 FROM sqlite_master WHERE name='request_contexts'").fetchone() else set()
  latest_sid=max(sessions,key=lambda x:x['started_at'])['id'] if sessions else None
  for session in sessions:
   sid=session['id']
   if sid in present:
    # A previous interrupted backfill may need better evidence than a transcript.
    with sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True) as prior_db:
     source=prior_db.execute('SELECT provenance FROM request_contexts WHERE session_id=? ORDER BY id DESC LIMIT 1',(sid,)).fetchone()[0]
    if source in ('request','pretransport'):continue
    if source=='saved_request' and not (dbpath.parent.name in official and sid!=latest_sid):continue
    if source=='transcript_estimate' and sid not in dumps:continue
   payload=None;tools=None;provenance='transcript_estimate'
   item=official.get(dbpath.parent.name)
   if item and sid==latest_sid:
    art,cfg=item
    for snapshot in sorted((art/'context_snapshots').glob('*.json'),reverse=True):
     try:
      data=json.loads(snapshot.read_text())
      if data.get('kind')=='agent':payload=data['messages'];tools=[BASH_TOOL];provenance='saved_request';break
     except (OSError,ValueError,KeyError):pass
   if payload is None and sid in dumps:
    body=dumps[sid][1];payload=body['messages'];tools=body.get('tools');provenance='saved_failed_request'
   writer=SessionDB(dbpath)
   if payload is None:
    messages=writer.get_messages_as_conversation(sid)
    indices=[i for i,m in enumerate(messages) if m.get('role')=='assistant']
    if not indices:writer.close();continue
    payload=messages[:indices[-1]]
    if not any(m.get('role')=='system' for m in payload) and session.get('system_prompt'):
     payload=[{'role':'system','content':session['system_prompt']}]+payload
   capacity=None
   try:
    import yaml
    config=yaml.safe_load((dbpath.parent/'config.yaml').read_text()) or {};capacity=(config.get('model') or {}).get('context_length')
   except (OSError,ValueError):pass
   writer.db_path  # One canonical schema for all producers.
   writer._execute_write(lambda db:db.execute("DELETE FROM request_contexts WHERE session_id=? AND provenance NOT IN ('request','pretransport')",(sid,)))
   writer.record_request_context(sid,payload,tools,capacity_tokens=capacity,provenance=provenance);writer.close()
   report.append({'profile':dbpath.parent.name,'session_id':sid,'source':provenance,'tools_available':tools is not None})
 except (OSError,sqlite3.Error,ValueError) as e:report.append({'profile':dbpath.parent.name,'error':type(e).__name__})
output=ROOT/'reports/clm-hermes-schema-20261108/context-backfill.json';output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'backfilled':sum('session_id' in x for x in report),'with_tools':sum(x.get('tools_available',False) for x in report),'errors':sum('error' in x for x in report)}))
