"""Stable wall-clock origin for both live and completed trajectory exports."""
from datetime import datetime
import math
import sqlite3


def restore_session_origin(session, db_path):
    """Repair old resume snapshots using only this conversation's persisted IDs.

    Keep pauses in elapsed wall time. Never use unrelated delegate/retry rows,
    or infer a duration for an interrupted model/tool call.
    """
    candidates = []
    try:
        candidates.append(datetime.fromisoformat(session['session_start']).timestamp())
    except (KeyError, TypeError, ValueError):
        pass
    ids = set(session.get('_viewer_session_ids', []))
    ids.update(m.get('_viewer_session_id') for m in session.get('messages', []))
    ids.add(session.get('session_id'))
    ids.discard(None)
    ids.discard('')
    if ids and db_path.is_file():
        try:
            with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as db:
                for sid in ids:
                    row = db.execute('SELECT started_at FROM sessions WHERE id=?', (sid,)).fetchone()
                    if row and row[0] is not None:
                        candidates.append(float(row[0]))
        except (sqlite3.Error, TypeError, ValueError):
            # Older exports may have messages only, with no sessions table.
            pass
    candidates = [x for x in candidates if math.isfinite(x) and x > 0]
    if candidates:
        origin = min(candidates)
        session['session_start'] = datetime.fromtimestamp(origin).isoformat()
        session['_viewer_timing'] = {
            'origin': datetime.fromtimestamp(origin).astimezone().isoformat(),
            'clock': 'wall', 'includesPauses': True,
        }


def apply_message_timings(session, steps, db_path):
    """Read producer-neutral Hermes timing records; no inference for absent anchors."""
    import hashlib,json
    if not db_path.is_file():return False
    try:
        origin=datetime.fromisoformat(session['session_start']).timestamp()
        with sqlite3.connect(f'file:{db_path}?mode=ro',uri=True) as db:
            rows=db.execute('SELECT m.id,m.session_id,m.role,m.tool_call_id,m.content,m.tool_calls,t.phase,t.duration_seconds,t.started_at,t.ended_at,t.provenance FROM messages m JOIN message_timings t ON m.id=t.message_id WHERE COALESCE(m.inherited_from_previous_session,0)=0').fetchall()
        for step in steps:
            sid,cid=step['sourceSessionId'],step['toolCallId']
            for phase in ('model','tool'):
                matches=[]
                for row in rows:
                    mid,rsid,role,call_id,content,calls,rphase,duration,start,end,provenance=row
                    if rsid!=sid or rphase!=phase:continue
                    ids=[x.get('id') or x.get('call_id') for x in json.loads(calls or '[]')]
                    match=(cid in ids if cid else not ids and hashlib.sha256((content or '').encode()).hexdigest()==step['sourceContentHash']) if phase=='model' else bool(cid and call_id==cid)
                    if match:matches.append(row)
                if len(matches)!=1:continue
                row=matches[0];duration,start,end,provenance=row[7:]
                step[phase+'_seconds']=duration
                step[phase+'TimingProvenance']=provenance
                if start is not None and end is not None:
                    step[phase+'Start']=max(0,start-origin);step[phase+'End']=max(0,end-origin)
        return any(s.get('modelStart') is not None or s.get('toolStart') is not None for s in steps)
    except (OSError,sqlite3.Error,KeyError,TypeError,ValueError):return False


def request_context_stats(session, db_path):
    """Latest saved request, never the post-response transcript or toolset guess."""
    import json
    ids=set(session.get('_viewer_session_ids',[]));ids.add(session.get('session_id'));ids.discard(None)
    if not ids or not db_path.is_file():return {}
    try:
        with sqlite3.connect(f'file:{db_path}?mode=ro',uri=True) as db:
            placeholders=','.join('?' for _ in ids)
            row=db.execute(f'SELECT context_json,provenance FROM request_contexts WHERE session_id IN ({placeholders}) ORDER BY id DESC LIMIT 1',list(ids)).fetchone()
        if not row:return {}
        context=json.loads(row[0]);context['provenance']=row[1]
        return context
    except (OSError,sqlite3.Error,ValueError):return {}


def producer_alive(session):
    """Common local producer status, guarded against stale/reused PIDs."""
    from pathlib import Path
    producer=session.get('producer') or {}
    if session.get('session_end') or not producer.get('pid'):return False
    try:
        stat=Path(f"/proc/{int(producer['pid'])}/stat").read_text().split(') ',1)[1].split()
        return stat[0]!='Z' and stat[19]==str(producer['start_ticks'])
    except (OSError,ValueError,KeyError,IndexError):return False
