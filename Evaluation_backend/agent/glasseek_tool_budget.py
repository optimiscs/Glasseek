"""Opt-in, persisted tool invocation budget shared by a question's agents."""
import fcntl,json,os,time
from pathlib import Path

def _path():
    value=os.environ.get('GLASSEEK_TOOL_CALL_BUDGET_FILE')
    return Path(value) if value else None

def initialize(path, limit, used=0):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            state=json.loads(path.read_text())
            if state['limit']!=limit:raise ValueError('Tool budget limit changed on resume')
            return state
        state={'limit':limit,'used':used,'initial_used':used,'denied':0,'reservations':[]}
        _save(path,state);return state

def _save(path,state):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state));os.replace(tmp,path)

def reserve(count):
    path=_path()
    if path is None:return count
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(path.read_text());allowed=min(count,max(0,state['limit']-state['used']))
        state['used']+=allowed;state['denied']+=count-allowed
        state['reservations'].append({'at':time.time(),'pid':os.getpid(),'admitted':allowed,'denied':count-allowed})
        _save(path,state);return allowed

def exhausted():
    path=_path()
    if path is None:return False
    state=json.loads(path.read_text());return state['used']>=state['limit']
