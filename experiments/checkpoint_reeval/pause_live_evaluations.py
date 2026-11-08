"""Pause/resume evaluator process trees without killing their live sessions."""
import json,os,signal,sys,time
from pathlib import Path
state=Path(os.environ.get('REEVAL_PAUSE_STATE','/root/autodl-tmp/Glasseek/checkpoint-reeval/analysis/checkpoint_reeval/paused-evaluations.json'))
def processes():
 out={}
 for p in Path('/proc').glob('[0-9]*'):
  try:
   stat=p.joinpath('stat').read_text();tail=stat[stat.rfind(')')+2:].split();args=p.joinpath('cmdline').read_bytes().split(b'\0')
   out[int(p.name)]={'ppid':int(tail[1]),'state':tail[0],'starttime':tail[19],'args':args}
  except (OSError,IndexError,ValueError):pass
 return out
if sys.argv[1]=='pause':
 if state.exists():raise SystemExit('pause state already exists; inspect before acting')
 ps=processes(); roots=[]
 for pid,d in ps.items():
  a=d['args']; cmd=b' '.join(a)
  if b'browse_comp_eval.run' in a and b'glasseek_step' in cmd:roots.append(pid)
 if len(roots)!=8:raise SystemExit(f'expected 8 evaluators, found {len(roots)}')
 for pid in roots:os.kill(pid,signal.SIGSTOP)
 ps=processes();nodes=set(roots)
 for _ in range(20):
  extended=nodes|{pid for pid,d in ps.items() if d['ppid'] in nodes}
  if extended==nodes:break
  nodes=extended
 stopped=[]
 for pid in nodes-set(roots):
  try:os.kill(pid,signal.SIGSTOP);stopped.append(pid)
  except ProcessLookupError:pass
 ps2=processes();records=[{'pid':pid,'starttime':ps2[pid]['starttime']} for pid in nodes if pid in ps2]
 tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps({'created_at':time.time(),'roots':roots,'processes':records},indent=2)+'\n');os.replace(tmp,state)
 print('paused evaluators',len(roots),'total live processes',len(records),'not stopped',[(i,ps2[i]['state']) for i in nodes if i in ps2 and ps2[i]['state'] not in ('T','t','Z')])
elif sys.argv[1]=='pause-services':
 data=json.loads(state.read_text());ps=processes();roots=[]
 for pid,d in ps.items():
  a=d['args'];cmd=b' '.join(a)
  if b'vllm' in cmd and b'serve' in a and any((b'step'+str(i).encode()+b'=') in cmd for i in range(10,81,10)):roots.append(pid)
 if len(roots)!=8:raise SystemExit(f'expected 8 vLLM services, found {len(roots)}')
 for pid in roots:os.kill(pid,signal.SIGSTOP)
 ps=processes();nodes=set(roots)
 for _ in range(20):
  extended=nodes|{pid for pid,d in ps.items() if d['ppid'] in nodes}
  if extended==nodes:break
  nodes=extended
 for pid in nodes-set(roots):
  try:os.kill(pid,signal.SIGSTOP)
  except ProcessLookupError:pass
 ps=processes();data['service_processes']=[{'pid':pid,'starttime':ps[pid]['starttime']} for pid in nodes if pid in ps]
 tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,state)
 print('paused services',len(roots),'total service processes',len(data['service_processes']),'not stopped',[(i,ps[i]['state']) for i in nodes if i in ps and ps[i]['state'] not in ('T','t','Z')])
elif sys.argv[1]=='resume':
 data=json.loads(state.read_text()); ps=processes();missing=[]
 all_processes=data.get('service_processes',[])+data['processes']
 for d in all_processes:
  pid=d['pid'];p=ps.get(pid)
  if not p or p['starttime']!=d['starttime']:missing.append(pid);continue
  try:os.kill(pid,signal.SIGCONT)
  except ProcessLookupError:missing.append(pid)
 print('resumed',len(all_processes)-len(missing),'missing',missing)
 if not missing:state.unlink()
else:raise SystemExit('usage: pause_live_evaluations.py pause|resume')
