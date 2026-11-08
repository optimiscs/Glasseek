import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path('/root/autodl-tmp/Glasseek');OUT=ROOT/'reports/url-modes-34-20261107/repeat-20261107-221522'
sys.path.insert(0,str(ROOT/'Evaluation/services'));from eval_run import build
os.environ['GLASSEEK_EVAL_PROFILES']=str(OUT/'profiles.json')
state=json.loads((OUT/'status.json').read_text());state['resumed_at']=time.time();jobs={};judges={}
def save():
 p=OUT/'status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(state,indent=2));q.replace(p)
for mode,r in state['runs'].items():
 os.environ['GLASSEEK_URL_MODE']=mode
 argv,env=build(mode,'full');env['GLASSEEK_URL_STATE_DIR']=r['url_state_dir'];env['PYTHONPATH']=str(ROOT/'Evaluation_backend')+os.pathsep+env.get('PYTHONPATH','')
 log=(OUT/(mode+'.resume.log')).open('a');proc=subprocess.Popen(argv,cwd=ROOT/'Evaluation',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);jobs[mode]=(proc,log)
 r.update(pid=proc.pid,status='running',resume_started_at=time.time());r.pop('exit_code',None);r.pop('finished_at',None)
 ja,je=build(mode,'watch-judge');ja+=['--parent-pid',str(proc.pid),'--viewer-model','glasseek-base-9b']
 jlog=(OUT/(mode+'.resume.judge.log')).open('a');jp=subprocess.Popen(ja,cwd=ROOT/'Evaluation',env=je,stdin=subprocess.DEVNULL,stdout=jlog,stderr=subprocess.STDOUT,start_new_session=True);judges[mode]=(jp,jlog)
 r['judge']={'pid':jp.pid,'status':'running','log':str(OUT/(mode+'.resume.judge.log'))}
save()
while jobs or judges:
 for mode,(proc,log) in list(jobs.items()):
  code=proc.poll()
  if code is not None:state['runs'][mode].update(status='completed' if code==0 else 'failed',exit_code=code,finished_at=time.time());log.close();del jobs[mode]
 for mode,(proc,log) in list(judges.items()):
  code=proc.poll()
  if code is not None:state['runs'][mode]['judge'].update(status='completed' if code==0 else 'failed',exit_code=code,finished_at=time.time());log.close();del judges[mode]
 save()
 if jobs or judges:time.sleep(5)
if any(r['exit_code'] or r['judge'].get('exit_code',1) for r in state['runs'].values()):raise SystemExit(1)
