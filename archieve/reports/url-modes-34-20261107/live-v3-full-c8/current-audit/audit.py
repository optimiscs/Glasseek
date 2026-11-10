import sys,json,sqlite3,hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime
ROOT=Path('/root/autodl-tmp/Glasseek')
sys.path.insert(0,str(ROOT/'scripts/trajectory_benchmark'))
from audit_url_mode_run import audit
BASE=ROOT/'reports/url-modes-34-20261107'
report={'generated_at':datetime.now().astimezone().isoformat(),'conditions':{}}
for condition, dirname in [('old','live-c16'),('new_c16','live-v3-two-c16'),('new_c8','live-v3-full-c8')]:
 state=json.loads((BASE/dirname/'status.json').read_text());rows=[]; progress={}
 for mode,run in state['runs'].items():
  folder=ROOT/'Evaluation/output/preds'/run['run']/state['dataset']
  cases=[(p,json.loads(p.read_text())) for p in sorted((folder/'conv').rglob('*.json'))]
  progress[mode]=dict(Counter(c.get('status') for p,c in cases))
  for path,case in cases:
   if case.get('status')!='success':continue
   dbpath=ROOT/'Evaluation/data/runtime/hermes/profiles'/case['profile']/'state.db'
   with sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True) as db:
    db.row_factory=sqlite3.Row
    sessions=[dict(r) for r in db.execute('SELECT * FROM sessions')]
    messages=[dict(r) for r in db.execute('SELECT * FROM messages ORDER BY timestamp,id')]
   regs={}
   for s in sessions:
    p=Path(run['url_state_dir'])/(hashlib.sha256(s['id'].encode()).hexdigest()+'.json')
    if p.exists():regs[s['id']]=json.loads(p.read_text())
   row=audit(mode,case,messages,sessions,regs)
   row['conversation_path']=str(path);row['db_path']=str(dbpath)
   row['extract_errors']=[p['error'] for c in row['extract_calls'] for p in c.get('page_results',[]) if p['error']]
   row['exact_answer_section']=(row['final_answer'] or '').split('Exact Answer:')[-1][-900:]
   rows.append(row)
 report['conditions'][condition]={'progress':progress,'rows':rows}
paired={}
for condition,data in report['conditions'].items():
 paired[condition]={}
 for mode in ['raw','hybrid','ref','evidence']:
  selected=[r for r in data['rows'] if r['mode']==mode and r['question_id'] in ['row0026','row0048']]
  counts=Counter()
  for r in selected:counts.update(r['counts'])
  paired[condition][mode]={'n':len(selected),'counts':dict(counts),**{k:sum(r[k] for r in selected) for k in ['input_tokens_including_cache','uncached_input_tokens','cache_read_tokens','output_tokens','duration_seconds']}}
report['paired']=paired
out=BASE/'live-v3-full-c8/current-audit/report.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
for condition,data in report['conditions'].items():
 print(condition,data['progress'])
 if condition=='new_c8':
  for mode in ['raw','hybrid','ref','evidence']:
   rows=[r for r in data['rows'] if r['mode']==mode];counts=Counter()
   for r in rows:counts.update(r['counts'])
   print(mode,len(rows),dict(counts))
print('PAIRED',json.dumps(paired,ensure_ascii=False))
