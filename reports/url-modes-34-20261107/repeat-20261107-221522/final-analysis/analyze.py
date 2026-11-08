import sys,json,sqlite3,hashlib,statistics,math
from pathlib import Path
from collections import Counter
from datetime import datetime
ROOT=Path('/root/autodl-tmp/Glasseek'); BASE=ROOT/'reports/url-modes-34-20261107/repeat-20261107-221522';OUT=BASE/'final-analysis'
sys.path.insert(0,str(ROOT/'scripts/trajectory_benchmark'))
from audit_url_mode_run import audit
state=json.loads((BASE/'status.json').read_text()); rows=[]
recovered=state['started_at']
for mode,run in state['runs'].items():
 folder=ROOT/'Evaluation/output/preds'/run['run']/state['dataset']
 for path in sorted((folder/'conv').rglob('*.json')):
  case=json.loads(path.read_text()); dbpath=ROOT/'Evaluation/data/runtime/hermes/profiles'/case['profile']/'state.db'
  with sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True) as db:
   db.row_factory=sqlite3.Row
   sessions=[dict(x) for x in db.execute('SELECT * FROM sessions')]
   messages=[dict(x) for x in db.execute('SELECT * FROM messages ORDER BY timestamp,id')]
  regs={}
  for session in sessions:
   p=Path(run['url_state_dir'])/(hashlib.sha256(session['id'].encode()).hexdigest()+'.json')
   if p.exists():regs[session['id']]=json.loads(p.read_text())
  row=audit(mode,case,messages,sessions,regs)
  row['last_session_duration_seconds']=row['duration_seconds']
  row['duration_seconds']=case.get('wall_seconds') or row['duration_seconds']
  row.update(conversation_path=str(path),db_path=str(dbpath),started_at=min(s['started_at'] for s in sessions),error=case.get('error'),timed_out=case.get('timed_out'),retry_attempts=len(case.get('retry_attempts') or []))
  row['started_after_recovery']=row['started_at']>=recovered
  jp=folder/'judge_official'/case['question_id']/(case['run_id']+'.json'); judge=json.loads(jp.read_text())
  row['judge']={k:judge.get(k) for k in ['status','score','judge_reason','pred_fingerprint']}
  row['correct']=judge.get('score')==1
  row['errors']=dict(Counter(p['error'] for c in row['extract_calls'] for p in c.get('page_results',[]) if p.get('error')))
  row['retrieval_error_count']=sum(n for e,n in row['errors'].items() if 'embedding retrieval unavailable' in e)
  row['api_calls']=sum(s.get('api_call_count') or 0 for s in sessions)
  row['registry_entries']=sum(len(r.get('entries',{})) for r in regs.values())
  row['truncated_pages']=sum(bool(p.get('truncated')) for c in row['extract_calls'] for p in c.get('page_results',[]))
  row['short_noerror_pages']=sum(not p.get('error') and p.get('content_chars',0)<150 for c in row['extract_calls'] for p in c.get('page_results',[]))
  row['unresolved_ids']=sum(t.get('kind') in ('bare_id_argument','xml_argument') and not t.get('resolved_url') for c in row['extract_calls'] for t in c['targets'])
  rows.append(row)

def aggregate(rs):
 counts=Counter();errors=Counter()
 for r in rs:counts.update(r['counts']);errors.update(r['errors'])
 vals=sorted(r['duration_seconds'] for r in rs);n=len(rs)
 return {'n':n,'success':sum(r['status']=='success' for r in rs),'correct':sum(r['correct'] for r in rs),'accuracy':sum(r['correct'] for r in rs)/n if n else None,'timeouts':sum(bool(r['timed_out']) for r in rs),'counts':dict(counts),'errors':dict(errors), 'duration_median_seconds':statistics.median(vals) if vals else None,'duration_p90_seconds':vals[math.ceil(.9*n)-1] if n else None, 'retrieval_affected_cases':sum(r['retrieval_error_count']>0 for r in rs), **{k:sum(r[k] for r in rs) for k in ['input_tokens_including_cache','uncached_input_tokens','cache_read_tokens','output_tokens','duration_seconds','api_calls','truncated_pages','short_noerror_pages','unresolved_ids']}}
summary={m:aggregate([r for r in rows if r['mode']==m]) for m in state['runs']}
for m,v in summary.items():
 pause=json.loads((BASE/'paused.json').read_text())
 v['wall_seconds']=(pause['paused_at']-state['started_at'])+(state['runs'][m]['finished_at']-state['resumed_at']) if state.get('resumed_at') else state['runs'][m]['finished_at']-state['started_at']
 v['input_change_vs_raw_pct']=100*(v['input_tokens_including_cache']/summary['raw']['input_tokens_including_cache']-1)
 v['output_change_vs_raw_pct']=100*(v['output_tokens']/summary['raw']['output_tokens']-1)
 v['cache_hit_ratio']=v['cache_read_tokens']/v['input_tokens_including_cache']
 v['tools_per_case']=v['counts']['tool_calls']/v['n']
 v['generated_tokens_per_correct']=v['output_tokens']/v['correct'] if v['correct'] else None
common_ids=set.intersection(*[{r['question_id'] for r in rows if r['mode']==m and r['started_after_recovery']} for m in state['runs']])
clean_ids=set.intersection(*[{r['question_id'] for r in rows if r['mode']==m and not r['retrieval_error_count']} for m in state['runs']])
paired={};bykey={(r['mode'],r['question_id']):r for r in rows}
for mode in ['hybrid','ref','evidence']:
 wins=[q for q in state['question_ids'] if bykey[mode,q]['correct'] and not bykey['raw',q]['correct']]
 losses=[q for q in state['question_ids'] if not bykey[mode,q]['correct'] and bykey['raw',q]['correct']]
 n=len(wins)+len(losses);k=min(len(wins),len(losses));p=min(1.,2*sum(math.comb(n,i) for i in range(k+1))/2**n) if n else 1.
 paired[mode]={'wins':wins,'losses':losses,'mcnemar_exact_two_sided_p':p}
metric_rows=[]
with (BASE/'vllm_metrics.jsonl').open() as f:
 for line in f:
  try:m=json.loads(line)
  except ValueError:continue
  if m.get('timestamp',0)<=max(x['finished_at'] for x in state['runs'].values()):metric_rows.append(m)
metrics={}
for k in ['num_requests_running','num_requests_waiting','kv_cache_usage_perc','num_preemptions_total']:
 values=[x[k] for x in metric_rows if k in x]
 if values: metrics[k]={'min':min(values),'max':max(values),'first':values[0],'last':values[-1]}
manifest=json.loads((BASE/'implementation_sha256.json').read_text());hashes={k:hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in manifest.items()}
report={'generated_at':datetime.now().astimezone().isoformat(),'summary':summary,'rows':rows,'paired_vs_raw':paired,'post_recovery_common_ids':sorted(common_ids),'post_recovery_common':{m:aggregate([r for r in rows if r['mode']==m and r['question_id'] in common_ids]) for m in state['runs']},'no_retrieval_error_common_ids':sorted(clean_ids),'no_retrieval_error_common':{m:aggregate([r for r in rows if r['mode']==m and r['question_id'] in clean_ids]) for m in state['runs']},'metrics':metrics,'source_hash_still_matches_start':hashes,'limitations':['136 terminal trajectories include two one-hour timeouts; cost totals include those failed attempts.','Session token counters include child agents and compression continuation sessions stored in the profile; not an invoice or end-to-end measurement of auxiliary summarizer models.','The experiment was paused and resumed after services restarted; group wall time excludes the paused interval. Check per-case retrieval errors.','All modes share one GPU and dynamic web services; this is one stochastic run, not a controlled causal estimate.','Check status.json and logs for process failures; do not infer completion from judge counts alone.','Answer accuracy denominator is all 34 assigned cases, including inference failures.']}
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
for m,v in summary.items():print(m,json.dumps(v,ensure_ascii=False))
print('POST_RECOVERY',sorted(common_ids));print({m:(v['correct'],v['n'],v['retrieval_affected_cases']) for m,v in report['post_recovery_common'].items()})
print('PAIRED',paired);print('METRICS',metrics);print('HASHES',hashes)
