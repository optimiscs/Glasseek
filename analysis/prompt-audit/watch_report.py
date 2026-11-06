import collections,json,os,pathlib,time
p=pathlib.Path('/root/autodl-tmp/Glasseek/checkpoint-reeval/analysis/prompt-audit');o=pathlib.Path('/root/autodl-tmp/Glasseek/Evaluation/output/preds/glasseek_base9b_prompt_audit_20261105/browsecomp_rl9b_non_timeout_34')
while True:
 try:
  a=[json.loads(x) for x in (p/'requests.jsonl').read_text().splitlines()];primary=[r for r in a if r['system_count']>0];aux=[r for r in a if r['system_count']==0];fs=list(o.glob('conv/*/*.json'));js=list(o.glob('judge_official/*/*.json'));report={'updated_at':time.time(),'requests':len(a),'primary_requests':len(primary),'primary_system_mismatches':sum(not x['system_matches'] for x in primary),'primary_tools_mismatches':sum(not x['tools_match'] for x in primary),'no_system_requests':len(aux),'completed_trajectories':len(fs),'judged_trajectories':len(js)}
  if fs:
   x=json.loads(fs[0].read_text());report['trajectory_status']=x.get('status');report['rounds']=x.get('rounds');report['duration_seconds']=x.get('duration_seconds');report['tool_calls']=x.get('tool_calls')
  if js:report['judge']=json.loads(js[0].read_text())
  tmp=p/'report.json.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));os.replace(tmp,p/'report.json')
  if js:break
 except (OSError,ValueError,KeyError):pass
 time.sleep(30)
