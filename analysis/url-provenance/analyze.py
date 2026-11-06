import json,re,collections,html,urllib.parse
from pathlib import Path
D=Path('/root/autodl-tmp/Quest/web/trace-viewer/datasets');P=Path('/root/autodl-tmp/Glasseek/analysis/url-provenance');U=re.compile(r'https?://[^\s<>"\'\[\]{}\\]+',re.I)
CFG=[('QUEST-4B / GAIA 32','gaia-quest4b-32-20261027-r2'),('Glasseek SFT-9B / BrowseComp 128','glasseek-sft9b-browsecomp-128'),('Glasseek RL-9B / BrowseComp 128','glasseek-rl9b-browsecomp-128-32')]
def extract(s):
 for m in U.finditer(s or ''):
  a,b=m.span();v=s[a:b].rstrip('.,;:!?`*')
  while v.endswith(')') and v.count(')')>v.count('('):v=v[:-1]
  if v:yield v,a,min(a+len(v),b)
def norm(s):
 try:
  p=urllib.parse.urlsplit(html.unescape(s));host=(p.hostname or '').lower();port=p.port
  if not host:return ''
  net=host+((':'+str(port)) if port and not (p.scheme.lower()=='http' and port==80 or p.scheme.lower()=='https' and port==443) else '')
  return urllib.parse.urlunsplit((p.scheme.lower(),net,p.path.rstrip('/') or '/',p.query,''))
 except ValueError:return ''
def main():
 overall={}
 for label,d in CFG:
  cases=json.load((D/d/'traces.json').open())['cases'];occ=[];case_stats=[]
  for case in cases:
   sources={norm(u) for u,_,_ in extract(case.get('question') or '')};thoughts=[];excluded=0;caseoc=[]
   def generated(text,phase,step):
    for url,a,b in extract(text):
     n=norm(url)
     if not n:continue
     rec={'dataset':label,'case':case['id'],'step':step,'phase':phase,'url':url,'normalized':n,'observed_before':n in sources,'context':text[max(0,a-240):min(len(text),b+240)]}
     occ.append(rec);caseoc.append(rec)
   for step in case.get('steps',[]):
    thought=step.get('thought') or ''
    if '[CONTEXT COMPACTION' in thought:excluded+=1
    else:generated(thought,'answer_step' if step.get('kind')=='answer' else 'step_'+str(step.get('kind') or 'other'),step.get('n'));thoughts.append(thought)
    raw=step.get('rawToolOutput') or (step.get('tool') or {}).get('rawOutput') or ''
    if not isinstance(raw,str):raw=json.dumps(raw,ensure_ascii=False)
    sources.update(norm(u) for u,_,_ in extract(raw))
   final=case.get('finalAnswer') or ''
   if final and '[CONTEXT COMPACTION' not in final and not any(final in x for x in thoughts):generated(final,'final',None)
   case_stats.append({'case':case['id'],'occurrences':len(caseoc),'unseen':sum(not x['observed_before'] for x in caseoc),'excluded_compaction_steps':excluded})
  allunique={x['normalized'] for x in occ};novelunique={x['normalized'] for x in occ if not x['observed_before']};final=[x for x in occ if x['phase']=='final']
  summary={'cases':len(cases),'cases_with_url':sum(bool(x['occurrences']) for x in case_stats),'occurrences':len(occ),'occurrence_unseen':sum(not x['observed_before'] for x in occ),'unique_urls':len(allunique),'unique_unseen_at_least_once':len(novelunique),'final_occurrences':len(final),'final_unseen':sum(not x['observed_before'] for x in final),'excluded_compaction_steps':sum(x['excluded_compaction_steps'] for x in case_stats)}
  overall[label]=summary
  with (P/(d+'.jsonl')).open('w') as f:
   for x in occ:f.write(json.dumps(x,ensure_ascii=False)+'\n')
  print(label,json.dumps(summary,ensure_ascii=False),flush=True)
 (P/'summary.json').write_text(json.dumps(overall,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
