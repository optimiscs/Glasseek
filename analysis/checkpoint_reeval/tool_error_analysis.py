"""Summarize observable tool responses in viewer snapshots, excluding in-flight calls."""
import collections,json,pathlib,re,sys
p=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else '/root/autodl-tmp/Glasseek/checkpoint-reeval/viewer')
def obj(s):
 for i,c in enumerate(s):
  if c=='{':
   try:
    v,_=json.JSONDecoder().raw_decode(s[i:]); return v if isinstance(v,dict) else {}
   except ValueError: pass
 return {}
def classify(s):
 raw=s.get('rawToolOutput') or ''; name=s['tool'].get('name','');
 if not raw:return 'pending',[],[]
 if raw.startswith('Skipped:'):return 'skipped',[],[]
 if "does not exist. Available tools:" in raw:return 'invalid_tool',[],[]
 o=obj(raw); errors=[]; page=[]
 if name=='web_search':
  if o.get('success') is False or o.get('error'):return 'failed',[str(o.get('error') or o.get('message') or o)[:300]],[]
  return ('success' if o.get('success') is True else 'unknown'),[],[]
 if name in ('web_extract','fetch_server'):
  results=o.get('results');
  if not isinstance(results,list):return ('failed' if o.get('error') else 'unknown'),[str(o.get('error'))] if o.get('error') else [],[]
  for x in results:
   if isinstance(x,dict):
    e=x.get('error');page.append((bool(e),str(e or ''),str(x.get('url') or '')))
  errors=[e for bad,e,_ in page if bad]
  return ('partial' if errors and len(errors)<len(page) else 'failed' if errors else 'success' if page else 'unknown'),errors,page
 if name=='execute_code':
  if o.get('status')=='error' or o.get('error'):return 'failed',[str(o.get('error') or '')],[]
  if o.get('status')=='success':
   output=str(o.get('output') or '')
   if re.search(r'(?im)^\s*(error:|traceback \(most recent call last\)|[a-z]*error:)',output):return 'output_error',[output[:300]],[]
   return 'success',[],[]
  return 'unknown',[],[]
 if name=='delegate_task':
  if o.get('error'):return 'failed',[str(o['error'])],[]
  results=o.get('results')
  if isinstance(results,list):
   bad=[str(x.get('error') or x.get('status')) for x in results if isinstance(x,dict) and x.get('status') not in ('completed','success')]
   return ('partial' if bad and len(bad)<len(results) else 'failed' if bad else 'success'),bad,[]
 return 'unknown',[],[]
all_results={}
for tag in ['base']+list(range(10,81,10)):
 path=p/('traces.json' if tag=='base' else f'glasseek-step{tag}-rl9b-34.json')
 d=json.loads(path.read_text()); totals=collections.defaultdict(collections.Counter); reasons=collections.Counter(); pages=collections.Counter(); cases=collections.Counter(); samples={}
 for case in d['cases']:
  for step in case.get('steps',[]):
   if not step.get('tool'):continue
   name=step['tool'].get('name',''); status,errs,urls=classify(step)
   totals[name][status]+=1
   if status in ('failed','partial','output_error','invalid_tool','skipped'): cases[case['id']]+=1
   for err in errs:
    text=err.lower()
    label=('private_or_internal_address' if 'private or internal network' in text else 'missing_bs4' if "no module named 'bs4'" in text else 'tls_certificate' if 'certificate' in text else 'dns' if 'name or service not known' in text or 'name resolution' in text else 'anti_cheat_block' if 'anti-cheat guard' in text else 'timeout' if 'timed out' in text or 'timeout' in text else 'other')
    reasons[(name,label)]+=1; samples.setdefault(f'{name}:{label}',{'case':case['id'],'error':err[:400]})
   for bad,err,url in urls:pages['failed' if bad else 'success']+=1
 def pack(c):return dict(c)
 all_results[str(tag)]={'generatedAt':d.get('generatedAt'),'caseCount':len(d['cases']),'runningCases':sum(bool(x.get('inProgress')) for x in d['cases']),'calls':{k:pack(v) for k,v in totals.items()},'pages':dict(pages),'errorReasons':{':'.join(k):v for k,v in reasons.items()},'casesWithObservedError':len(cases),'errorExamples':samples}
print(json.dumps(all_results,ensure_ascii=False,indent=2))
