import sys,json,collections,re,urllib.parse
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/Glasseek/analysis/url-provenance');from analyze import norm
D=Path('/root/autodl-tmp/Quest/web/trace-viewer/datasets');P=Path('/root/autodl-tmp/Glasseek/analysis/url-provenance')
cfg=['gaia-quest4b-32-20261027-r2','glasseek-sft9b-browsecomp-128','glasseek-rl9b-browsecomp-128-32']
def obj(s):
 for m in re.finditer(r'\{',s):
  try:
   x,_=json.JSONDecoder().raw_decode(s[m.start():])
   if isinstance(x,dict):return x
  except (ValueError,RecursionError):continue
 return {}
def searchengine(url):
 try:return any((urllib.parse.urlsplit(url).hostname or '').lower().endswith(h) for h in ['duckduckgo.com','bing.com','google.com'])
 except ValueError:return False
for name in cfg:
 occ=[json.loads(x) for x in (P/(name+'.jsonl')).open()];unseen=[r for r in occ if not r['observed_before'] and r['phase']=='step_read'];data=json.load((D/name/'traces.json').open());lookup={(c['id'],s['n']):s for c in data['cases'] for s in c['steps']};counter=collections.Counter();details=[]
 for r in unseen:
  s=lookup.get((r['case'],r['step']),{});raw=s.get('rawToolOutput') or (s.get('tool') or {}).get('rawOutput') or '';raw=raw if isinstance(raw,str) else json.dumps(raw,ensure_ascii=False);o=obj(raw);items=o.get('results')
  if not isinstance(items,list):
   d=o.get('data') or {};items=d.get('results',d.get('web',[])) if isinstance(d,dict) else []
  if isinstance(items,dict):items=[items]
  if not isinstance(items,list):items=[]
  matched=[i for i in items if isinstance(i,dict) and norm(str(i.get('url') or ''))==r['normalized']]
  if matched:
   item=matched[0];content=item.get('content') or item.get('text') or '';err=item.get('error') or ''
   if err:outcome='error'
   elif not content:outcome='empty'
   elif re.search(r'(?i)(captcha|verify you are human|unusual traffic|access denied)',content[:2000]):outcome='challenge_or_denied'
   else:outcome='content_returned'
   sz=len(content)
  else:outcome='unmatched_or_unstructured';err='';sz=0
  group='search_engine' if searchengine(r['url']) else 'other_site';counter[(group,outcome)]+=1;details.append({'case':r['case'],'step':r['step'],'url':r['url'],'group':group,'outcome':outcome,'content_chars':sz,'error':str(err)[:300]})
 (P/(name+'-unseen-access.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in details))
 print(name,'total',len(unseen),'counter',dict((str(k),v) for k,v in counter.items()))
 print('error reasons',collections.Counter(x['error'] for x in details if x['outcome']=='error').most_common(5))
