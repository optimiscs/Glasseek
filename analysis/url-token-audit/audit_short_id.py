import json,re,collections,hashlib
from pathlib import Path
from functools import lru_cache
from urllib.parse import urlsplit,urlunsplit
from tokenizers import Tokenizer
root=Path('/root/autodl-tmp/Glasseek/Evaluation/data/runtime'); run='glasseek_rl9b_quest128_32'; base=root/'hermes/sessions'/run/'quest_qwen27b_128'; tok=Tokenizer.from_file(str(root/'models/Glasseeker-9B-RL/tokenizer.json'))
@lru_cache(maxsize=10000)
def nt(s):return len(tok.encode(s,add_special_tokens=False).ids)
pat=re.compile(r'https?://[^\s"<>\\]+',re.I)
tracking=re.compile(r'^(?:utm_(?:source|medium|campaign|term|content|id)|gclid|dclid|fbclid|msclkid|mc_cid|mc_eid|igshid|yclid|_ga)$',re.I)
counts=collections.Counter();skip=collections.Counter(); case_set=set();sessions=0; idmaps={}

def clean_url(u):
 try:
  p=urlsplit(u)
  if not p.query:return u
  fields=p.query.split('&');keep=[x for x in fields if not tracking.fullmatch(x.split('=',1)[0])]
  if len(keep)==len(fields):return u
  counts['tracking_fields']+=len(fields)-len(keep)
  return urlunsplit((p.scheme,p.netloc,p.path,'&'.join(keep),p.fragment))
 except ValueError:return u

def parse_search(s):
 i=s.find('{')
 if i<0:return None
 try:obj,n=json.JSONDecoder().raw_decode(s[i:]);rows=obj.get('data',{}).get('web')
 except (ValueError,AttributeError,TypeError):return None
 if not isinstance(rows,list):return None
 return i,n,obj,rows

def dedup(s,seen):
 x=parse_search(s)
 if x is None:return s
 i,n,obj,rows=x; old=s[i:i+n]; newrows=[]
 for r in rows:
  if not isinstance(r,dict) or not isinstance(r.get('url'),str) or not r['url']:
   newrows.append(r);continue
  u=r['url']
  if u in seen:counts['duplicate_results']+=1;continue
  seen.add(u);newrows.append(r)
 if len(newrows)==len(rows):return s
 obj['data']['web']=newrows
 new=json.dumps(obj,ensure_ascii=False)
 # Avoid changing serialization: only use results whose original uses the same JSON formatting.
 orig_obj=json.loads(old);orig_obj['data']['web']=rows
 if json.dumps(orig_obj,ensure_ascii=False)!=old:skip['dedup_serialization']+=1;return s
 return s[:i]+new+s[i+n:]

for f in sorted(base.rglob('session_*.json')):
 try:
  j=json.loads(f.read_text());msgs=j['messages']
  if not isinstance(msgs,list):continue
 except (OSError,ValueError,KeyError,TypeError):skip['snapshot_unreadable']+=1;continue
 case=f.parent.parent.name; case_set.add(case); sessions+=1; idmap=idmaps.setdefault(case, {}); cumul_id=0; seen=set();searches=[]; cumul_track=0;cumul_dedup=0;cumul_search_tokens=0; running_input=0
 for m in msgs:
  role=m.get('role'); content=m.get('content') if isinstance(m.get('content'),str) else ''
  if role=='assistant' and not m.get('_inherited_from_previous_session'):
   counts['model_requests']+=1;counts['input_tool_tokens_weighted']+=running_input;counts['track_input_weighted']+=cumul_track;counts['dedup_input_weighted']+=cumul_dedup;counts['id_input_weighted']+=cumul_id
   # Remove old tool-result text but preserve one short placeholder per result.
   old=searches[:-2]; counts['prune_input_weighted']+=sum(max(0,t-nt('[Earlier search result omitted]')) for t in old)
   out=content or ''
   reason=m.get('reasoning') or m.get('reasoning_content') or ''
   calls=m.get('tool_calls') or []
   out+='\n'+str(reason)+'\n'+'\n'.join(str(c.get('function',{}).get('arguments','')) for c in calls if isinstance(c,dict))
   counts['assistant_output_tokens']+=nt(out);counts['assistant_output_url_occurrences']+=len(pat.findall(out)); counts['id_output_saved']+=nt(out)-nt(pat.sub(lambda z:'S'+str(idmap.setdefault(z.group(),len(idmap)+1)),out))
  if role!='tool':continue
  counts['tool_messages']+=1; t=nt(content);counts['tool_tokens_unique']+=t;running_input+=t
  matches=pat.findall(content);counts['tool_url_occurrences']+=len(matches)
  if matches:
   idtext=pat.sub(lambda z:'S'+str(idmap.setdefault(z.group(),len(idmap)+1)),content); iddelta=t-nt(idtext);counts['id_unique_saved']+=iddelta;cumul_id+=iddelta
  after=pat.sub(lambda z:clean_url(z.group()),content) if matches else content
  delta=t-nt(after) if after!=content else 0; counts['track_unique_saved']+=delta;cumul_track+=delta
  name=m.get('tool_name') or m.get('name') or ''
  if name=='web_search':
   counts['search_messages']+=1;counts['search_tokens_unique']+=t;searches.append(t)
   d=dedup(content,seen); dd=t-nt(d) if d!=content else 0; counts['dedup_unique_saved']+=dd;cumul_dedup+=dd
 counts['sessions']+=1
print(json.dumps({'run':run,'cases':len(case_set),'snapshots':sessions,'counts':dict(counts),'skipped':dict(skip)},ensure_ascii=False))
