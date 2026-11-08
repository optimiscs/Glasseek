import json,re,collections,time,bisect
from pathlib import Path
from tokenizers import Tokenizer
p=Path('/root/autodl-tmp/Glasseek/experiments/remaining2899-token-distribution'); src=p.parent/'no-tool-top10/explicit-matches/remaining_shortest_trajectories.jsonl'
t=Tokenizer.from_file('/autodl-fs/data/models/Qwen3.5-9B/tokenizer.json')
patterns=[('url',re.compile(r'https?://[^\s<>"\'\[\]{}\\]+',re.I)),('title',re.compile(r'(?im)^[ \t]{0,3}#{1,6}[ \t]+[^\n]{1,300}|["\'](?:title|page_title)["\']\s*:\s*["\'](?:\\.|[^"\'\\]){1,300}["\']')),('number',re.compile(r'\d+(?:[.,:/-]\d+)*')),('markup',re.compile(r'</?[A-Za-z_][^<>\n]{0,199}>|[{}\[\]`#]'))]
roles=collections.Counter(); cats=collections.Counter(); matrix=collections.defaultdict(collections.Counter); rows=0; start=time.time(); chars=collections.Counter(); title_messages=0
for line in src.open():
 row=json.loads(line);rows+=1
 for m in row['messages']:
  s=m.get('content') or ''
  if not isinstance(s,str) or not s:continue
  role=m.get('role','other');role='tool_result' if role in ('tool','function') or (role=='user' and s.lstrip().startswith('<tool_response>')) else role
  if role=='system': continue
  spans=[]
  for priority,(name,pat) in enumerate(patterns):
   for z in pat.finditer(s):
    a,b=z.span()
    if name=='url':
     while b>a and s[b-1] in '.,;:!?':b-=1
    if b>a:spans.append((a,b,priority))
  if any(x[2]==1 for x in spans):title_messages+=1
  spans.sort();j=0; active=[]
  offsets=t.encode(s,add_special_tokens=False).offsets
  roles[role]+=len(offsets)
  for a,b in offsets:
   while j<len(spans) and spans[j][0]<b:active.append(spans[j]);j+=1
   if active: active=[x for x in active if x[1]>a]
   category=patterns[min((x[2] for x in active if x[0]<b and x[1]>a),default=4)][0] if active and any(x[0]<b and x[1]>a for x in active) else 'other'
   cats[category]+=1;matrix[role][category]+=1
 if rows%200==0:print(json.dumps({'rows':rows,'seconds':round(time.time()-start)}),flush=True)
result={'rows':rows,'roles':dict(roles),'categories':dict(cats),'matrix':{k:dict(v) for k,v in matrix.items()},'title_messages':title_messages,'seconds':round(time.time()-start)}
(p/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
