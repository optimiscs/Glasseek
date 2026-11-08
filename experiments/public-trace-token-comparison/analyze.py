import collections,glob,json,random,re,sys,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
P=Path('/root/autodl-tmp/Glasseek/experiments/public-trace-token-comparison');R=random.Random(20261105);N=500
T=Tokenizer.from_file('/autodl-fs/data/models/Qwen3.5-9B/tokenizer.json')
URL=re.compile(r'https?://[^\s<>"\'\[\]{}\\]+',re.I); NUM=re.compile(r'\d+(?:[.,:/-]\d+)*');MARKUP=re.compile(r'</?[A-Za-z_][^<>\n]{0,199}>|[{}\[\]`#]')
patterns=[('url',URL),('number',NUM),('markup',MARKUP)]

def stats(messages):
 roles=collections.Counter();cat=collections.Counter();urlrole=collections.Counter()
 for m in messages:
  s=m.get('content') or ''
  if not isinstance(s,str) or not s:continue
  role=m.get('role','other');role='tool_result' if role in ('tool','function') or (role=='user' and s.lstrip().startswith('<tool_response>')) else role
  e=T.encode(s,add_special_tokens=False);offs=e.offsets;roles[role]+=len(offs)
  if role=='system':continue
  # Give URL priority over numbers and markup; every encoded token gets one label.
  spans=[]
  for rank,(name,pat) in enumerate(patterns):
   for match in pat.finditer(s):
    a,b=match.span()
    if name=='url':
     while b>a and s[b-1] in '.,;:!?':b-=1
    if b>a:spans.append((a,b,rank,name))
  spans.sort();j=0;active=[]
  for a,b in offs:
   if b<=a:cat['other']+=1;continue
   while j<len(spans) and spans[j][0]<b:active.append(spans[j]);j+=1
   active=[x for x in active if x[1]>a]
   hit=min((x for x in active if x[0]<b and x[1]>a),key=lambda x:x[2],default=None)
   name=hit[3] if hit else 'other';cat[name]+=1
   if name=='url':urlrole[role]+=1
 return {'roles':dict(roles),'categories':dict(cat),'url_by_role':dict(urlrole)}

def quest(d):
 files=sorted(glob.glob('/root/autodl-tmp/Quest/workspace/datasets/'+d+'/data/*.parquet'));total=sum(pq.ParquetFile(p).metadata.num_rows for p in files);chosen=set(R.sample(range(total),min(N,total)));i=0
 for f in files:
  pf=pq.ParquetFile(f)
  for batch in pf.iter_batches(batch_size=32,columns=['messages']):
   for row in batch.to_pylist():
    if i in chosen:yield i,row['messages']
    i+=1
 assert i==total

def glass():
 f='/root/autodl-tmp/Glasseek/data/Glasseek-SFT-5.7K/train.jsonl';chosen=set(R.sample(range(5762),N))
 with open(f) as fin:
  for i,l in enumerate(fin):
   if i in chosen:yield i,json.loads(l)['messages']

def main():
 result={};start=time.monotonic()
 for label,it,pop in [('Quest objective',quest('QUEST-SFT-Data-Objective'),39861),('Quest open-ended',quest('QUEST-SFT-Data-Open-ended'),11903),('Glasseek SFT',glass(),5762)]:
  rows=[]
  for idx,messages in it:
   v=stats(messages);v['index0']=idx;rows.append(v)
   if len(rows)%50==0:print(json.dumps({'dataset':label,'n':len(rows),'elapsed_s':round(time.monotonic()-start)}),flush=True)
  result[label]={'population':pop,'sample':len(rows),'rows':rows}
  (P/'partial.json').write_text(json.dumps(result,ensure_ascii=False))
 (P/'results.json').write_text(json.dumps({'seed':20261105,'tokenizer':'Qwen3.5-9B tokenizer.json','data':result},ensure_ascii=False))
 print('complete',flush=True)
if __name__=='__main__':main()
