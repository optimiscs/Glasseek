import json, os, multiprocessing as mp, numpy as np
from pathlib import Path
from transformers import AutoTokenizer
os.environ['TOKENIZERS_PARALLELISM']='false'
ROOT=Path('/root/autodl-tmp/Glasseek')
MODEL='/root/autodl-tmp/models/Qwen3.5-9B'
tok=None
def init():
 global tok
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
def count(item):
 i, line=item
 try:
  row=json.loads(line)
  ids=tok.apply_chat_template(row['messages'],tools=row.get('tools'),add_generation_prompt=False,tokenize=True)
  if isinstance(ids,dict): ids=ids['input_ids']
  elif hasattr(ids,'input_ids'): ids=ids.input_ids
  if len(ids)==1 and isinstance(ids[0],list): ids=ids[0]
  return i,len(ids),None
 except Exception as e:return i,None,f'{type(e).__name__}: {e}'
def main():
 source=ROOT/'data/Glasseek-SFT-5.7K/train.jsonl'
 with source.open() as f, mp.Pool(8,initializer=init) as pool:
  rows=[]
  for x in pool.imap_unordered(count,enumerate(f,1),chunksize=8):
   rows.append(x)
   if len(rows)%500==0: print(f'Processed {len(rows)}',flush=True)
 errors=[x for x in rows if x[2]]
 lengths=[x[1] for x in rows if x[1] is not None]
 result={'records':len(rows),'errors':errors,'method':'whole Qwen3.5 chat template; per-turn equivalence established only for first 42 records', 'min':min(lengths),'mean':float(np.mean(lengths)),'max':max(lengths),'percentiles':{str(p):float(np.percentile(lengths,p)) for p in [50,75,90,95,99,99.9]},'exceeds':{str(t):sum(n>t for n in lengths) for t in [32768,65536,98304,131072]},'largest':sorted([(n,i) for i,n,e in rows if n is not None],reverse=True)[:10], 'length_by_line':{str(i):n for i,n,e in rows if n is not None}}
 (ROOT/'analysis/sft-4gpu-feasibility/full_lengths.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k != 'length_by_line'},indent=2),flush=True)
if __name__=='__main__':main()
