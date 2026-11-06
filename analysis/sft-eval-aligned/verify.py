import collections,json,random,re
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
root=Path('/root/autodl-tmp/Glasseek');out=root/'data/glasseek/sft-eval-aligned-128k'
audit={x['source_line']:x for x in map(json.loads,(out/'audit.jsonl').open())}
new=iter(map(json.loads,(out/'all_filtered.jsonl').open()));total=0;sample=[]
rng=random.Random(42);chosen=set(rng.sample([k for k,v in audit.items() if 'original_system_tokens' in v and v['original_split']],16))
ref=json.loads((root/'analysis/checkpoint_reeval/remote_snapshot/checkpoints/glasseek_step10_rl9b_subset/output/browsecomp_rl9b_non_timeout_34/conv/q000001/q000001__r1.json').read_text())['history'][0]
tok=AutoTokenizer.from_pretrained(root/'checkpoints/glasseek/sft_128k_length_curriculum_20261104/global_step_80/huggingface',local_files_only=True,trust_remote_code=True)
r=tok.apply_chat_template([{'role':'system','content':'SENTINEL'},{'role':'user','content':'hi'}],tools=ref['tools'],tokenize=False,add_generation_prompt=False)
prefix=r.split('<|im_start|>system\n',1)[1].split('\n\nSENTINEL<|im_end|>',1)[0]
def fields(s):
 p=s.split('</tools>',1)[-1];d={}
 for z in p.split('\n\n'):
  for k in ('Terminal backend:', 'Workspace (snapshot','Active Hermes profile:'):
   if z.startswith(k):d[k]=z
 for k in ('Conversation started','Model','Provider'):
  m=re.findall(r'(?m)^'+k+r':[^\n]*',p)
  if m:d[k]=m[-1]
 return d
for i,line in enumerate((root/'data/Glasseek-SFT-5.7K/train.jsonl').open(),1):
 info=audit[i]
 if 'original_system_tokens' not in info:continue
 old=json.loads(line);newrow=next(new)
 assert old['messages'][1:]==newrow['messages'][1:],i
 s=old['messages'][0]['content'];t=newrow['messages'][0]['content']
 assert t.startswith(prefix+'\n\n') and fields(s)==fields(t),i
 if i in chosen:
  a=tok.apply_chat_template(old['messages'],tokenize=True,add_generation_prompt=False)
  b=tok.apply_chat_template(newrow['messages'],tokenize=True,add_generation_prompt=False)
  a=a['input_ids']; b=b['input_ids']
  assert len(b)-len(a)==info['aligned_system_tokens']-info['original_system_tokens'],(i,len(b)-len(a),info)
  sample.append(i)
 total+=1
try:next(new);raise AssertionError('extra output')
except StopIteration:pass
counts={s:pq.ParquetFile(out/f'{s}.parquet').metadata.num_rows for s in ('train','val')}
assert counts=={'train':3488,'val':168}
print(json.dumps({'all_retained_checked':total,'unchanged_non_system_messages':True,'tool_prefix_and_runtime_fields_checked':True,'full_chat_token_delta_sample':sample,'parquet_counts':counts},ensure_ascii=False))
