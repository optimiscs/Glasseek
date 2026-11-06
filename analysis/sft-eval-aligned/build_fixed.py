import json,os,re,hashlib,collections
from pathlib import Path
import pyarrow as pa,pyarrow.parquet as pq
from transformers import AutoTokenizer
root=Path('/root/autodl-tmp/Glasseek');old=root/'data/glasseek/sft-eval-aligned-128k';out=root/'data/glasseek/sft-eval-fixed-128k';out.mkdir(parents=True,exist_ok=True)
ref=root/'analysis/checkpoint_reeval/remote_snapshot/checkpoints/glasseek_step10_rl9b_subset/output/browsecomp_rl9b_non_timeout_34/conv/q000001/q000001__r1.json';h=json.loads(ref.read_text())['history'][0];parts=[]
for part in h['system_prompt'].split('\n\n'):
 if part.startswith(('Terminal backend:','Workspace (snapshot','Active Hermes profile:','Conversation started:')):continue
 parts.append(part)
body='\n\n'.join(parts);(out/'eval_system_body.txt').write_text(body,encoding='utf8');(out/'original_eval_system_prompt.txt').write_text(h['system_prompt'],encoding='utf8')
tok=AutoTokenizer.from_pretrained(root/'checkpoints/glasseek/sft_128k_length_curriculum_20261104/global_step_80/huggingface',local_files_only=True,trust_remote_code=True)
t=tok.apply_chat_template([{'role':'system','content':body},{'role':'user','content':'TEST'}],tools=h['tools'],tokenize=False,add_generation_prompt=False)
system=t.split('<|im_start|>system\n',1)[1].split('<|im_end|>',1)[0];(out/'fixed_rendered_system_prompt.txt').write_text(system,encoding='utf8')
nt=len(tok.encode(system,add_special_tokens=False));auditold=iter(x for x in map(json.loads,(old/'audit.jsonl').open()) if 'original_system_tokens' in x);counts=collections.Counter();diff=collections.Counter();lens=[]
with (out/'all_filtered.jsonl.tmp').open('w') as w, (out/'audit.jsonl.tmp').open('w') as a:
 for line in (old/'all_filtered.jsonl').open():
  x=json.loads(line);record=next(auditold);prev=x['messages'][0]['content'];x['messages'][0]['content']=system;w.write(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n');counts['retained']+=1
  record['fixed_system_tokens']=nt
  if 'estimated_total_tokens' in record:record['fixed_total_tokens_estimate']=record['estimated_total_tokens']+nt-record['aligned_system_tokens'];counts['over_128k']+=record['fixed_total_tokens_estimate']>131072
  a.write(json.dumps(record,ensure_ascii=False)+'\n');diff[record['original_system_tokens']]+=1
for name in ['all_filtered.jsonl','audit.jsonl']:os.replace(out/(name+'.tmp'),out/name)
for split in ('train','val'):
 p=pq.ParquetFile(old/f'{split}.parquet');schema=p.schema_arrow;writer=pq.ParquetWriter(out/f'{split}.parquet.tmp',schema,compression='zstd')
 for batch in p.iter_batches(batch_size=64):
  rows=batch.to_pylist()
  for row in rows:
   m=json.loads(row['messages']);m[0]['content']=system;row['messages']=json.dumps(m,ensure_ascii=False)
  writer.write_table(pa.Table.from_pylist(rows,schema=schema));counts[split]+=len(rows)
 writer.close();os.replace(out/f'{split}.parquet.tmp',out/f'{split}.parquet')
meta={'counts':dict(counts),'system_tokens':nt,'system_characters':len(system),'sha256':hashlib.sha256(system.encode()).hexdigest(),'prompt_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'reference':str(ref),'tool_names':sorted(t['function']['name'] for t in h['tools']),'note':'Exact at chat-template system serialization when using eval_system_body.txt, same tool schemas and matching Qwen tokenizer/chat template; requests not yet captured or verified.'}
(out/'manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n');print(json.dumps(meta,ensure_ascii=False),flush=True)
