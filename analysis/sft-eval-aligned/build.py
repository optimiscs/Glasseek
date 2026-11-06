#!/usr/bin/env python3
"""Build a separate evaluation-aligned dataset; never modify source files."""
import collections, hashlib, json, os, re, statistics
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT=Path('/root/autodl-tmp/Glasseek')
OUT=ROOT/'data/glasseek/sft-eval-aligned-128k'
SRC=ROOT/'data/Glasseek-SFT-5.7K/train.jsonl'
SELECT=ROOT/'data/glasseek/sft-128k/selection.jsonl'
REF=ROOT/'analysis/checkpoint_reeval/remote_snapshot/checkpoints/glasseek_step10_rl9b_subset/output/browsecomp_rl9b_non_timeout_34/conv/q000001/q000001__r1.json'
TOK=ROOT/'checkpoints/glasseek/sft_128k_length_curriculum_20261104/global_step_80/huggingface'
OUT.mkdir(parents=True,exist_ok=True)
ref=json.loads(REF.read_text())['history'][0]; tools=ref['tools']; template=ref['system_prompt']
spec={t['function']['name']:t['function']['parameters'] for t in tools}
tok=AutoTokenizer.from_pretrained(TOK,local_files_only=True,trust_remote_code=True)
probe=tok.apply_chat_template([{'role':'system','content':'EVAL_SYSTEM_SENTINEL'},{'role':'user','content':'HELLO'}],tools=tools,tokenize=False,add_generation_prompt=False)
prefix=probe.split('<|im_start|>system\n',1)[1].split('\n\nEVAL_SYSTEM_SENTINEL<|im_end|>',1)[0]
assert len(prefix)>1000 and all(name in prefix for name in spec)
# User explicitly requested these original per-session fields; retain their
# original paragraphs byte-for-byte, and never fabricate missing fields.
def extract_fields(s):
 body=s.split('</tools>',1)[-1]
 parts=body.split('\n\n');out={}
 for part in parts:
  for key in ('Terminal backend:', 'Workspace (snapshot', 'Active Hermes profile:'):
   if part.startswith(key):out[key]=part
 for key in ('Conversation started','Model','Provider'):
  matches=re.findall(r'(?m)^'+re.escape(key)+r':[^\n]*',body)
  if matches:out[key]=matches[-1]
 return out

def aligned_body(fields):
 blocks=[]
 for block in template.split('\n\n'):
  selected=False
  for key in ('Terminal backend:', 'Workspace (snapshot', 'Active Hermes profile:'):
   if block.startswith(key):
    if key in fields:blocks.append(fields[key])
    selected=True;break
  if selected:continue
  if block.startswith('Conversation started:'):
   selected_lines=[fields[k] for k in ('Conversation started','Model','Provider') if k in fields]
   if selected_lines:blocks.append('\n'.join(selected_lines))
   continue
  blocks.append(block)
 return '\n\n'.join(blocks)

def analyze(messages):
 badnames=set();badparams=[]; malformed=[]
 for idx,m in enumerate(messages):
  if m.get('role')!='assistant':continue
  s=m.get('content') or ''
  opens=s.count('<tool_call>');closed=s.count('</tool_call>')
  calls=re.findall(r'<tool_call>(.*?)</tool_call>',s,re.S)
  if opens!=closed or len(calls)!=opens:malformed.append(idx)
  for block in calls:
   matches=re.findall(r'<function=([^>\s]+)>(.*?)</function>',block,re.S)
   if len(matches)!=1:malformed.append(idx);continue
   name,body=matches[0]
   if name not in spec:badnames.add(name);continue
   params=re.findall(r'<parameter=([^>\s]+)>(.*?)</parameter>',body,re.S)
   names=[n for n,_ in params];valid=set(spec[name].get('properties',{}));required=set(spec[name].get('required',[]))
   if (len(names)!=len(set(names)) or not required<=set(names) or set(names)-valid):badparams.append((idx,name,names))
 return badnames,badparams,malformed

selection={r['source_line']:r for r in map(json.loads,SELECT.open())}
assert len(selection)==5762
schema=pa.schema([('messages',pa.string()),('tools',pa.string()),('enable_thinking',pa.bool_())])
writers={s:pq.ParquetWriter(str(OUT/f'{s}.parquet.tmp'),schema,compression='zstd') for s in ('train','val')};batches={s:[] for s in writers}
def flush(s):
 if batches[s]:writers[s].write_table(pa.Table.from_pylist(batches[s],schema=schema));batches[s].clear()
summary=collections.Counter(); retained=[];system_cache={};length_cache={};before=[];after=[];filtered=[]
with (OUT/'all_filtered.jsonl.tmp').open('w',encoding='utf8') as allout,(OUT/'audit.jsonl.tmp').open('w',encoding='utf8') as audit:
 for i,line in enumerate(SRC.open(encoding='utf8'),1):
  obj=json.loads(line);messages=obj['messages'];ss=[m for m in messages if m.get('role')=='system']
  if len(ss)!=1 or messages[0] is not ss[0]:raise ValueError(f'{i}: expected exactly one leading system')
  old=ss[0]['content'];badnames,badparams,malformed=analyze(messages);prior=selection[i];split=prior.get('split')
  reasons=[]
  if badnames:reasons.append('out_of_whitelist_tool')
  if badparams:reasons.append('invalid_tool_parameters')
  if malformed:reasons.append('malformed_tool_call')
  summary['source']+=1
  for z in reasons:summary[z]+=1
  rec={'source_line':i,'original_split':split,'reasons':reasons,'extra_tools':sorted(badnames),'invalid_parameter_calls':badparams[:5]}
  if reasons:
   audit.write(json.dumps(rec,ensure_ascii=False)+'\n');continue
  if old not in system_cache:
   fields=extract_fields(old);body=aligned_body(fields);new=prefix+'\n\n'+body
   # Dynamic fields are retained, present exactly once, with no fabricated values.
   assert extract_fields(new)==fields
   system_cache[old]=(new,fields)
  new,fields=system_cache[old];messages[0]=dict(messages[0],content=new)
  if old not in length_cache:
   # System content is isolated by chat special tokens in both old and new.
   oldlen=len(tok.encode(old,add_special_tokens=False));newlen=len(tok.encode(new,add_special_tokens=False));length_cache[old]=(oldlen,newlen)
  oldlen,newlen=length_cache[old];before.append(oldlen);after.append(newlen)
  rec.update(original_system_tokens=oldlen,aligned_system_tokens=newlen,retained_fields=list(fields))
  obj['messages']=messages
  allout.write(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n');summary['retained_all']+=1
  if split:
   assert prior['exact']
   aligned_total=prior['tokens']+(newlen-oldlen)
   rec['estimated_total_tokens']=aligned_total
   if aligned_total<=131072:
    batches[split].append({'messages':json.dumps(messages,ensure_ascii=False),'tools':None,'enable_thinking':None});summary[split]+=1
    if len(batches[split])>=64:flush(split)
   else:summary['over_128k_after_transform']+=1;rec['reasons'].append('over_128k_after_transform')
  else:summary['originally_excluded']+=1
  audit.write(json.dumps(rec,ensure_ascii=False)+'\n')
  if i%500==0:print('processed',i,dict(summary),flush=True)
for split in writers:flush(split);writers[split].close();os.replace(OUT/f'{split}.parquet.tmp',OUT/f'{split}.parquet')
for name in ['all_filtered.jsonl','audit.jsonl']:os.replace(OUT/(name+'.tmp'),OUT/name)
manifest={'counts':dict(summary),'system_token_stats':{'before_mean':statistics.mean(before),'after_mean':statistics.mean(after),'before_median':statistics.median(before),'after_median':statistics.median(after),'before_min':min(before),'before_max':max(before),'after_min':min(after),'after_max':max(after)},'whitelist':sorted(spec),'reference':str(REF),'source':str(SRC),'selection':str(SELECT),'tokenizer':str(TOK),'train_val_rule':'preserve original selection split; no added question leakage','sequence_length':'original exact verl length + isolated system-token difference; verify separately','dynamic_fields':'preserved when originally present; absent fields remain absent','warnings':['full prompts differ where preserved runtime fields differ','tool response content and first user research instructions are not rewritten','no guarantee model outputs meet semantics or original context remains accurate']}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
