import json,collections,statistics,re
from pathlib import Path
P=Path('/root/autodl-tmp/Quest/web/trace-viewer/datasets');O=Path('/root/autodl-tmp/Glasseek/experiments/extract-128-audit');D=['glasseek-sft9b-browsecomp-128','glasseek-rl9b-browsecomp-128-32']
def parse(s):
 if not isinstance(s,str):return None
 i=s.find('{')
 if i<0:return None
 try:return json.JSONDecoder().raw_decode(s[i:])[0]
 except ValueError:return None
for d in D:
 data=json.load((P/d/'traces.json').open());counts=collections.Counter();reason=collections.Counter();counts_calls=collections.Counter();sample=[];rat=[];no_meta=[];examples={}
 for case in data['cases']:
  for s in case['steps']:
   tool=s.get('tool') or {}
   if tool.get('name')!='web_extract':continue
   counts_calls['calls']+=1;raw=s.get('rawToolOutput') or tool.get('rawOutput') or '';o=parse(raw)
   if not isinstance(o,dict) or not isinstance(o.get('results'),list):counts_calls['unparseable']+=1;continue
   counts_calls['parseable']+=1
   for item in o['results']:
    if not isinstance(item,dict):counts['non_dict']+=1;continue
    counts['result_entries']+=1;v=item.get('retrieval') if isinstance(item.get('retrieval'),dict) else {};err=item.get('error');body=item.get('content') or ''; r=v.get('reason') or '';method=v.get('method') or ''
    if err:counts['error']+=1
    elif not body:counts['empty_no_error']+=1
    else:
     counts['nonempty_success']+=1
     if re.search(r'(?i)(captcha|verify you are human|unusual traffic|access denied|security verification)',body[:3000]):counts['captcha_or_block_text']+=1
    if v:
     counts['with_retrieval_meta']+=1;reason[r or '(missing)']+=1;counts['method_'+str(method)]+=1
     if type(v.get('input_tokens')) in (int,float) and type(v.get('output_tokens')) in (int,float) and v['input_tokens']>0:
      rat.append((r,v['input_tokens'],v['output_tokens']))
     if r not in examples:examples[r]={'case':case['id'],'url':item.get('url'),'retrieval':v,'error':err,'content_len':len(body)}
    else:
     counts['no_retrieval_meta']+=1
     if len(no_meta)<3:no_meta.append({'case':case['id'],'error':err,'content_prefix':body[:200]})
    if '[Summary of content beyond' in body:counts['explicit_overflow_summary_marker']+=1
    if re.search(r'(?i)\[\.\.\. summary truncated|LLM summarization timed out',body):counts['explicit_legacy_summary_marker']+=1
  
 sel=[(a,b) for r,a,b in rat if r=='selected'];agg={'measured_pairs':len(rat),'selected_measured':len(sel)}
 if sel:
  agg.update({'selected_input_tokens':sum(a for a,b in sel),'selected_output_tokens':sum(b for a,b in sel),'ratio_of_sums':sum(b for a,b in sel)/sum(a for a,b in sel),'median_input':statistics.median(a for a,b in sel),'median_output':statistics.median(b for a,b in sel),'median_ratio':statistics.median(b/a for a,b in sel)})
 result={'dataset':d,'calls':dict(counts_calls),'entries':dict(counts),'reason_counts':dict(reason),'token_stats':agg,'examples':examples,'no_metadata_examples':no_meta};(O/(d+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
