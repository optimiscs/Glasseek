import collections,json,re,random
from pathlib import Path
P=Path('/root/autodl-tmp/Glasseek');R=random.Random(20261105); CALL=re.compile(r'<tool_call>(.*?)</tool_call>',re.S);FN=re.compile(r'<function=([^>]+)>');RESP=re.compile(r'<tool_response>(.*?)</tool_response>',re.S);MARKER=re.compile(r'(?im)^\s*(?:#{1,4}\s*|\*\*\s*)?(?:summary|summari[sz]ed|key (?:points|takeaways|findings)|overview|tldr|tl;dr|摘要|总结|要点|主要内容)(?:\s*\*\*)?\s*[:：\-\n]');ERROR=re.compile(r'(?i)\b(?:error|blocked|failed|timeout|forbidden|not found|captcha|access denied)\b')
meta=[x for x in map(json.loads,(P/'data/glasseek/sft-eval-fixed-128k/audit.jsonl').open()) if not x['reasons']];keep={x['source_line'] for x in meta};stats={k:collections.Counter() for k in ('all','fixed')};reason={k:collections.Counter() for k in stats};sample={k:[] for k in stats}
for i,l in enumerate((P/'data/Glasseek-SFT-5.7K/train.jsonl').open(),1):
 ms=json.loads(l)['messages'];groups=['all']+(['fixed'] if i in keep else []);seen={k:set() for k in groups}
 for j,m in enumerate(ms):
  if m.get('role')!='assistant':continue
  calls=[FN.search(x).group(1) if FN.search(x) else '?' for x in CALL.findall(m.get('content') or '')];n=calls.count('web_extract')
  if not n:continue
  for k in groups:stats[k]['calls']+=n;seen[k].add('trajectory')
  chunks=[];z=j+1
  while z<len(ms) and ms[z].get('role')!='assistant':
   if ms[z].get('role') in ('user','tool'):
    v=ms[z].get('content') or ''; blocks=RESP.findall(v)
    if blocks:chunks.extend(blocks)
    elif ms[z].get('role')=='tool':chunks.append(v)
   z+=1
  if len(chunks)!=len(calls):
   for k in groups:stats[k]['unpaired_calls']+=n
   continue
  for call,resp in zip(calls,chunks):
   if call!='web_extract':continue
   for k in groups:stats[k]['paired_calls']+=1
   try:o,_=json.JSONDecoder().raw_decode(resp[resp.index('{'):])
   except (json.JSONDecodeError,TypeError,ValueError):
    for k in groups:
     stats[k]['non_json_calls']+=1
     if re.match(r'\s*\[web_extract\]',resp):stats[k]['redacted_response_stubs']+=1
    continue
   if not isinstance(o,dict) or not isinstance(o.get('results'),list):
    for k in groups:stats[k]['nonstandard_calls']+=1
    continue
   for item in o['results']:
    if not isinstance(item,dict):continue
    content=item.get('content') or '';error=item.get('error');retrieval=item.get('retrieval');
    for k in groups:
     stats[k]['page_entries']+=1
     if error:stats[k]['page_error']+=1
     elif not isinstance(content,str) or not content.strip():stats[k]['page_empty']+=1
     else:
      stats[k]['page_nonempty']+=1
      if MARKER.search(content):stats[k]['explicit_summary_marker_pages']+=1
      if '[... summary truncated for context management ...]' in content:stats[k]['summary_truncated_marker_pages']+=1;seen[k].add('explicit_truncated_summary')
      if MARKER.search(content) and '[... summary truncated for context management ...]' in content:stats[k]['marker_and_summary_truncated']+=1
      if '[Content truncated — showing first' in content:stats[k]['fallback_truncated_marker_pages']+=1
      if '[Content too large to process:' in content:stats[k]['too_large_marker_pages']+=1
      if '<untrusted_tool_result' in content:stats[k]['nested_wrapper']+=1
      if isinstance(retrieval,dict):stats[k]['retrieval_metadata']+=1;reason[k][retrieval.get('reason','unknown')]+=1
      if len(sample[k])<1000:sample[k].append({'source_line':i,'content':content[:2000],'url':item.get('url',''),'summary_marker':bool(MARKER.search(content))})
 for k in groups:
  if seen[k]:stats[k]['trajectories_with_calls']+=1
  if 'explicit_truncated_summary' in seen[k]:stats[k]['trajectories_with_truncated_summary']+=1
for k in stats:
 rows=sample[k];R.shuffle(rows);(P/f'experiments/extract-128-audit/teacher-{k}-sample.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows[:100]));print(k,json.dumps(dict(stats[k]),ensure_ascii=False),'reasons',dict(reason[k]))
