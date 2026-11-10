import collections,json,sqlite3,time
from pathlib import Path
D=Path(__file__).resolve().parent
H=D.parents[1]/'Evaluation/data/runtime/hermes/profiles'
decoder=json.JSONDecoder()
def parse(text):
 for i,ch in enumerate(text or ''):
  if ch=='{':
   try:
    value=decoder.raw_decode(text[i:])[0]
    if isinstance(value,dict):return value
   except ValueError:pass
 return {}
def reason(e):
 s=str(e).lower()
 if 'private or internal network' in s:return 'private_address_block'
 if 'anti-cheat' in s:return 'anti_cheat_block'
 if 'timeout' in s or 'timed out' in s:return 'timeout'
 if '429' in s:return 'http_429'
 if '403' in s:return 'http_403'
 if '404' in s:return 'http_404'
 if 'max_concurrent_children' in s:return 'too_many_subtasks'
 if 'json' in s:return 'invalid_json_arguments'
 if 'remote sandbox exited' in s:return 'sandbox_execution_error'
 return str(e)[:180]
report={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'models':{}}
for label in ['step39','step76','sft9b']:
 stats=collections.defaultdict(collections.Counter);errors=collections.defaultdict(collections.Counter);extras=collections.Counter();hits=[];impacted=collections.defaultdict(set)
 profiles=list(H.glob(f'fair120_20261106_{label}_*/state.db'))
 for db in profiles:
  with sqlite3.connect(f'file:{db}?mode=ro',uri=True,timeout=10) as c:
   rows=c.execute("SELECT id,session_id,role,tool_name,tool_call_id,tool_calls,content FROM messages WHERE COALESCE(inherited_from_previous_session,0)=0 ORDER BY id").fetchall()
  calls={}
  for rid,sid,role,name,tid,tcs,content in rows:
   if role=='assistant':
    for tc in json.loads(tcs or '[]'):calls[tc.get('id')]=tc
  seen=set()
  for rid,sid,role,name,tid,tcs,content in rows:
   if role!='tool':continue
   key=tid or ('row',rid)
   if key in seen:extras['duplicate_results_excluded']+=1;continue
   seen.add(key);text=content or '';obj=parse(text);tc=calls.get(tid,{});name=name or tc.get('function',{}).get('name','unknown')
   if obj.get('status')=='tool_budget_denied':extras['budget_denied_excluded']+=1;continue
   if text.startswith('Skipped:'):extras['invalid_batch_skipped_excluded']+=1;continue
   if 'does not exist. Available tools:' in text:extras['unregistered_tool_calls']+=1;errors['unregistered_tool'][name]+=1;continue
   st=stats[name];st['returned_calls']+=1
   failure=obj.get('error') or (str(obj) if obj.get('success') is False or obj.get('status') in ('error','failed') else None)
   if name=='web_extract' and isinstance(obj.get('results'),list):
    pages=obj['results'];bad=[x for x in pages if x.get('error')];st['returned_pages']+=len(pages);st['failed_pages']+=len(bad);st['empty_pages_without_error']+=sum(not x.get('error') and not str(x.get('content') or '').strip() for x in pages)
    if bad:
     st['calls_with_page_error']+=1;st['all_pages_failed' if len(bad)==len(pages) else 'partial_page_failure']+=1
     for x in bad:errors['web_extract_pages'][reason(x['error'])]+=1
    if not pages:st['empty_result_calls']+=1
   if name=='delegate_task' and isinstance(obj.get('results'),list):
    st['subtasks']+=len(obj['results']);st['failed_subtasks']+=sum(x.get('status') in ('error','failed') or bool(x.get('error')) for x in obj['results'])
   if failure:
    st['explicit_error_calls']+=1;errors[name][reason(failure)]+=1;impacted[name].add(db.parent.name)
   elif name=='web_search':
    web=obj.get('data',{}).get('web') if isinstance(obj.get('data',{}),dict) else None
    if isinstance(web,list):st['success_with_results' if web else 'success_zero_results']+=1
    else:st['unclassified_calls']+=1
   elif not obj:st['unclassified_calls']+=1
   if 'Belvedere' in str(tc) and 'Da Silva' in str(tc):hits.append({'profile':db.parent.name,'call':tc,'result':obj})
 report['models'][label]={'profiles':len(profiles),'tools':{k:dict(v) for k,v in stats.items()},'errors':{k:dict(v.most_common()) for k,v in errors.items()},'affected_profiles':{k:len(v) for k,v in impacted.items()},'excluded_or_invalid':dict(extras),'query_matches':hits}
(D/'tool-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
for label,r in report['models'].items():print(label,json.dumps({k:v for k,v in r.items() if k!='query_matches'},ensure_ascii=False))
print(report['time'])
