"""Offline reproduction of current URL normalization and Reader URL construction."""
import collections,glob,json,sys
from pathlib import Path
from urllib.parse import urlsplit,parse_qsl
from yarl import URL
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Evaluation_backend'))
sys.path.insert(0,str(ROOT/'scripts/trajectory_benchmark'))
from tools.url_safety import normalize_url_for_request
from analyze_browsecomp_url_transfer import extract_urls,redact
BASE=ROOT/'Evaluation/data/runtime/hermes/sessions'
RUNS={'step39_20261106_c64':'step39_browsecomp128_c64_20261106_154711','rl9b_20261103_c32':'glasseek_rl9b_quest128_32'}

def run(name):
 counts=collections.Counter();seen=set();examples=[]
 paths=sorted(glob.glob(str(BASE/name/'quest_qwen27b_128/*/main/*.json')))
 for path in paths:
  case=Path(path).parent.parent.name
  with open(path) as f:record=json.load(f)
  for m in record.get('messages',[]):
   for call in m.get('tool_calls') or []:
    func=call.get('function') or {};cid=call.get('id') or call.get('call_id');key=(case,cid)
    if func.get('name')!='web_extract' or key in seen:continue
    seen.add(key)
    try:a=json.loads(func['arguments']) if isinstance(func.get('arguments'),str) else func.get('arguments')
    except (TypeError,ValueError,KeyError):continue
    for index,url in enumerate(extract_urls(a)):
     counts['attempted_url_items']+=1
     current_handler_accepts=isinstance(a,dict) and isinstance(a.get('urls'),list) and index<5
     if current_handler_accepts:counts['current_handler_shape_eligible_url_items']+=1
     if not isinstance(url,str):counts['invalid_input']+=1;continue
     try:
      original=urlsplit(url)
      if original.scheme not in ('http','https') or not original.hostname:counts['invalid_input']+=1;continue
      original.port  # Verify malformed ports consistently with the trajectory matcher.
      normalized=normalize_url_for_request(url)
      inner=urlsplit(normalized)
      # The current FetchProvider.request calls client.get(reader_base + payload['url']).
      reader=URL('https://r.jina.ai/'+normalized)
      q0=original.query;q1=inner.query;q2=reader.raw_query_string
      if original.fragment:counts['url_with_fragment_not_sent_over_http']+=1
      counts['valid_http_url_items']+=1
      if not q0:continue
      counts['with_raw_query']+=1
      if current_handler_accepts:counts['current_handler_shape_eligible_with_query']+=1
      counts['raw_query_equal_after_hermes_normalize' if q0==q1 else 'raw_query_changed_after_hermes_normalize']+=1
      counts['raw_query_equal_at_reader_request' if q0==q2 else 'raw_query_changed_at_reader_request']+=1
      before=parse_qsl(q0,keep_blank_values=True);after=parse_qsl(q2,keep_blank_values=True)
      if before==after:
       counts['ordered_decoded_query_pairs_equal_at_reader_request']+=1
       if current_handler_accepts:counts['current_handler_shape_eligible_decoded_pairs_equal']+=1
      else:
       counts['ordered_decoded_query_pairs_changed_at_reader_request']+=1
       if len(examples)<10:examples.append({'case':case,'input_redacted':redact(url),'normalized_redacted':redact(normalized),'query_names_before':[k for k,_ in before],'query_names_after':[k for k,_ in after]})
     except (TypeError,ValueError,UnicodeError):counts['parse_failure']+=1
 return {'source':name,'counts':dict(counts),'semantic_query_change_examples_redacted':examples}

def main():
 x={'scope':'Offline simulation of current code only; no HTTP traffic; cannot prove Jina forwarded query or verify historical code version.',
    'normalizer_source':'Evaluation_backend/tools/url_safety.py:normalize_url_for_request',
    'transport_source':'Evaluation/services/jina_fetch.py:FetchProvider.request',
    'comparison':'Raw query bytes vs ordered parse_qsl decoded pairs; ordered-pair equality is not proof that the origin server received them. shape_eligible checks only list type and first-five cap, not URL validity or successful network delivery.',
    'runs':{k:run(v) for k,v in RUNS.items()}}
 (ROOT/'reports/browsecomp-url-transfer-20261107/offline_transport.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(x,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
