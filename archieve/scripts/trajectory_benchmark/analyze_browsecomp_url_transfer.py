"""Count search→extract URL transfer using recorded BrowseComp main-agent sessions.

Comparisons are limited to search results recorded earlier in the same saved session;
repeated tool-call IDs in resumed snapshots are counted once per case. No network calls.
"""
import collections
import glob
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl, unquote, quote

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'Evaluation/data/runtime/hermes/sessions'
OUT=ROOT/'reports/browsecomp-url-transfer-20261107'
RUNS={
 'step39_20261106_c64': 'step39_browsecomp128_c64_20261106_154711',
 'rl9b_20261103_c32':'glasseek_rl9b_quest128_32',
 'fair120_step39_20261106':'fair120_20261106_step39',
 'fair120_step76_20261106':'fair120_20261106_step76',
}
BLOCK=re.compile(r'captcha|cloudflare|access denied|access forbidden|just a moment|verify (?:that )?you are human|robot check|blocked request|enable javascript',re.I)
TEXT_URL=re.compile(r'https?://[^\s"<>\\]+')
REDIRECT_KEYS={'url','uri','u','target','redirect','redirect_uri','redirect_url','destination','dest','next','continue','link'}
UNRESERVED=set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~')

def decode_unreserved(text):
    return re.sub(r'%([0-9A-Fa-f]{2})',lambda m: chr(int(m.group(1),16)) if chr(int(m.group(1),16)) in UNRESERVED else m.group(0).upper(),text)

def urlparts(url):
    try:
        v=urlsplit(url)
        if v.scheme.lower() not in ('http','https') or not v.hostname:return None
        port=v.port
        authority=v.hostname.lower().rstrip('.')
        if port and not ((v.scheme.lower()=='http' and port==80) or (v.scheme.lower()=='https' and port==443)):
            authority+=':'+str(port)
        path=decode_unreserved(v.path or '/').rstrip('/') or '/'
        return {'base':(authority,path),'scheme':v.scheme.lower(),'query':tuple(sorted(parse_qsl(v.query,keep_blank_values=True))),
                'fragment':v.fragment}
    except (ValueError,TypeError):return None

def classify(url,seen):
    """seen: (exact set, canonical base -> earlier URLs). No external lookup."""
    exact,bybase=seen
    if url in exact:return 'exact',url
    v=urlparts(url)
    if not v:return 'invalid_url',None
    choices=bybase.get(v['base'],[])
    if not choices:return 'not_seen',None
    for old in choices:
        o=urlparts(old)
        if o and o['query']==v['query']:
            return 'normalized_same_query',old
    # Same host/path alone is a candidate relation, not proof of same document.
    def overlap(old):
        o=urlparts(old)
        return sum((collections.Counter(o['query']) & collections.Counter(v['query'])).values()) if o else -1
    return 'query_changed_same_host_path',max(choices,key=overlap)

def change_type(old,new):
    a=urlparts(old);b=urlparts(new)
    if not a or not b:return 'unknown'
    qa=collections.Counter(a['query']);qb=collections.Counter(b['query'])
    removed=qa-qb;added=qb-qa
    if added and removed:return 'changed_or_mixed'
    if added:return 'added_only'
    if removed:return 'removed_only'
    return 'equivalent'

def changes_redirect_target(old,new):
    # Changing an embedded destination can designate a wholly different page.
    a=urlparts(old);b=urlparts(new)
    if not a or not b:return False
    keys=REDIRECT_KEYS
    aa=collections.defaultdict(list);bb=collections.defaultdict(list)
    for k,v in a['query']:aa[k.casefold()].append(v)
    for k,v in b['query']:bb[k.casefold()].append(v)
    return any(aa[k]!=bb[k] for k in keys)

def response_url_key(url):
    """Normalize Unicode/percent spelling without decoding reserved delimiters."""
    try:
        v=urlsplit(url)
        if v.scheme.lower() not in ('http','https') or not v.hostname:return None
        path=quote(v.path or '/',safe="/%:@!$&'()*+,;=-._~")
        path=re.sub(r'%[0-9a-fA-F]{2}',lambda m:m.group(0).upper(),path)
        port=v.port
        authority=v.hostname.lower().rstrip('.')
        if port and not ((v.scheme.lower()=='http' and port==80) or (v.scheme.lower()=='https' and port==443)):authority+=':'+str(port)
        return (v.scheme.lower(),authority,path,tuple(sorted(parse_qsl(v.query,keep_blank_values=True))))
    except (TypeError,ValueError):return None

def response_json(content):
    if isinstance(content,dict):return content
    if not isinstance(content,str):return None
    pos=content.find('{')
    if pos<0:return None
    try:return json.JSONDecoder().raw_decode(content[pos:])[0]
    except (ValueError,TypeError):return None

def parse_args(value):
    try:
        if isinstance(value,str):value=json.loads(value)
        return value if isinstance(value,dict) else None
    except (ValueError,TypeError):return None

def extract_urls(args):
    if not isinstance(args,dict):return []
    value=args.get('urls',args.get('url',[]))
    if isinstance(value,str):
        # A JSON-encoded array is a common generated argument; do not infer arbitrary text.
        if value.lstrip().startswith('['):
            try:value=json.loads(value)
            except ValueError:return []
        else:value=[value]
    return value if isinstance(value,list) else []

def output_status(response,url,index,total):
    if response is None:return 'missing_result','missing',None
    if not isinstance(response,dict):return 'unparseable_result','missing',None
    results=response.get('results')
    if not isinstance(results,list):return ('call_error' if response.get('error') or response.get('success') is False else 'unparseable_result'),'missing',None
    matches=[x for x in results if isinstance(x,dict) and x.get('url')==url]
    normalized=False
    if not matches and response_url_key(url):
        matches=[x for x in results if isinstance(x,dict) and response_url_key(x.get('url'))==response_url_key(url)]
        normalized=bool(matches)
    if len(matches)>1:
        if index<len(results) and isinstance(results[index],dict) and results[index] in matches:matches=[results[index]]
        else:matches=[]
    if not matches and len(results)==total and index<len(results) and isinstance(results[index],dict):
        return 'unverified_result_url','different_or_missing',results[index].get('url')
    if not matches:return 'missing_url_result','missing',None
    row=matches[0]
    returned=row.get('url');identity='normalized' if normalized else 'exact'
    if row.get('error') or row.get('success') is False:return 'error',identity,returned
    content=row.get('content')
    if not isinstance(content,str) or not content.strip():return 'empty',identity,returned
    if BLOCK.search(content[:12000]):return 'content_with_block_signal',identity,returned
    return 'content_without_block_signal',identity,returned

def redact(url):
    # Show structure and query names, not values which may contain credentials.
    try:
        v=urlsplit(url);names=[k for k,_ in parse_qsl(v.query,keep_blank_values=True)]
        return v.scheme+'://'+v.netloc+v.path+('?'+'&'.join(quote(k)+'=<redacted>' for k in names) if names else '')
    except ValueError:return '<invalid>'

def analyze(label,run):
    paths=sorted(glob.glob(str(BASE/run/'quest_qwen27b_128/*/main/*.json')))
    out=collections.Counter();cases=set();tool_calls_seen=set();search_records={};extract_records={};examples=[]
    data_errors=[];duplicate_calls=0;idless=0;search_response_unparseable=0;orphan_responses=0
    seen_by_case=collections.defaultdict(set)
    for path in paths:
        case=Path(path).parent.parent.name;cases.add(case)
        try:
            with open(path) as f:session=json.load(f)
        except (ValueError,OSError) as exc:
            data_errors.append({'file':path,'error':type(exc).__name__});continue
        seen_exact=set();seen_base=collections.defaultdict(list);seen_search_embedded_targets=set();seen_extract_urls=set();seen_extract_content_urls=set();callmap={};pending=[]
        for mi,m in enumerate(session.get('messages',[])):
            if m.get('role')=='assistant':
                for ci,c in enumerate(m.get('tool_calls') or []):
                    f=c.get('function') or {};name=f.get('name');cid=c.get('id') or c.get('call_id')
                    if not cid:
                        idless+=1;continue
                    key=(case,cid);args=parse_args(f.get('arguments'));callmap[cid]=(name,args)
                    if key in tool_calls_seen:duplicate_calls+=1
                    else:tool_calls_seen.add(key)
                    if name=='web_extract':
                        urls=extract_urls(args)
                        if key not in extract_records:
                            extract_records[key]={'case':case,'session':session.get('session_id'),'call_id':cid,'args':args,'items':[],'response_urls':None}
                            for j,url in enumerate(urls):
                                if not isinstance(url,str):
                                    cat,old='invalid_url',None
                                else:cat,old=classify(url,(seen_exact,seen_base))
                                change=change_type(old,url) if cat=='query_changed_same_host_path' else None
                                extract_records[key]['items'].append({'url':url,'category':cat,'original':old,'change_type':change,'status':'missing_result','returned_url_identity':'missing','returned_url':None,
                                    'previous_search_embedded_target':isinstance(url,str) and url in seen_search_embedded_targets,
                                    'previous_extract_result_url':isinstance(url,str) and url in seen_extract_urls,
                                    'previous_extracted_content_url':isinstance(url,str) and url in seen_extract_content_urls})
                        # Result may be in a later, overlapping snapshot.
            elif m.get('role')=='tool':
                cid=m.get('tool_call_id');name,args=callmap.get(cid,(m.get('name') or m.get('tool_name'),None));key=(case,cid)
                if not cid or cid not in callmap:orphan_responses+=1
                value=response_json(m.get('content'))
                if name=='web_search':
                    if value is None:search_response_unparseable+=1;continue
                    data=value.get('data') if isinstance(value,dict) else None
                    web=data.get('web') if isinstance(data,dict) else None
                    urls=[z.get('url') for z in web if isinstance(z,dict) and isinstance(z.get('url'),str)] if isinstance(web,list) else []
                    for u in urls:
                        seen_exact.add(u);part=urlparts(u)
                        if part and u not in seen_base[part['base']]:seen_base[part['base']].append(u)
                        if part:
                            for k,target in part['query']:
                                if k.casefold() in REDIRECT_KEYS and urlparts(target):seen_search_embedded_targets.add(target)
                        seen_by_case[case].add(u)
                    if key not in search_records:search_records[key]=list(urls)
                elif name=='web_extract' and key in extract_records:
                    rec=extract_records[key]
                    if rec['response_urls'] is None and isinstance(value,dict) and isinstance(value.get('results'),list):
                        rec['response_urls']=[z.get('url') for z in value['results'] if isinstance(z,dict)]
                    for j,item in enumerate(rec['items']):
                        if item['status']=='missing_result':
                            item['status'],item['returned_url_identity'],item['returned_url']=output_status(value,item['url'],j,len(rec['items']))
                    if isinstance(value,dict) and isinstance(value.get('results'),list):
                        for result in value['results']:
                            if not isinstance(result,dict):continue
                            u=result.get('url')
                            if isinstance(u,str):seen_extract_urls.add(u)
                            content=result.get('content')
                            if isinstance(content,str):seen_extract_content_urls.update(m.group(0).rstrip('.,;)]') for m in TEXT_URL.finditer(content))
        # Mark earlier search results actually selected later in each visible snapshot.
        # (extract classification already records only searches preceding that call.)
    search_unique={(case,u) for (case,_),urls in search_records.items() for u in urls}
    used_exact=set();used_norm=set();used_params=set();percase=collections.defaultdict(collections.Counter)
    argkeys=collections.Counter();status_by_class=collections.defaultdict(collections.Counter);identity_by_class=collections.defaultdict(collections.Counter);change=collections.Counter();examples_by_category=collections.Counter();query_preservation=collections.Counter();redirect_candidate_status=collections.Counter();returned_mismatch=[];schema_issues=collections.Counter();not_seen_context=collections.Counter();over_cap_calls=[]
    for (case,cid),r in extract_records.items():
        args=r['args'];argkeys[str(tuple(sorted(args))) if args is not None else 'invalid']+=1
        if args is None:out['invalid_call_args']+=1
        if isinstance(args,dict):
            if 'urls' not in args:schema_issues['missing_urls']+=1
            if set(args)-{'urls'}:schema_issues['extra_arguments']+=1
            urlsarg=args.get('urls')
            if 'urls' in args and not isinstance(urlsarg,list):schema_issues['urls_not_array']+=1
            if isinstance(urlsarg,list) and len(urlsarg)>5:schema_issues['exceeds_five_urls']+=1
            if isinstance(urlsarg,list) and len(urlsarg)>5:
                response_keys={response_url_key(z) for z in (r['response_urls'] or [])};response_keys.discard(None)
                def matched(u):return response_url_key(u) is not None and response_url_key(u) in response_keys
                over_cap_calls.append({'case':case,'requested_items':len(urlsarg),'recorded_result_items':len(r['response_urls'] or []),
                    'first_five_with_matching_result':sum(matched(u) for u in urlsarg[:5]),
                    'after_five_items':len(urlsarg)-5,'after_five_with_matching_result':sum(matched(u) for u in urlsarg[5:])})
            if isinstance(urlsarg,list) and any(not isinstance(u,str) for u in urlsarg):schema_issues['url_item_not_string']+=1
        if not r['items']:out['calls_without_url_items']+=1
        out['extract_calls']+=1;percase[case]['extract_calls']+=1
        for item in r['items']:
            out['extract_url_occurrences']+=1;cat=item['category'];out['category:'+cat]+=1;percase[case]['category:'+cat]+=1
            status_by_class[cat][item['status']]+=1
            identity_by_class[cat][item['returned_url_identity']]+=1
            if cat=='not_seen':
                if item['previous_search_embedded_target']:not_seen_context['earlier_search_embedded_target_exact']+=1
                if item['previous_extract_result_url']:not_seen_context['earlier_extract_result_url_exact']+=1
                if item['previous_extracted_content_url']:not_seen_context['earlier_extract_content_url_exact_heuristic']+=1
                if item['previous_extract_result_url'] or item['previous_extracted_content_url']:not_seen_context['either_earlier_extract']+=1
            if item['returned_url_identity']=='different_or_missing' and len(returned_mismatch)<20:
                returned_mismatch.append({'case':case,'requested':redact(item['url']) if isinstance(item['url'],str) else '<invalid>',
                                          'recorded_result_url':redact(item['returned_url']) if isinstance(item['returned_url'],str) else None,'status':item['status']})
            old=item['original']
            if old is not None:
                prior=urlparts(old);current=urlparts(item['url'])
                if prior and current and prior['query']:
                    query_preservation['extract_occurrences_prior_search_has_query']+=1
                    query_preservation['category:'+cat]+=1
                    query_preservation['query_pairs_equal' if prior['query']==current['query'] else 'query_pairs_changed']+=1
                    query_preservation['status:'+item['status']]+=1
                if cat=='query_changed_same_host_path' and changes_redirect_target(old,item['url']):
                    redirect_candidate_status[item['status']]+=1
                pair=(case,old)
                if cat=='exact':used_exact.add(pair)
                elif cat=='normalized_same_query':used_norm.add(pair)
                elif cat=='query_changed_same_host_path':used_params.add(pair);change[item['change_type']]+=1
            if cat in ('query_changed_same_host_path','normalized_same_query','not_seen') and examples_by_category[cat]<25:
                examples.append({'case':case,'session_id':r['session'],'category':cat,'change_type':item['change_type'],
                                 'source_url_redacted':redact(old) if old else None,
                                 'extract_url_redacted':redact(item['url']) if isinstance(item['url'],str) else '<invalid>',
                                 'status':item['status']})
                examples_by_category[cat]+=1
    # Only source URL picked by matching recorded; lower bound when multiple source URLs share base path.
    used_any=(used_exact|used_norm|used_params) & search_unique
    search_query_unique={(case,u) for case,u in search_unique if urlparts(u) and urlparts(u)['query']}
    return {'run':run,'sessions':len(paths),'cases':len(cases),'unique_tool_calls_all_names':len(tool_calls_seen),
            'duplicate_tool_call_occurrences_skipped':duplicate_calls,'idless_calls_skipped':idless,'orphan_tool_responses':orphan_responses,
            'unparseable_search_response_occurrences':search_response_unparseable,'file_errors':data_errors,
            'search_calls_with_parsed_result':len(search_records),'search_url_occurrences':sum(map(len,search_records.values())),
            'search_urls_unique_per_case':len(search_unique),'search_urls_later_requested_exact_unique':len(used_exact & search_unique),
            'search_urls_with_query_unique_per_case':len(search_query_unique),
            'search_urls_with_query_later_requested_exact_unique':len(used_exact & search_query_unique),
            'search_urls_later_requested_any_candidate_matching_unique':len(used_any),'extract_counts':dict(out),
            'status_by_category':{k:dict(v) for k,v in status_by_class.items()},
            'returned_url_identity_by_category':{k:dict(v) for k,v in identity_by_class.items()},'query_change_types':dict(change),
            'extract_occurrences_where_prior_search_has_query':dict(query_preservation),
            'query_changed_redirect_target_candidates':dict(redirect_candidate_status),
            'extract_argument_keys':dict(argkeys),'extract_schema_issue_counts_overlapping':dict(schema_issues),
            'over_five_url_calls':over_cap_calls,
            'not_seen_earlier_extract_context':dict(not_seen_context),
            'returned_url_mismatch_examples_redacted':returned_mismatch,'per_case':dict(percase),'examples_redacted':examples}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    allruns={label:analyze(label,run) for label,run in RUNS.items()}
    result={'scope':'BrowseComp Quest 128 subset; saved main-agent messages only; real recorded outcomes, no replay.',
      'analysis_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'method':{'unit_search':'distinct (case, original result URL); repetitions across search calls collapsed',
                'unit_extract':'each URL occurrence in a distinct extract tool call',
                'ordering':'matches only search results recorded earlier in the same saved session; this does not establish exactly what survived in the model context after compaction',
                'snapshots':'tool-call IDs deduplicated per case across overlapping sessions; earliest recorded extract context',
                'exact':'byte-for-byte string equality',
                'normalized_same_query':'same host and normalized path plus same decoded query pairs; scheme, slash, encoding, query order may differ; not proof equivalence',
                'query_changed_same_host_path':'same host and normalized path but query parameters differ; plausible candidate, not proof of same resource',
                'success':'non-empty content, no reported error; separately count potential block/CAPTCHA text. No factual correctness validation.',
                'returned_url_identity':'match extract input URL to result.url literally, or by URL percent-encoding normalization; other positional matches are unverified and excluded from success. It does not verify the final HTTP redirect URL.',
                'prior_extract_content':'exact URL appears in text of an earlier extract response in the same snapshot, using a heuristic URL parser; not proof that this occurrence caused the later action.',
                'embedded_target':'extract URL exactly appeared as an http(s) value of a redirect-like query parameter in a preceding search result URL; not proof that the wrapper actually redirects.',
                'limits':'Exact preceding-search matches are a lower bound because compacted histories and child-agent histories can be invisible. Same host/path with changed query is only a candidate and may be a different resource; it is neither a lower nor upper bound.',
                'example_privacy':'URL query values are redacted.'},'runs':allruns}
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for k,v in allruns.items():
        print(k,json.dumps({z:v[z] for z in ('sessions','cases','search_calls_with_parsed_result','search_url_occurrences','search_urls_unique_per_case','search_urls_later_requested_exact_unique','search_urls_later_requested_any_candidate_matching_unique','extract_counts','status_by_category','query_change_types','extract_argument_keys','duplicate_tool_call_occurrences_skipped','file_errors')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
