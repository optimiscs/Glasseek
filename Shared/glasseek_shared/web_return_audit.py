"""Separate batch identity, query labels and reported execution outcomes.

Historical return text is never rewritten. Labels are protocol evidence, not
proof that every query ran or that any returned claim is factually correct.
"""
import collections
import hashlib
import json
import re

from glasseek_shared.historical_tools import load,search_sections,unwrap
from glasseek_shared.search_protocol import validate_queries,decode_query_array

VERSION='search-return-audit-v3'
ERROR=re.compile(r'^\s*(?:Error\s*:|Tool error\s*:|Error (?:searching|executing|fetching)\b)',re.I)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def coverage_report(queries,label_hashes,shared,unlabeled):
    query_hashes=[fingerprint(q) for q in queries]
    requested=collections.Counter(query_hashes);observed=collections.Counter(label_hashes)
    unmatched=observed-requested;missing=requested-observed
    if shared:status='shared_batch_echo'
    elif not label_hashes:status='not_reported'
    elif unmatched:status='rewritten_or_unmatched_labels'
    elif missing:status='partial_labels'
    elif unlabeled:status='complete_labels_with_unlabeled_components'
    elif len(set(queries))!=len(queries):status='duplicate_query_labels'
    else:status='complete_labels'
    return {'status':status,'requested_query_count':len(queries),'observed_label_count':len(label_hashes),
            'observed_label_sha256':label_hashes,
            'matched_query_indices':[[i for i,q in enumerate(query_hashes) if q==label] for label in label_hashes],
            'unmatched_label_count':sum(unmatched.values()),'missing_label_count':sum(missing.values()),
            'unlabeled_result_parts':unlabeled,'per_query_association_verified':status=='complete_labels',
            'needs_review':status!='complete_labels'}


def execution_report(outcomes):
    if outcomes['failure'] and (outcomes['success'] or outcomes['no_error']):status='mixed_reports'
    elif outcomes['failure']:status='reported_failure'
    elif outcomes['success']:status='reported_success'
    elif outcomes['no_error']:status='reported_no_error'
    else:status='unreported'
    return {'status':status,'scope':'historical_reported_fields_only','reported_success_count':outcomes['success'],
            'reported_failure_count':outcomes['failure'],'reported_no_error_count':outcomes['no_error'],'replayed':False}


def payload_presence(value):
    if isinstance(value,str):
        if ERROR.match(value):return 'reported_error'
        try:decoded=load(value)
        except (ValueError,TypeError):decoded=value
        if isinstance(decoded,str):return 'empty_results' if not value.strip() else 'payload_only'
        value=decoded
    for _ in range(4):
        if isinstance(value,list):return 'nonempty_results' if value else 'empty_results'
        if not isinstance(value,dict):return 'unverified'
        if value.get('success') is False or (isinstance(value.get('error'),str) and value['error'].strip()):return 'reported_error'
        if isinstance(value.get('data'),(dict,list)):value=value['data'];continue
        for key in ('web','results','organic','items'):
            if isinstance(value.get(key),list):return 'nonempty_results' if value[key] else 'empty_results'
        return 'unverified'
    return 'unverified'


def query_results_report(queries,coverage,observations):
    hashes={fingerprint(q) for q in queries};states=collections.defaultdict(set)
    for observation in observations:
        if observation['query_label_sha256'] in hashes:states[observation['query_label_sha256']].add(observation['state'])
    failures={q for q,s in states.items() if s & {'reported_error','empty_results'}}
    nonempty={q for q,s in states.items() if 'nonempty_results' in s}
    if len(hashes)==len(queries) and failures:status='some_reported_empty_or_error';all_present=False
    elif coverage['per_query_association_verified'] and nonempty==hashes:status='all_reported_nonempty';all_present=True
    else:status='unverified';all_present=None
    return {'status':status,'all_queries_have_reported_nonempty_results':all_present,
            'reported_nonempty_query_count':len(nonempty),'reported_empty_or_error_query_count':len(failures),
            'scope':'explicit_per_query_payloads_or_reported_result_counts','observations':observations}


def analyze_search_return(query,text,link,components=None):
    if components:
        audits=[analyze_search_return(query,part,link) for part in components]
        result=analyze_search_return(query,text,link)
        labels=[label for audit in audits for label in audit['coverage']['observed_label_sha256']]
        shared=any(audit['coverage']['status']=='shared_batch_echo' for audit in audits)
        unlabeled=sum(audit['coverage']['unlabeled_result_parts'] or int(not audit['coverage']['observed_label_count']) for audit in audits)
        result['coverage']=coverage_report(validate_queries(query),labels,shared,unlabeled)
        observations=[observation for audit in audits for observation in audit['query_results']['observations']]
        result['query_results']=query_results_report(validate_queries(query),result['coverage'],observations)
        outcomes=collections.Counter()
        for audit in audits:
            for key in ('success','failure','no_error'):outcomes[key]+=audit['execution']['reported_'+key+'_count']
        result['execution']=execution_report(outcomes)
        result.update(representation='concatenated_call_returns',decoding='original_return_components',
                      component_representations=[audit['representation'] for audit in audits],source_component_count=len(audits))
        return result
    queries=validate_queries(query);labels=[];shared=False;unlabeled=0;outcomes=collections.Counter();observations=[]
    decoding='parsed';representation='plain_text'
    try:value=load(text)
    except (ValueError,TypeError):value=text;decoding='opaque_preserved'
    if isinstance(value,(dict,list)):representation='json_unlabeled'

    def reported(node):
        if isinstance(node,str):
            if ERROR.match(node):outcomes['failure']+=1
            return
        if not isinstance(node,dict):return
        if any(k in node for k in ('title','link','description','content','snippet')) and 'query' not in node:return
        success=node.get('success');error=node.get('error')
        if success is True:outcomes['success']+=1
        if success is False or (isinstance(error,str) and error.strip()):outcomes['failure']+=1
        elif 'error' in node and error is None and success is not True:outcomes['no_error']+=1

    def entry(node):
        nonlocal shared,unlabeled
        if isinstance(node,str):
            try:decoded=load(node)
            except (ValueError,TypeError):decoded=node
            if not isinstance(decoded,str):node=decoded
        if not isinstance(node,dict) or 'query' not in node:
            unlabeled+=1;reported(node);return
        q=decode_query_array(node['query'])
        if isinstance(q,str):labels.append(q)
        elif isinstance(q,list) and all(isinstance(v,str) for v in q):labels.extend(q);shared=True
        else:unlabeled+=1
        reported(node)
        if isinstance(q,str):
            field=next((key for key in ('result','results','data','content') if key in node),None)
            state='reported_error' if node.get('success') is False or (isinstance(node.get('error'),str) and node['error'].strip()) else payload_presence(node[field]) if field else 'unverified'
            observations.append({'query_label_sha256':fingerprint(q),'state':state})
        if 'result' in node:
            result=node['result']
            if isinstance(result,str):
                try:result=load(result)
                except (ValueError,TypeError):pass
            reported(result)

    if not isinstance(value,dict) or 'query' not in value:reported(value)
    if decoding!='opaque_preserved':decoding='structured_json' if isinstance(value,(dict,list)) else 'text_preserved'
    sections=search_sections(text) if isinstance(value,str) else None
    if sections is not None:
        representation='query_grouped_text'
        counts=[int(m[1]) for m in re.finditer(r"(?m)^(?:#{1,6}\s+)?A [A-Za-z0-9 ._-]+ search for '[^\n]*' found ([0-9]+) results:[ \t]*$",unwrap(text))]
        for q,body,index in sections:
            decoded=decode_query_array(q)
            if isinstance(decoded,list) and all(isinstance(v,str) for v in decoded):
                labels.extend(decoded);shared=True;representation='shared_query_array_title'
            else:
                labels.append(q)
                state='reported_error' if ERROR.match(body) else 'nonempty_results' if index<len(counts) and counts[index]>0 else 'empty_results' if index<len(counts) else 'unverified'
                observations.append({'query_label_sha256':fingerprint(q),'state':state})
            reported(body)
    else:
        node=value
        for _ in range(4):
            if isinstance(node,dict) and 'query' not in node and isinstance(node.get('data'),(dict,list)):
                node=node['data']
            else:break
        if isinstance(node,dict) and 'query' in node:
            representation='json_query_results';entry(node)
        elif isinstance(node,dict) and isinstance(node.get('results'),list):
            representation='json_query_results' if any(isinstance(part,dict) and 'query' in part for part in node['results']) else 'json_result_list'
            for part in node['results']:entry(part)
        elif isinstance(node,list):
            representation='json_string_list' if node and all(isinstance(part,str) for part in node) else 'json_query_results' if any(isinstance(part,dict) and 'query' in part for part in node) else 'json_result_list'
            for part in node:entry(part)
        elif isinstance(node,str) and ERROR.match(node):representation='tool_error_text'

    coverage=coverage_report(queries,[fingerprint(label) for label in labels],shared,unlabeled)
    return {'version':VERSION,'request_sha256':fingerprint(query),'raw_response_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'association':{'status':'verified','scope':'call_batch','call_id':link['call_id'],'identity_origin':link.get('id_origin','linked'),
                           'individual_source_response_mapping':link.get('association_scope')!='closed_round_call_batch'},
            'representation':representation,'decoding':decoding,'response_body_preserved':True,
            'coverage':coverage,'query_results':query_results_report(queries,coverage,observations),
            'execution':execution_report(outcomes)}
