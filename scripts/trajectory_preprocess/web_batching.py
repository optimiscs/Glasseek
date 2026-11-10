"""Same-turn search/extract batches; no invented per-item response mapping."""
import collections
import copy
import hashlib
import json
from urllib.parse import urlsplit

from glasseek_shared.extract_protocol import MAX_URLS, response_body
from glasseek_shared.historical_tools import canonical_tool, response_hints
from glasseek_shared.search_protocol import MAX_QUERIES,validate_queries
from tool_xml import CALL, THINK, serialize

VERSION='same-turn-web-batch-v2'


def canonical_batch_call(call):
    """Only lossless aliases/options with no cross-call result references."""
    name=canonical_tool(call['name']);args=call['arguments']
    if name=='web_search':
        if set(args)-{'query','limit','topn'}:return None
        try:validate_queries(args.get('query'))
        except ValueError:return None
        limit=args.get('limit',args.get('topn',5))
        if type(limit) is not int or not 1<=limit<=100:return None
        if 'limit' in args and 'topn' in args and args['limit']!=args['topn']:return None
        return {'name':name,'arguments':{'query':copy.deepcopy(args['query']),'limit':limit}}
    if name!='web_extract' or set(args)-{'url','urls','goal','query'}:return None
    urls=args.get('urls',args.get('url'));urls=[urls] if isinstance(urls,str) else urls
    if not isinstance(urls,list) or not 1<=len(urls)<=MAX_URLS:return None
    if 'url' in args and 'urls' in args:
        other=args['url'];other=[other] if isinstance(other,str) else other
        if other!=urls:return None
    if not all(isinstance(u,str) and urlsplit(u).scheme in ('http','https') and urlsplit(u).netloc
               and '${' not in u and '{{' not in u for u in urls):return None
    target={'urls':copy.deepcopy(urls)}
    if 'goal' in args and 'query' in args and args['goal']!=args['query']:return None
    if 'goal' in args or 'query' in args:
        goal=args.get('goal',args.get('query'))
        if not isinstance(goal,str):return None
        target['goal']=goal
    return {'name':name,'arguments':target}


def values(call):
    return validate_queries(call['arguments']['query']) if call['name']=='web_search' else call['arguments']['urls']


def batch_key(call):
    field='query' if call['name']=='web_search' else 'urls'
    return call['name'],json.dumps({k:v for k,v in call['arguments'].items() if k!=field},sort_keys=True,ensure_ascii=False)


def closed_round_batches(steps,unassigned,contexts,instructions,reserved_ids,answered_ids):
    """Recover a whole batch only when its source turn is uniquely determined.

    Unlike individual matching, this makes no claim that one opaque component
    belongs to a particular query/URL. Mixed tools/goals, interruptions and
    conflicting explicit identifiers remain unresolved.
    """
    repaired=set();flags=[]
    barriers=contexts+instructions
    for si,step in enumerate(steps):
        if not step['calls']:continue
        try:mapped=[canonical_batch_call(c) for c in step['calls']]
        except ValueError:continue
        if any(c is None for c in mapped) or len({batch_key(c) for c in mapped})!=1:continue
        limit=MAX_QUERIES if mapped[0]['name']=='web_search' else MAX_URLS
        if sum(len(values(c)) for c in mapped)>limit:continue
        start=step['source_message_index']
        stop=steps[si+1]['source_message_index'] if si+1<len(steps) else float('inf')
        unresolved=[(i,r) for i,r in enumerate(unassigned) if i not in repaired and start<r.get('source_message_index',-1)<stop]
        if not unresolved:continue
        if any(not start<r.get('source_message_index',-1)<stop for r in step['responses']):continue
        responses=sorted(step['responses']+[r for _,r in unresolved],key=lambda r:(r['source_message_index'],r.get('source_content_start',0)))
        if len(mapped)==1 and len(values(mapped[0]))==1 and len(responses)!=1:continue
        if len({(r.get('source_message_index'),r.get('source_content_start'),r.get('source_content_end')) for r in responses})!=len(responses):continue
        end=max(r['source_message_index'] for r in responses)
        if any(start<m.get('source_message_index',-1)<end for m in barriers):continue
        # An earlier unresolved request would make the batch's turn ambiguous.
        answered_before={r['response_to'] for prior in steps for r in prior['responses'] if r.get('source_message_index',float('inf'))<start}
        if any(not link.get('call_id') or link['call_id'] not in answered_before
               for prior in steps[:si] for link in prior['call_links']):continue
        expected_ids={link['call_id'] for link in step['call_links'] if link.get('call_id')}
        explicit_ids=set();seen_explicit=set();query_union=collections.Counter(q for c in mapped for q in values(c))
        bad=False
        for response in responses:
            if response['content'].lstrip().startswith('<tool_response>') and not response['content'].rstrip().endswith('</tool_response>'):
                bad=True;break
            try:
                body=response_body(response['content']);hints=response_hints(response['content'])
            except ValueError:bad=True;break
            if not body.strip():bad=True;break
            native_ids={v for v in (response.get('source_response_id'),hints.get('id')) if v}
            if native_ids & seen_explicit:bad=True;break
            seen_explicit.update(native_ids);explicit_ids.update(native_ids)
            if hints.get('name') and canonical_tool(hints['name'])!=mapped[0]['name']:bad=True;break
            labels=hints.get('queries' if mapped[0]['name']=='web_search' else 'urls')
            if len(mapped)>1 and mapped[0]['name']=='web_extract' and labels and collections.Counter(labels)-query_union:bad=True;break
        if bad or (explicit_ids and (explicit_ids!=expected_ids or len(expected_ids)!=len(mapped))):continue
        # Rewriting requires executable calls with no intervening instructions.
        protected=[(m.start(),m.end()) for m in THINK.finditer(step['assistant'])]
        matches=[m for m in CALL.finditer(step['assistant']) if not any(a<=m.start()<b for a,b in protected)]
        if len(matches)!=len(mapped) or any(step['assistant'][a.end():b.start()].strip() for a,b in zip(matches,matches[1:])):continue
        before=copy.deepcopy({k:step[k] for k in ('assistant','calls','call_links','responses','observation')})
        before.update(version=VERSION,unassigned_responses=copy.deepcopy([r for _,r in unresolved]))
        ids=[]
        for i,link in enumerate(step['call_links']):
            if not link.get('call_id'):
                cid=f'generated:m{start}:c{i}'
                while cid in reserved_ids:cid+=':generated'
                reserved_ids.add(cid);link['call_id']=cid
                link['id_origin']='source_position_with_batch_association'
            ids.append(link['call_id']);answered_ids.add(link['call_id'])
        cid=ids[0]
        if len(mapped)>1:
            cid+=':web_batch'
            while cid in reserved_ids:cid+=':derived'
            reserved_ids.add(cid)
        answered_ids.add(cid)
        call=copy.deepcopy(mapped[0]);field='query' if call['name']=='web_search' else 'urls'
        if len(mapped)>1:call['arguments'][field]=[v for c in mapped for v in values(c)]
        chunks=[];parts=[];offset=0
        for response in responses:
            body=response_body(response['content'])
            if chunks:chunks.append('\n\n');offset+=2
            parts.append({'start':offset,'end':offset+len(body),'content_sha256':hashlib.sha256(body.encode()).hexdigest(),
                          'source_response':{k:copy.deepcopy(v) for k,v in response.items() if k!='content'},
                          'source_response_sha256':hashlib.sha256(response['content'].encode()).hexdigest()})
            chunks.append(body);offset+=len(body)
        response={'content':''.join(chunks),'response_to':cid,'source_message_index':responses[0]['source_message_index'],
                  'source_message_indices':[r['source_message_index'] for r in responses],
                  'source_call_ids':ids,'batch_response_parts':parts,'association_scope':'closed_round_call_batch',
                  'source_content_start':None,'source_content_end':None,'source_role':responses[0].get('source_role'),
                  'page_boundary_status':'not_required_for_batch'}
        if len(mapped)==1 and len(responses)==1:
            response={**copy.deepcopy(responses[0]),'response_to':cid,'association_scope':'unique_closed_round_call'}
        step['assistant']=step['assistant'][:matches[0].start()]+serialize(call)+step['assistant'][matches[-1].end():]
        step['calls']=[call];step['call_links']=[{**copy.deepcopy(call),'call_id':cid,'source_call_ids':ids,
            'source_call_indices':list(range(len(mapped))),'source_call_index':0,'source_message_index':start,
            'id_origin':'closed_round_call_batch','association_scope':'closed_round_call_batch','merge_policy_version':VERSION}]
        step['responses']=[response];step['observation']=response['content'];step['web_batch_provenance']=before
        repaired.update(i for i,_ in unresolved);flags.append('closed_round_'+call['name']+'_batch_associated')
    return [r for i,r in enumerate(unassigned) if i not in repaired],sorted(set(flags))


def coalesce_step(item, step, original_calls, reserved_ids, max_urls=MAX_URLS):
    if step.get('association_status')!='linked':return []
    calls,links,responses=step['calls'],step.get('call_links',[]),step.get('responses',[])
    if len(calls)!=len(links) or len(calls)<2:return []
    text=step['assistant'];protected=[(m.start(),m.end()) for m in THINK.finditer(text)]
    matches=[m for m in CALL.finditer(text) if not any(a<=m.start()<b for a,b in protected)]
    if len(matches)!=len(calls):return []
    call_ids=[link['call_id'] for link in links]
    if len(set(call_ids))!=len(call_ids):return []
    by_id={cid:[r for r in responses if r['response_to']==cid] for cid in call_ids}
    if any(len(rs)!=1 for rs in by_id.values()):return []
    # A return arriving after another assistant/context message is not a purely
    # parallel batch. Preserve chronology, including user reminders.
    start=step.get('source_message_index')
    indices=[r.get('source_message_index') for r in responses]
    if type(start) is int and all(type(i) is int for i in indices):
        end=max(indices)
        barriers=[m.get('source_message_index') for key in ('context_messages','instruction_messages') for m in item.get(key,[])]
        barriers += [s.get('source_message_index') for s in item['steps'] if s is not step]
        if any(type(i) is int and start<i<end for i in barriers):return []
    groups=[];active={}
    def finish():
        groups.extend(g for g in active.values() if len(g)>1);active.clear()
    for index,(call,link) in enumerate(zip(calls,links)):
        if index and calls[index-1]['name']!=call['name']:finish()
        if index and text[matches[index-1].end():matches[index].start()].strip():finish()
        args=call['arguments'];source_index=link.get('source_call_index',index)
        source=original_calls[source_index] if type(source_index) is int and 0<=source_index<len(original_calls) else None
        eligible=source is not None and canonical_batch_call(source) is not None and call['name'] in ('web_search','web_extract')
        if eligible:
            try:
                eligible=not any(k in by_id[link['call_id']][0] for k in ('batch_response_parts','merged_response_parts'))
                response_body(by_id[link['call_id']][0]['content'])
            except ValueError:eligible=False
        if not eligible:finish();continue
        key=batch_key(call);limit=MAX_QUERIES if call['name']=='web_search' else max_urls
        pending=active.get(key,[])
        if pending and sum(len(values(calls[i])) for i in pending)+len(values(call))>limit:
            if len(pending)>1:groups.append(pending)
            pending=[]
        active[key]=pending+[index]
    finish()
    if not groups:return []
    before={'version':VERSION,'assistant':text,'source_calls':copy.deepcopy(original_calls),
            'calls':copy.deepcopy(calls),'call_links':copy.deepcopy(links),'responses':copy.deepcopy(responses)}
    replacement,removed,new_responses={},set(),{}
    response_order={r['response_to']:i for i,r in enumerate(responses)}
    for indices in sorted(groups,key=lambda g:g[0]):
        ids=[links[i]['call_id'] for i in indices]
        ordered=sorted((by_id[cid][0] for cid in ids),key=lambda r:response_order[r['response_to']])
        bodies=[response_body(r['content']) for r in ordered]
        if any(not body.strip() for body in bodies):continue
        cid=ids[0]+':web_batch'
        while cid in reserved_ids:cid+=':derived'
        reserved_ids.add(cid)
        name=calls[indices[0]]['name'];args=copy.deepcopy(calls[indices[0]]['arguments'])
        field='query' if name=='web_search' else 'urls';args[field]=[v for i in indices for v in values(calls[i])]
        call={'name':name,'arguments':args}
        link={**copy.deepcopy(links[indices[0]]),**copy.deepcopy(call),'call_id':cid,
              'id_origin':'same_turn_independent_web_batch','source_call_ids':ids,
              'source_call_indices':[links[i].get('source_call_index',i) for i in indices],
              'merge_policy_version':VERSION}
        chunks=[];parts=[];offset=0
        for r,body in zip(ordered,bodies):
            if chunks:chunks.append('\n\n');offset+=2
            source={k:copy.deepcopy(v) for k,v in r.items() if k not in ('content','original_content','merged_response_parts')}
            parts.append({'start':offset,'end':offset+len(body),'source_call_id':r['response_to'],
                          'source_response':source,'source_response_sha256':hashlib.sha256(r['content'].encode()).hexdigest()})
            chunks.append(body);offset+=len(body)
        response={k:copy.deepcopy(v) for k,v in ordered[0].items() if k not in ('original_content','merged_response_parts')}
        response.update(content=''.join(chunks),response_to=cid,merged_response_parts=parts,
                        source_content_start=None,source_content_end=None,
                        source_message_indices=[r.get('source_message_index') for r in ordered],
                        extract_response_scope='call_batch',page_boundary_status='not_required_for_batch')
        replacement[indices[0]]=(call,link);removed.update(indices[1:])
        new_responses[ordered[0]['response_to']]=response
        removed_ids=set(ids)-{ordered[0]['response_to']}
        for old_id in removed_ids:new_responses[old_id]=None
    if not replacement:return []
    segments=[];cursor=0;target_calls=[];target_links=[]
    for i,match in enumerate(matches):
        segments.append(text[cursor:match.start()]);cursor=match.end()
        if i in removed:continue
        call,link=replacement.get(i,(calls[i],links[i]))
        segments.append(serialize(call));target_calls.append(copy.deepcopy(call));target_links.append(copy.deepcopy(link))
    segments.append(text[cursor:]);step['assistant']=''.join(segments)
    step['calls'],step['call_links']=target_calls,target_links
    step['responses']=[new_responses.get(r['response_to'],r) for r in responses if new_responses.get(r['response_to'],r) is not None]
    step['web_batch_provenance']=before
    return sorted({'same_turn_'+call['name'].removeprefix('web_')+'_calls_merged' for call,link in replacement.values()})
