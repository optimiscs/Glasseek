"""Remove known delegate execution telemetry, never recursively edit evidence.

Task identity, summary, status, error and unknown fields are preserved. Source
files/provenance remain the archive. Callers can persist returned audit events
outside model-visible messages. Malformed or ambiguous responses stay unchanged.
"""
import collections
import copy
import hashlib
import json
from tool_xml import normalize_text

VERSION='delegate-runtime-metadata-v1'
TOOLS={'delegate_task','call_sub_agent'}
# Fixed, versioned training-data policy. Unlike the live runtime, offline SFT
# does not need historical latency, API usage or billing fields in its context.
TOP_FIELDS={'total_duration_seconds','duration_seconds','api_calls','tokens','model','cost_usd',
            'input_tokens','output_tokens','reasoning_tokens','total_cost_usd'}
ENTRY_FIELDS=TOP_FIELDS | {'exit_reason','tool_trace','_child_role','_child_cost_usd'}


def _object(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise ValueError('duplicate_json_key')
        result[k]=v
    return result


def clean_response(text):
    """Only call for an observation linked unambiguously to a delegate call."""
    opening='<tool_response>';closing='</tool_response>'
    if opening in text:
        start=text.index(opening)+len(opening);end=text.rfind(closing)
        if end<start:return text,{'status':'skipped','reason':'unclosed_tool_response'}
        body=text[start:end]
    else:start=0;end=len(text);body=text
    # Literal \\n at the envelope boundary is a legacy transport representation.
    left=0;right=len(body)
    while left<right:
        if body[left].isspace():left+=1
        elif body.startswith('\\n',left):left+=2
        else:break
    while right>left:
        if body[right-1].isspace():right-=1
        elif body[left:right].endswith('\\n'):right-=2
        else:break
    raw=body[left:right]
    if not raw.startswith('{'):return text,None  # prose reports have no structured telemetry
    try:payload=json.loads(raw,object_pairs_hook=_object)
    except (ValueError,TypeError):return text,{'status':'skipped','reason':'invalid_or_ambiguous_response_json'}
    if not isinstance(payload,dict) or not isinstance(payload.get('results'),list):return text,None
    for entry in payload['results']:
        if not isinstance(entry,dict) or type(entry.get('task_index')) is not int or not (
            isinstance(entry.get('summary'),str) or isinstance(entry.get('error'),str)):
            return text,{'status':'skipped','reason':'unrecognized_delegate_result_schema'}
    removed=[]
    for key in sorted(TOP_FIELDS & payload.keys()):
        payload.pop(key);removed.append('$.'+key)
    for i,entry in enumerate(payload['results']):
        for key in sorted(ENTRY_FIELDS & entry.keys()):
            entry.pop(key);removed.append(f'$.results[{i}].'+key)
    if not removed:return text,None
    replacement=json.dumps(payload,ensure_ascii=False)
    cleaned=text[:start+left]+replacement+text[start+right:]
    return cleaned,{'status':'cleaned','removed_paths':removed,'removed_fields':len(removed),
                    'source_response_sha256':hashlib.sha256(text.encode()).hexdigest(),
                    'chars_removed':len(text)-len(cleaned)}


def clean_item(item):
    """Normalize observations before semantic hashing, deduplication and budgeting."""
    events=[]
    for step in item['steps']:
        calls=step.get('calls',[])
        if len(calls)!=1 or calls[0]['name'].casefold() not in TOOLS:continue
        observation,event=clean_response(step['observation'])
        if event:
            step['observation']=observation;events.append({'step_id':step['step_id'],**event})
        for response in step.get('responses', []):
            response['content'], _ = clean_response(response['content'])
    if events:item['runtime_metadata_cleaning']={'version':VERSION,'events':events}
    return events


def clean_messages(messages):
    """Apply the same policy to already-exported XML/native message sequences."""
    output=copy.deepcopy(messages);events=[];pending=[];by_id={}
    for i,message in enumerate(output):
        role=message.get('role');text=message.get('content')
        if not isinstance(text,str):pending=[];continue
        if role=='assistant':
            try:_,pending=normalize_text(text,message.get('tool_calls'))
            except (ValueError,TypeError,KeyError):pending=[]
            for raw in message.get('tool_calls') or []:
                if raw.get('id'):by_id[raw['id']]=raw.get('function',{}).get('name','').casefold()
            for link in message.get('call_links') or []:
                if link.get('call_id'):by_id[link['call_id']]=link.get('name','').casefold()
        elif role=='tool' or (role=='user' and text.lstrip().startswith('<tool_response>')):
            cid=message.get('tool_call_id')
            matched=by_id.get(cid) in TOOLS if cid else len(pending)==1 and pending[0]['name'].casefold() in TOOLS
            if message.get('name') and message['name'].casefold() not in TOOLS:matched=False
            if matched:
                message['content'],event=clean_response(text)
                if event:events.append({'message_index':i,**event})
        else:pending=[]
    return output,events
