"""Lossless JSON/XML tool-call normalization, independent of quality decisions.

Uses Hermes/Qwen-style <function=name> and <parameter=name> blocks. Canonical
string parameters are raw text, matching the installed Qwen chat template;
non-string parameters use JSON literals. The typed internal call object is retained
for unambiguous round-trip validation. Delimiter collisions fail closed.
"""
import copy
import json
import re

NAME=re.compile(r'[A-Za-z_][A-Za-z0-9_.:-]*\Z')
CALL=re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>',re.S)
FUNCTION=re.compile(r'<function=([^>]+)>\s*(.*?)\s*</function>',re.S)
PARAM=re.compile(r'<parameter=([^>]+)>(.*?)</parameter>',re.S)
THINK=re.compile(r'<think>.*?</think>',re.S)

def checked_call(value):
    if not isinstance(value,dict):raise ValueError('Tool call must be an object')
    call=copy.deepcopy(value.get('function',value))
    name=call.get('name');args=call.get('arguments',{})
    if isinstance(args,str):args=json.loads(args)
    if not isinstance(name,str) or not NAME.fullmatch(name):raise ValueError('Invalid tool function name')
    if not isinstance(args,dict) or any(not isinstance(k,str) or not NAME.fullmatch(k) for k in args):raise ValueError('Invalid tool arguments/names')
    # Decode only schema-defined container/numeric fields. Ordinary query strings
    # and quoted code remain text; malformed bracketed text is not guessed at.
    for key in ('query','url','urls'):
        value=args.get(key)
        if isinstance(value,str) and value.strip().startswith('['):
            try:decoded=json.loads(value)
            except json.JSONDecodeError:continue
            if isinstance(decoded,list) and all(isinstance(v,str) for v in decoded):args[key]=decoded
    for key in ('limit','topn'):
        value=args.get(key)
        if isinstance(value,str) and re.fullmatch(r'[0-9]+',value):args[key]=int(value)
    return {'name':name,'arguments':args}

def serialize(call):
    call=checked_call(call);parts=['<tool_call>',f"<function={call['name']}>"]
    for key,value in call['arguments'].items():
        body=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,allow_nan=False)
        if re.search(r'</?(?:tool_call|function|parameter)(?:[=>\s])',body):
            raise ValueError('Tool parameter collides with XML delimiters; cannot represent losslessly')
        parts.extend([f'<parameter={key}>',body,'</parameter>'])
    return '\n'.join(parts+['</function>','</tool_call>'])

def parse_body(body,expected=None):
    body=body.strip()
    if not body:raise ValueError('Empty tool call')
    if body.startswith('{'):
        value,end=json.JSONDecoder().raw_decode(body)
        tail=body[end:].strip()
        if not tail:return checked_call(value)
        # Some tool protocols store code after the JSON call header. Preserve it
        # as an argument; unsupported interpreters still fail the harness gate.
        call=checked_call(value)
        code=re.fullmatch(r'<code>(.*?)</code>',tail,re.S)
        if code:
            extra={'code':code.group(1)}
        else:
            extra=json.loads(tail)
            if not isinstance(extra,dict) or set(extra)!={'code'} or not isinstance(extra['code'],str):
                raise ValueError('Unsupported multipart tool payload')
        if 'code' in call['arguments'] and call['arguments']['code']!=extra['code']:
            raise ValueError('Conflicting multipart code payload')
        call['arguments'].update(extra)
        return checked_call(call)
    function=FUNCTION.fullmatch(body)
    if not function:raise ValueError('Malformed XML/JSON tool call')
    name,parameters=function.groups();args={};end=0
    for match in PARAM.finditer(parameters):
        if parameters[end:match.start()].strip():raise ValueError('Unparsed XML tool parameters')
        key,value=match.groups()
        if value.startswith('\n'):value=value[1:]
        if value.endswith('\n'):value=value[:-1]
        if key in args:raise ValueError('Duplicate XML tool parameter')
        if expected is not None:
            if key not in expected['arguments']:raise ValueError('XML/native parameter mismatch')
            typed=expected['arguments'][key]
            rendered=typed if isinstance(typed,str) else json.dumps(typed,ensure_ascii=False,allow_nan=False)
            if value!=rendered:raise ValueError('XML/native parameter value mismatch')
            args[key]=typed
        elif key in ('query','url','goal') and not value.lstrip().startswith('['):
            args[key]=value
        else:
            try:args[key]=json.loads(value)
            except json.JSONDecodeError:args[key]=value
        end=match.end()
    if parameters[end:].strip():raise ValueError('Unparsed XML tool parameters')
    return checked_call({'name':name,'arguments':args})

def normalize_text(text,structured_calls=None):
    """Convert executable calls only; preserve quoted syntax within reasoning."""
    protected=[(m.start(),m.end()) for m in THINK.finditer(text)]
    native=[checked_call(c) for c in structured_calls or []]
    calls=[];parts=[];cursor=0
    for match in CALL.finditer(text):
        if any(a<=match.start()<b for a,b in protected):continue
        expected=native[len(calls)] if len(calls)<len(native) else None
        call=parse_body(match.group(1),expected);calls.append(call)
        parts.extend([text[cursor:match.start()],serialize(call)]);cursor=match.end()
    parts.append(text[cursor:]);converted=''.join(parts)
    if native and calls and native!=calls:raise ValueError('Conflicting text and structured tool calls')
    if native and not calls:
        calls=native;converted+=('\n' if converted else '')+'\n'.join(serialize(c) for c in calls)
    # Unclosed executable tags cannot be silently converted into a final answer.
    visible=THINK.sub('',converted)
    if visible.count('<tool_call>')!=len(calls) or visible.count('</tool_call>')!=len(calls):
        raise ValueError('Unbalanced or nested tool call tags')
    return converted,calls
