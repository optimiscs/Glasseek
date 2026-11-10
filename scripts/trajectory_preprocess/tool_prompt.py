"""Reconcile explicit tool schemas/examples without rewriting task requirements."""
import hashlib
import json
import re

from glasseek_shared.hermes.qwen_xml_protocol import strip_batch_wrappers
from tool_xml import CALL,serialize,normalize_text
from web_batching import canonical_batch_call

VERSION='source-tool-prompt-xml-v3'
TOOLS=re.compile(r'<tools>(.*?)</tools>',re.S)
PLACEHOLDER=re.compile(r'''\s*\{\s*["']name["']\s*:\s*<[\w-]+>\s*,\s*["']arguments["']\s*:\s*<[\w-]+>\s*\}\s*''')
EXAMPLE=serialize({'name':'example_function_name','arguments':{'example_parameter':'value'}})
DECLARATION_HEADER=re.compile(r'^\s*\{\s*(?:"type"\s*:\s*"function"\s*,\s*"function"\s*:\s*\{\s*)?"name"\s*:\s*"[A-Za-z_][A-Za-z0-9_.:-]*"')


def check_declaration(value):
    if not isinstance(value,dict):raise ValueError('non_object_tool_declaration')
    value=value.get('function',value)
    if not isinstance(value,dict) or not isinstance(value.get('name'),str) or not isinstance(value.get('parameters'),dict):
        raise ValueError('unrecognized_tool_declaration')


def schema_block(body):
    """Consume recognized JSON or XML declarations, not arbitrary tagged text."""
    if len(body)>2_000_000:raise ValueError('tool_declarations_too_large')
    if body.lstrip().startswith('<function>'):
        pattern=re.compile(r'<function>\s*<name>([A-Za-z_][A-Za-z0-9_.:-]*)</name>(.*?)</function>',re.S)
        cursor=0;count=0
        for match in pattern.finditer(body):
            if body[cursor:match.start()].strip():raise ValueError('unrecognized_tool_declaration')
            declaration=match[2]
            if declaration.count('<parameters>')!=1 or declaration.count('</parameters>')!=1:
                raise ValueError('unrecognized_tool_declaration')
            cursor=match.end();count+=1
        if not count or count>128 or body[cursor:].strip():raise ValueError('unrecognized_tool_declaration')
        return count
    # Some serialized prompt schemas retain literal newlines in descriptions.
    # Relax only JSON string control-character escaping in this metadata parser;
    # executable tool arguments continue to use the strict JSON decoder.
    decoder=json.JSONDecoder(strict=False);rest=body.strip();values=[]
    while rest:
        try:value,end=decoder.raw_decode(rest)
        except ValueError:
            # The explicit <tools> container and a formal function-declaration
            # header identify obsolete metadata even when description quoting is
            # broken. Never repair or execute it; replace it with the trusted
            # runtime declaration and separately retain the syntax uncertainty.
            if DECLARATION_HEADER.match(rest) and rest.rstrip().endswith('}') and re.search(r'"parameters"\s*:\s*\{',rest):
                for previous in values:check_declaration(previous)
                return None
            raise
        values.extend(value if isinstance(value,list) else [value]);rest=rest[end:].strip()
        if len(values)>128:raise ValueError('too_many_tool_declarations')
    if not values:raise ValueError('empty_tool_declarations')
    for value in values:
        check_declaration(value)
    return len(values)


def reconcile(item):
    events=[];flags=[]
    for instruction in item.get('instruction_messages',[]):
        original=instruction['content'];text=original;rules=[]
        def replace_tools(match):
            if not match[1].strip():return match[0]  # Inline notation, not a declaration.
            try:declarations=schema_block(match[1])
            except (ValueError,TypeError):
                flags.append('tool_prompt_protocol_review_required');return match[0]
            if declarations is None:flags.append('source_tool_declaration_syntax_unverified')
            rules.append('source_tool_declarations_use_runtime_schema')
            return 'Tool definitions are supplied by the current runtime tool schema.'
        text=TOOLS.sub(replace_tools,text)
        # Only standalone, balanced wrappers around examples are removed. A
        # mention of a token inside arbitrary prose is not silently deleted.
        try:text,framing=strip_batch_wrappers(text)
        except ValueError:
            flags.append('tool_prompt_protocol_review_required');framing=[]
        rules.extend(event['rule'] for event in framing)
        def replace_example(match):
            body=match[1]
            if PLACEHOLDER.fullmatch(body):
                rules.append('symbolic_json_call_example_to_xml');return EXAMPLE
            if not body.lstrip().startswith(('{','<function=')):return match[0]
            try:_,calls=normalize_text(match[0])
            except (ValueError,TypeError,KeyError):
                flags.append('tool_prompt_protocol_review_required');return match[0]
            if len(calls)!=1:return match[0]
            call=canonical_batch_call(calls[0]) or calls[0]
            replacement=serialize(call)
            if replacement!=match[0]:rules.append('tool_call_example_to_canonical_xml')
            return replacement
        text=CALL.sub(replace_example,text)
        patterns=[
            (r'(?im)^[ \t]*All tool calls must be wrapped inside\s+<tool_calls_begin>\s+and\s+</?tool_calls_end>\.[ \t]*$',
             'Write each tool call directly as one XML block; use no outer batch wrapper.'),
            (r'(?im)^[ \t]*Inside this block, each individual tool call must be wrapped with\s+<tool_call>\s+and\s+</tool_call>\.[ \t]*$',
             'Each <tool_call> contains one <function=...> block and its <parameter=...> fields.'),
            (r'(?im)^[ \t]*For each function call,? return a json object with function name and arguments within\s+<tool_call></tool_call> XML tags:[ \t]*$',
             'For each function call, use a <function=...> block with <parameter=...> fields inside <tool_call></tool_call>:')]
        for pattern,replacement in patterns:
            text,n=re.subn(pattern,replacement,text)
            if n:rules.append('tool_protocol_directive_to_xml')
        if re.search(r'(?im)^[^\n]*(?:function call|tool calls?)[^\n]*(?:return a json object|wrapped[^\n]*tool_calls_begin)',text):
            flags.append('tool_prompt_protocol_review_required')
        if text!=original:
            instruction['content']=text
            events.append({'source_message_index':instruction.get('source_message_index'),
                           'before_step':instruction['before_step'],'source_role':instruction['role'],
                           'original_content':original,'source_sha256':hashlib.sha256(original.encode()).hexdigest(),
                           'replacement_sha256':hashlib.sha256(text.encode()).hexdigest(),'rules':sorted(set(rules))})
    if events:
        item['tool_prompt_provenance']={'version':VERSION,'events':events}
        flags.append('source_tool_prompt_format_normalized')
    return sorted(set(flags))
