"""Recognize transport wrappers independently of task text and tool arguments."""
import hashlib
import re

VERSION='qwen-xml-framing-v2'
CALL=re.compile(r'<tool_call>\s*.*?\s*</tool_call>',re.S)
THINK=re.compile(r'<think>.*?</think>',re.S)
OPEN=re.compile(r'(?m)^[ \t]*<tool_calls_begin>[ \t]*(?=\r?$)')
END=re.compile(r'(?m)^[ \t]*(?:</tool_calls_end>|<tool_calls_end>|</tool_calls_begin>)[ \t]*(?=\r?$)')


def strip_batch_wrappers(text):
    """Remove only a complete outer frame containing tool calls and whitespace.

    Quoted reasoning and argument values are protected. Unknown/non-call payload
    between delimiters is not discarded. Original offsets are returned for audit.
    """
    protected=[(m.start(),m.end()) for regex in (THINK,CALL) for m in regex.finditer(text)]
    visible=lambda m:not any(a<=m.start()<b for a,b in protected)
    opens=[m for m in OPEN.finditer(text) if visible(m)]
    ends=[m for m in END.finditer(text) if visible(m)]
    if not opens and not ends:return text,[]
    if len(opens)!=len(ends):raise ValueError('unsupported_or_unbalanced_tool_batch_wrapper')
    edits=[];events=[];previous=-1
    for opening,closing in zip(opens,ends):
        if not previous<opening.start()<closing.start():raise ValueError('unsupported_or_unbalanced_tool_batch_wrapper')
        body=text[opening.end():closing.start()];matches=list(CALL.finditer(body));cursor=0
        if not matches:raise ValueError('unsupported_or_unbalanced_tool_batch_wrapper')
        for match in matches:
            if body[cursor:match.start()].strip():raise ValueError('unsupported_or_unbalanced_tool_batch_wrapper')
            cursor=match.end()
        if body[cursor:].strip():raise ValueError('unsupported_or_unbalanced_tool_batch_wrapper')
        edits.extend([(opening.start(),opening.end()),(closing.start(),closing.end())])
        events.append({'rule':'legacy_tool_batch_wrapper_removed','start':opening.start(),'end':closing.end(),
                       'source_sha256':hashlib.sha256(text[opening.start():closing.end()].encode()).hexdigest()})
        previous=closing.end()
    for start,end in sorted(edits,reverse=True):text=text[:start]+text[end:]
    return text,events


def has_batch_wrapper(text):
    outside=CALL.sub('',THINK.sub('',text))
    return bool(OPEN.search(outside) or END.search(outside))
