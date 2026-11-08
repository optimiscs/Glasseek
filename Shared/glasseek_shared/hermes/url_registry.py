"""Causal URL presentation protocol shared by SFT, RL and Hermes.

raw is an identity path. Only observations publish new references. Assistant
text/actions never allocate references; an assistant turn resolves against a
frozen pre-turn view. Original URLs, evidence and sampled actions are retained.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from urllib.parse import urlsplit

MODES = ('raw', 'hybrid', 'ref', 'evidence')
VERSION = 'glasseek-url-v3-text-xml-args-id'
URL = re.compile(r'https?://[^\s<>"\x27\\]+')
REF = re.compile(r'\b(?:U[1-9]\d*|D[1-9]\d*@v[1-9]\d*(?:#E[1-9]\d*)?)\b')
XML_REF = re.compile(r'<url>\s*(U[1-9]\d*|D[1-9]\d*@v[1-9]\d*(?:#E[1-9]\d*)?)\s*</url>')
CALL = re.compile(r'<tool_call>.*?</tool_call>|```.*?```', re.S)
_LOCK = threading.RLock()
_REGISTRIES = {}


def reference_id(value):
    """Decode only an entire XML address; never interpret arbitrary XML."""
    if not isinstance(value, str):
        raise ValueError('URL reference must be a string')
    match = XML_REF.fullmatch(value.strip())
    if match:
        return match[1]
    if '<url' in value or '</url' in value:
        raise ValueError('Pass exactly <url> ID </url>, not an ID embedded inside a URL or other text')
    return value


def xml_reference(ref):
    return '<url>' + ref + '</url>'


def mode_value(value=None):
    value = value if value is not None else os.environ.get('GLASSEEK_URL_MODE', 'raw')
    if value not in MODES:raise ValueError('url-mode must be one of '+', '.join(MODES))
    return value


def add_url_argument(parser):
    parser.add_argument('--url-mode', choices=MODES, default=mode_value(), help='URL protocol; default raw preserves existing behavior')
    parser.add_argument('--url-prompt', default=os.environ.get('GLASSEEK_URL_PROMPT', 'auto'),
                        help='System prompt plugin: auto (built-in), off, or a JSON config path; raw is always unchanged')


def http_domain(value):
    try:
        parsed=urlsplit(value)
        return parsed.netloc if parsed.scheme in ('http','https') else ''
    except (ValueError,TypeError):return ''


def urls(text):
    for m in URL.finditer(text):
        value=m.group().rstrip('.,;!?)\u3002\uff0c')
        # Keep balanced URL parentheses (common in encyclopaedia links).
        tail=m.group()[len(value):]
        while tail.startswith(')') and value.count('(')>value.count(')'):
            value+=')';tail=tail[1:]
        if value:yield m.start(),m.start()+len(value),value


class URLRegistry:
    def __init__(self, mode='raw', state=None):
        self.mode=mode_value(mode);self.entries={};self.documents={};self.step=0
        self.visible=set();self.lock=threading.RLock();self.events=[]
        if state:
            if state['version']!=VERSION or state['mode']!=self.mode:raise ValueError('URL protocol snapshot mismatch')
            self.entries=copy.deepcopy(state['entries']);self.documents=copy.deepcopy(state.get('documents',{}));self.step=state['step']
            self.events=copy.deepcopy(state.get('events',[]))
        self.by_url={v['url']:k for k,v in self.entries.items()}
        self.begin_turn()

    def snapshot(self):
        return {'version':VERSION,'mode':self.mode,'step':self.step,'entries':copy.deepcopy(self.entries),
                'documents':copy.deepcopy(self.documents),'events':copy.deepcopy(self.events)}

    def begin_turn(self):
        with self.lock:self.visible=set(self.entries)|set(self.documents)

    def register(self,url):
        if not isinstance(url,str) or not http_domain(url):return url
        if url not in self.by_url:
            ref='U'+str(len(self.entries)+1)
            self.by_url[url]=ref;self.entries[ref]={'url':url,'introduced_at':self.step}
            self.events.append({'event':'publish','ref':ref,'step':self.step})
        return self.by_url[url]

    def resolve(self,value):
        if self.mode=='raw':return value
        value=reference_id(value)
        if value.startswith(('http://','https://')):return value
        base=value.split('#E')[0]
        if base not in self.visible:raise ValueError('Unknown or not-yet-visible URL reference: '+value)
        if base in self.entries:return self.entries[base]['url']
        doc=self.documents[base]
        if '#E' in value:
            n=int(value.split('#E')[1])
            if not 1<=n<=len(doc['chunks']):raise ValueError('Unknown evidence span: '+value)
        return self.entries[doc['url_ref']]['url']

    def resolve_args(self,name,args):
        if self.mode=='raw':return args
        out=copy.deepcopy(args)
        if name=='web_extract':
            for key in ('url','urls'):
                if key in out:
                    value=out[key]
                    if isinstance(value,str) and value.lstrip().startswith('['):value=json.loads(value)
                    out[key]=[self.resolve(v) for v in value] if isinstance(value,list) else self.resolve(value)
        elif name=='vision_analyze' and 'image_url' in out:
            if not str(out['image_url']).startswith('data:'):out['image_url']=self.resolve(out['image_url'])
        if name=='web_extract' and 'url' in out and 'urls' not in out:
            out['urls']=[out.pop('url')]
        return out

    def known_text(self,text):
        if self.mode=='raw' or not isinstance(text,str):return text
        def replace_plain(s):
            parts=[];last=0
            for start,end,url in urls(s):
                ref=self.by_url.get(url)
                parts.extend([s[last:start],xml_reference(ref) if ref in self.visible else url]);last=end
            return ''.join(parts)+s[last:]
        # Tool query/code blocks are not free-text URL references.
        parts=[];last=0
        for m in CALL.finditer(text):
            parts.extend([replace_plain(text[last:m.start()]),m.group()]);last=m.end()
        return ''.join(parts)+replace_plain(text[last:])

    def encode_args(self,name,args):
        if self.mode=='raw':return args
        out=copy.deepcopy(args)
        if name in ('web_extract','vision_analyze'):
            for key in ('url','urls','image_url'):
                if key not in out:continue
                def one(v):
                    ref=self.by_url.get(v) if isinstance(v,str) else None
                    return ref if ref in self.visible else v
                v=out[key];out[key]=[one(x) for x in v] if isinstance(v,list) else one(v)
        if name=='web_extract' and 'url' in out and 'urls' not in out:
            out['urls']=[out.pop('url')]
        return out

    def _observe_text(self,text,announced):
        parts=[];last=0
        for start,end,url in urls(text):
            fresh=url not in self.by_url;ref=self.register(url)
            if ref==url:shown=url
            elif self.mode=='hybrid' and fresh:shown=f'{url} {xml_reference(ref)}'
            else:shown=(xml_reference(ref)+' ('+urlsplit(url).netloc+')') if fresh and ref!=url else xml_reference(ref)
            if ref!=url:announced[ref]=urlsplit(url).netloc
            parts.extend([text[last:start],shown]);last=end
        return ''.join(parts)+text[last:]

    def _walk(self,value,announced):
        if isinstance(value,str):return self._observe_text(value,announced)
        if isinstance(value,list):return [self._walk(v,announced) for v in value]
        if isinstance(value,dict):
            address=value.get('url')
            page=self.mode=='evidence' and isinstance(address,str) and isinstance(value.get('content'),str) and not value.get('error')
            out={k:self._walk(v,announced) for k,v in value.items() if not (page and k=='content') and k!='url'}
            if 'url' in value:
                if isinstance(address,str) and http_domain(address):
                    fresh=address not in self.by_url;ref=self.register(address);announced[ref]=urlsplit(address).netloc
                    out['url']=address if self.mode=='hybrid' and fresh else xml_reference(ref)
                    if self.mode=='hybrid' and fresh:out['source_ref']=xml_reference(ref)
                    else:out.setdefault('domain',urlsplit(address).netloc)
                else:out['url']=address
            if isinstance(address,str) and address in self.by_url:
                if self.mode=='evidence' and isinstance(value.get('content'),str) and not value.get('error'):
                    u=self.by_url[address];content=value['content'];sha=hashlib.sha256(content.encode()).hexdigest()
                    stem='D'+u[1:]+'@v'
                    prior=[(k,v) for k,v in self.documents.items() if v['url_ref']==u]
                    doc=next((k for k,v in prior if v['sha256']==sha),None)
                    if doc is None:
                        doc=stem+str(len(prior)+1)
                        chunks=[content[i:i+1600] for i in range(0,len(content),1600)] or ['']
                        self.documents[doc]={'url_ref':u,'sha256':sha,'chunks':chunks,'introduced_at':self.step}
                    chunks=self.documents[doc]['chunks'];out['document_ref']=xml_reference(doc)
                    out['content']='\n'.join(xml_reference(f'{doc}#E{i+1}')+' '+self._observe_text(c,announced) for i,c in enumerate(chunks[:2]))
                    out['available_spans']=len(chunks)
                    out['truncated']=len(chunks)>2
            return out
        return value

    def observe(self,name,result):
        if self.mode=='raw' or name not in ('web_search','web_extract'):return result
        with self.lock:
            self.step+=1;announced={}
            if isinstance(result,str):
                # Preserve external-result trust wrappers verbatim.
                start=result.find('{');end=result.rfind('}')
                try:
                    obj=json.loads(result[start:end+1]) if start>=0 else json.loads(result)
                except (ValueError,TypeError):
                    if result.lstrip().startswith(('{','<untrusted_tool_result')) and start>=0:
                        # A truncated JSON URL is not a valid new address. Leave
                        # the original observation intact instead of guessing.
                        return result
                    text=self._observe_text(result,announced)
                else:
                    text=result[:start]+json.dumps(self._walk(obj,announced),ensure_ascii=False)+result[end+1:]
                return text
            return self._walk(result,announced)

    def cached_evidence(self,name,args):
        if self.mode!='evidence' or name!='web_extract':return None
        # Stored snippets may have been selected for a different objective.
        # Resolve the reference to its URL and perform a fresh goal-aware read.
        if args.get('goal'):return None
        values=args.get('urls',args.get('url',[]))
        if isinstance(values,str):
            values=json.loads(values) if values.lstrip().startswith('[') else [values]
        values=[reference_id(value) for value in values]
        stored=[isinstance(v,str) and v.split('#E')[0] in self.documents for v in values]
        if any(stored) and not all(stored):raise ValueError('Use separate calls for stored evidence snapshots and live URLs')
        if not values or not all(stored):return None
        results=[]
        for ref in values:
            self.resolve(ref);doc=self.documents[ref.split('#E')[0]];chunks=doc['chunks']
            indices=[int(ref.split('#E')[1])-1] if '#E' in ref else list(range(min(2,len(chunks))))
            results.append({'source_ref':xml_reference(doc['url_ref']),'document_ref':xml_reference(ref.split('#E')[0]),
                            'content':'\n'.join(xml_reference(f'{ref.split("#E")[0]}#E{i+1}')+' '+chunks[i] for i in indices),
                            'available_spans':len(chunks),'snapshot_sha256':doc['sha256'],'cached':True})
        return json.dumps({'results':results},ensure_ascii=False)

    def expand_references(self,text):
        if self.mode=='raw' or not isinstance(text,str):return text
        def expand(ref):
            base=ref.split('#E')[0]
            if base in self.entries:return self.entries[base]['url']
            if base in self.documents:return self.entries[self.documents[base]['url_ref']]['url']
            return ref
        # Expand whole XML elements without leaving tags around literal URLs.
        # Do not run a second regex over the expanded URLs (which may contain U IDs).
        pattern=re.compile(XML_REF.pattern+'|'+REF.pattern)
        def replace(m):
            ref=m[1] or m.group();expanded=expand(ref)
            return m.group() if expanded==ref else expanded
        return pattern.sub(replace,text)


def annotate_schemas(schemas,mode=None):
    mode=mode_value(mode)
    if mode=='raw':return schemas
    from .url_prompt import prompt_module
    descriptions=prompt_module(mode).get('tool_descriptions',{})
    result=copy.deepcopy(schemas)
    for wrapper in result or []:
        s=wrapper.get('function',wrapper)
        if s.get('name') not in ('web_search','web_extract','vision_analyze'):continue
        text=descriptions.get(s['name'],'')
        if text:s['description']=s.get('description','')+'\n\n'+text
        if s.get('name')=='web_extract':
            params=s.setdefault('parameters',{'type':'object'})
            properties=params.setdefault('properties',{})
            properties.pop('url',None)
            properties['urls']={'type':'array','items':{'type':'string'},'maxItems':5}
            params['required']=['urls']+[k for k in params.get('required',[]) if k not in ('url','urls')]
    return result


def runtime_registry(scope,mode=None):
    mode=mode_value(mode)
    if not scope:raise ValueError('URL modes require an isolated session/rollout identifier')
    root=Path(os.environ.get('GLASSEEK_URL_STATE_DIR',str(Path(os.environ.get('HERMES_HOME', '/tmp/glasseek-hermes'))/'url_registry')))
    key=(str(root),scope,mode)
    with _LOCK:
        if key not in _REGISTRIES:
            path=root/(hashlib.sha256(scope.encode()).hexdigest()+'.json')
            state=json.loads(path.read_text()) if path.exists() else None
            reg=URLRegistry(mode,state);reg.state_path=path;_REGISTRIES[key]=reg
        return _REGISTRIES[key]


def persist(reg):
    with reg.lock:
        path=getattr(reg,'state_path',None)
        if path:
            path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(reg.snapshot(),ensure_ascii=False));tmp.replace(path)


def runtime_resolve(scope,name,args):
    if mode_value()=='raw':return args,None
    reg=runtime_registry(scope)
    cached=reg.cached_evidence(name,args)
    return reg.resolve_args(name,args),cached


def runtime_observe(scope,name,result):
    if mode_value()=='raw':return result
    reg=runtime_registry(scope);result=reg.observe(name,result);persist(reg);return result


def carry_registry(old_scope,new_scope):
    """Only compression/session continuation calls this; never share across tasks."""
    source=runtime_registry(old_scope);target=runtime_registry(new_scope)
    with source.lock,target.lock:
        if target.entries and target.entries!=source.entries:raise ValueError('Conflicting URL state on session rotation')
        clone=URLRegistry(source.mode,source.snapshot())
        for key in ('entries','documents','by_url','visible','step','events'):
            setattr(target,key,getattr(clone,key))
        persist(target)


def expand_tree(value,reg):
    if isinstance(value,str):return reg.expand_references(value)
    if isinstance(value,list):return [expand_tree(v,reg) for v in value]
    if isinstance(value,dict):return {k:expand_tree(v,reg) for k,v in value.items()}
    return value


def rewrite_assistant(text,reg):
    """Offline SFT only. This function is never applied to RL sampled tokens."""
    if not isinstance(text,str) or reg.mode=='raw':return text
    def call_sub(match):
        block=match.group()
        if block.startswith('```'):return block
        body=block[len('<tool_call>'):-len('</tool_call>')].strip()
        if body.startswith('{'):
            try:
                call=json.loads(body);fn=call.get('function',call);args=fn.get('arguments',{})
                typed=json.loads(args) if isinstance(args,str) else args
                fn['arguments']=reg.encode_args(fn.get('name'),typed)
                return '<tool_call>'+json.dumps(call,ensure_ascii=False)+'</tool_call>'
            except (ValueError,TypeError):return block
        name=re.search(r'<function=([^>]+)>',body)
        if not name or name[1] not in ('web_extract','vision_analyze'):return block
        def param(m):
            if m[1] not in ('url','urls','image_url'):return m.group()
            value=m[2].strip()
            try:typed=json.loads(value)
            except ValueError:typed=value
            key='urls' if name[1]=='web_extract' and m[1]=='url' else m[1]
            converted=reg.encode_args(name[1],{m[1]:typed})[key]
            if converted==typed and key==m[1]:return m.group()
            return '<parameter='+key+'>'+ (converted if isinstance(converted,str) else json.dumps(converted,ensure_ascii=False))+'</parameter>'
        return re.sub(r'<parameter=([^>]+)>(.*?)</parameter>',param,block,flags=re.S)
    parts=[];last=0
    for m in CALL.finditer(text):
        parts.extend([reg.known_text(text[last:m.start()]),call_sub(m)]);last=m.end()
    return ''.join(parts)+reg.known_text(text[last:])


def convert_messages(messages,mode='raw'):
    if mode=='raw':return copy.deepcopy(messages),URLRegistry('raw').snapshot()
    reg=URLRegistry(mode);result=[];names={};last_name=None
    for source in messages:
        msg=copy.deepcopy(source);role=msg.get('role')
        if role=='assistant':
            reg.begin_turn()
            for field in ('content','reasoning_content','reasoning'):
                if isinstance(msg.get(field),str):msg[field]=rewrite_assistant(msg[field],reg)
            for call in msg.get('tool_calls') or []:
                fn=call.get('function',call);name=fn.get('name');args=fn.get('arguments',{})
                was_string=isinstance(args,str);args=json.loads(args) if was_string else args
                converted=reg.encode_args(name,args)
                # Check the actual executor receives the original arguments.
                if reg.resolve_args(name,converted)!=reg.resolve_args(name,args):raise ValueError('URL conversion changed a tool target')
                fn['arguments']=json.dumps(converted,ensure_ascii=False) if was_string else converted
                names[call.get('id')]=name;last_name=name
            if not msg.get('tool_calls'):
                found=re.findall(r'<function=([^>]+)>',msg.get('content') or '')
                if found:last_name=found[-1]
        elif role=='tool' or (role=='user' and str(msg.get('content','')).lstrip().startswith('<tool_response>')):
            name=msg.get('name') or msg.get('tool_name') or names.get(msg.get('tool_call_id')) or last_name
            msg['content']=reg.observe(name,msg.get('content',''))
        result.append(msg)
    return result,reg.snapshot()
