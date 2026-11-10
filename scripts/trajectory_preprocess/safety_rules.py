"""Offline safety signals, never credential probes, DNS requests or code execution.

Findings contain positions and keyed fingerprints, not source snippets. Quoted
content and executed tool arguments receive different dispositions.
"""
import ast
import copy
from datetime import date
import hashlib
import hmac
import html
import ipaddress
import json
from pathlib import Path
import re
import socket
from urllib.parse import parse_qsl, unquote, urlsplit

from client import digest
from glasseek_contract import validate_args, private_temp_target
from safety_context import build_context, View
from answer_leakage import external_signals, question_text, guard, PLUGIN_SHA256

VERSION = 'safety-rules-v5-web-batch-response-scope'
URL = re.compile(r'(?:https?|ftp|file|gopher)://[^\s<>"\x27\\]+', re.I)
SECRET = re.compile(r'(?<![\w-])(?:sk-(?:ant-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{30,}|xox[baprs]-[A-Za-z0-9-]{16,})')
PRIVATE_KEY = re.compile(r'-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----[\s\S]*?(?:-----END (?:[A-Z0-9]+ )?PRIVATE KEY-----|$)')
ASSIGNMENT = re.compile(r'''(?ix)["']?(?:api[_-]?key|access_token|refresh_token|client_secret|password|authorization)["']?\s*[:=]\s*(?:"([^"\n]{6,})"|'([^'\n]{6,})'|([^\s"'<>;,}&\\]{6,})(?=\s*(?:$|[\n,}\#])))''')
BEARER = re.compile(r'(?i)(?:\bBearer|\bAuthorization\s*:\s*Basic)\s+([A-Za-z0-9._~+/=-]{16,})')
UNQUOTED = re.compile(r'''(?ix)\b(?:api[_-]?key|access_token|refresh_token|client_secret|password|passwd)\s*:\s*([^\s"'<>;,}&\\]{6,})''')
EMAIL = re.compile(r'(?<![\w.+-])[A-Za-z0-9.!#$%&\x27*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])')
PHONE = re.compile(r'(?<!\w)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\w)')
IDENTITY = re.compile(r'(?<!\w)[1-9]\d{16}[0-9Xx](?!\w)')
SHELL = re.compile(r'(?i)(?:\brm\s+-[a-z]*r[a-z]*f[a-z]*\s+/(?:\s|$)|\b(?:curl|wget)\b[^\n]{0,300}\|\s*(?:ba)?sh\b)')
AUTH_KEYS = {'api_key','apikey','access_token','refresh_token','token','password','passwd','client_secret','authorization','x-amz-signature','x-goog-signature'}
PLACEHOLDER = re.compile(r'(?i)^(?:\$\{[A-Z0-9_]+\}|<[^>]+>|YOUR_[A-Z_]+|REDACTED|\[REDACTED(?:_[0-9a-f]{12})?\])$')


def decoded(text, layers=3):
    for _ in range(layers):
        changed = html.unescape(unquote(re.sub(r'\\u([0-9a-fA-F]{4})', lambda m:chr(int(m[1],16)), text)))
        if changed == text: break
        text = changed
    return text


def identity_valid(value):
    try: date(int(value[6:10]), int(value[10:12]), int(value[12:14]))
    except ValueError: return False
    weights = (7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2)
    return '10X98765432'[sum(int(c)*w for c,w in zip(value[:17],weights)) % 11] == value[-1].upper()


def unsafe_url(value):
    try:
        p = urlsplit(decoded(value)); port = p.port
        if p.scheme.lower() not in ('http','https'): return 'scheme'
        host = (p.hostname or '').rstrip('.').lower()
        if not host: return 'malformed'
        if host == 'localhost' or host.endswith(('.localhost','.local','.internal')): return 'local_target'
        try: ip = ipaddress.ip_address(host)
        except ValueError:
            if re.fullmatch(r'(?:0x[0-9a-f]+|[0-9.]+)',host):
                try: ip = ipaddress.ip_address(socket.inet_aton(host))
                except OSError: return 'ambiguous_numeric_host'
            else: return None  # unresolved public hostname, not a historical DNS verdict
        if isinstance(ip,ipaddress.IPv6Address) and ip.ipv4_mapped: ip = ip.ipv4_mapped
        if not ip.is_global: return 'non_global_target'
    except ValueError: return 'malformed'
    return None


def code_capabilities(code):
    try: tree = ast.parse(code)
    except (ValueError,SyntaxError): return {'unparseable'}
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            for a in node.names: aliases[a.asname or a.name.split('.')[0]] = a.name
        if isinstance(node,ast.ImportFrom):
            for a in node.names: aliases[a.asname or a.name] = (node.module or '')+'.'+a.name
    def name(node):
        if isinstance(node,ast.Name): return aliases.get(node.id,node.id)
        if isinstance(node,ast.Attribute): return name(node.value)+'.'+node.attr
        if isinstance(node,ast.Call): return name(node.func)+'()'
        return ''
    result = set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            n = name(node.func)
            if n in {'eval','exec','compile','__import__','getattr','setattr'}: result.add('dynamic_execution')
            if n.startswith(('subprocess.','os.system','os.popen','os.exec')): result.add('subprocess')
            if n.startswith(('requests.','httpx.','urllib.','socket.','http.client.')): result.add('network')
            if n in {'open','io.open'}:
                target=node.args[0] if node.args else next((k.value for k in node.keywords if k.arg=='file'),None)
                if not private_temp_target(target):result.add('filesystem')
            elif n.startswith(('shutil.','os.remove','os.unlink','os.rmdir')) or n.endswith(('.write_text','.write_bytes','.read_text','.read_bytes','.open')): result.add('filesystem')
        if isinstance(node,ast.Attribute) and name(node).startswith(('os.environ','os.getenv')): result.add('environment_access')
    return result


class Scanner:
    def __init__(self, policy, contract, key):
        if not isinstance(key,bytes) or len(key) < 32: raise ValueError('Safety HMAC key must contain at least 32 bytes')
        for name in ('max_view_chars','max_findings','max_decode_layers'):
            if not isinstance(policy[name],int) or isinstance(policy[name],bool) or policy[name]<1:
                raise ValueError('Safety rule limits must be positive integers')
        self.policy, self.contract, self.key = policy, contract, key
        self.injection = [re.compile(p) for p in policy['injection_patterns']]
        self.benchmarks = [re.compile(p) for p in policy['benchmark_patterns']]
        # Runtime review budgets/providers can change without repeating the rule scan.
        # Each model decision separately records its review settings and config hash.
        shared_hash=hashlib.sha256((Path(__file__).resolve().parents[2]/'Shared/glasseek_shared/historical_tools.py').read_bytes()).hexdigest()
        extract_hash=hashlib.sha256((Path(__file__).resolve().parents[2]/'Shared/glasseek_shared/extract_protocol.py').read_bytes()).hexdigest()
        self.policy_hash = digest([VERSION,{k:v for k,v in policy.items() if k!='llm'},contract,PLUGIN_SHA256,shared_hash,extract_hash,{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                 for n in ('safety_rules.py','safety_context.py','answer_leakage.py')}])

    def fingerprint(self, text):
        return hmac.new(self.key,text.encode(),hashlib.sha256).hexdigest()

    def sensitive_spans(self, text):
        spans=[]
        for rule, regex in [('credential.private_key',PRIVATE_KEY),('credential.token',SECRET),
                            ('privacy.email',EMAIL),('privacy.phone',PHONE),('privacy.identity',IDENTITY)]:
            if rule in self.policy.get('disabled_rules', []): continue
            for m in regex.finditer(text):
                if rule == 'privacy.identity' and not identity_valid(m[0]): continue
                spans.append((m.start(),m.end(),rule))
        for regex in (ASSIGNMENT, BEARER):
            for m in regex.finditer(text):
                group=next(i for i,value in enumerate(m.groups(),1) if value is not None)
                if not PLACEHOLDER.fullmatch(m[group]):spans.append((m.start(group),m.end(group),'credential.assignment'))
        for m in UNQUOTED.finditer(text):
            value=m[1]
            if PLACEHOLDER.fullmatch(value):continue
            token_shape=any(c.isdigit() or not c.isalpha() for c in value) or (any(c.isupper() for c in value[1:]) and any(c.islower() for c in value))
            next_field=re.match(r'\s+[A-Za-z][A-Za-z ]{0,40}:',text[m.end(1):])
            if token_shape or next_field:spans.append((m.start(1),m.end(1),'credential.unquoted_value'))
        for m in URL.finditer(text):
            try:
                p=urlsplit(m[0])
                if p.username or p.password or any(k.casefold() in AUTH_KEYS and v for k,v in parse_qsl(p.query)):
                    spans.append((m.start(),m.end(),'credential.url_auth'))
            except ValueError: pass
        return spans

    def scrub(self, text):
        # Preserve JSON structure in logs/prompts, including escaped quotes and keys.
        try: value=json.loads(text)
        except (ValueError,TypeError): value=None
        if isinstance(value,(dict,list)):
            def walk(v):
                if isinstance(v,str): return self._scrub_text(v)
                if isinstance(v,list): return [walk(x) for x in v]
                if isinstance(v,dict): return {self._scrub_text(k):walk(x) for k,x in v.items()}
                return v
            revised=walk(value)
            return text if revised==value else json.dumps(revised,ensure_ascii=False)
        return self._scrub_text(text)

    def _scrub_text(self, text):
        original=text
        text=decoded(text,self.policy['max_decode_layers'])
        spans = self.sensitive_spans(text)
        if not spans: return original
        intervals=[]
        for start,end,_ in sorted(spans):
            if intervals and start <= intervals[-1][1]: intervals[-1][1]=max(end,intervals[-1][1])
            else: intervals.append([start,end])
        for start,end in reversed(intervals):
            text=text[:start]+'[REDACTED_'+self.fingerprint(text[start:end])[:12]+']'+text[end:]
        return text

    def scan(self, item):
        views, actions, issues = build_context(item)
        question = question_text(item)
        findings=[]; known=set(); overflow=False
        def add(rule, view, start=0, end=0, disposition='review', details=None):
            nonlocal overflow
            identity=(rule,view.pointer,start,end,view.derived)
            if identity in known: return
            known.add(identity)
            if len(findings) >= self.policy['max_findings']:
                overflow=True; return
            value=view.text[start:end]
            finding={'rule_id':rule,'category':rule.split('.')[0], 'pointer':self.scrub(view.pointer),
                     'view_id':self.fingerprint(view.pointer + str(view.derived))[:24],
                     'message_index':view.message_index,'origin':view.kind,'derived_view':view.derived,
                     'start':start,'end':end,'offset_unit':'unicode_codepoint',
                     'evidence_hmac':self.fingerprint(value),'disposition':disposition}
            if details: finding['details']=details
            finding['finding_id']='F'+digest(finding)[:20]
            findings.append(finding)
        for issue in issues:
            add('schema.'+issue,View('/messages','','structure'),disposition='quarantine')
        tools=item.get('tools')
        try:
            if isinstance(tools,str): tools=json.loads(tools)
            declared={t['function']['name']:t['function'] for t in tools}
            expected={n:s for n,s in self.contract['schemas'].items() if item.get('agent_role')!='child' or n not in self.contract['delegation']['blocked_child_tools']}
            if declared != expected or len(declared)!=len(tools): add('schema.tool_contract_mismatch',View('/tools','','structure'),disposition='quarantine')
        except (ValueError,TypeError,KeyError): add('schema.tool_contract_missing',View('/tools','','structure'),disposition='quarantine')
        destinations=set(); executable_code=set()
        for action in actions:
            i=action['message_index']; name=action['name']; args=action['arguments']
            v=View(f'/messages/{i}/parsed_calls/{action["call_index"]}',json.dumps(args,ensure_ascii=False),'argument',i,True)
            if name not in self.contract['schemas']:
                add('schema.unknown_tool',v,disposition='quarantine'); continue
            errors=validate_args(name,args,self.contract)
            if errors: add('schema.arguments_or_capability',v,disposition='quarantine')
            if item.get('agent_role')=='child' and name in self.contract['delegation']['blocked_child_tools']:
                add('schema.child_tool_blocked',v,disposition='quarantine')
            if name in ('web_extract','vision_analyze'):
                base=v.pointer+'/arguments'
                for key in ('url','image_url'):
                    if key in args: destinations.add(base+'/'+key)
                if isinstance(args.get('urls'),list):
                    destinations.update(base+'/urls/'+str(j) for j in range(len(args['urls'])))
            if name=='execute_code' and isinstance(args.get('code'),str):
                executable_code.add(v.pointer+'/arguments/code')
                for capability in code_capabilities(args['code']):
                    add('code.'+capability,v,0,len(v.text),'quarantine')
        for original in views:
            if len(original.text)>self.policy['max_view_chars']:
                add('coverage.view_limit',original,disposition='review'); continue
            variants=[original]
            text=decoded(original.text,self.policy['max_decode_layers'])
            if text != original.text:
                variants.append(View(original.pointer+'/decoded',text,original.kind,original.message_index,True))
            if original.kind=='external_part':
                # Parse each recorded call return independently for query/goal
                # echo removal. The full concatenation is still scanned below
                # for credentials, unsafe URLs, code and prompt injection.
                for v in variants:
                    for rule,details in external_signals(v.text,question,self.policy.get('answer_overlap_max_chars',120000)):
                        add(rule,v,0,len(v.text),details=details)
                continue
            for v in variants:
                for start,end,rule in self.sensitive_spans(v.text):
                    disposition = 'quarantine' if rule.startswith('credential.') else 'review'
                    if rule=='credential.unquoted_value':disposition='review'
                    # A token-looking slug is not enough to declare a URL path a key.
                    if rule == 'credential.token' and any(m.start() <= start < end <= m.end() for m in URL.finditer(v.text)):
                        disposition = 'review'
                    add(rule,v,start,end,disposition)
                if v.kind=='metadata': continue
                if v.pointer.removesuffix('/decoded') in destinations and not v.text.startswith('data:image/'):
                    problem=unsafe_url(v.text)
                    if problem: add('url.'+problem,v,0,len(v.text),'quarantine')
                for match in URL.finditer(v.text):
                    problem=unsafe_url(match[0])
                    if problem: add('url.'+problem,v,match.start(),match.end(),'quarantine' if v.pointer.removesuffix('/decoded') in destinations else 'review')
                if v.kind == 'argument' and v.pointer.removesuffix('/decoded').rsplit('/',1)[-1] != 'goal':
                    if guard._BENCHMARK_RE.search(v.text) or any(regex.search(v.text) for regex in self.benchmarks):
                        add('benchmark.resource_requested',v,0,len(v.text),details={'actual_tool_argument':True,'proof_of_answer_use':False})
                if v.kind == 'external':
                    for rule, details in external_signals(v.text, question, self.policy.get('answer_overlap_max_chars',120000)):
                        add(rule,v,0,len(v.text),details=details)
                for match in SHELL.finditer(v.text):
                    add('code.shell_pattern',v,match.start(),match.end(),'quarantine' if v.pointer.removesuffix('/decoded') in executable_code else 'review')
                if v.kind in ('external','external_batch'):
                    for regex in self.injection:
                        for match in regex.finditer(v.text):
                            targets={m[0].rstrip('.,;)') for m in URL.finditer(v.text)}
                            later=[a['message_index'] for a in actions if a['message_index']>v.message_index and any(t in json.dumps(a['arguments'],ensure_ascii=False) for t in targets)]
                            add('injection.suspected',v,match.start(),match.end(),details={
                                'later_same_url_call_messages':sorted(set(later)),
                                'causality':'matching target is a signal, not proof of instruction following'})
        if overflow:
            findings.append({'finding_id':'coverage-limit','rule_id':'coverage.finding_limit','category':'coverage','disposition':'review'})
        state='quarantine' if any(f['disposition']=='quarantine' for f in findings) else 'review' if findings else 'pass_rules'
        return {'state':state,'findings':findings,'policy_hash':self.policy_hash,'content_hash':digest(item),
                'coverage_complete':not overflow and not any(f['category']=='coverage' for f in findings),
                'training_ready':False}

    def redact_preview(self, item):
        changes=[]
        def walk(value,pointer=''):
            if isinstance(value,str):
                sanitized=self.scrub(value)
                if sanitized != value:
                    changes.append({'pointer':self.scrub(pointer),'source_hmac':self.fingerprint(value),'replacement_hash':digest(sanitized)})
                    return sanitized
                return value
            if isinstance(value,list): return [walk(v,pointer+'/'+str(i)) for i,v in enumerate(value)]
            if isinstance(value,(int,float)) and not isinstance(value,bool):
                sanitized=self.scrub(str(value))
                if sanitized!=str(value):
                    changes.append({'pointer':self.scrub(pointer),'source_hmac':self.fingerprint(str(value)),'replacement_hash':digest(sanitized)})
                    return sanitized
            if isinstance(value,dict):
                result={}
                for k,v in value.items():
                    new_key=walk(k,pointer+'/@key')
                    if new_key in result: raise ValueError('Redacted dictionary keys collide')
                    result[new_key]=walk(v,pointer+'/'+str(k))
                return result
            return value
        revised=walk(copy.deepcopy(item))
        revised['training_ready']=False
        revised['safety']={'state':'redacted_pending_validation','source_hash':digest(item),'policy_hash':self.policy_hash,'changes':changes}
        if 'messages' in revised and 'tools' in revised: revised['payload_sha256']=digest({'messages':revised['messages'],'tools':revised['tools']})
        revised['safety']['rescan_state']=self.scan(revised)['state']
        return revised
