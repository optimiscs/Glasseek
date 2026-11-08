"""Offline, deterministic audit. Flags are review candidates, not correctness labels."""
import collections as C
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SRC = ROOT / 'data/Glasseek-SFT-5.7K/train.jsonl'
CALL = re.compile(r'<tool_call>(.*?)</tool_call>', re.S)
RESP = re.compile(r'<tool_response>(.*?)</tool_response>', re.S)
THINK = re.compile(r'<think>(.*?)</think>', re.S)
URL = re.compile(r'https?://[^\s<>"\\]+')
ERR = re.compile(r'"(?:success|ok)"\s*:\s*false|"error"\s*:\s*"[^"\s]|Traceback \(most recent call last\)|(?:^|\n)Error:|"(?:status|status_code)"\s*:\s*(?:4\d\d|5\d\d)', re.I)
def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def norm(s): return ' '.join(s.split())
def urls(s):
    out=set()
    for u in URL.findall(s):
        u=u.rstrip('.,;:)]}*')
        try:
            p=urlsplit(u);out.add(urlunsplit((p.scheme.lower(),p.netloc.lower(),p.path.rstrip('/'),p.query,'')))
        except ValueError: pass
    return out
def stats(v):
    if not v: return {}
    return dict(n=len(v),min=min(v),mean=float(np.mean(v)),p50=float(np.percentile(v,50)),p90=float(np.percentile(v,90)),p95=float(np.percentile(v,95)),p99=float(np.percentile(v,99)),max=max(v),sum=sum(v))
def question_body(q):
    if '\nQuestion:\n' in q:
        q=q.split('\nQuestion:\n',1)[1]
        for sep in ['\n\nWhen you produce your answer','\n\nReturn the final answer','\n\nAfter receiving tool results','\n\nReturn your final answer']:
            q=q.split(sep,1)[0]
    return norm(q)
def family(q):
    if q.startswith('You are answering one hard multi-constraint'): return 'multi_constraint'
    if q.startswith('You are running one browse-comparison'): return 'browse_comparison'
    return 'other'

def main():
    rows=[]; qs=C.defaultdict(list); bodies=C.defaultdict(list); full=C.defaultdict(list); nosys=C.defaultdict(list)
    roles=C.Counter(); tools=C.Counter(); sysgroups=C.Counter(); sys_examples={}; signals=C.defaultdict(list)
    selections={}
    for name in ['sft-64k','sft-128k']:
        selections[name]={x['source_line']:x for x in map(json.loads,(ROOT/'data/glasseek'/name/'selection.jsonl').open())}
    source_hash=hashlib.sha256()
    with SRC.open('rb') as f:
        for line,raw in enumerate(f,1):
            source_hash.update(raw)
            x=json.loads(raw); ms=x['messages']; q=next(m['content'] for m in ms if m['role']=='user'); sy=ms[0]['content']
            final=THINK.sub('',ms[-1]['content']).strip(); body=question_body(q); fam=family(q)
            rolechars=C.Counter(); counts=C.Counter(); callseq=[]; toolset=set(); observed_urls=set(); extra_users=[]; errors=[]; structural=[]
            for j,m in enumerate(ms):
                role=m['role']; s=m.get('content') or '';roles[role]+=1;rolechars[role]+=len(s)
                if role=='assistant':
                    ts=THINK.findall(s); counts['think_chars']+=sum(map(len,ts));counts['empty_thinks']+=sum(not t.strip() for t in ts)
                    counts['assistant_messages']+=1
                    if s.count('<think>')!=s.count('</think>'): structural.append([j,'unbalanced_think'])
                    if s.count('<tool_call>')!=s.count('</tool_call>'): structural.append([j,'unbalanced_tool_call'])
                    calls=CALL.findall(s);counts['calls']+=len(calls)
                    for call in calls:
                        n=re.search(r'<function=([^>]+)>',call)
                        if not n:
                            structural.append([j,'unparsed_tool_call']);continue
                        name=n.group(1);tools[name]+=1;toolset.add(name);callseq.append(norm(call))
                        counts['call_chars']+=len(call)
                    if calls:
                        nexts=ms[j+1]['content'] if j+1<len(ms) else ''
                        responses=RESP.findall(nexts)
                        if len(calls)!=len(responses): structural.append([j,'call_response_count',len(calls),len(responses)])
                elif role=='user' and j>1:
                    blocks=RESP.findall(s);counts['responses']+=len(blocks)
                    if not blocks: extra_users.append([j,s[:450]])
                    observed_urls.update(urls(s))
                    for b in blocks:
                        if ERR.search(b): errors.append([j,b[:250]])
            declared=set(re.findall(r'<function>\s*<name>([^<]+)</name>',sy))
            # Newer traces declare one JSON schema per line, while calls remain XML.
            for schema in re.findall(r'<tools>(.*?)</tools>',sy,re.S):
                for sl in schema.splitlines():
                    try:
                        d=json.loads(sl)
                        if isinstance(d,dict) and isinstance(d.get('function'),dict):declared.add(d['function']['name'])
                    except (ValueError,KeyError):pass
            undeclared=toolset-declared
            ccounts=C.Counter(callseq)
            answer=re.findall(r'FINAL ANSWER:\s*(.*)',final,re.I)
            finalurls=urls(final); unseen=finalurls-observed_urls-urls(q)
            flags=[]
            if structural: flags.append('structure_review')
            if ms[-1]['role']!='assistant' or '<tool_call>' in ms[-1]['content'] or not final: flags.append('incomplete_final')
            if errors: flags.append('tool_error_marker')
            if undeclared: flags.append('undeclared_tool')
            if extra_users: flags.append('extra_user_context')
            if any('CONTEXT COMPACTION' in s for _,s in extra_users):flags.append('compacted_context')
            if any('empty response' in s for _,s in extra_users):flags.append('empty_response_reprompt')
            if len(callseq)-len(ccounts)>=3: flags.append('at_least_3_repeated_calls')
            if finalurls and unseen: flags.append('final_url_unseen_in_observations')
            if not finalurls: flags.append('no_final_url')
            if 'FINAL ANSWER:' in q and not answer: flags.append('missing_requested_final_marker')
            if re.search(r'cannot (?:determine|identify|find|verify)|unable to (?:determine|identify|find|verify)|could not (?:determine|identify|find|verify)|couldn.t (?:determine|identify|find|verify)',final,re.I): flags.append('final_uncertainty')
            if re.search(r'already (?:passed|known|resolved)|retrospective|after.the.fact|not a (?:true|genuine|live) forecast|in the past|realized close',final,re.I): flags.append('retrospective_language')
            if re.search(r'forecast|predict(?:ed|ion)? probability|probability.*(?:YES|NO)',q,re.I): flags.append('forecast_question')
            if re.search(r'/root/|/workspace/|/tmp/',final): flags.append('final_local_path')
            if re.search(r'\b(?:browsecomp|gaia|xbench|livebrowsecomp|AIDABench)\b', '\n'.join(m['content'] for m in ms[2:] if m['role']=='assistant'),re.I): flags.append('benchmark_name_in_assistant')
            for flag in flags: signals[flag].append(line)
            r=dict(line=line,family=fam,question_hash=digest(norm(q)),body_hash=digest(body),question=body,system_chars=len(sy),role_chars=dict(rolechars),messages=len(ms),**counts,
                   repeated_calls=len(callseq)-len(ccounts),max_identical_call=max(ccounts.values(),default=0),tools=sorted(toolset),declared_tools=len(declared),undeclared_tools=sorted(undeclared),
                   final_chars=len(final),final_answer=answer[-1].strip() if answer else None,final_urls=len(finalurls),unseen_final_urls=sorted(unseen),error_blocks=len(errors),extra_users=extra_users,
                   structural=structural,flags=flags,split64=selections['sft-64k'][line]['split'],split128=selections['sft-128k'][line]['split'],tokens128=selections['sft-128k'][line]['tokens'],tokens_exact=selections['sft-128k'][line]['exact'])
            rows.append(r);qs[r['question_hash']].append(line);bodies[r['body_hash']].append(line)
            full[digest(json.dumps(ms,sort_keys=True,ensure_ascii=False))].append(line)
            nosys[digest(json.dumps(ms[1:],sort_keys=True,ensure_ascii=False))].append(line)
            sh=digest(sy);sysgroups[sh]+=1
            if sh not in sys_examples: sys_examples[sh]=dict(line=line,chars=len(sy),declared_tools=len(declared))
    groups={}
    for fam in ['all','other','browse_comparison','multi_constraint']:
        rr=rows if fam=='all' else [r for r in rows if r['family']==fam]
        groups[fam]=dict(rows=len(rr),unique_questions=len({r['question_hash'] for r in rr}),unique_bodies=len({r['body_hash'] for r in rr}),
          system_chars=stats([r['system_chars'] for r in rr]),tokens_exact_only=stats([r['tokens128'] for r in rr if r['tokens_exact']]),
          calls=stats([r.get('calls',0) for r in rr]),assistant_chars=stats([r['role_chars']['assistant'] for r in rr]),
          think_fraction_of_assistant_chars=sum(r.get('think_chars',0) for r in rr)/sum(r['role_chars']['assistant'] for r in rr),
          role_chars=dict(sum((C.Counter(r['role_chars']) for r in rr),C.Counter())),flags=dict(C.Counter(flag for r in rr for flag in r['flags'])),
          split64=dict(C.Counter(str(r['split64']) for r in rr)),split128=dict(C.Counter(str(r['split128']) for r in rr)))
    disagreements=[];leaks=[]
    for h,ls in bodies.items():
        answers={norm(rows[i-1]['final_answer']).lower() for i in ls if rows[i-1]['final_answer']}
        if len(answers)>1: disagreements.append(dict(lines=ls,answers=sorted(answers)))
        splits={rows[i-1]['split128'] for i in ls}-{None}
        if len(splits)>1: leaks.append(ls)
    report=dict(source=str(SRC),source_sha256=source_hash.hexdigest(),rows=len(rows),roles=dict(roles),groups=groups,
                tool_calls=dict(tools.most_common()),unique_system_prompts=len(sysgroups),top_system_prompts=[dict(**sys_examples[h],count=n) for h,n in sysgroups.most_common(10)],
                duplicate_question_multiplicity=dict(C.Counter(map(len,qs.values()))),duplicate_body_multiplicity=dict(C.Counter(map(len,bodies.values()))),
                exact_duplicate_groups=[v for v in full.values() if len(v)>1],duplicates_ignoring_system=[v for v in nosys.values() if len(v)>1],
                body_split_overlap=leaks,answer_string_disagreement_candidates=disagreements,signals=dict(signals),
                caveats=['No external fact verification or reference answers. Flags are candidates, not error rates.',
                         'URL presence is lexical provenance only, not claim entailment; terminal-written URLs may be unseen.',
                         'Token lengths from existing selection128; 13 excluded rows have lower-bound counts.',
                         'Question body extraction is conservative and template based; semantic duplicates may remain.'])
    (OUT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'rows.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['signals','answer_string_disagreement_candidates','duplicates_ignoring_system','exact_duplicate_groups']},ensure_ascii=False,indent=2))
    print('EXACT DUP GROUPS',len(report['exact_duplicate_groups']),'NOSYS DUP GROUPS',len(report['duplicates_ignoring_system']),'ANSWER DISAGREEMENTS',len(disagreements))

if __name__=='__main__':main()
