"""Supplementary error taxonomy, leakage examples and internal consistency evidence."""
import collections as C
import json
import re
from pathlib import Path
from audit import CALL,RESP,THINK,ERR,ROOT,OUT,question_body

rows={r['line']:r for r in map(json.loads,(OUT/'rows.jsonl').open())}
errors=C.Counter();examples={};error_rows=C.defaultdict(set);extras=C.Counter();traces={}
selected={1,10,1253,1588,1339,1628,1354,1468,1380,1475,1434,1664,3020,3022,4124,4231,4647,5257}
repeat_thoughts=[];max_think=[];constraint_flags=[];forecasts=[]
for line,raw in enumerate((ROOT/'data/Glasseek-SFT-5.7K/train.jsonl').open(),1):
    ms=json.loads(raw)['messages'];r=rows[line];paragraphs=[];largest=0
    final=THINK.sub('',ms[-1]['content']).strip()
    if line in selected:
        traces[line]=dict(question=question_body(ms[1]['content']),final=final)
        if line in [4647,5257]:
            target='ZSM 1688/2012' if line==4647 else '2B'
            traces[line]['observation_evidence']=[]
            for j,m in enumerate(ms):
                if m['role']=='user' and j>1 and target in m['content']:
                    at=m['content'].index(target);traces[line]['observation_evidence'].append(dict(message=j,snippet=m['content'][max(0,at-150):at+250]))
    if r['family']=='multi_constraint' and re.search(r'(?:clue|puzzle|question|constraint).{0,90}(?:incorrect|inaccur|imprecis|contradict|inconsisten|mismatch|does not|doesn.t)|(?:does not|doesn.t|cannot).{0,60}(?:fit|match).{0,40}(?:clue|constraint)',final,re.I):
        constraint_flags.append(line)
    if 'forecast_question' in r['flags'] and 'retrospective_language' in r['flags']:forecasts.append(line)
    for j,m in enumerate(ms):
        s=m['content']
        if m['role']=='assistant':
            for t in THINK.findall(s):
                largest=max(largest,len(t))
                paragraphs += [' '.join(p.split()) for p in t.split('\n\n') if len(p.strip())>=150]
            calls=CALL.findall(s)
            blocks=RESP.findall(ms[j+1]['content']) if j+1<len(ms) else []
            for call,b in zip(calls,blocks):
                if not ERR.search(b):continue
                n=re.search(r'<function=([^>]+)>',call);name=n.group(1) if n else 'unknown'
                clean=re.sub(r'</?untrusted_tool_result[^>]*>','',b).strip()
                try:
                    obj=json.loads(clean)
                    def find_errors(obj):
                        if isinstance(obj,dict):
                            for k,v in obj.items():
                                if k=='error' and v:yield str(v)
                                elif isinstance(v,(dict,list)):yield from find_errors(v)
                        elif isinstance(obj,list):
                            for v in obj:yield from find_errors(v)
                    ee=list(find_errors(obj))
                    if not ee:ee=['error marker, no structured error field']
                except ValueError:ee=['non-JSON error marker']
                for e in ee:
                    e=re.sub(r'https?://\S+','<URL>',e)
                    e=re.sub(r'\b\d{4,}\b','<NUM>',e)
                    key=name+': '+e[:180];errors[key]+=1;error_rows[key].add(line)
                    examples.setdefault(key,dict(line=line,message=j+1,snippet=b[:650]))
        elif m['role']=='user' and j>1 and not RESP.findall(s):
            extras['compaction' if 'CONTEXT COMPACTION' in s else 'empty_response' if 'empty response' in s else 'todo_restored' if 'task list was preserved' in s else 'other']+=1
    pc=C.Counter(paragraphs);dups=sum(len(p)*(n-1) for p,n in pc.items());total=sum(map(len,paragraphs))
    if total and dups/total>=.1:repeat_thoughts.append(dict(line=line,duplicate_paragraph_chars=dups,paragraph_chars=total,ratio=dups/total))
    max_think.append((largest,line))

out=dict(error_types=[dict(type=k,blocks=n,rows=len(error_rows[k]),example=examples[k]) for k,n in errors.most_common(25)],
         extra_user_messages=dict(extras),constraint_inconsistency_candidates=constraint_flags,
         forecast_with_retrospective_language=forecasts,thought_paragraph_duplicate_fraction_ge_10pct=repeat_thoughts,
         largest_think_chars=sorted(max_think,reverse=True)[:20],manual_evidence=traces)
(OUT/'diagnostics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['manual_evidence','error_types','thought_paragraph_duplicate_fraction_ge_10pct']},ensure_ascii=False,indent=2))
print('REPEATING_THOUGHT_ROWS',len(repeat_thoughts))
for e in out['error_types'][:12]:print(e['blocks'],e['rows'],e['type'],e['example']['line'])
