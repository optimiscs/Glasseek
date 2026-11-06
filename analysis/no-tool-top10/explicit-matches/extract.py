import collections,hashlib,json,re,sqlite3,unicodedata
from pathlib import Path
P=Path('/root/autodl-tmp/Glasseek');O=P/'analysis/no-tool-top10/explicit-matches';c=sqlite3.connect(P/'analysis/no-tool-top10/snapshot.sqlite3');answers={r[0]:r[1:] for r in c.execute('select qid,status,finish,reasoning,content,candidates,error from answers')};reviews=collections.defaultdict(list)
for r in c.execute('select rowid,qid,status,verdict,teacher_answer,method,matches,reasoning_match,reason,error from reviews'):reviews[r[0]].append(r)
c.close()
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s or '').casefold()).strip()
def explicit(s):
 s=re.sub(r'<think\b[^>]*>.*?</think\s*>','',s,flags=re.I|re.S);found=list(re.finditer(r'(?i)\bfinal\s+answer[ \t]*:[ \t]*([^\r\n]*)',s))
 if not found:return None
 arr=[x.group(1).strip().strip('*_`').strip() for x in found]
 if any(not x or len(x)>200 or norm(x) in ('unknown','uncertain','none','n/a') for x in arr) or len({norm(x) for x in arr})!=1:return None
 return arr[-1]
def in_text(needle,text):
 needle=norm(needle);text=norm(text)
 if not needle or len(needle)<3:return False
 left=r'(?<!\w)' if needle[0].isascii() and needle[0].isalnum() else ''
 right=r'(?!\w)' if needle[-1].isascii() and needle[-1].isalnum() else ''
 return bool(re.search(left+re.escape(needle)+right,text))
meta=[x for x in map(json.loads,(P/'data/glasseek/sft-eval-fixed-128k/audit.jsonl').open()) if not x['reasons']];groups=collections.defaultdict(list)
for i,line in enumerate((P/'data/glasseek/sft-eval-fixed-128k/all_filtered.jsonl').open(),1):
 row=json.loads(line);ms=row['messages'];q=next(m['content'] for m in ms if m['role']=='user');qid=hashlib.sha256(q.encode()).hexdigest();teacher=next(m['content'] for m in reversed(ms) if m['role']=='assistant');groups[qid].append((i,meta[i-1]['source_line'],teacher))
hits={};counts=collections.Counter()
for qid,items in groups.items():
 a=answers.get(qid)
 if not a:continue
 status,finish,reasoning,content,candidate_json,error=a
 try:cs=json.loads(candidate_json or '[]')
 except ValueError:cs=[]
 matches=[]
 for i,source,teacher in items:
  ta=explicit(teacher)
  rev=reviews.get(i,[])
  for x in rev:
   _,_,st,verdict,judged,method,raw,rm,reason,err=x
   if st=='ok' and not err and judged and 'invalid_teacher_evidence' not in (reason or ''):
    for v in json.loads(raw or '[]'):
     if v.get('match_type') in ('exact','equivalent'):matches.append({'source_line':source,'teacher_answer':judged,'kind':'candidate_judged_'+v['match_type'],'rank':v.get('rank'),'answer':v.get('answer')})
    rv=json.loads(rm or '{}');ev=rv.get('evidence') or ''
    if rv.get('found') and norm(judged) and len(norm(judged))>=3 and norm(judged) in norm(ev):matches.append({'source_line':source,'teacher_answer':judged,'kind':'reasoning_exact_verified','rank':None})
  if ta:
   for x in cs:
    for field in [x.get('answer',''),*(x.get('aliases') or [])]:
     if norm(field)==norm(ta):matches.append({'source_line':source,'teacher_answer':ta,'kind':'candidate_literal','rank':x.get('rank'),'answer':x.get('answer')});break
   if in_text(ta,reasoning or ''):matches.append({'source_line':source,'teacher_answer':ta,'kind':'reasoning_literal','rank':None})
   if not cs and in_text(ta,content or ''):matches.append({'source_line':source,'teacher_answer':ta,'kind':'raw_output_literal','rank':None})
 if matches:
  hits[qid]={'question_id':qid,'answer_status':status,'finish':finish,'matches':matches};counts[status+'/'+finish]+=1
  for kind in {m['kind'] for m in matches}:counts['kind/'+kind]+=1
with (O/'questions.jsonl').open('w') as f:
 for v in hits.values():f.write(json.dumps(v,ensure_ascii=False)+'\n')
print(json.dumps({'questions':len(groups),'matched_questions':len(hits),'counts':dict(counts),'path':str(O/'questions.jsonl')},ensure_ascii=False,indent=2))
