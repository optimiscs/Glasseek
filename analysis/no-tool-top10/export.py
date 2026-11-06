import collections,hashlib,json,re,sqlite3
from pathlib import Path
P=Path(__file__).resolve().parent
rows=[json.loads(x) for x in (P/'all_filtered.jsonl').open()];meta=[x for x in map(json.loads,(P/'audit.jsonl').open()) if not x['reasons']];assert len(rows)==len(meta)==3666
c=sqlite3.connect(P/'results.sqlite3');reviews={r[0]:r for r in c.execute('select rowid,qid,status,verdict,teacher_answer,method,matches,reasoning_match,reason,error from reviews')};c.close();assert len(reviews)==len(rows),f'reviews incomplete {len(reviews)}'
groups=collections.defaultdict(list)
for i,(row,m) in enumerate(zip(rows,meta),1):
 q=next(x['content'] for x in row['messages'] if x['role']=='user');qid=hashlib.sha256(q.encode()).hexdigest();groups[qid].append((i,row,m))
counts=collections.Counter()
with (P/'decisions.jsonl').open('w') as d,(P/'retained.jsonl').open('w') as out:
 for qid,group in groups.items():
  rr=[reviews[r[0]] for r in group];vals={re.sub(r'\s+',' ',str(r[4]).strip()).casefold() for r in rr};eligible=all(r[2]=='ok' and r[3]=='keep' for r in rr) and len(vals)==1
  winner=min(group,key=lambda x:(sum(m['role']=='assistant' for m in x[1]['messages']),x[2].get('fixed_total_tokens_estimate',float('inf')),x[2]['source_line'])) if eligible else None
  for i,row,m in group:
   r=reviews[i];state='keep' if winner and i==winner[0] else 'duplicate' if eligible else 'exclude_group';counts[state]+=1
   result={'index':i,'source_line':m['source_line'],'original_split':m['original_split'],'qid':qid,'decision':state,'status':r[2],'verdict':r[3],'teacher_answer':r[4],'method':r[5],'matched_candidates':json.loads(r[6]),'reasoning_match':json.loads(r[7]),'reason':r[8],'error':r[9]};d.write(json.dumps(result,ensure_ascii=False)+'\n')
   if state=='keep':out.write(json.dumps({'source_line':m['source_line'],'original_split':m['original_split'],'messages':row['messages']},ensure_ascii=False)+'\n')
(P/'manifest.json').write_text(json.dumps({'input_rows':len(rows),'questions':len(groups),'counts':counts,'note':'matched_candidates list contains answer and candidate rank; original split preserved'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps(counts,ensure_ascii=False))
