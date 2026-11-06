import collections,hashlib,json,re,sqlite3
from pathlib import Path
P=Path(__file__).resolve().parent
rows=[json.loads(x) for x in (P/'all_filtered.jsonl').open()]
metadata=[x for x in map(json.loads,(P/'audit.jsonl').open()) if not x['reasons']]
assert len(rows)==len(metadata)==3666
con=sqlite3.connect(P/'results.sqlite3');reviews={r[0]:r for r in con.execute('select rowid,qid,status,verdict,teacher_answer,teacher_evidence,student_evidence,reason,error from reviews')};answers={r[0]:r for r in con.execute('select question_id,status,finish,reasoning,content,error from answers')};con.close()
assert len(reviews)==len(rows),f'incomplete reviews: {len(reviews)}'
groups=collections.defaultdict(list)
for i,(row,meta) in enumerate(zip(rows,metadata),1):
 q=next(m['content'] for m in row['messages'] if m['role']=='user');qid=hashlib.sha256(q.encode()).hexdigest();groups[qid].append((i,row,meta))
counts=collections.Counter()
with (P/'decisions.jsonl').open('w') as decisions, (P/'retained.jsonl').open('w') as out:
 for qid,group in groups.items():
  rr=[reviews[x[0]] for x in group]; a=answers[qid]; vals={re.sub(r'\s+',' ',str(x[4]).strip()).casefold() for x in rr}
  eligible=all(x[2]=='ok' and x[3]=='keep' for x in rr) and len(vals)==1 and a[1]=='ok' and a[2]=='stop'
  winner=min(group,key=lambda x:(sum(m['role']=='assistant' for m in x[1]['messages']),x[2].get('fixed_total_tokens_estimate',float('inf')),x[2]['source_line'])) if eligible else None
  for i,row,meta in group:
   state=('keep' if winner and i==winner[0] else 'duplicate' if eligible else 'exclude_group'); counts[state]+=1
   decisions.write(json.dumps({'index':i,'source_line':meta['source_line'],'original_split':meta['original_split'],'qid':qid,'decision':state,'review':{'status':reviews[i][2],'verdict':reviews[i][3],'teacher_answer':reviews[i][4],'reason':reviews[i][7],'error':reviews[i][8]}},ensure_ascii=False)+'\n')
   if state=='keep':out.write(json.dumps({'source_line':meta['source_line'],'original_split':meta['original_split'],'messages':row['messages']},ensure_ascii=False)+'\n')
(P/'manifest.json').write_text(json.dumps({'input_rows':len(rows),'questions':len(groups),'counts':counts,'warning':'Local model audit is fallible; exclude all uncertain groups. Unassigned original split is preserved.'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps(counts,ensure_ascii=False))
