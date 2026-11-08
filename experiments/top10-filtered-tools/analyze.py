import collections,csv,hashlib,json,re,sqlite3,statistics
from pathlib import Path
P=Path('/root/autodl-tmp/Glasseek');source=P/'data/glasseek/sft-eval-fixed-128k';out=P/'experiments/top10-filtered-tools';db=sqlite3.connect(P/'reports/experiments/no-tool-top10/tool_counts_snapshot.sqlite3');reviews={r[0]:r for r in db.execute('select rowid,qid,status,verdict,teacher_answer,method,matches,reasoning_match,reason,error from reviews')};answers=dict(db.execute('select qid,status from answers'));db.close()
metadata=[x for x in map(json.loads,(source/'audit.jsonl').open()) if not x['reasons']];groups=collections.defaultdict(list)
with (source/'all_filtered.jsonl').open() as f:
 for i,line in enumerate(f,1):
  row=json.loads(line);q=next(m['content'] for m in row['messages'] if m['role']=='user');qid=hashlib.sha256(q.encode()).hexdigest();groups[qid].append((i,metadata[i-1],sum(m['role']=='assistant' for m in row['messages'])))
selected={};groups_status=collections.Counter()
for qid,g in groups.items():
 rr=[reviews.get(x[0]) for x in g]
 if qid not in answers:groups_status['unanswered']+=1;continue
 if any(r is None for r in rr):groups_status['review_incomplete']+=1;continue
 if any(r[2]!='ok' or r[3]!='keep' for r in rr):groups_status['excluded_nonkeep']+=1;continue
 vals={re.sub(r'\s+',' ',str(r[4]).strip()).casefold() for r in rr}
 if len(vals)!=1:groups_status['excluded_conflicting_teachers']+=1;continue
 win=min(g,key=lambda x:(x[2],x[1].get('fixed_total_tokens_estimate',float('inf')),x[1]['source_line']));selected[win[0]]=(qid,win[1]);groups_status['retained']+=1
CALL=re.compile(r'<tool_call>(.*?)</tool_call>',re.S);FN=re.compile(r'<function=([^>]+)>');toolcounts=collections.Counter();trajectory_counts=collections.Counter();rows=[];malformed=0
with (source/'all_filtered.jsonl').open() as f:
 for i,l in enumerate(f,1):
  if i not in selected:continue
  row=json.loads(l);qid,meta=selected[i];ts=collections.Counter();broken=0
  for m in row['messages']:
   if m['role']!='assistant':continue
   content=m['content'] or ''
   for c in CALL.findall(content):
    m2=FN.search(c)
    if m2:ts[m2.group(1)]+=1
    else:broken+=1
  n=sum(ts.values());toolcounts.update(ts);trajectory_counts.update(ts.keys());malformed+=broken
  rows.append({'source_line':meta['source_line'],'split':meta['original_split'],'question_id':qid,'total_calls':n,**{k:ts[k] for k in ['web_search','web_extract','execute_code','delegate_task','vision_analyze']},'unknown_or_unparsed':broken})
assert len(rows)==len(selected)
vals=[r['total_calls'] for r in rows];hist=collections.Counter(vals);count=len(rows)
result={'reviewed_questions':sum(1 for q in groups if q in answers and all(x[0] in reviews for x in groups[q])),'questions':len(groups),'reviews':len(reviews),'answers':len(answers),'group_status':dict(groups_status),'selected_trajectories':count,'tool_calls_total':sum(vals),'tool_counts':dict(toolcounts),'tool_trajectory_counts':dict(trajectory_counts),'unparsed_calls':malformed,'summary':{'mean':sum(vals)/count if count else None,'median':statistics.median(vals) if count else None,'min':min(vals) if count else None,'max':max(vals) if count else None},'histogram':dict(sorted(hist.items()))};(out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));
with (out/'per_trajectory.csv').open('w',newline='') as f:
 if rows:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(result,ensure_ascii=False,indent=2))
