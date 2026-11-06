#!/usr/bin/env python3
"""Export conservatively screened train/val data without modifying input files."""
import collections,json,sqlite3,time
from pathlib import Path
import pyarrow as pa,pyarrow.parquet as pq
P=Path('/root/autodl-tmp/Glasseek/analysis/no-tool-screen');SRC=Path('/root/autodl-tmp/Glasseek/data/glasseek/sft-eval-fixed-64k');OUT=Path('/root/autodl-tmp/Glasseek/data/glasseek/sft-no-tool-screened-64k');OUT.mkdir(parents=True,exist_ok=True)
questions=[json.loads(x) for x in (P/'questions.jsonl').open()];rows=[json.loads(x) for x in (P/'rows.jsonl').open()];db=sqlite3.connect(P/'results.sqlite3');answers={x[0]:x[1:] for x in db.execute('select question_id,status,response,finish_reason,error from answers')};judges={x[0]:x[1:] for x in db.execute('select source_line,status,label,confidence,reason,error from judgements')};second={x[0]:x[1:] for x in db.execute('select question_id,verdict,confidence,reason,error from second_reviews')};db.close()
if len(answers)!=len(questions):raise RuntimeError(f'Inference not complete: {len(answers)}/{len(questions)}')
group=collections.defaultdict(list)
for r in rows:group[r['question_id']].append(r)
counter=collections.Counter({'candidate_easy_remove':0,'keep_unsolved':0,'keep_review':0});result={}
for q,rr in group.items():
 answer=answers[q];scores=[judges.get(r['source_line']) for r in rr]
 rev=second.get(q)
 easy=answer[0]=='ok' and all(j and j[0]=='ok' and j[1]=='easy' and j[2]>=0.90 for j in scores) and bool(rev and rev[0]=='clean' and rev[1]>=0.95 and not rev[3])
 for r,j in zip(rr,scores):
  status='candidate_easy_remove' if easy else ('keep_unsolved' if answer[0]=='ok' and j and j[0]=='ok' and j[1]=='not_solved' else 'keep_review')
  result[r['source_line']]={'source_line':r['source_line'],'split':r['split'],'question_id':q,'decision':status,'model_status':answer[0],'model_error':answer[3],'judge_status':j[0] if j else 'not_run','judge_label':j[1] if j else None,'judge_confidence':j[2] if j else None,'judge_reason':j[3] if j else None,'judge_error':j[4] if j else None,'second_review_verdict':rev[0] if rev else None,'second_review_confidence':rev[1] if rev else None,'second_review_reason':rev[2] if rev else None,'second_review_error':rev[3] if rev else None};counter[status]+=1
selection=[json.loads(l) for l in (SRC/'selection.jsonl').open()]
for split in ('train','val'):
 ids=[r['source_line'] for r in selection if r['split']==split];table=pq.read_table(SRC/f'{split}.parquet');assert len(ids)==table.num_rows
 indices=[i for i,n in enumerate(ids) if result[n]['decision']!='candidate_easy_remove'];easy_indices=[i for i,n in enumerate(ids) if result[n]['decision']=='candidate_easy_remove']
 pq.write_table(table.take(pa.array(indices,type=pa.int64())),OUT/f'{split}.parquet',compression='zstd');pq.write_table(table.take(pa.array(easy_indices,type=pa.int64())),OUT/f'{split}-candidate-easy.parquet',compression='zstd');counter[split+'_retained']=len(indices);counter[split+'_removed']=len(easy_indices)
with (OUT/'decisions.jsonl').open('w') as f:
 for r in rows:f.write(json.dumps(result[r['source_line']],ensure_ascii=False)+'\n')
# Include the actual no-tool answer, teacher reference and judge outcome for review.
question_by_id={q['question_id']:q['question'] for q in questions}
with (OUT/'review.jsonl').open('w') as f:
 for r in rows:
  d=dict(result[r['source_line']]);d['question']=question_by_id[r['question_id']];d['model_answer']=answers[r['question_id']][1];d['teacher_answer']=r['teacher_answer'];f.write(json.dumps(d,ensure_ascii=False)+'\n')
summary={'counts':dict(counter),'questions':len(questions),'trajectories':len(rows),'model':'Qwen3.5-9B base','mode':'one local generation per normalized unique user input; no tools supplied; temperature 0','judge':'DeepSeek Flash comparing the teacher response as a fallible reference','threshold':{'first_pass':0.90,'second_pass':0.95},'group_rule':'remove only if every associated teacher comparison is easy >=0.90 and a second audit is clean >=0.95; otherwise retain','warning':'candidate ease depends on fallible teacher answers and model judgement; review before training'};(OUT/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False))
