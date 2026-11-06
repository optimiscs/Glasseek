import collections,hashlib,json,os
from pathlib import Path
import pyarrow.parquet as pq
root=Path('/root/autodl-tmp/Glasseek');src=root/'data/glasseek/sft-eval-fixed-64k';out=root/'analysis/no-tool-screen';out.mkdir(parents=True,exist_ok=True)
selection=[json.loads(l) for l in (src/'selection.jsonl').open()];groups={};rows=[]
for split in ('train','val'):
 metas=[x for x in selection if x['split']==split];table=pq.read_table(src/f'{split}.parquet',columns=['messages']);assert len(metas)==table.num_rows
 for meta,row in zip(metas,table.to_pylist()):
  messages=json.loads(row['messages']);question=next(x['content'] for x in messages if x['role']=='user');answer=next(x['content'] for x in reversed(messages) if x['role']=='assistant');key=' '.join(question.split());qid=hashlib.sha256(key.encode()).hexdigest();groups.setdefault(qid,{'question_id':qid,'question':question,'row_ids':[]})['row_ids'].append(meta['source_line']);rows.append({'source_line':meta['source_line'],'split':split,'question_id':qid,'teacher_answer':answer})
assert len(rows)==3265
for name,vals in [('questions',groups.values()),('rows',rows)]:
 with (out/(name+'.jsonl')).open('w') as f:
  for x in vals:f.write(json.dumps(x,ensure_ascii=False)+'\n')
print(json.dumps({'rows':len(rows),'questions':len(groups),'train':sum(x['split']=='train' for x in rows),'val':sum(x['split']=='val' for x in rows)},ensure_ascii=False))
