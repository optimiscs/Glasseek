import collections,hashlib,json
from pathlib import Path
P=Path('/root/autodl-tmp/Glasseek');O=P/'experiments/no-tool-top10/explicit-matches';removed={r['question_id'] for r in map(json.loads,(O/'strict_literal_questions.jsonl').open())};assert len(removed)==96
meta=[x for x in map(json.loads,(P/'data/glasseek/sft-eval-fixed-128k/audit.jsonl').open()) if not x['reasons']];groups={}
for i,l in enumerate((P/'data/glasseek/sft-eval-fixed-128k/all_filtered.jsonl').open()):
 x=json.loads(l);q=next(m['content'] for m in x['messages'] if m['role']=='user');qid=hashlib.sha256(q.encode()).hexdigest();groups.setdefault(qid,{'question_id':qid,'question':q,'source_lines':[]})['source_lines'].append(meta[i]['source_line'])
assert len(groups)==2995 and removed<=groups.keys()
with (O/'remaining_questions.jsonl').open('w') as f:
 for qid,row in groups.items():
  if qid not in removed:f.write(json.dumps(row,ensure_ascii=False)+'\n')
print(json.dumps({'original_questions':len(groups),'removed_questions':len(removed),'remaining_questions':len(groups)-len(removed),'removed_trajectories':sum(len(groups[q]['source_lines']) for q in removed),'remaining_trajectories':sum(len(r['source_lines']) for q,r in groups.items() if q not in removed)},ensure_ascii=False))
