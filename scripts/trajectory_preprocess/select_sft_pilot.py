"""Quality-ranked, dataset-balanced difficulty strata; candidate export only."""
import argparse,json,sqlite3,collections,random,hashlib
from pathlib import Path
from rubric_batch import unpack
from glasseek_shared.trajectory_release import runtime_schemas

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--backfill',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 out=args.output;out.mkdir(parents=True,exist_ok=True)
 s=sqlite3.connect('file:'+str(args.source/'batch.sqlite')+'?mode=ro',uri=True);b=sqlite3.connect('file:'+str(args.backfill/'batch.sqlite')+'?mode=ro',uri=True)
 results={i:json.loads(r) for i,r in s.execute("SELECT id,result FROM results WHERE state='completed'")}
 for i,r in b.execute("SELECT id,payload FROM results WHERE state='completed'"):results[i]=json.loads(r)
 groups=collections.defaultdict(lambda:collections.defaultdict(list));excluded=collections.Counter();tools_schema=runtime_schemas()
 for r in sorted(results.values(),key=lambda r:(-(r.get('score') or 0),r['id'])):
  if r.get('status')!='rankable' or r.get('score',0)<85:excluded['quality_or_gate']+=1;continue
  d=r.get('difficulty') or {}
  if d.get('score') is None or d.get('confidence') is None or float(d['confidence'])<70:excluded['difficulty_missing_or_uncertain']+=1;continue
  if r.get('eligibility',{}).get('safety',{}).get('state')!='pass_rules':excluded['safety']+=1;continue
  row=s.execute('SELECT sample,canonical FROM inputs WHERE id=?',(r['id'],)).fetchone();item,canonical=map(unpack,row)
  from normalize_training_text import normalize
  canonical=normalize(canonical, tool_schemas=tools_schema)
  if canonical['text_normalization']['holds']:excluded['text_normalization_hold']+=1;continue
  if canonical.get('parent_id') or canonical.get('agent_role')=='child' or any(c['name']=='delegate_task' for st in item['steps'] for c in st['calls']):excluded['dependent_agent_bundle']+=1;continue
  difficulty=float(d['score']);level='easy' if difficulty<30 else 'medium' if difficulty<70 else 'hard'
  groups[level][r['source']].append((r,item,canonical))
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/root/autodl-fs/models/Qwen3.5-9B',trust_remote_code=True)
 picked={};seen=set();dataset_totals=collections.Counter()
 for level,target in [('hard',330),('easy',220),('medium',550)]:
  selected=[];pools=groups[level];positions=collections.Counter()
  while len(selected)<target:
   available=[name for name in pools if positions[name]<len(pools[name])]
   if not available:raise ValueError(f'Insufficient {level}: {len(selected)}/{target}; do not relax gates')
   name=min(available,key=lambda n:(dataset_totals[n],n));r,item,c=pools[name][positions[name]];positions[name]+=1
   group=c.get('split_group_id') or item.get('question_hash') or item['question'].strip()
   question_group=hashlib.sha256(item['question'].split('RESEARCH STATE SUMMARY')[0].rstrip('= \n').strip().encode()).hexdigest()
   if group in seen or question_group in seen:excluded['duplicate_question_group']+=1;continue
   messages=c['messages'];tools=c['tools']
   if isinstance(messages,str):messages=json.loads(messages)
   if isinstance(tools,str):tools=json.loads(tools)
   encoded=tokenizer.apply_chat_template(messages,tools=tools,enable_thinking=c.get('enable_thinking',True),tokenize=True,add_generation_prompt=False)
   ids=encoded['input_ids'] if hasattr(encoded,'keys') else encoded
   n=len(ids[0]) if len(ids) and isinstance(ids[0],list) else len(ids)
   assert n>2, 'Invalid rendered token count'
   if n>32768:excluded['over_32768_tokens']+=1;continue
   seen.update((group,question_group));dataset_totals[name]+=1;selected.append((r,c,n,group))
  picked[level]=selected
  print('Selected',level,len(selected),flush=True)
 rng=random.Random(42);stats={};audits=[]
 for split in ('train','val'):
  with (out/(split+'.candidates.jsonl')).open('w') as f:
   count=collections.Counter();levels=collections.Counter();lengths=[]
   for level,items in picked.items():
    if split=='train':rng.shuffle(items)
    validation={'easy':20,'medium':50,'hard':30}[level];chosen=items[validation:] if split=='train' else items[:validation]
    for r,c,n,group in chosen:
     # No reference answer, judge rationale, or difficulty labels in model input.
     c=dict(c);c.pop('reference_answer',None)
     f.write(json.dumps(c,ensure_ascii=False)+'\n');count[r['source']]+=1;levels[level]+=1;lengths.append(n)
     audits.append({'id':r['id'],'split':split,'dataset':r['source'],'quality':r['score'],'difficulty':r['difficulty']['score'],'confidence':r['difficulty']['confidence'],'tokens':n,'group':group})
   stats[split]={'rows':sum(count.values()),'datasets':dict(count),'difficulty':dict(levels),'tokens':sum(lengths),'max_tokens':max(lengths)}
 (out/'selection.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audits))
 report={'seed':42,'quality_min':85,'confidence_min':70,'max_length':32768,'splits':stats,'excluded':dict(excluded),'training_ready':False,'required':['hash-bound replay and review acceptance','benchmark overlap audit','publish_release before training']}
 (out/'selection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
