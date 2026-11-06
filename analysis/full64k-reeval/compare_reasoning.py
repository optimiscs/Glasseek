import collections,json,statistics
from pathlib import Path
from tokenizers import Tokenizer
T=Tokenizer.from_file('/autodl-fs/data/models/Qwen3.5-9B/tokenizer.json')
old=Path('/root/autodl-tmp/Glasseek/Evaluation/data/runtime/hermes/sessions/glasseek_rl9b_quest128_32/quest_qwen27b_128');new=Path('/root/autodl-tmp/Glasseek/analysis/full64k-reeval/session-audit');both=sorted({p.name for p in old.iterdir() if p.is_dir()} & {p.name for p in new.iterdir() if p.is_dir()})
def get(p):
 paths=sorted((p/'main').glob('session_*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True);snap=[]
 for f in paths:
  try:snap.append(json.load(f.open()))
  except (ValueError,OSError):pass
 if not snap:return None
 byid={s.get('session_id'):s for s in snap};cur=snap[0];chain=[];seen=set()
 while cur and cur.get('session_id') not in seen:seen.add(cur.get('session_id'));chain.append(cur);cur=byid.get(cur.get('parent_session_id'))
 chain.reverse();msgs=[m for i,s in enumerate(chain) for m in s.get('messages',[]) if i==0 or not m.get('_inherited_from_previous_session')]; ass=[m for m in msgs if m.get('role')=='assistant'];lens=[];char=[]
 for m in ass:
  r=m.get('reasoning') or m.get('reasoning_content') or ''
  if r:char.append(len(r));lens.append(len(T.encode(r,add_special_tokens=False).ids))
 return {'assistants':len(ass),'reasoning_turns':len(lens),'total_tokens':sum(lens),'mean_tokens':statistics.mean(lens) if lens else None,'median_tokens':statistics.median(lens) if lens else None,'lengths':lens,'sessions':len(chain),'first_reasoning':next(((m.get('reasoning') or m.get('reasoning_content')) for m in ass if (m.get('reasoning') or m.get('reasoning_content'))),'')}
allr={};alllens={'old':[],'new':[]}
for q in both:
 a=get(old/q);b=get(new/q)
 if not a or not b:continue
 allr[q]={'old':{k:v for k,v in a.items() if k!='lengths'},'new':{k:v for k,v in b.items() if k!='lengths'}}
 alllens['old']+=a['lengths'];alllens['new']+=b['lengths']
for label in ('old','new'):
 lens=alllens[label];print(label,'cases',len(allr),'turns',len(lens),'total reasoning tokens',sum(lens),'median',statistics.median(lens),'mean',statistics.mean(lens),'p90',sorted(lens)[int(.9*(len(lens)-1))])
for q in ['row0114','row0001','row0031']:
 if q in allr:
  print(q,[(k,allr[q][k]['assistants'],allr[q][k]['reasoning_turns'],allr[q][k]['total_tokens'],allr[q][k]['median_tokens'],allr[q][k]['mean_tokens']) for k in ('old','new')])
Path('/root/autodl-tmp/Glasseek/analysis/full64k-reeval/reasoning-comparison.json').write_text(json.dumps(allr,ensure_ascii=False,indent=2))
