import collections,json,re,time
from pathlib import Path
from tokenizers import Tokenizer
P=Path('/root/autodl-tmp/Glasseek/experiments/released-model-inference-url');D=Path('/root/autodl-tmp/Quest/web/trace-viewer/datasets');T=Tokenizer.from_file('/autodl-fs/data/models/Qwen3.5-9B/tokenizer.json');U=re.compile(r'https?://[^\s<>"\'\[\]{}\\]+',re.I)

def count(text):
 if not isinstance(text,str) or not text:return (0,0)
 e=T.encode(text,add_special_tokens=False);offs=e.offsets;sp=[]
 for m in U.finditer(text):
  a,b=m.span()
  while b>a and text[b-1] in '.,;:!?':b-=1
  if b>a:sp.append((a,b))
 n=0;j=0
 for a,b in offs:
  while j<len(sp) and sp[j][1]<=a:j+=1
  if j<len(sp) and sp[j][0]<b and sp[j][1]>a:n+=1
 return len(offs),n

def main():
 cfg=[('QUEST-4B / GAIA 32','gaia-quest4b-32-20261027-r2'),('Glasseek SFT-9B / BrowseComp 128','glasseek-sft9b-browsecomp-128'),('Glasseek RL-9B / BrowseComp 128','glasseek-rl9b-browsecomp-128-32')]
 out={};start=time.monotonic()
 for label,d in cfg:
  data=json.load((D/d/'traces.json').open());rr=[]
  for case in data['cases']:
   sums=collections.Counter();thoughts=[];compact=0
   for step in case.get('steps',[]):
    s=step.get('thought') or ''
    if '[CONTEXT COMPACTION' in s:compact+=1
    else:
     t,u=count(s);sums['model_tokens']+=t;sums['model_url']+=u;thoughts.append(s)
    raw=step.get('rawToolOutput') or (step.get('tool') or {}).get('rawOutput') or ''
    if not isinstance(raw,str):raw=json.dumps(raw,ensure_ascii=False)
    t,u=count(raw);sums['tool_tokens']+=t;sums['tool_url']+=u
   final=case.get('finalAnswer') or ''
   if final and '[CONTEXT COMPACTION' not in final and not any(final in t for t in thoughts):
    t,u=count(final);sums['model_tokens']+=t;sums['model_url']+=u;sums['final_added']=1
   t,u=count(case.get('question') or '');sums['question_tokens']+=t;sums['question_url']+=u
   sums['compaction_excluded_steps']=compact;sums['step_count']=len(case.get('steps',[]));sums['case_id']=case.get('id');sums['has_final']=bool(final);sums['correct']=case.get('correct');rr.append(dict(sums))
   if len(rr)%25==0:print(json.dumps({'dataset':label,'cases':len(rr),'seconds':round(time.monotonic()-start)}),flush=True)
  out[label]={'id':d,'run':data.get('run'),'cases':rr};(P/'partial.json').write_text(json.dumps(out,ensure_ascii=False))
 (P/'results.json').write_text(json.dumps(out,ensure_ascii=False));print('complete',flush=True)
if __name__=='__main__':main()
