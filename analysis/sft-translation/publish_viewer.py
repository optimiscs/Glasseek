"""Publish completed bilingual SFT trajectories into Quest viewer workspaces."""
import json,os,sys,tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT=Path('/root/autodl-tmp/Glasseek/analysis/sft-translation')
SOURCE=ROOT/'translated'
SITE=Path('/root/autodl-tmp/Quest/web/trace-viewer')
INDEX=SITE/'datasets/index.json'

def atomic(path, data):
 path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.sft-zh-')
 try:
  with os.fdopen(fd,'w',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,separators=(',',':'))
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)

def make_case(row):
 n=row['source_row']; msgs=row['translated']['messages']; original_msgs=row['original']['messages']
 if len(msgs)!=len(original_msgs):raise ValueError('message count mismatch')
 def content(m):return str(m.get('content') or '')
 prompt=next((content(x) for x in msgs if x.get('role')=='user' and not content(x).lstrip().startswith('<tool_response>')),'')
 answer=next((content(x) for x in reversed(msgs) if x.get('role')=='assistant'),'')
 steps=[]
 for i,m in enumerate(msgs,1):
  role=m.get('role','unknown');txt=content(m)
  label={'system':'系统提示','user':'用户或工具返回','assistant':'助手'}.get(role,role)
  steps.append({'n':i,'kind':'answer','label':label,'thought':f'【{label} · 原始消息 {i}】\n{txt}','thoughtOriginal':f'【{label} · 原始消息 {i}】\n{content(original_msgs[i-1])}','sourceRole':role,'model_seconds':None,'tool_seconds':None,'modelStart':None,'modelEnd':None,'toolStart':None,'toolEnd':None,'tool':None})
 original_prompt=next((content(x) for x in original_msgs if x.get('role')=='user' and not content(x).lstrip().startswith('<tool_response>')),'')
 original_answer=next((content(x) for x in reversed(original_msgs) if x.get('role')=='assistant'),'')
 return {'questionOriginal':original_prompt,'answerOriginal':original_answer,'finalAnswerOriginal':original_answer,'id':f'sft-{n:05d}','index':n,'field':'SFT','fieldZh':'SFT 训练轨迹','topic':f'原始第 {n} 条','topicZh':f'原始第 {n} 条','question':prompt,'answer':answer,'answerNote':'离线机器翻译，未人工校对；原文保存在双语结果文件。','judge':'训练数据，未评分','correct':None,'evaluationStatus':'training_sample','inProgress':False,'rounds':sum(x.get('role')=='assistant' for x in msgs),'durationSeconds':0,'durationText':'未记录','model':'SFT 示例 · Hy-MT2-1.8B 中文译文','risk':{'level':'unknown','label':'未审计'},'riskKey':'not_audited','stats':{'steps':len(steps),'searchCalls':0,'readCalls':0,'computeCalls':0,'searchQueries':0,'resultsScanned':0,'pagesRead':0,'uniqueSources':0,'uniqueDomains':0,'averageStepSeconds':None,'longestStepSeconds':None},'timing':{},'timingAvailable':False,'domains':[],'steps':steps,'finalAnswer':answer,'keyIndices':[],'sources':[]}

groups=defaultdict(list)
for f in sorted(SOURCE.glob('trace-*.json')):
 try:
  n=int(f.stem.split('-')[1]);groups[(n-1)//100].append(f)
 except ValueError:continue
index=json.loads(INDEX.read_text());existing={x['id'] for x in index['datasets']};new=changed=0
for g,files in sorted(groups.items()):
 lo=g*100+1;hi=min(lo+99,5762);name=f'glasseek-sft-zh-{lo:04d}-{hi:04d}'
 dest=SITE/'datasets'/name/'traces.json'
 if dest.exists() and dest.stat().st_mtime_ns>=max(f.stat().st_mtime_ns for f in files):continue
 cases=[]
 for f in files:
  try:
   x=json.loads(f.read_text());
   if x.get('status')=='machine_translated_unreviewed':cases.append(make_case(x))
  except (ValueError,OSError,KeyError):continue
 if not cases:continue
 now=datetime.now().astimezone().isoformat()
 run={'name':f'Glasseek SFT 中文 · 原始 {lo}–{hi}','model':'离线 Hy-MT2-1.8B 机器翻译','benchmark':'Glasseek-SFT-5.7K','date':now[:10],'dateShort':now[:10],'totalCases':len(cases),'requestedCases':hi-lo+1,'judgePassed':0,'judgeScored':0,'rounds':sum(x['rounds'] for x in cases),'toolCalls':0,'pagesRead':0,'searchQueries':0,'searchCalls':0,'inProgress':False,'wiki':'中文为离线机器翻译，未人工校对；系统标记、代码块和 URL 保留原文。'}
 atomic(dest,{'generatedAt':now,'run':run,'cases':cases});changed+=1
 if name not in existing:
  index['datasets'].append({'id':name,'label':f'Glasseek SFT 中文 {lo}–{hi}','description':'完整原始消息的离线机器翻译；已发布轨迹随翻译进度增加','folder':f'datasets/{name}','data':f'datasets/{name}/traces.json','benchmark':'Glasseek-SFT-5.7K','visible':True});existing.add(name);new+=1
if new:atomic(INDEX,index)
print(json.dumps({'translated_files':sum(map(len,groups.values())),'updated_workspaces':changed,'new_workspaces':new}),flush=True)
