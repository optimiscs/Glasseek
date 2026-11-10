import asyncio, collections, hashlib, json, re, sqlite3, time, unicodedata
from pathlib import Path
import httpx
from glasseek_shared.screening import policy, normalize, parse_json
POLICY = policy('strong')
INFERENCE, JUDGE = POLICY.inference, POLICY.judge
normal = normalize
extract_final_answer = POLICY.teacher_answer
extract_json = parse_json
P=Path(__file__).resolve().parent




def answer_appears(answer, output):
 return normal(answer) in normal(output)

def setup():
 db=sqlite3.connect(P/'results.sqlite3');db.execute('PRAGMA journal_mode=WAL');db.executescript('CREATE TABLE IF NOT EXISTS answers(qid TEXT PRIMARY KEY,status TEXT,finish TEXT,reasoning TEXT,content TEXT,error TEXT,seconds REAL); CREATE TABLE IF NOT EXISTS reviews(rowid INTEGER PRIMARY KEY,qid TEXT,status TEXT,verdict TEXT,teacher_answer TEXT,teacher_evidence TEXT,student_evidence TEXT,reason TEXT,error TEXT);');db.commit();return db

async def main():
 groups=collections.defaultdict(list)
 for i,l in enumerate((P/'all_filtered.jsonl').open()):
  x=json.loads(l);m=x['messages'];q=next(y['content'] for y in m if y['role']=='user');qid=hashlib.sha256(q.encode()).hexdigest();teacher=next(y['content'] for y in reversed(m) if y['role']=='assistant');groups[qid].append({'id':i+1,'q':q,'teacher':teacher,'rounds':sum(y['role']=='assistant' for y in m)})
 db=setup();ans={r[0]:r[1:] for r in db.execute('select * from answers')};done={r[0] for r in db.execute('select rowid from reviews')};queue=asyncio.Queue()
 for qid,rows in groups.items():
  if POLICY.initially_pending(qid,rows,ans,done): queue.put_nowait((qid,rows))
 print(json.dumps({'event':'start','questions':len(groups),'rows':sum(map(len,groups.values())),'pending':queue.qsize()}),flush=True)
 client=httpx.AsyncClient(timeout=httpx.Timeout(900,connect=10),trust_env=False)
 async def request(gpu,payload):
  return await POLICY.request(client,gpu,payload)
 async def worker(gpu):
  while True:
   try:qid,rows=queue.get_nowait()
   except asyncio.QueueEmpty:return
   if qid not in ans:
    start=time.monotonic();status='error';finish='';reasoning='';content='';error=''
    try:
     payload={'model':'glasseek-base-9b','messages':[{'role':'system','content':INFERENCE},{'role':'user','content':rows[0]['q']}],'temperature':0,'max_tokens':8192,'stream':False,'chat_template_kwargs':{'enable_thinking':True}}
     c=await request(gpu,payload);msg=c['message'];reasoning=msg.get('reasoning_content') or msg.get('reasoning') or '';content=msg.get('content') or '';finish=c.get('finish_reason') or ''
     if msg.get('tool_calls') or re.search(r'<tool_call>|<function=|\b(?:web_search|web_extract)\s*\(',reasoning+'\n'+content):error='tool_call_or_simulation'
     elif not content.strip():error='missing_final_answer'
     else:status='ok'
    except Exception as e:error=type(e).__name__+': '+str(e)[:500]
    a=(status,finish,reasoning,content,error,time.monotonic()-start);ans[qid]=a;db.execute('insert or replace into answers values(?,?,?,?,?,?,?)',(qid,*a));db.commit()
   a=ans[qid];output=(a[2] or '')+'\n'+(a[3] or '')
   for row in rows:
    if row['id'] in done:continue
    status='error';verdict='exclude';ta=te=se=None;reason='';error=''
    if a[0]!='ok' or a[1]!='stop':error='student_error_or_incomplete: '+str(a[4] or a[1])
    elif (rule_answer:=extract_final_answer(row['teacher'])) is not None:
     ta=rule_answer;te=rule_answer;status='ok';verdict='exclude' if answer_appears(ta,output) else 'keep';reason='rule_final_answer_match' if verdict=='exclude' else 'rule_final_answer_no_match'
    else:
     try:
      data=json.dumps({'QUESTION':row['q'],'TEACHER_RESPONSE':row['teacher'],'STUDENT_OUTPUT':output},ensure_ascii=False)
      payload={'model':'glasseek-base-9b','messages':[{'role':'system','content':JUDGE},{'role':'user','content':data}],'temperature':0,'max_tokens':POLICY.judge_tokens,'stream':False,'chat_template_kwargs':{'enable_thinking':POLICY.judge_thinking}}
      c=await request(gpu,payload)
      if c.get('finish_reason')!='stop':raise ValueError('judge_incomplete')
      v=extract_json(c['message'].get('content') or '');ta=v.get('teacher_answer');te=v.get('teacher_evidence');se=v.get('student_evidence');reason=str(v.get('reason',''))[:1000];verdict=v.get('verdict')
      if verdict not in ('keep','exclude'):raise ValueError('bad_verdict')
      if not isinstance(ta,str) or not ta or not isinstance(te,str) or not te or te not in row['teacher'] or ta not in row['teacher']:verdict='exclude';reason='missing_or_invalid_teacher_evidence; '+reason
      if isinstance(ta,str) and ta and ta.casefold() in output.casefold():verdict='exclude';reason='literal_match; '+reason
      if se is not None and (not isinstance(se,str) or se not in output):verdict='exclude';reason='invalid_student_evidence; '+reason
      reason='local_judge; '+reason;status='ok'
     except Exception as e:error=type(e).__name__+': '+str(e)[:500];verdict='exclude'
    db.execute('insert or replace into reviews values(?,?,?,?,?,?,?,?,?)',(row['id'],qid,status,verdict,ta,te,se,reason,error));db.commit();done.add(row['id'])
   print(json.dumps({'event':'done','qid':qid,'rows':len(rows),'student_status':a[0],'finish':a[1],'pending':queue.qsize()},ensure_ascii=False),flush=True);queue.task_done()
 try:
  await asyncio.gather(*(worker(g) for g in range(4) for _ in range(4)))
  print(json.dumps({'event':'reaudit_complete','reviews':len(done)}),flush=True)
  for qid,rows in groups.items():
   if qid not in ans: queue.put_nowait((qid,rows))
  print(json.dumps({'event':'remaining_start','pending':queue.qsize()}),flush=True)
  await asyncio.gather(*(worker(g) for g in range(4) for _ in range(4)))
 finally:await client.aclose();db.close()
if __name__=='__main__':asyncio.run(main())
