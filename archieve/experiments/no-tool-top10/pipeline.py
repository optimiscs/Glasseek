import asyncio,collections,hashlib,json,re,sqlite3,time,unicodedata,os
from pathlib import Path
import httpx
from glasseek_shared.screening import policy, normalize, parse_json
POLICY = policy('top10')
INFERENCE, JUDGE = POLICY.inference, POLICY.judge
norm = normalize
teacher_answer = POLICY.teacher_answer
parse_candidates = POLICY.parse_candidates
P=Path(__file__).resolve().parent


def setup():
 db=sqlite3.connect(P/'results.sqlite3');db.execute('PRAGMA journal_mode=WAL');db.executescript('CREATE TABLE IF NOT EXISTS answers(qid TEXT PRIMARY KEY,status TEXT,finish TEXT,reasoning TEXT,content TEXT,candidates TEXT,error TEXT,seconds REAL); CREATE TABLE IF NOT EXISTS reviews(rowid INTEGER PRIMARY KEY,qid TEXT,status TEXT,verdict TEXT,teacher_answer TEXT,method TEXT,matches TEXT,reasoning_match TEXT,reason TEXT,error TEXT);');db.commit();return db
async def main():
 groups=collections.defaultdict(list)
 for i,l in enumerate((P/'all_filtered.jsonl').open(),1):
  m=json.loads(l)['messages'];q=next(y['content'] for y in m if y['role']=='user');qid=hashlib.sha256(q.encode()).hexdigest();teacher=next(y['content'] for y in reversed(m) if y['role']=='assistant');groups[qid].append({'id':i,'q':q,'teacher':teacher})
 db=setup();ans={r[0]:r[1:] for r in db.execute('select * from answers')};done={r[0] for r in db.execute('select rowid from reviews')};queue=asyncio.Queue()
 for qid,rows in groups.items():
  if POLICY.initially_pending(qid,rows,ans,done):queue.put_nowait((qid,rows))
 print(json.dumps({'event':'start','questions':len(groups),'rows':sum(map(len,groups.values())),'pending':queue.qsize()}),flush=True)
 client=httpx.AsyncClient(timeout=httpx.Timeout(900,connect=10),trust_env=False)
 async def request(gpu,payload):
  return await POLICY.request(client,gpu,payload)
 async def worker(gpu):
  while True:
   try:qid,rows=queue.get_nowait()
   except asyncio.QueueEmpty:return
   if qid not in ans:
    start=time.monotonic();status='error';finish='';reasoning='';content='';cs=[];error=''
    try:
     p={'model':'glasseek-base-9b','messages':[{'role':'system','content':INFERENCE},{'role':'user','content':rows[0]['q']}],'temperature':0,'max_tokens':8192,'stream':False,'chat_template_kwargs':{'enable_thinking':True}}
     ch=await request(gpu,p);msg=ch['message'];reasoning=msg.get('reasoning_content') or msg.get('reasoning') or '';content=msg.get('content') or '';finish=ch.get('finish_reason') or ''
     if finish!='stop':raise ValueError('incomplete_'+finish)
     if msg.get('tool_calls') or re.search(r'<tool_call>|<function=|\b(?:web_search|web_extract)\s*\(',reasoning+'\n'+content):raise ValueError('tool_call_or_simulation')
     cs=parse_candidates(content);status='ok'
    except Exception as e:error=type(e).__name__+': '+str(e)[:400]
    a=(status,finish,reasoning,content,json.dumps(cs,ensure_ascii=False),error,time.monotonic()-start);ans[qid]=a;db.execute('insert or replace into answers values(?,?,?,?,?,?,?,?)',(qid,*a));db.commit()
   a=ans[qid];cs=json.loads(a[4] or '[]')
   for row in rows:
    if row['id'] in done:continue
    status='error';verdict='exclude';ta=None;method='';matches=[];rm={};reason='';error=''
    if a[0]!='ok':error='student_error: '+a[5];method='student_error'
    else:
     ta=teacher_answer(row['teacher']); method='exact'
     if ta:
      for candidate in cs:
       for field in [candidate['answer'],*candidate['aliases']]:
        if norm(field)==norm(ta):matches.append({'rank':candidate['rank'],'answer':candidate['answer'],'matched_field':field,'match_type':'exact'});break
      if norm(ta) in norm(a[2]):rm={'found':True,'evidence':ta}
     literal_reasoning_match=bool(rm)
     if matches:
      status='ok';verdict='exclude';reason='exact_candidate_match'
     else:
      method='local_judge'
      try:
       data=json.dumps({'QUESTION':row['q'],'TEACHER_RESPONSE':row['teacher'],'STUDENT_REASONING':a[2],'STUDENT_CANDIDATES':cs},ensure_ascii=False)
       p={'model':'glasseek-base-9b','messages':[{'role':'system','content':JUDGE},{'role':'user','content':data}],'temperature':0,'max_tokens':POLICY.judge_tokens,'stream':False,'chat_template_kwargs':{'enable_thinking':POLICY.judge_thinking}}
       ch=await request(gpu,p)
       if ch.get('finish_reason')!='stop':raise ValueError('judge_incomplete')
       v=parse_json(ch['message'].get('content') or '');jt=v.get('teacher_answer');te=v.get('teacher_evidence');verdict=v.get('verdict');matches=v.get('matched_candidates');rm=v.get('reasoning_mention');reason=str(v.get('reason',''))[:1000]
       if verdict not in ('keep','exclude') or not isinstance(matches,list) or not isinstance(rm,dict) or type(rm.get('found')) is not bool:raise ValueError('judge_schema')
       if not isinstance(jt,str) or not jt.strip() or not isinstance(te,str) or not te or te not in row['teacher'] or jt not in row['teacher']:verdict='exclude';reason='invalid_teacher_evidence; '+reason
       else:ta=jt
       ranks={c['rank']:c for c in cs}
       for item in matches:
        if not isinstance(item,dict) or item.get('rank') not in ranks or item.get('answer')!=ranks[item['rank']]['answer'] or item.get('match_type') not in ('exact','equivalent','possible') or not isinstance(item.get('evidence'),str) or item['evidence'] not in json.dumps(cs,ensure_ascii=False):raise ValueError('invalid_candidate_evidence')
       if rm.get('found') and (not isinstance(rm.get('evidence'),str) or not rm['evidence'] or rm['evidence'] not in a[2]):raise ValueError('invalid_reasoning_evidence')
       if matches or rm.get('found') or literal_reasoning_match:verdict='exclude'
       if literal_reasoning_match:rm={'found':True,'evidence':ta,'method':'literal_normalized'}
       status='ok'
      except Exception as e:error=type(e).__name__+': '+str(e)[:500];verdict='exclude';matches=[];rm={}
    db.execute('insert or replace into reviews values(?,?,?,?,?,?,?,?,?,?)',(row['id'],qid,status,verdict,ta,method,json.dumps(matches,ensure_ascii=False),json.dumps(rm,ensure_ascii=False),reason,error));db.commit();done.add(row['id'])
   print(json.dumps({'event':'done','qid':qid,'rows':len(rows),'student_status':a[0],'finish':a[1],'pending':queue.qsize()},ensure_ascii=False),flush=True);queue.task_done()
 try:await asyncio.gather(*(worker(g) for g in range(4) for _ in range(int(os.getenv("PER_GPU", "4")))))
 finally:await client.aclose();db.close()
if __name__=='__main__':asyncio.run(main())
