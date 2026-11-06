import asyncio,concurrent.futures,json,os,statistics,sys,time,urllib.request
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parents[1]
os.environ.update(HERMES_HOME=str(D/'hermes'),GLASSEEK_DOH_PROXY='http://127.0.0.1:18200',GLASSEEK_FETCH_DOH_SAFETY='1',FETCH_SERVER_BASE_URL='http://127.0.0.1:18081',SEARCH_SERVER_ENDPOINT='http://127.0.0.1:18083/search',SEARCH_SERVER_API_KEY='rotation',SEARCH_SERVER_MAX_ATTEMPTS='2',GLASSEEK_SEARCH_TOP10='1',GLASSEEK_QUEST_PAGE_RETRIEVAL='1',GLASSEEK_QUEST_RETRIEVAL_URL='http://127.0.0.1:18217',GLASSEEK_QUEST_RETRIEVAL_TIMEOUT='60',BROWSE_COMP_QUESTION='What verified facts does this page provide about its main subject, history, dates and people?')
sys.path.insert(0,str(P/'Evaluation_backend'))
from plugins.web.search_server.provider import _search_server_response
from tools.url_safety import check_external_fetch_url
from tools.web_tools import web_extract_tool
queries=['OpenAI official website','Jamie Uys The Gods Must Be Crazy','Trefor Jenkins Benjamin Pogrund Medal','Singh Is Bliing Lara Dutta translator review','Kid Trunks real name Minh Nguyen','"Da Silva" Belvedere academy president federation re-elected researcher education','New Castle Court House Museum history','Art Gallery New South Wales bequest 1951','Belvedere academy president education','Domenico Modugno biography','University of Oxford history','NASA Mars mission']
urls=['https://example.com','https://en.wikipedia.org/wiki/The_Gods_Must_Be_Crazy','https://en.wikipedia.org/wiki/Jamie_Uys','https://en.wikipedia.org/wiki/Singh_Is_Bliing','https://en.wikipedia.org/wiki/Trefor_Jenkins','https://www.nature.com','https://history.delaware.gov/new-castle-court-house-museum/','https://www.nasa.gov']
report={'started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'search':[],'fetch':[],'dns':[]}
def save(): (D/'benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
def search(q):
 t=time.monotonic();r=_search_server_response(q);return {'query':q,'success':r.get('success') is True,'results':len(r.get('data',{}).get('web',[])),'error':r.get('error'),'error_code':r.get('error_code'),'seconds':time.monotonic()-t}
if '--fetch-only' in sys.argv:
 report=json.loads((D/'benchmark.json').read_text());report['fetch']=[];report['dns']=[]
for concurrency in ([] if '--fetch-only' in sys.argv else [1,4,8]):
 start=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:rows=list(pool.map(search,queries*2))
 entry={'concurrency':concurrency,'requests':len(rows),'success':sum(x['success'] for x in rows),'zero_results':sum(x['success'] and x['results']==0 for x in rows),'mean_seconds':statistics.mean(x['seconds'] for x in rows),'elapsed_seconds':time.monotonic()-start,'rows':rows};report['search'].append(entry);save();print('SEARCH',json.dumps({k:v for k,v in entry.items() if k!='rows'}),flush=True)
 if entry['success']<len(rows):break
for u in urls+['http://127.0.0.1','http://169.254.169.254']:
 t=time.monotonic();allowed,code,error=check_external_fetch_url(u);report['dns'].append({'url':u,'allowed':allowed,'error_code':code,'error':error,'seconds':time.monotonic()-t})
save();print('DNS',json.dumps(report['dns']),flush=True)
async def batch(concurrency):
 sem=asyncio.Semaphore(concurrency)
 async def fetch(u):
  async with sem:
   t=time.monotonic()
   try:
    raw=await asyncio.wait_for(web_extract_tool([u],use_llm_processing=False),timeout=120);r=json.loads(raw);pages=r.get('results',[]);valid=[p for p in pages if not p.get('error') and p.get('content')];return {'url':u,'success':bool(valid),'errors':[p.get('error') for p in pages if p.get('error')]+([r['error']] if r.get('error') else []),'content_chars':sum(len(p.get('content','')) for p in valid),'seconds':time.monotonic()-t}
   except Exception as e:return {'url':u,'success':False,'error':type(e).__name__,'seconds':time.monotonic()-t}
 return await asyncio.gather(*(fetch(u) for u in urls))
for concurrency in [1,4]:
 rows=asyncio.run(batch(concurrency));entry={'concurrency':concurrency,'requests':len(rows),'success':sum(x['success'] for x in rows),'rows':rows};report['fetch'].append(entry);save();print('FETCH',json.dumps(entry),flush=True)
report['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z');save()
