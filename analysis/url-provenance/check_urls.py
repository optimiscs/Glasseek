import asyncio,collections,json,random,socket,ipaddress,re
from urllib.parse import urlsplit,urljoin
from pathlib import Path
import httpx
P=Path('/root/autodl-tmp/Glasseek/analysis/url-provenance');rng=random.Random(20261105)
FILES=['gaia-quest4b-32-20261027-r2','glasseek-sft9b-browsecomp-128','glasseek-rl9b-browsecomp-128-32']
async def safe(url):
 try:
  p=urlsplit(url)
  if p.scheme not in ('http','https') or not p.hostname or p.username or p.password:return False
  if re.search(r'(?i)(?:api_?key|token|signature|password|credential)=',p.query):return False
  infos=await asyncio.wait_for(asyncio.get_running_loop().getaddrinfo(p.hostname,p.port or (443 if p.scheme=='https' else 80),type=socket.SOCK_STREAM),timeout=5)
  return bool(infos) and all(ipaddress.ip_address(x[4][0]).is_global for x in infos)
 except Exception:return False
async def main():
 targets=[]
 for name in FILES:
  rows=[json.loads(x) for x in (P/(name+'.jsonl')).open()];seen=set();unique=[]
  for r in rows:
   k=(r['observed_before'],r['normalized'])
   if k not in seen:unique.append(r);seen.add(k)
  for observed in (True,False):
   group=[r for r in unique if r['observed_before']==observed]
   targets.extend(rng.sample(group,min(30,len(group))))
 sem=asyncio.Semaphore(30);client=httpx.AsyncClient(timeout=httpx.Timeout(8,connect=5),follow_redirects=False,trust_env=True,headers={'User-Agent':'Mozilla/5.0 (compatible; research evaluation)'})
 async def one(r):
  async with sem:
   url=r['url'];result={'dataset':r['dataset'],'case':r['case'],'url':url,'observed_before':r['observed_before']}
   for _ in range(5):
    if not await safe(url):result['outcome']='dns_or_address_unavailable';break
    try:
     async with client.stream('GET',url,headers={'Range':'bytes=0-2047'}) as resp:
      status=resp.status_code;result['status']=status
      if status in (301,302,303,307,308):
       loc=resp.headers.get('location')
       if not loc:result['outcome']='redirect_without_location';break
       url=urljoin(url,loc);continue
      result['outcome']='http_2xx' if 200<=status<300 else 'not_found_404_410' if status in (404,410) else 'blocked_401_403_429' if status in (401,403,429) else 'other_http';break
    except Exception as e:result['outcome']='network_error';result['error']=type(e).__name__;break
   else:result['outcome']='redirect_limit'
   return result
 out=await asyncio.gather(*(one(r) for r in targets));await client.aclose()
 with (P/'access_sample.jsonl').open('w') as f:
  for x in out:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 c=collections.Counter((r['dataset'],r['observed_before'],r['outcome']) for r in out)
 for k,v in sorted(c.items()):print(json.dumps({'dataset':k[0],'observed_before':k[1],'outcome':k[2],'count':v},ensure_ascii=False))
asyncio.run(main())
