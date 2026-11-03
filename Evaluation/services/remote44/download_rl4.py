"""Resume an immutable HF revision through hf-mirror into one sparse file."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import random
import threading
import time
import requests

ROOT=Path('/home/moxu/glasseek-rl4-eval')
DEST=ROOT/'model'; DEST.mkdir(parents=True,exist_ok=True)
REV='c7fcb9a8bb478e75b46e04fa1d68c82287c047dd'
BASE=f'https://hf-mirror.com/Glasseek/Glasseeker-4B-RL/resolve/{REV}/'
MANIFEST=json.loads((ROOT/'metadata.json').read_text())
if MANIFEST.get('sha') != REV: raise RuntimeError('unexpected model revision')

def digest(path, lfs):
 h=hashlib.sha256() if lfs else hashlib.sha1()
 if not lfs: h.update(f'blob {path.stat().st_size}\0'.encode())
 with path.open('rb') as f:
  while block:=f.read(8*1024*1024): h.update(block)
 return h.hexdigest()

def expected(item):
 lfs=item.get('lfs') or {}
 return (lfs.get('sha256') or lfs.get('oid') or item.get('blobId') or item.get('oid') or '').removeprefix('sha256:')

def verified(path,item):
 lfs=bool(item.get('lfs')); hash_value=expected(item)
 return path.is_file() and path.stat().st_size==item['size'] and bool(hash_value) and digest(path,lfs)==hash_value

def download(item):
 name=item['rfilename'];size=item['size'];target=DEST/name
 target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  if verified(target,item):print('verified',name,flush=True);return
  raise RuntimeError(f'existing model file differs; preserved: {name}')
 temp=target.with_name(target.name+'.partial');progress=target.with_name(target.name+'.progress.json')
 chunk=64*1024*1024;count=(size+chunk-1)//chunk
 completed=set(json.loads(progress.read_text()).get('done',[])) if progress.exists() and temp.exists() else set()
 if not temp.exists():
  completed.clear()
  with temp.open('wb') as f:f.truncate(size)
 fd=os.open(temp,os.O_RDWR); lock=threading.Lock()
 def job(index):
  start=index*chunk;end=min(size,start+chunk)-1;length=end-start+1
  for attempt in range(20):
   try:
    response=requests.get(BASE+name,params={'download':'true','part':index},headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=(20,180))
    if response.status_code==429:
     delay=min(90,10*(attempt+1))+random.random()*3;print('rate limited',name,index,'sleep',round(delay,1),flush=True);time.sleep(delay);continue
    response.raise_for_status()
    partial_ok = response.status_code == 206 and response.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/')
    full_ok = response.status_code == 200 and start == 0 and end == size - 1
    if not (partial_ok or full_ok) or len(response.content)!=length:
     raise ValueError(f'invalid range response {response.status_code} {response.headers.get("Content-Range")} length={len(response.content)} expected={length}')
    data=memoryview(response.content);offset=start
    while data:
     n=os.pwrite(fd,data,offset)
     if n<=0:raise OSError('short write')
     offset+=n;data=data[n:]
    with lock:
     completed.add(index);tmp=progress.with_suffix('.tmp');tmp.write_text(json.dumps({'done':sorted(completed),'count':count}));os.replace(tmp,progress)
     print('progress',name,len(completed),'/',count,flush=True)
    return
   except (requests.RequestException,ValueError,OSError) as exc:
    print('retry',name,index,attempt+1,type(exc).__name__,str(exc)[:120],flush=True);time.sleep(min(60,3*(attempt+1))+random.random()*2)
  raise RuntimeError(f'failed chunk {index} for {name}')
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   futures=[pool.submit(job,i) for i in range(count) if i not in completed]
   for future in concurrent.futures.as_completed(futures):future.result()
 finally:os.close(fd)
 if not verified(temp,item):raise RuntimeError(f'hash mismatch for {name}; partial retained')
 os.replace(temp,target);progress.unlink(missing_ok=True);print('verified downloaded',name,flush=True)

for item in MANIFEST['siblings']:
 if item.get('size') is None:raise RuntimeError('metadata missing file size')
 download(item)
(ROOT/'download.success').write_text(REV+'\n')
print('all files verified',flush=True)
