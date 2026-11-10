"""Resumable bounded-range downloads of the pinned BC+ corpus via network_turbo."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

REPO='Tevatron/browsecomp-plus-corpus'
REVISION='b27b02bc3e45511b8b82a13e6f90ce761df726f6'
CHUNK=4*1024*1024


def acquire(destination,workers=24):
    import requests
    from huggingface_hub import HfApi
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    info=HfApi().dataset_info(REPO,revision=REVISION,files_metadata=True,timeout=60)
    files=[f for f in info.siblings if f.rfilename.endswith('.parquet')]
    jobs=[];urls={};lock=threading.Lock();done=0;thread_local=threading.local()
    def resolve(name):
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}'
        r=requests.get(url,headers={'Range':'bytes=0-0'},allow_redirects=False,timeout=(15,30))
        r.raise_for_status();return r.headers.get('Location',url)
    for file in files:
        path=destination/file.rfilename;parts=path.with_suffix('.parts');parts.mkdir(parents=True,exist_ok=True)
        expected=file.lfs.sha256
        if path.is_file() and sha(path)==expected:continue
        # Reuse verified-identity contiguous Hugging Face partials; final SHA256
        # checks every byte before any downloaded shard becomes visible.
        cached=list((destination/'.cache').rglob('*'+expected+'*.incomplete'))
        if cached:
            cached=max(cached,key=lambda p:p.stat().st_size)
            with cached.open('rb') as f:
                for offset in range(0,min(cached.stat().st_size,file.size)//CHUNK*CHUNK,CHUNK):
                    part=parts/str(offset)
                    if not part.exists():part.write_bytes(f.read(CHUNK))
                    else:f.seek(CHUNK,1)
        pending=[]
        for offset in range(0,file.size,CHUNK):
            size=min(CHUNK,file.size-offset);part=parts/str(offset)
            if not part.exists() or part.stat().st_size!=size:pending.append((file,offset,size,part))
        if pending:urls[file.rfilename]=resolve(file.rfilename);jobs.extend(pending)
    def download(job):
        nonlocal done
        file,offset,size,part=job
        if not hasattr(thread_local,'session'):thread_local.session=requests.Session()
        for attempt in range(10):
            try:
                temporary=part.with_suffix('.tmp');received=temporary.stat().st_size if temporary.exists() else 0
                if received==size:temporary.replace(part);return
                if received>size:temporary.unlink();received=0
                start=offset+received
                with thread_local.session.get(urls[file.rfilename],headers={'Range':f'bytes={start}-{offset+size-1}'},stream=True,timeout=(20,40)) as response:
                    if response.status_code==403:
                        with lock:urls[file.rfilename]=resolve(file.rfilename)
                    response.raise_for_status()
                    if response.headers.get('Content-Range')!=f'bytes {start}-{offset+size-1}/{file.size}':raise ValueError('server did not honor byte range')
                    with temporary.open('ab') as f:
                        for block in response.iter_content(256*1024):f.write(block)
                    if temporary.stat().st_size!=size:raise ValueError('incomplete range')
                    temporary.replace(part)
                with lock:
                    done+=1
                    if done%10==0:print(json.dumps({'stage':'download','ranges_completed':done,'ranges_needed':len(jobs)}),flush=True)
                return
            except Exception as exc:
                if attempt==9:raise RuntimeError(f'corpus range failed: {file.rfilename} offset={offset}, {type(exc).__name__}') from None
                time.sleep(min(10,attempt+1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(download,jobs):pass
    for file in files:
        path=destination/file.rfilename
        if path.is_file() and sha(path)==file.lfs.sha256:continue
        temporary=path.with_suffix('.assembling');parts=path.with_suffix('.parts')
        with temporary.open('wb') as output:
            for offset in range(0,file.size,CHUNK):
                with (parts/str(offset)).open('rb') as src:shutil.copyfileobj(src,output)
        if sha(temporary)!=file.lfs.sha256:raise ValueError('corpus shard SHA256 mismatch')
        temporary.replace(path);shutil.rmtree(parts)
        print(json.dumps({'verified_shard':file.rfilename,'bytes':file.size}),flush=True)
    (destination/'download.manifest.json').write_text(json.dumps({'repo':REPO,'revision':REVISION,
        'files':{f.rfilename:{'bytes':f.size,'sha256':f.lfs.sha256} for f in files}},indent=2))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);p.add_argument('--workers',type=int,default=24)
    a=p.parse_args();acquire(a.destination,a.workers)
