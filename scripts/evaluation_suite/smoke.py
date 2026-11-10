"""Fixed-count, stratified child releases without changing the frozen parent."""
import json
import shutil
from pathlib import Path
from suite import dump,jsonl,read,select,sha,verify


def prepare(parent,count,seed=20261108):
    parent=Path(parent).resolve();target=parent.parent/f'smoke-{count}-seed{seed}'
    if target.exists():
        m=json.loads((target/'manifest.json').read_text())
        if m.get('parent_manifest_sha256')!=sha(parent/'manifest.json'):raise ValueError('smoke parent changed')
        verify(target);return target
    members=list(read(parent/'private/membership.jsonl'));refs=list(read(parent/'private/references.jsonl'))
    summary=json.loads((parent/'summary.json').read_text());picked=[];outsummary=[]
    for row in summary:
        name=row['benchmark'];pool=[dict(r,id=r['id']) for r in members if r['benchmark']==name]
        chosen,quota=select(pool,count,seed,'smoke:'+name);picked.extend(chosen)
        outsummary.append(dict(row,selected=count,parent_selected=row['selected'],target=count,sampling='fixed-count smoke from frozen release',stratum_quota=quota))
    ids={r['opaque_id'] for r in picked};target.mkdir();(target/'private').mkdir(mode=0o700)
    for row in summary:
        name=row['benchmark'];jsonl(target/'inference'/(name+'.jsonl'),[r for r in read(parent/'inference'/(name+'.jsonl')) if r['id'] in ids])
    jsonl(target/'private/references.jsonl',[r for r in refs if r['opaque_id'] in ids]);jsonl(target/'private/membership.jsonl',picked)
    shutil.copytree(parent/'prompts',target/'prompts');shutil.copyfile(parent/'assets.lock.json',target/'assets.lock.json')
    dump(target/'summary.json',outsummary)
    dump(target/'manifest.json',{'version':'fixed-count-smoke-v1','seed':seed,'per_benchmark':count,
         'parent':str(parent),'parent_manifest_sha256':sha(parent/'manifest.json'),
         'files':{str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*')) if p.is_file()}})
    verify(target);return target
