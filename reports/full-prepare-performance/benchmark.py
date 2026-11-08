import json,sys,time,hashlib
from pathlib import Path
ROOT=Path('/root/autodl-tmp/Glasseek')
sys.path.insert(0,str(ROOT/'scripts/trajectory_preprocess'))
import full_prepare as fp

def main():
 out=ROOT/'reports/full-prepare-performance'
 manifest=json.loads((ROOT/'data/processed/trajectory-full-offline-v2/manifest.json').read_text())
 specs=[s for s in manifest['inputs'] if s['source']=='OpenSeeker']
 results=[]; baseline=None
 for workers in [1,4,8,16]:
  dest=out/f'bench600-w{workers}'
  start=time.perf_counter()
  result=fp.run(specs,manifest['config'],dest,workers=workers,stop_after=600)
  seconds=time.perf_counter()-start
  hashes={name:hashlib.sha256((dest/(name+'.jsonl')).read_bytes()).hexdigest() for name in fp.OUTPUTS}
  if baseline is None: baseline=hashes
  record={'workers':workers,'records':600,'wall_seconds':seconds,'rows_per_second':600/seconds,
          'byte_identical':hashes==baseline,'counts':result['counts']}
  assert record['byte_identical'],record
  results.append(record)
  (out/'benchmark.json').write_text(json.dumps(results,indent=2))
  print('BENCHMARK',json.dumps(record),flush=True)
if __name__=='__main__':main()
