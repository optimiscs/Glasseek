"""Benchmark intake worker counts with byte-identical output verification."""
import argparse,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/trajectory_preprocess'))
import full_prepare as fp

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--baseline',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--source',required=True)
 parser.add_argument('--rows',type=int,default=600)
 parser.add_argument('--workers',type=int,nargs='+',default=[1,4,8,16])
 args=parser.parse_args()
 if args.rows<1 or any(w<1 for w in args.workers):parser.error('rows/workers must be positive')
 out=args.output;out.mkdir(parents=True,exist_ok=False)
 manifest=json.loads((args.baseline/'manifest.json').read_text())
 specs=[s for s in manifest['inputs'] if s['source']==args.source]
 if not specs:parser.error('source not found in baseline')
 results=[]; baseline=None
 for workers in args.workers:
  dest=out/f'bench-w{workers}'
  start=time.perf_counter()
  result=fp.run(specs,manifest['config'],dest,workers=workers,stop_after=args.rows)
  seconds=time.perf_counter()-start
  hashes={name:hashlib.sha256((dest/(name+'.jsonl')).read_bytes()).hexdigest() for name in fp.OUTPUTS}
  if baseline is None: baseline=hashes
  record={'workers':workers,'records':result['counts']['scanned'],'wall_seconds':seconds,'rows_per_second':result['counts']['scanned']/seconds,
          'byte_identical':hashes==baseline,'counts':result['counts']}
  assert record['byte_identical'],record
  results.append(record)
  (out/'benchmark.json').write_text(json.dumps(results,indent=2))
  print('BENCHMARK',json.dumps(record),flush=True)
if __name__=='__main__':main()
