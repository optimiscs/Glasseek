import json,time,re,urllib.request,sys,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;stage=sys.argv[1];duration=int(sys.argv[2]) if len(sys.argv)>2 else 60
pattern=re.compile(r'^(vllm:[^ {]+)(?:\{(.*?)\})?\s+(\S+)$')
def sample():
 out={'time':time.time(),'engines':{}}
 for p in (19139,19239):
  metrics=urllib.request.urlopen(f'http://127.0.0.1:{p}/metrics',timeout=10).read().decode();v={}
  for l in metrics.splitlines():
   m=pattern.match(l)
   if not m:continue
   k,labels,value=m.groups()
   if k in ('vllm:prompt_tokens_total','vllm:generation_tokens_total','vllm:num_requests_running','vllm:num_requests_waiting','vllm:num_preemptions_total','vllm:kv_cache_usage_perc','vllm:request_success_total') or k=='vllm:prompt_tokens_by_source_total' and 'source="local_compute"' in labels:v[k]=v.get(k,0)+float(value)
  out['engines'][str(p)]=v
 out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
 return out
rows=[sample()];end=time.monotonic()+duration
while time.monotonic()<end:time.sleep(min(5,max(0,end-time.monotonic())));rows.append(sample())
(D/f'{stage}.json').write_text(json.dumps(rows,indent=2))
result={'seconds':rows[-1]['time']-rows[0]['time'],'engines':{}}
for p in rows[0]['engines']:
 a,b=rows[0]['engines'][p],rows[-1]['engines'][p];result['engines'][p]={k:(b.get(k,0)-a.get(k,0))/result['seconds'] for k in ['vllm:prompt_tokens_total','vllm:prompt_tokens_by_source_total','vllm:generation_tokens_total']};result['engines'][p]['preemptions']=b.get('vllm:num_preemptions_total',0)-a.get('vllm:num_preemptions_total',0)
(D/f'{stage}-summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
