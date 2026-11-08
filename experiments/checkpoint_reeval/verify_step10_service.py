"""Fail closed unless step10 service agrees with the original-environment reference."""
import argparse, json, urllib.request
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--port', type=int, required=True); p.add_argument('--reference',type=Path,default=Path(__file__).with_name('step10_reference.json')); p.add_argument('--report',type=Path); args=p.parse_args()
r=json.loads(args.reference.read_text())
def request(model):
    data=json.dumps({'model':model,'prompt':r['input_ids'],'max_tokens':1,'temperature':0,'logprobs':20}).encode()
    req=urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/completions',data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=90) as response: body=json.load(response)
    if body.get('model')!=model: raise RuntimeError(f'model mismatch: {body.get("model")} != {model}')
    return body['choices'][0]['logprobs']['top_logprobs'][0]
b=request('glasseek-base-9b'); l=request('step10')
tokens=r['top_token_text']; observed=[x for x in tokens[:10] if x in b and x in l]
if len(observed)<8: raise RuntimeError(f'insufficient top-token overlap: {len(observed)}')
rows=[]
for word in observed:
    i=tokens.index(word)
    rows.append({'token':word,'reference_base':r['base_logprobs'][i], 'reference_step10':r['lora_logprobs'][i], 'vllm_base':b[word], 'vllm_step10':l[word]})
max_err=max(max(abs(x['reference_base']-x['vllm_base']),abs(x['reference_step10']-x['vllm_step10'])) for x in rows)
if max_err > 0.20: raise RuntimeError(f'logprob difference {max_err:.4f} exceeds 0.20')
if tokens[0] != max(l,key=l.get): raise RuntimeError('top token differs from original reference')
effect=max(abs(l[x]-b[x]) for x in observed)
if effect < 0.005: raise RuntimeError(f'LoRA effect not detectable: {effect:.5f}')
report={'port':args.port,'shared_top_tokens':len(rows),'max_abs_logprob_difference':max_err,'max_vllm_lora_logprob_delta':effect,'reference_max_logit_delta':r['max_abs_logit_delta'],'comparison':rows}
if args.report: args.report.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
