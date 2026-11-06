import ast
import hashlib
import json
from pathlib import Path
import shutil
import struct

P=Path('/root/autodl-tmp/Glasseek')
D=Path(__file__).resolve().parent
shutil.copyfile(P/'analysis/clean2899-full64k/browsecomp128.jsonl',D/'questions.jsonl')
shutil.copyfile(P/'data/glasseek/sft-eval-fixed-128k/eval_system_body.txt',D/'system_prompt.txt')
qs=list(map(json.loads,(D/'questions.jsonl').read_text().splitlines()))
assert len(qs)==len({q['id'] for q in qs})==128
files=[D/'questions.jsonl',D/'system_prompt.txt',
       P/'Evaluation/browse_comp_eval/hermes_client.py',P/'Evaluation/browse_comp_eval/resume_checkpoint.py',
       P/'Evaluation/config/answer_prompts/browsecomp.txt',P/'Evaluation_backend/agent/glasseek_tool_budget.py',
       P/'Evaluation_backend/agent/conversation_loop.py',P/'Evaluation_backend/agent/chat_completion_helpers.py',
       P/'Evaluation_backend/run_agent.py']
models={}
for label,path in [('step39',P/'analysis/clean2899-full64k/step39-hf'),
                   ('step76',P/'analysis/full64k-reeval/step76-hf'),
                   ('sft9b',P/'Evaluation/data/runtime/models/Glasseeker-9B-SFT')]:
    index=path/'model.safetensors.index.json'
    idx=json.loads(index.read_text())['weight_map'] if index.exists() else {}
    shard_names=sorted(set(idx.values())) if idx else ['model.safetensors']
    shards={}
    for name in shard_names:
        f=path/name
        with f.open('rb') as stream:
            size=struct.unpack('<Q',stream.read(8))[0]
            raw=stream.read(size)
            header=json.loads(raw)
        assert all(k in header for k,v in idx.items() if v==name)
        assert f.stat().st_size==8+size+max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
        shards[name]={'bytes':f.stat().st_size,'header_sha256':hashlib.sha256(raw).hexdigest()}
    models[label]={'path':str(path),'shards':shards,
                   'config_sha256':hashlib.sha256((path/'config.json').read_bytes()).hexdigest()}
manifest={'models':models,'order':['step39','step76','sft9b'],
    'questions_per_model':128,'repeats_per_model':1,'total_questions':384,
    'tool_call_budget':120,'budget_scope':'shared main and delegated agents; force final answer at limit',
    'per_gpu_concurrency':64,'global_concurrency':128,'tp':1,'context':131072,'max_output':16384,
    'temperature':1.0,'top_p':1.0,'top_k':-1,'min_p':0.0,'repetition_penalty':1.0,'seed':42,
    'thinking':True,'kv_dtype':'fp8','gpu_memory_utilization':.92,
    'agent_iteration_safety_guard':200,'question_timeout':None,
    'protocol_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(D/'manifest.json').write_text(json.dumps(manifest,indent=2))
mod=ast.parse((P/'Evaluation/browse_comp_eval/hermes_client.py').read_text())
cs={n.targets[0].id:ast.literal_eval(n.value) for n in mod.body
    if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
    and n.targets[0].id in ['RESEARCH_GUIDANCE','RESEARCH_TASK_PREAMBLE']}
(D/'user_prompt_template.txt').write_text(cs['RESEARCH_TASK_PREAMBLE']+'\n\n'+cs['RESEARCH_GUIDANCE']+
    '\n\nQuestion:\n{question}\n\n'+(P/'Evaluation/config/answer_prompts/browsecomp.txt').read_text())
print('PASS: 128 unique questions; three model shard headers and lengths verified; protocol frozen')
