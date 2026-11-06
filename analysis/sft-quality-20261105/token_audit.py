"""Count raw message tokens with the local training tokenizer, without GPU use.

These counts exclude chat wrappers; total rendered sequence lengths live in selection.jsonl.
"""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import collections as C
import hashlib
import json
import re
from pathlib import Path
from tokenizers import Tokenizer

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
tokenizer=Tokenizer.from_file('/autodl-fs/data/models/Qwen3.5-9B/tokenizer.json')
think=re.compile(r'<think>(.*?)</think>',re.S)
cache={}; totals=C.Counter();groups=C.defaultdict(C.Counter)
with (OUT/'tokens.jsonl').open('w') as f:
    for line,raw in enumerate((ROOT/'data/Glasseek-SFT-5.7K/train.jsonl').open(),1):
        ms=json.loads(raw)['messages'];q=ms[1]['content']
        family='multi_constraint' if q.startswith('You are answering one hard multi-constraint') else 'browse_comparison' if q.startswith('You are running one browse-comparison') else 'other'
        c=C.Counter()
        for m in ms:
            role=m['role'];s=m['content']
            if role=='system':
                h=hashlib.sha256(s.encode()).hexdigest()
                if h not in cache:cache[h]=len(tokenizer.encode(s,add_special_tokens=False).ids)
                c[role]+=cache[h]
            else:c[role]+=len(tokenizer.encode(s,add_special_tokens=False).ids)
            if role=='assistant':
                for t in think.findall(s):c['think']+=len(tokenizer.encode(t,add_special_tokens=False).ids)
        c['final_visible']=len(tokenizer.encode(think.sub('',ms[-1]['content']),add_special_tokens=False).ids)
        totals.update(c);groups[family].update(c)
        f.write(json.dumps(dict(line=line,family=family,**c))+'\n')
        if line%500==0:print('processed',line,flush=True)
(OUT/'token_metrics.json').write_text(json.dumps(dict(totals=dict(totals),groups={k:dict(v) for k,v in groups.items()},method='Qwen3.5-9B raw content encoding; no chat wrappers. think and final_visible are subsets of assistant.'),indent=2)+'\n')
print('DONE',dict(totals),flush=True)
