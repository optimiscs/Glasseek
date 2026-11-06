import json, pandas as pd
from pathlib import Path
p=Path('analysis/32k-parallel-benchmark')
a=sorted((x for x in map(json.loads,open('data/glasseek/sft/selection.jsonl')) if x.get('split')=='train' and x.get('exact') and x['tokens']<=32768),key=lambda x:x['tokens'])
chosen=[a[round(i*(len(a)-1)/63)] for i in range(64)]
ids={x['source_line'] for x in chosen}
lines={i:json.loads(s) for i,s in enumerate(open('data/Glasseek-SFT-5.7K/train.jsonl'),1) if i in ids}
records=[{'messages':json.dumps(lines[x['source_line']]['messages'],ensure_ascii=False)} for x in chosen]
pd.DataFrame(records*2).to_parquet(p/'representative-128.parquet',index=False)
(p/'representative.json').write_text(json.dumps(chosen,indent=2))
print('n=',len(records),'min=',chosen[0]['tokens'],'median=',chosen[32]['tokens'],'max=',chosen[-1]['tokens'],'total tokens per set=',sum(x['tokens'] for x in chosen))
