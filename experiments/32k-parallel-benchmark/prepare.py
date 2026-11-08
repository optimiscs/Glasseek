import json, pandas as pd
from pathlib import Path
p=Path('experiments/32k-parallel-benchmark')
ids=[427,3489,3321,2854]
lines={i:json.loads(s) for i,s in enumerate(open('data/Glasseek-SFT-5.7K/train.jsonl'),1) if i in ids}
records=[{'messages':json.dumps(lines[i]['messages'],ensure_ascii=False)} for i in ids]
for n in (8,128):
 pd.DataFrame([records[i%4] for i in range(n)]).to_parquet(p/f'data-{n}.parquet',index=False)
(p/'samples.json').write_text(json.dumps([{'source_line':i,'tokens':next(x['tokens'] for x in map(json.loads,open('data/glasseek/sft/selection.jsonl')) if x['source_line']==i)} for i in ids],indent=2))
