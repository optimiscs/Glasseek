import json,sys,pandas as pd
from pathlib import Path
ids={int(x) for x in sys.argv[1:]}; p=Path('experiments/sp4-30g')
for i,line in enumerate(open('data/Glasseek-SFT-5.7K/train.jsonl'),1):
 if i in ids:
  row=json.loads(line); d={'messages':json.dumps(row['messages'],ensure_ascii=False)}
  if row.get('tools') is not None:d['tools']=json.dumps(row['tools'],ensure_ascii=False)
  if row.get('enable_thinking') is not None:d['enable_thinking']=bool(row['enable_thinking'])
  pd.DataFrame([d]).to_parquet(p/f'line-{i}.parquet',index=False)
