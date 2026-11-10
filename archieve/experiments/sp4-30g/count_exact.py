import json,multiprocessing as mp
from pathlib import Path
from glasseek_training.filter_sft_length import initialize, process
p=Path('experiments/sp4-30g'); audit=[json.loads(x) for x in open('data/glasseek/sft/selection.jsonl')]; wanted={x['source_line'] for x in audit if not x['exact']}
def items():
 for i,line in enumerate(open('data/Glasseek-SFT-5.7K/train.jsonl'),1):
  if i in wanted: yield i,line,10**9,42,.05
with (p/'lengths-extra.jsonl').open('w') as f, mp.Pool(8, initializer=initialize,initargs=('/root/autodl-tmp/models/Qwen3.5-9B',)) as pool:
 for n,(line,length,exact,*_) in enumerate(pool.imap_unordered(process,items(),chunksize=1),1):
  f.write(json.dumps({'source_line':line,'tokens':length,'exact':exact})+'\n');f.flush()
  if n%100==0: print('completed',n,'of',len(wanted),flush=True)
print('DONE',flush=True)
