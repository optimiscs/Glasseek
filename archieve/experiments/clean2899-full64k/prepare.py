import json,hashlib,shutil,collections,time
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from glasseek_training.sft_dataset import GlasseekSFTDataset
P=Path('/root/autodl-tmp/Glasseek');D=P/'data/glasseek/sft-clean2899-fixed';S=P/'reports/experiments/no-tool-top10/explicit-matches/remaining_shortest_trajectories.jsonl'
shutil.copyfile(S,D/'all_2899.jsonl');rows=list(map(json.loads,(D/'all_2899.jsonl').open()));assert len(rows)==2899
pd.DataFrame({'messages':[json.dumps(x['messages'],ensure_ascii=False) for x in rows]}).to_parquet(D/'all_2899.parquet',index=False)
t=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen3.5-9B',local_files_only=True)
c=OmegaConf.create(dict(max_length=65536,truncation='error',pad_mode='no_padding',ignore_input_ids_mismatch=True,enable_thinking_default=None))
ds=GlasseekSFTDataset(str(D/'all_2899.parquet'),t,c)
splits={'train':[],'val':[]};audit=[];start=time.time()
for i,row in enumerate(rows):
 total=loss=0
 for j,m in enumerate(row['messages']):
  ids,mask,_,_=ds._process_single_message(j,m,row['messages']);total+=len(ids);loss+=int(mask.sum())
 split='val' if row['original_split']=='val' else 'train';keep=total<=65536
 audit.append(dict(source_line=row['source_line'],tokens=total,supervised_tokens=loss,split=split,keep=keep))
 if keep:splits[split].append(dict(messages=json.dumps(row['messages'],ensure_ascii=False),source_line=row['source_line']))
 if (i+1)%200==0:print(i+1,round(time.time()-start),flush=True)
for split,r in splits.items():pd.DataFrame(r).to_parquet(D/f'{split}.parquet',index=False)
(D/'selection.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in audit))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=dict(source=str(S),source_sha256=sha(S),max_length=65536,counts={**{s:len(r) for s,r in splits.items()},'source':2899,'excluded':sum(not x['keep'] for x in audit)},output_sha256={s:sha(D/f'{s}.parquet') for s in splits},maximum_retained_tokens=max(x['tokens'] for x in audit if x['keep']),length_method='Exact training dataset _process_single_message, no truncation; original validation split retained; previously unassigned rows assigned train')
(D/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
