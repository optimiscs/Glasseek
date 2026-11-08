import json,collections,hashlib
from pathlib import Path
import pyarrow as pa,pyarrow.parquet as pq
r=Path('/root/autodl-tmp/Glasseek');src=r/'data/glasseek/sft-eval-fixed-128k';dst=r/'data/glasseek/sft-eval-fixed-64k';dst.mkdir(parents=True,exist_ok=True)
a=[json.loads(x) for x in (src/'audit.jsonl').open()];c={};picks=[]
for split in ('train','val'):
 meta=[x for x in a if x.get('original_split')==split];t=pq.read_table(src/f'{split}.parquet');assert len(meta)==t.num_rows,(len(meta),t.num_rows)
 keep=[i for i,x in enumerate(meta) if x['fixed_total_tokens_estimate']<=65536];filtered=t.take(pa.array(keep));pq.write_table(filtered,dst/f'{split}.parquet',compression='zstd');c[split]=len(keep)
 if split=='train':
  j=max(keep,key=lambda i:meta[i]['fixed_total_tokens_estimate']);sample=t.slice(j,1).to_pylist()[0];pq.write_table(pa.Table.from_pylist([sample]*4,schema=t.schema),dst/'probe-longest-4.parquet',compression='zstd');c['probe_source_line']=meta[j]['source_line'];c['probe_tokens_estimate']=meta[j]['fixed_total_tokens_estimate']
 for i in keep:picks.append({'split':split,'source_line':meta[i]['source_line'],'length':meta[i]['fixed_total_tokens_estimate']})
(dst/'selection.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in picks));(dst/'manifest.json').write_text(json.dumps({'source':str(src),'length_limit':65536,'length_method':'original verl per-turn exact + system token difference; sample exact verify before use','counts':c,'data_sha256':{s:hashlib.sha256((dst/f'{s}.parquet').read_bytes()).hexdigest() for s in ('train','val')}},indent=2)+'\n');print(json.dumps(c))
