"""Create an immutable source-filtered REVIEW snapshot; never reapprove old data."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'Shared'))
from glasseek_shared.dataset_policy import excluded_source, VERSION
from glasseek_shared.trajectory_release import file_hash


def filter_snapshot(source, output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    output.mkdir(parents=True,exist_ok=False)
    counts=collections.Counter();removed=collections.Counter();ids=set();files=[]
    for filename in ('candidates.jsonl','origins.jsonl','exact_duplicates.jsonl'):
        path=source/filename
        if not path.exists():continue
        h=hashlib.sha256();n=0
        with path.open('rb') as src,(output/filename).open('wb') as dst:
            for line in src:
                row=json.loads(line)
                if excluded_source(row.get('source')):
                    if filename=='candidates.jsonl':removed[row['source']]+=1
                    continue
                dst.write(line);h.update(line);n+=1
                if filename=='candidates.jsonl':
                    counts[row['source']]+=1;ids.add(row['id'])
                    if n%20000==0:print(json.dumps({'file':filename,'retained':n}),flush=True)
        files.append({'path':filename,'rows':n,'sha256':h.hexdigest()})
    pf=pq.ParquetFile(source/'review_sft.parquet');n=0;parquet_ids=set()
    with pq.ParquetWriter(output/'review_sft.parquet',pf.schema_arrow,compression='zstd') as writer:
        for batch in pf.iter_batches(batch_size=64):
            table=pa.Table.from_batches([batch])
            keep=pa.array([not excluded_source(s) for s in table['source'].to_pylist()])
            table=table.filter(keep);n+=table.num_rows
            parquet_ids.update(table['id'].to_pylist());writer.write_table(table)
    if n!=sum(counts.values()) or parquet_ids!=ids:raise ValueError('JSONL/Parquet ID or count mismatch')
    files.append({'path':'review_sft.parquet','rows':n,'sha256':file_hash(output/'review_sft.parquet')})
    for filename in ('tools.json','delegation_links.jsonl'):
        if (source/filename).exists():
            shutil.copyfile(source/filename,output/filename)
            files.append({'path':filename,'sha256':file_hash(output/filename)})
    with (output/'dataset_summary.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['source','retained_candidates','removed'])
        for name in sorted(counts.keys()|removed.keys()):writer.writerow([name,counts[name],removed[name]])
    manifest={'status':'completed','version':VERSION,'operation':'source_exclusion_only',
        'source':str(source.resolve()),'candidates':n,'removed':dict(removed),'by_source':dict(counts),
        'training_ready':False,'llm_calls':0,'live_tool_calls':0,'files':files,
        'note':'Historical Qwen user/tool_response messages preserved byte-for-byte. No protocol remigration, quality approval or replay.',
        'script_sha256':file_hash(__file__)}
    (output/'summary.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (output/'release.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    manifest=add_content_inventory(output)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)
    return manifest


def add_content_inventory(output):
    """Preserve each existing v7 payload hash; do not relabel it as a new approval."""
    import pyarrow.parquet as pq
    manifest=json.loads((output/'summary.json').read_text())
    path=output/'content_hashes.jsonl'
    count=0
    with path.open('x') as stream:
        for batch in pq.ParquetFile(output/'review_sft.parquet').iter_batches(columns=['id','source','payload_sha256']):
            for row in batch.to_pylist():
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');count+=1
    if count!=manifest['candidates']:raise ValueError('Content inventory count mismatch')
    manifest['content_inventory']={'path':path.name,'rows':count,'sha256':file_hash(path),
        'hash_scope':'Original v7 messages + tools payload_sha256, preserved without reapproval'}
    manifest['content_inventory_builder_sha256']=file_hash(__file__)
    for name in ('summary.json','release.json'):
        (output/name).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();filter_snapshot(args.input,args.output)
