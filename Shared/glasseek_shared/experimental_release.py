"""Explicit experimental pilot datasets; never claim historical runtime replay."""
import json
from pathlib import Path
from .trajectory_release import digest,payload,file_hash,runtime_schemas,schemas,validate_messages
from .dataset_policy import excluded_source
VERSION='glasseek-experimental-pilot-v1'

def validate(row, tools):
    acceptance=json.loads(row['acceptance']) if isinstance(row['acceptance'],str) else row['acceptance']
    if excluded_source(row.get('source')):raise ValueError('excluded_source')
    if row.get('acceptance_status')!='experimental_pilot' or row.get('harness_replayed') is not False:raise ValueError('invalid_pilot_status')
    if acceptance.get('review')!='not_performed' or acceptance.get('replay')!='not_performed':raise ValueError('invalid_pilot_acceptance')
    if acceptance.get('safety')!='pass_rules' or not acceptance.get('authorization'):raise ValueError('missing_safety_or_authorization')
    value=payload(row)
    if digest(value)!=row.get('content_sha256') or digest(value)!=acceptance.get('content_sha256'):raise ValueError('stale_pilot_hash')
    if schemas(value['tools'])!=tools:raise ValueError('pilot_tool_schema_mismatch')
    validate_messages(value['messages'],tools)
    import re
    for m in value['messages']:
        if m['role']=='assistant' and m['content'].count('<think>')!=m['content'].count('</think>'):raise ValueError('unbalanced_think')
    final=re.sub(r'^\s*<think>.*?</think>','',value['messages'][-1]['content'],count=1,flags=re.S)
    if not final.startswith('<answer>') or not final.endswith('</answer>') or final.count('<answer>')!=1 or final.count('</answer>')!=1:raise ValueError('invalid_answer_envelope')

def verify(path):
    import pyarrow.parquet as pq
    path=Path(path);manifest=json.loads((path.parent/'release.manifest.json').read_text());tools=runtime_schemas()
    if manifest.get('version')!=VERSION or manifest.get('acceptance_status')!='experimental_pilot':raise ValueError('not_experimental_pilot')
    if manifest.get('tool_schema_sha256')!=digest(tools) or manifest.get('sha256')!=file_hash(path):raise ValueError('pilot_file_hash_mismatch')
    count=0;ids=set()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            validate(row,tools)
            if row['id'] in ids:raise ValueError('duplicate_id')
            ids.add(row['id']);count+=1
    if count!=manifest['rows'] or count==0:raise ValueError('pilot_row_count')
    return manifest
