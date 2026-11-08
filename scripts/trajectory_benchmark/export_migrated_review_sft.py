"""Streaming review-only Parquet export for the existing Glasseek training loader."""
import argparse,collections,json,sys,hashlib
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'trajectory_preprocess'))
from migration_render import target_messages
from runtime_metadata import VERSION as METADATA_POLICY

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--migration',type=Path,required=True);args=ap.parse_args();out=args.migration
    contract=json.loads((out/'manifest.json').read_text())['contract'];counts=collections.Counter();transformation_counts=collections.Counter()
    mismatched=set()
    with (out/'report_mismatch_review.jsonl').open('w') as review:
        for line in (out/'links.jsonl').open():
            edge=json.loads(line)
            if edge['child_id'] and not edge['report_matches_child_final']:
                mismatched.update([edge['parent_id'],edge['child_id']]);review.write(line)
    schema=pa.schema([('id',pa.string()),('messages',pa.string()),('tools',pa.string()),('enable_thinking',pa.bool_()),
                      ('agent_role',pa.string()),('bundle_id',pa.string()),('split_group_id',pa.string()),('parent_id',pa.string()),
                      ('source',pa.string()),('source_row',pa.int64()),('training_ready',pa.bool_()),('harness_replayed',pa.bool_()),
                      ('message_loss_mask',pa.string()),('review_flags',pa.string()),('render_transforms',pa.string())])
    temp=out/'review_sft.parquet.tmp';writer=pq.ParquetWriter(temp,schema,compression='zstd');batch=[]
    try:
        for role in ('parent','child'):
            tools=[{'type':'function','function':s} for name,s in contract['schemas'].items() if role=='parent' or name not in contract['delegation']['blocked_child_tools']]
            tooltext=json.dumps(tools,ensure_ascii=False)
            for line in (out/(role+'_candidates.jsonl')).open():
                r=json.loads(line)
                assert not r['migration_errors'] and not r['training_ready']
                messages,transforms=target_messages(r)
                transformation_counts.update(transforms)
                flags=list(r['review_flags'])
                if r['id'] in mismatched:flags.append('linked_report_final_mismatch')
                batch.append({'id':r['id'],'messages':json.dumps(messages,ensure_ascii=False),'tools':tooltext,'enable_thinking':True,
                              'agent_role':role,'bundle_id':r['bundle_id'],'split_group_id':r['split_group_id'],'parent_id':r.get('parent_id'),
                              'source':'SearchSwarm','source_row':r['provenance']['row_index'],'training_ready':False,'harness_replayed':False,
                              'message_loss_mask':json.dumps(r['message_loss_mask']),'review_flags':json.dumps(flags,ensure_ascii=False),'render_transforms':json.dumps(transforms)})
                counts[role]+=1
                if len(batch)>=32:writer.write_table(pa.Table.from_pylist(batch,schema=schema));batch=[]
        if batch:writer.write_table(pa.Table.from_pylist(batch,schema=schema))
    finally:writer.close()
    assert pq.ParquetFile(temp).metadata.num_rows==sum(counts.values());temp.replace(out/'review_sft.parquet')
    result={'status':'completed','rows':dict(counts),'total':sum(counts.values()),'training_ready':False,
            'runtime_metadata_policy':METADATA_POLICY,
            'metadata_cleaner_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'trajectory_preprocess/runtime_metadata.py').read_bytes()).hexdigest(),
            'render_transforms':dict(transformation_counts),'report_mismatch_review_ids':len(mismatched),
            'renderer_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'trajectory_preprocess/migration_render.py').read_bytes()).hexdigest(),
            'required_before_training':['Target system/tool instructions reconciliation','Tool response semantics review/replay','Quality review','Full-corpus target-token length filtering and bundle-grouped split'],
            'mask_note':'message_loss_mask is role-level audit metadata. Existing loader computes actual token masks.'}
    (out/'export_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
