"""Verify full migration outputs, independent agent masks and source provenance."""
import argparse,collections,hashlib,json,sys,time
from pathlib import Path
import pyarrow.parquet as pq
from artifact_io import open_jsonl
from glasseek_contract import validate_args
from tool_xml import normalize_text,THINK
from client import digest

def rows(path):
    with open_jsonl(path) as handle:
        for line in handle:yield json.loads(line)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--migration',type=Path,required=True);args=ap.parse_args();out=args.migration
    start=time.monotonic();summary=json.loads((out/'summary.json').read_text());manifest=json.loads((out/'manifest.json').read_text());contract=manifest['contract'];counts=summary['counts']
    expected=min(manifest['limit'],manifest['expected_bundles']) if manifest['limit'] is not None else manifest['expected_bundles']
    assert summary['status']=='completed' and counts['bundles']==expected
    source=Path(manifest['source']);stat=source.stat();assert stat.st_size==manifest['bytes'] and stat.st_mtime_ns==manifest['mtime_ns']
    for name,sha in manifest['code_sha256'].items():assert hashlib.sha256((out/'frozen_source'/name).read_bytes()).hexdigest()==sha
    seen=set();eligible=set();tools=collections.defaultdict(collections.Counter);metrics=collections.Counter();groups={}
    for kind in ('parent','child'):
        n=0
        for item in rows(out/(kind+'_candidates.jsonl')):
            n+=1;assert item['id'] not in seen;seen.add(item['id']);eligible.add(item['id']);groups[item['id']]=item['split_group_id']
            assert not item['migration_errors'] and not item['harness']['errors'] and not item['training_ready'] and item['tools_materialized']
            assert item['message_loss_mask']==[int(m['role']=='assistant') for m in item['messages']]
            assert all(mask==0 for m,mask in zip(item['messages'],item['message_loss_mask']) if '<tool_response>' in m['content'] and m['role']!='assistant')
            if item['over_legacy_250k_char_budget']:metrics[kind+'_over_legacy_char_budget']+=1
            for step in item['steps']:
                assert normalize_text(step['assistant'],step['calls'])[1]==step['calls']
                for call in step['calls']:
                    assert call['name'] in contract['schemas'] and not validate_args(call['name'],call['arguments'],contract)
                    if kind=='child':assert call['name'] not in contract['delegation']['blocked_child_tools']
                    tools[kind][call['name']]+=1
            if kind=='parent' and not item['delegation_links_complete']:metrics['candidate_parents_with_unresolved_child_archive']+=1
            if n%2000==0:print(kind,n,flush=True)
        assert n==counts[kind+'_candidates'];metrics[kind+'_candidates_verified']=n
    allids={};linecounts={}
    for name in ('parents','children','held','links','delegation_events','duplicates'):
        n=0
        with open_jsonl(out/(name+'.jsonl')) as handle:
            for line in handle:
                n+=1
                if name in ('parents','children','held'):
                    r=json.loads(line);allids[r['id']]=True
                if name=='links':
                    r=json.loads(line)
                    if r['child_id'] in eligible and r['parent_id'] in eligible:assert groups[r['child_id']]==groups[r['parent_id']]
                    if not r['report_matches_child_final'] and r['child_id']:metrics['linked_report_final_mismatches']+=1
        linecounts[name]=n
    assert linecounts['parents']==counts['parent_normalized'] and linecounts['children']==counts['child_normalized']
    assert linecounts['held']==counts['parent_held']+counts['child_held']
    assert linecounts['links']==counts['delegated_tasks']
    for edge in rows(out/'links.jsonl'):
        assert edge['parent_id'] in allids
        if edge['child_id']:assert edge['child_id'] in allids
    result={'status':'passed','counts':counts,'metrics':dict(metrics),'tool_calls':dict(tools),'line_counts':linecounts,
            'source_preservation':'comparison examples are archived; this audit checks full-corpus structure and provenance','elapsed_seconds':time.monotonic()-start,'runtime_replayed':False,'training_ready':False,
            'audit_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
