"""Verify every emitted candidate against the frozen Glasseek contract (no tools/LLMs)."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'trajectory_preprocess'))
from glasseek_contract import validate_args
from ingest import semantic_hash
from tool_xml import normalize_text
from full_prepare import OUTPUTS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output.resolve();audit=out/'audit';audit.mkdir(exist_ok=True)
    state=json.loads((out/'cleaning_summary.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    contract=manifest['config']['harness']['contract'];start=time.monotonic()
    assert state['status']=='completed' and state['completed_files']==len(manifest['inputs'])
    assert not state['counts'].get('file_errors')
    for name,sha in manifest['code_sha256'].items():
        assert hashlib.sha256((out/'frozen_source'/name).read_bytes()).hexdigest()==sha
        assert hashlib.sha256((Path(__file__).resolve().parents[1]/'trajectory_preprocess'/name).read_bytes()).hexdigest()==sha
    counts=collections.Counter();calls=collections.Counter();trajectories=collections.Counter();flags=collections.Counter()
    strict_calls=collections.Counter()
    by_source=collections.defaultdict(collections.Counter);examples={};linecounts={}
    review_flags={'unmapped_search_arguments','extraction_semantics_require_replay','goal_conditioned_extraction_requires_replay','search_batch_split_requires_observation_alignment',
                  'extract_batch_split_requires_observation_alignment','browser_viewport_semantics_require_replay','search_backend_semantics_require_replay'}
    with (audit/'semantic_review_index.jsonl').open('w') as review, (out/'strict_tool_candidates.jsonl.tmp').open('w') as strict:
        for index,line in enumerate((out/'clean.jsonl').open()):
            item=json.loads(line);counts['candidates']+=1
            assert item['tools_materialized'] is True and item['training_ready'] is False
            assert item['harness']['contract_sha256']==contract['sha256']
            assert not item['program_errors'] and not item['harness']['errors'] and not item['harness']['replayed']
            assert item['content_hash']==semantic_hash(item)
            seen=set();planned=[];actual=[]
            for call in item['harness']['calls']:planned.append((call['step_id'],call['name'],call['arguments']))
            for step in item['steps']:
                assert normalize_text(step['assistant'],step['calls'])[1]==step['calls']
                for call in step['calls']:
                    name=call['name'];assert name in contract['schemas']
                    assert not validate_args(name,call['arguments'],contract)
                    actual.append((step['step_id'],name,call['arguments']))
                    seen.add(name);calls[name]+=1;by_source[item['source']]['calls:'+name]+=1
            assert actual==planned
            trajectories.update(seen);flags.update(set(item['harness']['flags']))
            by_source[item['source']]['candidates']+=1
            by_source[item['source']].update('trajectories:'+name for name in seen)
            if not seen:counts['without_tool_calls']+=1
            semantic=sorted(set(item['harness']['flags']) & review_flags)
            if item.get('instruction_messages'):semantic.append('source_instructions_require_reconciliation')
            if semantic:
                counts['semantic_review']+=1
                review.write(json.dumps({'id':item['id'],'source':item['source'],'clean_line_index':index,'reasons':semantic,'provenance':item['provenance']},ensure_ascii=False)+'\n')
            else:
                counts['without_known_semantic_conversion_flags']+=1
                by_source[item['source']]['strict_tool_candidates']+=1
                strict_calls.update(c['name'] for step in item['steps'] for c in step['calls'])
                strict.write(line)
            for name in seen:
                if name not in examples:examples[name]={'id':item['id'],'source':item['source'],'clean_line_index':index,'provenance':item['provenance']}
            if counts['candidates']%20000==0:print('AUDITED',counts['candidates'],flush=True)
    linecounts['clean']=counts['candidates']
    reasons=collections.Counter();runtime=collections.Counter();unsupported=collections.Counter()
    partitions={name:(audit/(name+'.jsonl')).open('w') for name in ('runtime_incompatible','disabled_or_unmapped_tools','confirmed_record_defects','structural_review')}
    partition_counts=collections.Counter()
    definite={'empty_trajectory','empty_tool_call','empty_question'}
    for name in OUTPUTS:
        if name=='clean':continue
        n=0
        for line in (out/(name+'.jsonl')).open():
            n+=1
            if name=='held':
                row=json.loads(line);reasons.update(set(row['reasons']))
                runtime.update(r for r in set(row['reasons']) if r.startswith('tool_runtime:'))
                unsupported.update(r for r in set(row['reasons']) if r.startswith('unsupported_tool:'))
                groups=[]
                if any(r.startswith('tool_runtime:') for r in row['reasons']):groups.append('runtime_incompatible')
                if any(r.startswith('unsupported_tool:') for r in row['reasons']):groups.append('disabled_or_unmapped_tools')
                if set(row['reasons']) & definite or any(s in row.get('error','') for s in ('Empty trajectory','Empty question','Empty tool call')):groups.append('confirmed_record_defects')
                elif 'quality_issues' in row['categories']:groups.append('structural_review')
                for group in groups:partitions[group].write(line);partition_counts[group]+=1
        linecounts[name]=n
    for file in partitions.values():file.close()
    assert linecounts['clean']==state['counts']['eligible']
    assert linecounts['held']==state['counts']['held']
    assert linecounts['duplicates']==state['counts']['exact_duplicates']
    assert linecounts['near_duplicates']==state['counts']['near_duplicate_flagged']
    assert linecounts['clean']+linecounts['held']+linecounts['duplicates']==state['counts']['scanned']
    db=sqlite3.connect('file:'+str(out/'dedup.sqlite')+'?mode=ro',uri=True)
    # The dedup table keeps one entry per unique normalized record, including holds.
    unique=db.execute('SELECT count(*) FROM exact').fetchone()[0];db.close()
    assert unique+linecounts['duplicates']==state['counts']['normalized']
    result={'status':'passed','contract_sha256':contract['sha256'],'completed_files':state['completed_files'],
            'audit_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'processing_counts':state['counts'],'sidecar_line_counts':linecounts,'unique_normalized_records':unique,
            'candidate_counts':dict(counts),'tool_call_counts':dict(calls),'tool_trajectory_counts':dict(trajectories),
            'strict_tool_candidate_call_counts':dict(strict_calls),
            'by_dataset':dict(by_source),'adapter_flags_overlapping':dict(flags),'examples':examples,
            'runtime_holds_overlapping':dict(runtime),'unsupported_tools_overlapping':dict(unsupported),
            'held_reasons_overlapping':dict(reasons),'audit_elapsed_seconds':time.monotonic()-start,
            'held_partitions_overlapping':dict(partition_counts),
            'strict_tool_candidate_policy':{'excluded_adapter_flags':sorted(review_flags),'source_instruction_reconciliation_required':True,
                                            'meaning':'No known tool-semantic conversion gap under the static screen. Runtime/factual/human review still pending; not a final training export.'},
            'limits':['Static/schema checks only; no dataset code executed or live replay.','Historical observations preserved; semantic review index lists transformations/source instructions needing reconciliation.','All candidates remain training_ready=false; factual quality, human review and replay are pending.']}
    (audit/'tool_contract_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (audit/'glasseek_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
    (audit/'audit_glasseek_contract.py').write_bytes(Path(__file__).read_bytes())
    (out/'strict_tool_candidates.jsonl.tmp').replace(out/'strict_tool_candidates.jsonl')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
