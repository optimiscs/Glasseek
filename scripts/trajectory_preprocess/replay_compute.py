"""Replay only recorded execute_code calls in the pinned isolated runtime.

No web calls, models, source repairs or observation replacement. Exact output
agreement is reported independently from execution success. Failed or different
returns remain review cases, not invented historical evidence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import threading
from concurrent.futures import Future

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'Evaluation_backend'))
from tools.remote_pure_compute import execute
from glasseek_shared.historical_tools import unwrap
from client import digest


def run(source, out, workers=3, reuse=None):
    out.mkdir(parents=True, exist_ok=True)
    manifest = (ROOT/'Evaluation_backend/tools/scientific_runtime.json').read_bytes()
    identity = {'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'runtime_sha256':hashlib.sha256(manifest).hexdigest(),
                'supervisor_sha256':hashlib.sha256((ROOT/'Evaluation_backend/tools/scientific_container_worker.py').read_bytes()).hexdigest(),
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    config=out/'manifest.json'
    if config.exists() and json.loads(config.read_text())!=identity:
        raise ValueError('Replay inputs changed; use a new directory')
    config.write_text(json.dumps(identity,indent=2))
    items=[json.loads(l) for l in source.open()]
    calls={};references={}
    for item in items:
        refs=[]
        for message in item['messages']:
            for call in message.get('call_links',[]):
                if call['name']!='execute_code':continue
                code=call['arguments']['code'];key=hashlib.sha256(code.encode()).hexdigest()
                calls[key]=code
                observed=[m['content'] for m in item['messages'] if m.get('response_to')==call['call_id']]
                refs.append({'code_sha256':key,'call_id':call['call_id'],
                             'expected':unwrap(observed[0]).strip() if len(observed)==1 else None})
        if not refs:raise ValueError('Selected trajectory has no code calls')
        references[item['id']]=refs
    receipts={}
    receipt_path=out/'calls.jsonl'
    if reuse is not None:
        previous=json.loads((reuse/'manifest.json').read_text())
        for field in ('input_sha256','runtime_sha256','supervisor_sha256'):
            if previous[field]!=identity[field]:raise ValueError('Cached runtime/input mismatch')
        # Only raw execution receipts are reused. All output comparisons below
        # are recomputed by this implementation, including mismatches.
        for line in (reuse/'calls.jsonl').open():
            row=json.loads(line)
            if row['code_sha256'] in calls:receipts[row['code_sha256']]=row
        (out/'reused_receipts.json').write_text(json.dumps({'source':str(reuse.resolve()),'manifest':previous,'count':len(receipts)},indent=2))
        if receipt_path.exists():raise ValueError('Reuse requires a fresh receipt ledger')
        with receipt_path.open('w') as f:
            for row in receipts.values():f.write(json.dumps(row,ensure_ascii=False)+'\n')
    if receipt_path.exists():
        for line in receipt_path.open():
            row=json.loads(line);receipts[row['code_sha256']]=row
    pending={};mutex=threading.Lock();completed_items=0
    with receipt_path.open('a') as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        def get_receipt(key):
            with mutex:
                if key in receipts:return receipts[key]['receipt']
                owner=key not in pending
                if owner:pending[key]=Future()
                future=pending[key]
            if not owner:return future.result()
            try:
                result=json.loads(execute(calls[key],'Lab3090',30))
                row={'code_sha256':key,'receipt':result}
                with mutex:
                    receipts[key]=row
                    handle.write(json.dumps(row,ensure_ascii=False)+'\n');handle.flush()
                future.set_result(result)
                return result
            except BaseException as exc:
                future.set_exception(exc);raise
        def replay_item(item):
            for ref in references[item['id']]:
                receipt=get_receipt(ref['code_sha256'])
                if receipt['status']!='success' or not receipt.get('cleanup_verified') or ref['expected'] is None or receipt['output'].strip()!=ref['expected']:
                    break
        futures=[pool.submit(replay_item,item) for item in items]
        for future in as_completed(futures):
            future.result();completed_items+=1
            if completed_items%50==0:print(json.dumps({'completed_trajectories':completed_items,'total':len(items),'executed_unique_calls':len(receipts)}),flush=True)
    counts={'input':len(items),'unique_code_calls':len(calls),'executed_unique_calls':len(receipts),'execution_passed':0,'exact_output_passed':0,'held_execution':0,'held_output':0}
    with (out/'decisions.jsonl').open('w') as decisions, (out/'recovered.jsonl').open('w') as released:
        for item in items:
            details=[]
            stopped=False
            for ref in references[item['id']]:
                if stopped:
                    details.append({'code_sha256':ref['code_sha256'],'call_id':ref['call_id'],'skipped':'prior_call_failed_or_differed'})
                    continue
                receipt=receipts[ref['code_sha256']]['receipt']
                success=receipt['status']=='success' and receipt.get('cleanup_verified')
                matches=success and ref['expected'] is not None and receipt['output'].strip()==ref['expected']
                details.append({k:v for k,v in ref.items() if k!='expected'} | {'execution_passed':bool(success),'output_matches':bool(matches),
                    'expected_sha256':digest(ref['expected']),'actual_sha256':digest(receipt.get('output','').strip())})
                stopped=not matches
            success=all(d.get('execution_passed',False) for d in details)
            matches=all(d.get('output_matches',False) for d in details)
            failed_execution=any(d.get('execution_passed') is False for d in details)
            counts['execution_passed']+=success;counts['exact_output_passed']+=matches
            counts['held_execution']+=failed_execution;counts['held_output']+=not matches and not failed_execution
            decisions.write(json.dumps({'id':item['id'],'execution_passed':success,'output_matches':matches,'calls':details})+'\n')
            if matches:
                item['compute_replay']={'status':'passed','output_agreement':'exact_after_outer_whitespace',**identity}
                # Web calls have NOT been replayed; do not set harness_replayed.
                released.write(json.dumps(item,ensure_ascii=False)+'\n')
    (out/'summary.json').write_text(json.dumps(counts,indent=2))
    print(json.dumps(counts),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--reuse-receipts-from',type=Path)
    args=parser.parse_args()
    if not 1<=args.workers<=3:parser.error('workers must be 1..3 (one runtime slot reserved)')
    run(args.input,args.output,args.workers,args.reuse_receipts_from)
