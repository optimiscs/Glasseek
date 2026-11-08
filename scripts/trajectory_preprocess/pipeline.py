"""Configurable offline curation. Run --help for the six-stage workflow.

No automatic human approvals, live tool replay, or deployment are implied.
"""
import argparse
import copy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import glob
import hashlib
import json
from pathlib import Path
import re
import sys

from client import Client, digest, public_config, save
from ingest import Dedup, harness_plan, normalize, records

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / 'scripts/trajectory_preprocess_pilot'
sys.path.insert(0, str(PILOT))
import run_v3 as policy


def append(path, value):
    with Path(path).open('a') as file:
        file.write(json.dumps(value, ensure_ascii=False) + '\n')


def policy_hash():
    files = [*[p for p in Path(__file__).resolve().parent.glob('*.py') if not p.name.startswith('test_')], PILOT/'run_v3.py', PILOT/'run_v2.py']
    return digest({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})


def prepare(config, out, limit):
    out.mkdir(parents=True, exist_ok=True)
    database = out/'dedup.sqlite'
    if database.exists(): database.unlink()
    dedup = Dedup(database, config['dedup'])
    counts = {'scanned': 0, 'kept': 0, 'exact_duplicates': 0, 'quarantined': 0, 'file_errors': 0}
    paths = {name: out/(name+'.jsonl.tmp') for name in ('clean', 'cleaning_log')}
    for path in paths.values(): path.write_text('')
    ids = set()
    try:
        for spec in config['inputs']:
            matched = sorted(glob.glob(str(ROOT/spec['path']), recursive=True))
            if not matched: raise ValueError('Input pattern matched no files: ' + spec['path'])
            for path in matched:
                try:
                    for index, row in enumerate(records(path)):
                        if counts['scanned'] >= limit: break
                        counts['scanned'] += 1
                        try:
                            item = normalize(row,spec['source'],path,index,spec.get('adapter','auto'))
                            item['harness'] = harness_plan(item, config['harness'])
                            item['program_errors'] = sorted(set(item['program_errors']+item['harness']['errors']))
                            if len(json.dumps(item,ensure_ascii=False)) > config['execution']['max_sample_chars']:
                                raise ValueError('Exceeds configured annotation size budget; no truncation performed')
                            duplicate = dedup.add(item)
                            item['dedup'] = duplicate
                            if duplicate['exact_duplicate_of']:
                                counts['exact_duplicates'] += 1
                                append(paths['cleaning_log'],{'id':item['id'],**duplicate,'provenance':item['provenance']})
                                continue
                            if item['id'] in ids: raise ValueError('ID reused by different contents')
                            ids.add(item['id']); counts['kept'] += 1
                            append(paths['clean'], item)
                        except (ValueError, TypeError, KeyError) as exc:
                            counts['quarantined'] += 1
                            append(paths['cleaning_log'],{'source':spec['source'],'path':path,'row':index,'error':str(exc)})
                except ValueError as exc:
                    counts['file_errors'] += 1
                    append(paths['cleaning_log'],{'path':path,'file_error':str(exc)})
                if counts['scanned'] >= limit: break
            if counts['scanned'] >= limit: break
    finally:
        dedup.close()
    for name, path in paths.items(): path.replace(out/(name+'.jsonl'))
    save(out/'cleaning_summary.json',counts)
    return counts


def annotate_one(item, client):
    try:
        if item['program_errors']: raise ValueError('Structural errors require repair before LLM annotation')
        def call(stage, prompt, validator):
            return client.call(stage, item['id'], prompt, policy.SYSTEM, validator)
        inventory = call('inventory',policy.inventory_prompt(item),lambda v:policy.validate_inventory(v,item))
        audit = call('audit',policy.audit_prompt(item,inventory),lambda v:policy.validate_audit(v,item,inventory))
        review = call('review',policy.review_prompt(inventory,audit),lambda v:policy.validate_review(v,audit))
        reviewed = policy.reviewed_audit(audit,review)
        behaviors = call('behaviors',policy.behavior_prompt(item,inventory,reviewed),
                         lambda v:policy.validate_behaviors(v,item,reviewed) if hasattr(policy,'validate_behaviors') else policy.v2.validate_behaviors(v,item))
        result = policy.compute_result(item,inventory,audit,review,behaviors)
        result['raw_behaviors'] = behaviors
    except Exception as exc:
        result = {'id':item['id'],'source':item['source'],'status':'failed','error':str(exc),
                  'offline_quality_candidate':False,'training_ready':False}
    result.update(sample_hash=digest(item), policy_hash=client.policy_hash, harness=item['harness'],
                  human_review='pending',training_ready=False)
    return result


def review_entry(item, result, result_path):
    return {'id':item['id'], 'sample_hash':digest(item), 'result_hash':digest(result),
            'result_path':str(result_path.resolve()), 'source':item['source'],
            'quality_issues':result.get('quality_issues',[]), 'evidence_gaps':result.get('evidence_gaps',[]),
            'annotation_issues':result.get('annotation_issues',[]),
            'decision':None, 'reviewer':None, 'reason':None,
            'allowed_decisions':['accept','reject','relabel','rewrite']}


def bounded_map(fn, items, workers):
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        while True:
            while len(pending) < workers*2:
                try: item = next(iterator)
                except StopIteration: break
                pending.add(pool.submit(fn,item))
            if not pending: return
            done,pending = wait(pending,return_when=FIRST_COMPLETED)
            for future in done: yield future.result()


def annotate(config, out, input_path, limit):
    client = Client(config,out,policy_hash())
    def selected():
        for index,item in enumerate(records(input_path)):
            if index >= limit: break
            yield item
    def run(item): return item,annotate_one(item,client)
    queue = out/'human_review_queue.jsonl'; queue.write_text('')
    index_path = out/'result_index.jsonl'; index_path.write_text('')
    counts = {'samples':0,'scored':0,'offline_candidates':0,'training_ready':0}
    for item,result in bounded_map(run,selected(),config['execution']['workers']):
        path = out/'results'/(digest(item['id'])[:24]+'.json')
        save(path,{'sample':item,'result':result})
        append(queue,review_entry(item,result,path))
        append(index_path,{'id':item['id'],'path':str(path.resolve()),'status':result['status']})
        counts['samples'] += 1; counts['scored'] += result['status']=='scored'
        counts['offline_candidates'] += result['offline_quality_candidate']
    save(out/'annotation_summary.json',counts)
    return counts


def validate_rewrite(value, item):
    answer = value.get('final_answer')
    if not isinstance(answer,str) or not answer.strip() or '<tool_call>' in answer or '<think>' in answer:
        raise ValueError('Rewrite must contain only a visible answer, not tool calls or hidden reasoning')
    if answer.strip() == policy.v2.final_text(item).strip(): raise ValueError('Rewrite made no change')
    if not isinstance(value.get('evidence'),list) or not value['evidence']: raise ValueError('Rewrite needs located evidence')
    for ref in value['evidence']:
        if not policy.locate_across(item,ref): raise ValueError('Rewrite cites unlocated evidence')
    return value


def validate_counterfactual(value, item):
    original = policy.v2.final_text(item)
    span, replacement = value.get('original_span'),value.get('replacement')
    if not isinstance(span,str) or not span.strip(): raise ValueError('Mutation span must be nonempty')
    # Match whitespace variants only, retaining raw offsets for a single precise edit.
    # No entity/date/punctuation substitutions or fuzzy semantic matching are permitted.
    pattern=r'\s+'.join(re.escape(part) for part in re.split(r'\s+',span.strip()))
    matches=list(re.finditer(pattern,original))
    if len(matches)!=1: raise ValueError('Mutation span must be unique in final answer_to_edit')
    match=matches[0]
    if not isinstance(replacement,str) or not replacement.strip() or replacement==span: raise ValueError('Mutation must change one nonempty span')
    if any(t in replacement for t in ('<tool_call>','<think>')): raise ValueError('No tool/reasoning injection in counterfactual')
    if not isinstance(value.get('evidence'),list) or not value['evidence']: raise ValueError('Counterfactual requires source evidence')
    for index,ref in enumerate(value['evidence']):
        if not policy.locate_across(item,ref):
            raise ValueError(f'evidence[{index}].quote is not a contiguous observation substring: '+
                             repr(str(ref.get('quote',''))[:250])+'. Copy an exact passage; do not concatenate titles/snippets or insert ellipses.')
    value['edit_location']={'start':match.start(),'end':match.end(),'original_text':match.group(),
                            'match':'exact' if match.group()==span else 'unicode_whitespace'}
    return value


def validate_negative_review(value, item):
    if value.get('verdict') not in ('contradicted','insufficient'): raise ValueError('Invalid counterfactual review verdict')
    if not isinstance(value.get('reason'),str): raise ValueError('Missing review reason')
    if not isinstance(value.get('evidence'),list): raise ValueError('Missing review evidence')
    if value['verdict']=='contradicted' and not value['evidence']: raise ValueError('Contradiction needs evidence')
    for ref in value['evidence']:
        if not policy.locate_across(item,ref): raise ValueError('Reviewer cites unlocated evidence')
    return value


def variants(config,out,input_path,limit,kinds):
    client = Client(config,out,policy_hash()); counts={'parents':0,'rewrites':0,'counterfactuals':0,'failed':0,'training_ready':0}
    queue=out/'variant_review_queue.jsonl'
    failures=out/'variant_failures.jsonl'
    for index,item in enumerate(records(input_path)):
        if index>=limit: break
        counts['parents']+=1
        for kind in kinds:
            try:
                parent_result_path=out/'results'/(digest(item['id'])[:24]+'.json')
                parent_result=json.loads(parent_result_path.read_text())['result'] if parent_result_path.exists() else None
                if parent_result and parent_result['sample_hash']!=digest(item): raise ValueError('Parent audit is stale')
                data={'trajectory':item,'audit':parent_result}
                if kind=='rewrite':
                    prompt='''Rewrite ONLY the final visible answer to improve clarity and evidence grounding.
Do not change the question, prior actions, tool calls or observations. Do not invent missing facts.
Remove unsupported specificity or clearly acknowledge unresolved constraints; never repair a gap by memory.
Return {"final_answer":"...","evidence":[{"step_id":1,"quote":"exact tool observation"}],"changes":["..."]}.
This is a candidate, not an accepted training example. DATA:\n'''+json.dumps(data,ensure_ascii=False)
                    generated=client.call(kind,item['id'],prompt,policy.SYSTEM,lambda v:validate_rewrite(v,item))
                    child=copy.deepcopy(item);child['steps'][-1]['assistant']=generated['final_answer']
                else:
                    prompt='''Construct ONE minimally changed final-answer negative for contrastive evaluation.
Choose a unique original_span and replacement that becomes directly contradicted by existing tool evidence.
The original_span MUST be taken from final_answer_to_edit, not from the observations or reasoning.
Do not modify observations, question, tool calls, or invent a real source. Do not use outside knowledge.
Every quote must be one contiguous observation substring; never concatenate titles and snippets.
Return {"original_span":"...","replacement":"...","failure_type":"...","reason":"...",
"evidence":[{"step_id":1,"quote":"exact tool observation"}]}.
The original answer is NOT assumed correct; this only proposes a synthetic negative. DATA:\n'''+json.dumps({
    'question':item['question'],'final_answer_to_edit':policy.v2.final_text(item),
    'observations':[{'step_id':s['step_id'],'text':s['observation']} for s in item['steps']]},ensure_ascii=False)
                    generated=client.call(kind,item['id'],prompt,policy.SYSTEM,lambda v:validate_counterfactual(v,item))
                    child=copy.deepcopy(item)
                    location=generated['edit_location'];original=policy.v2.final_text(item)
                    child['steps'][-1]['assistant']=original[:location['start']]+generated['replacement']+original[location['end']:]
                child['lineage']={'parent_id':item['id'],'parent_sample_hash':digest(item),'operation':kind,
                    'synthetic':True,'policy_hash':client.policy_hash,'generation_hash':digest(generated)}
                child['id']=digest([item['id'],kind,child['steps']])[:24]
                child['content_hash']=digest({'question':child['question'],'steps':child['steps']})
                # Existing prior observations must remain byte-for-byte identical.
                assert [s['observation'] for s in child['steps']]==[s['observation'] for s in item['steps']]
                if kind=='rewrite':
                    result=annotate_one(child,client)
                    artifact={'sample':child,'result':result,'generation':generated}
                    path=out/'variants'/(child['id']+'.json');save(path,artifact)
                    counts['rewrites']+=1
                else:
                    prompt='''Independently verify the changed assertion against the supplied observations ONLY.
Do not trust the generator label, reference answer, or model memory. Is the changed assertion directly
contradicted by a located observation? Otherwise return insufficient. The unchanged answer may also have errors.
Return {"verdict":"contradicted|insufficient","reason":"...","evidence":[{"step_id":1,"quote":"exact text"}]}.
DATA:\n'''+json.dumps({'original_span':generated['original_span'],'changed_assertion':generated['replacement'],
                            'changed_final_answer':child['steps'][-1]['assistant'],
                            'observations':[{'step_id':s['step_id'],'text':s['observation']} for s in item['steps']]},ensure_ascii=False)
                    review=client.call('counterfactual_review',child['id'],prompt,policy.SYSTEM,lambda v:validate_negative_review(v,item))
                    result={'id':child['id'],'status':'synthetic_negative_candidate','sample_hash':digest(child),
                            'review':review,'training_ready':False,'human_review':'pending',
                            'preference_pair_ready':False,'parent_quality_candidate':bool(parent_result and parent_result.get('offline_quality_candidate')),
                            'note':'Negative-only candidate. Positive side needs independent acceptance; never export as positive SFT.'}
                    path=out/'variants'/(child['id']+'.json');save(path,{'sample':child,'result':result,'generation':generated})
                    counts['counterfactuals']+=1
            except Exception as exc:
                counts['failed']+=1;append(failures,{'parent_id':item['id'],'kind':kind,'error':str(exc)})
    queue.write_text('');available={'rewrite':0,'counterfactual':0}
    for path in sorted((out/'variants').glob('*.json')):
        artifact=json.loads(path.read_text())
        append(queue,review_entry(artifact['sample'],artifact['result'],path))
        available[artifact['sample']['lineage']['operation']]+=1
    counts['available_candidates']=available
    save(out/'variant_summary.json',counts)
    return counts


def finalize(out, decisions_path, replay_path):
    """Consume actual human decisions and externally-produced harness replay receipts."""
    decisions=list(records(decisions_path)); receipts={r['id']:r for r in records(replay_path)}
    export=out/'accepted_candidates.jsonl'; export.write_text('')
    rejected=out/'finalization_pending.jsonl';rejected.write_text('')
    counts={'accepted':0,'pending_or_rejected':0};seen=set()
    for decision in decisions:
        try:
            if decision['id'] in seen: raise ValueError('Duplicate human decision')
            seen.add(decision['id'])
            if decision.get('decision')!='accept': raise ValueError('Not accepted by human reviewer')
            if not all(isinstance(decision.get(k),str) and decision[k].strip() for k in ('reviewer','reason')):
                raise ValueError('Human reviewer and rationale required')
            artifact=json.loads(Path(decision['result_path']).read_text());sample=artifact['sample'];result=artifact['result']
            if sample['id']!=decision['id'] or digest(sample)!=decision['sample_hash'] or digest(result)!=decision['result_hash']:
                raise ValueError('Stale or mismatched human review')
            if sample.get('lineage',{}).get('operation')=='counterfactual': raise ValueError('Synthetic negatives cannot be positive SFT')
            if not result.get('offline_quality_candidate'): raise ValueError('Unresolved annotation/quality items; relabel before acceptance')
            replay=receipts.get(sample['id'],{})
            if replay.get('sample_hash')!=digest(sample) or replay.get('status')!='passed' or not replay.get('artifact_path'):
                raise ValueError('Matching successful harness replay receipt required')
            replay_artifact=Path(replay['artifact_path'])
            if not replay_artifact.is_file() or hashlib.sha256(replay_artifact.read_bytes()).hexdigest()!=replay.get('artifact_sha256'):
                raise ValueError('Replay artifact is missing or hash does not match')
            # Keep auditable source trajectory, adaptation plan and replay receipt; no silent tool rewrites.
            append(export,{'sample':sample,'annotation':result,'human_review':decision,'replay':replay,'curation_accepted':True,'training_ready':False,
                           'remaining':'Export and re-audit the actual adapted/replayed messages before training; original legacy trace is retained for provenance.'})
            counts['accepted']+=1
        except (ValueError,KeyError,OSError) as exc:
            append(rejected,{'id':decision.get('id'),'reason':str(exc)});counts['pending_or_rejected']+=1
    save(out/'finalization_summary.json',counts);return counts


def main():
    global policy
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','annotate','variants','finalize'])
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.example.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--inputs',type=Path,help='Override config inputs with a dataset manifest JSON array')
    parser.add_argument('--limit',type=int,default=4,help='Explicit budget cap, not a production sample selection strategy')
    parser.add_argument('--kinds',nargs='+',choices=['rewrite','counterfactual'],default=['rewrite','counterfactual'])
    parser.add_argument('--decisions',type=Path);parser.add_argument('--replay',type=Path)
    args=parser.parse_args();config=json.loads(args.config.read_text());out=args.output;out.mkdir(parents=True,exist_ok=True)
    if config.get('policy_version') == 'evidence-id-v4':
        import policy_v4
        policy = policy_v4
    elif config.get('policy_version') not in (None, 'offline-pilot-v3.1-general'):
        raise ValueError('Unknown policy_version')
    if args.inputs:config['inputs']=json.loads(args.inputs.read_text())
    if args.limit<1 or config['execution']['workers']<1: raise SystemExit('Positive limit/workers required')
    batch_limit=config['execution'].get('max_batch_samples')
    if batch_limit is not None and args.limit>batch_limit:
        raise SystemExit('Requested sample limit exceeds configured max_batch_samples='+str(batch_limit))
    save(out/'config.redacted.json',public_config(config));save(out/'policy.json',{'code_hash':policy_hash(),'rubric':policy.VERSION,'deltas':policy.v2.DELTAS})
    if args.command=='prepare':summary=prepare(config,out,args.limit)
    elif args.command=='annotate':summary=annotate(config,out,args.input or out/'clean.jsonl',args.limit)
    elif args.command=='variants':summary=variants(config,out,args.input or out/'clean.jsonl',args.limit,args.kinds)
    else:
        if not args.decisions or not args.replay:parser.error('finalize requires --decisions and --replay')
        summary=finalize(out,args.decisions,args.replay)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
