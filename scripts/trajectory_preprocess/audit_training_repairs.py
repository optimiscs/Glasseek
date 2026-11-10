"""Read-only stratified audit of scored canonical trajectories; never rewrites training data."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import copy
import hashlib
import secrets
import sqlite3
import zlib

from glasseek_shared.trajectory_release import digest, payload, runtime_schemas, schemas, validate_messages
import normalize_training_text

LEGACY = re.compile(r'\b(?:functions\.(?:search|visit_urls|visit|browse|python)|browser\.(?:open|search|find|click)|web\.(?:run|search_query)|browse_page|visit_page|python_interpreter|search_engine)\b')
ALIASES = {'functions.search':'web_search','functions.visit':'web_extract','functions.visit_urls':'web_extract'}


def unpack(raw):
    return json.loads(zlib.decompress(raw))


def final_body(messages):
    if not messages or messages[-1]['role'] != 'assistant':
        return ''
    return re.sub(r'^\s*<think>.*?</think>', '', messages[-1]['content'], count=1, flags=re.S)


def inspect(row, result, tools):
    findings=[]
    def add(rule, route, message=None, excerpt='', detail=''):
        findings.append({'rule':rule,'route':route,'message':message,'excerpt':excerpt[:800],'detail':detail})
    messages=row['messages']
    try:
        allowed={k:v for k,v in tools.items() if row.get('agent_role')!='child' or k not in ('execute_code','delegate_task')}
        validate_messages(messages, allowed)
        if schemas(row['tools'])!=allowed:raise ValueError('declared_tool_schema_mismatch')
        protocol_ok=True
    except (ValueError,KeyError,TypeError) as error:
        protocol_ok=False;add('protocol_or_association','isolate',detail=str(error))
    normalized=normalize_training_text.normalize(row, tool_schemas=tools)
    for event in normalized['text_normalization']['events']:
        if event['rule'] in ('response_tool_label','benchmark_user_wrapper'):
            add(event['rule'],'replace',event['message'])
    counts=Counter()
    for i,m in enumerate(messages):
        if m['role']=='system':continue
        text=m['content']
        observation=text.lstrip().startswith('<tool_response>')
        if m['role']=='user' and not observation:
            if i and messages[i-1]['role']=='user' and text.startswith('Solve the following task:') and text.removeprefix('Solve the following task:').strip()==messages[i-1]['content'].strip():
                add('duplicate_adjacent_question','replace',i,detail='Exact duplicate after known wrapper; rebuild message-level supervision bindings if deduplicated.')
            if text.strip()=='Please continue to solve the task.':
                add('continuation_reminder','review',i,detail='Do not delete genuine user turns; require harness-origin provenance before removing repeated prompts.')
            if 'RESEARCH STATE SUMMARY' in text or re.search(r'\bprev_state\b',text):
                add('historical_context','retain_context',i,detail='Keep as continuation; do not rewrite as unsupported from-scratch research.')
            if re.search(r'FINAL ANSWER\s*:',text,re.I):
                add('explicit_final_prefix','replace',i,excerpt=re.search(r'.{0,100}FINAL ANSWER.{0,160}',text,re.I).group(),detail='Preserve requested prefix inside single answer wrapper; never blindly remove it.')
        if m['role']!='assistant':continue
        if text.count('<think>')!=text.count('</think>'):
            empty=bool(re.match(r'^\s*<think>\s*<tool_call>',text)) and text.count('<think>')==1 and text.count('</think>')==0
            duplicate=bool(re.match(r'^\s*<think>\s*<think>',text)) and text.count('<think>')==2 and text.count('</think>')==1
            kind='empty_think' if empty else 'duplicate_think' if duplicate else 'nonempty_think'
            counts[kind+'_messages']+=1
            add('empty_think_residue' if empty else 'duplicate_think_opening' if duplicate else 'reasoning_boundary_missing','replace' if (empty or duplicate) and protocol_ok else 'isolate',i,text[:250],detail='Audit recommendation only: remove only empty/redundant opening markup; keep nonempty reasoning boundary failures isolated.')
        for match in LEGACY.finditer(text):
            span=next((x for x in re.finditer(r'<think>(.*?)</think>',text,re.S) if x.start(1)<=match.start()<x.end(1)),None)
            quoted=False
            if span:
                quoted=any(x.start()<=match.start()-span.start(1)<x.end() for x in re.finditer(r'```.*?```|`[^`]*`|"[^"\n]*"|(?<!\w)\'[^\'\n]*\'(?!\w)',span.group(1),re.S))
            clear=span is not None and not quoted and match.group() in ALIASES
            add('legacy_tool_reasoning_clear' if clear else 'legacy_tool_reference_uncertain','replace' if clear else 'isolate',i,text[max(0,match.start()-100):match.end()+160])
    visible=final_body(messages)
    if visible:
        child=row.get('agent_role')=='child'
        required_report=any(m['role']=='user' and not m['content'].startswith('<tool_response>') and '<report>' in m['content'] for m in messages)
        if child:
            add('child_role_context','retain_context',detail='Keep child task/context/report semantics and bundle split; not a stand-alone parent answer.')
            if required_report and not re.fullmatch(r'\s*<report>.*</report>\s*',visible,re.S):
                add('required_report_envelope_missing','replace',len(messages)-1,detail='Restore the explicitly required child report envelope, preserving full report and references.')
        elif re.search(r'</?(?:answer|explanation)>',visible):
            if re.search(r'<think>|</think>|<tool_call>',visible):add('final_boundary_missing','isolate',len(messages)-1,visible[:350])
            elif any(re.search(r'</?(?:answer|explanation)>',part) for part in re.findall(r'```.*?```|`[^`]*`',visible,re.S)):
                add('literal_answer_tag','isolate',len(messages)-1,detail='Literal/code tags must not be stripped as envelope markup.')
            elif not re.fullmatch(r'\s*<answer>(?:(?!</?answer>|</?explanation>).)*</answer>\s*',visible,re.S):
                add('merge_or_repair_answer_envelope','replace',len(messages)-1,detail='Merge/repair wrappers in original order, preserving body and task-mandated prefix.')
        else:add('plain_answer_wrapper','replace',len(messages)-1,detail='Parent answer: add a single outer answer wrapper without changing body.')
        if re.search(r'^\s*(?:<answer>\s*)?### Prediction\s*\n',visible):
            add('prediction_heading','replace',len(messages)-1,detail='Remove known leading presentation heading only if no user requirement requests it.')
        if re.search(r'(?im)^\s*(?:\*\*)?FINAL ANSWER\s*:',visible) and not any(x['rule']=='explicit_final_prefix' for x in findings):
            add('unrequested_final_prefix','replace',len(messages)-1)
        if re.search(r'(?:【\d+[^】]*】|cite|turn\d+|\[\[\d+\]\])',visible):
            add('citation_reference_resolution','review',len(messages)-1,detail='Citation tokens need exact source mapping; appearance alone is not an error.')
        if re.search(r'(?:as an ai|language model|I (?:cannot|can\x27t|am unable to) (?:browse|access|search)|作为.{0,12}(?:模型|AI)|无法(?:联网|访问互联网))',visible,re.I):
            add('capability_disclaimer','review',len(messages)-1,detail='Check task relevance and consistency with recorded tools before constrained rewriting.')
    criteria=result.get('judgment',{}).get('criteria',{})
    gates=result.get('judgment',{}).get('gates',{})
    # These are model suggestions, not confirmed defects. They must be adjudicated.
    for key,c in criteria.items():
        score=c.get('score')
        if isinstance(score,(int,float)) and score<90:
            add('rubric_rewrite_candidate','llm_candidate',detail=json.dumps({'criterion':key,'score':score,'reason':c.get('reason'),'refs':c.get('refs')},ensure_ascii=False))
    return {'protocol_ok':protocol_ok,'normalizer_holds':normalized['text_normalization']['holds'],'findings':findings,'message_counts':dict(counts),'assistant_turns':sum(m['role']=='assistant' for m in messages),'agent_role':row.get('agent_role','parent'),'parent_id':row.get('parent_id'),'final_chars':len(visible),'content_sha256':digest(payload(row))}


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'sample.jsonl').exists():raise ValueError('Audit sample already exists; use a fresh output directory')
    source=sqlite3.connect(f'file:{args.source}/batch.sqlite?mode=ro',uri=True)
    backfill=sqlite3.connect(f'file:{args.backfill}/batch.sqlite?mode=ro',uri=True)
    difficulty={i:json.loads(d) for i,d in backfill.execute("SELECT id,json_extract(payload,'$.difficulty') FROM results WHERE state='completed'")}
    exclude={json.loads(line)['id'] for line in args.exclude.open()}
    pool=defaultdict(lambda:defaultdict(list))
    for ident,dataset,status,score,raw in source.execute("SELECT id,json_extract(result,'$.source'),json_extract(result,'$.status'),json_extract(result,'$.score'),json_extract(result,'$.difficulty') FROM results WHERE state='completed'"):
        d=difficulty.get(ident) or (json.loads(raw) if raw else {})
        if ident in exclude or status!='rankable' or score is None or d.get('score') is None or float(d.get('confidence') or 0)<70:continue
        dl='easy' if float(d['score'])<30 else 'medium' if float(d['score'])<70 else 'hard'
        ql='high_95_100' if score>=95 else 'middle_85_95' if score>=85 else 'lower_below85'
        pool[dataset][(dl,ql)].append({'id':ident,'source':dataset,'quality':score,'difficulty':float(d['score']),'confidence':float(d['confidence']),'difficulty_band':dl,'quality_band':ql})
    rng=random.Random(args.seed);selected=[];population={}
    for dataset,groups in sorted(pool.items()):
        population[dataset]={'rows':sum(map(len,groups.values())),'strata':{'/'.join(k):len(v) for k,v in groups.items()}}
        for group in groups.values():group.sort(key=lambda r:r['id']);rng.shuffle(group)
        used=Counter();count=0
        while count<args.per_dataset:
            available=[k for k,g in groups.items() if used[k]<len(g)]
            if not available:break
            key=min(available,key=lambda k:(used[k],k));selected.append(groups[key][used[key]]);used[key]+=1;count+=1
    tools=runtime_schemas()
    results=[]
    with (args.output/'sample.jsonl').open('w') as full,(args.output/'findings.jsonl').open('w') as findings:
        for meta in selected:
            raw=source.execute('SELECT canonical FROM inputs WHERE id=?',(meta['id'],)).fetchone()[0];canonical=unpack(raw)
            result=json.loads(source.execute('SELECT result FROM results WHERE id=?',(meta['id'],)).fetchone()[0])
            record={**meta,**inspect(canonical,result,tools)}
            full.write(json.dumps({'selection':meta,'canonical':canonical,'judgment':result.get('judgment'),'rubric_status':result['status']},ensure_ascii=False)+'\n')
            findings.write(json.dumps(record,ensure_ascii=False)+'\n');results.append(record)
    by_dataset={}
    for dataset in sorted(pool):
        rs=[r for r in results if r['source']==dataset];rules=Counter();routes=Counter()
        for r in rs:
            rules.update({f['rule'] for f in r['findings']});routes.update({f['route'] for f in r['findings']})
        by_dataset[dataset]={'rows':len(rs),'difficulty':dict(Counter(r['difficulty_band'] for r in rs)),'quality':dict(Counter(r['quality_band'] for r in rs)),'agent_roles':dict(Counter(r['agent_role'] for r in rs)),'rules':dict(rules),'routes_overlapping':dict(routes),'protocol_pass':sum(r['protocol_ok'] for r in rs)}
    report={'seed':args.seed,'sample_size':len(selected),'excluded_previous_ids':len(exclude),'population':population,'datasets':by_dataset,'rule_counts':dict(Counter(f for r in results for f in {x['rule'] for x in r['findings']})),'message_counts':dict(sum((Counter(r['message_counts']) for r in results),Counter())),'notes':['Read-only audit, no LLM calls and no edits to training data.','Balanced strata are not population prevalence estimates.','llm_candidate is a rubric-based screening hint, not a verified defect or permission to fabricate evidence.']}
    (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


def verify_cleaned(args):
    """Check every sampled ID, including quarantines, against its original text."""
    from safety_rules import Scanner
    from glasseek_shared.curation.supervision import validate_plan
    args.output.mkdir(parents=True, exist_ok=False)
    original = [json.loads(line)['canonical'] for line in args.sample.open()]
    routed = {}
    for route in ('clean', 'held'):
        for line in (args.verify_cleaned / (route + '.candidates.jsonl')).open():
            row = json.loads(line)
            if row['id'] in routed:
                raise ValueError('duplicate_output_id:' + row['id'])
            routed[row['id']] = (route, row)
    if len(original) != len(routed) or {r['id'] for r in original} != set(routed):
        raise ValueError('sample_output_id_mismatch')
    tools = runtime_schemas()
    policy = json.loads(Path(__file__).with_name('safety_policy.json').read_text())
    contract = json.loads(args.contract.read_text())
    scanner = Scanner(policy, contract, secrets.token_bytes(32))
    results, datasets, check_counts = [], defaultdict(Counter), Counter()
    # This comparison removes only explicitly authorized presentation differences.
    # All remaining characters, including whitespace and order, must match.
    def reasoning(text, rules):
        if rules & {'empty_think_residue', 'duplicate_think_opening'}:
            text = text.replace('<think>', '', 1)
        if 'assistant_tool_reference' in rules:
            def rename(match):
                body = match[1]
                for old, new in ALIASES.items():
                    body = normalize_training_text._unprotected_sub(
                        r'(?<![\w.])' + re.escape(old) + r'(?!\w|\.\w)', new, body)
                return '<think>' + body + '</think>'
            text = re.sub(r'<think>(.*?)</think>', rename, text, flags=re.S)
        return text
    def final_parts(text):
        thought = re.match(r'\s*<think>.*?</think>', text, re.S)
        end = thought.end() if thought else 0
        return text[:end], re.sub(r'</?(?:answer|explanation|report)>', '', text[end:])
    with (args.output / 'per-trajectory.jsonl').open('w') as audit, (args.output / 'safety.jsonl').open('w') as safety_file:
        for original_row in original:
            route, row = routed[original_row['id']]
            info = row['text_normalization']
            checks, errors = {}, []
            def check(name, condition):
                checks[name] = checks.get(name, True) and bool(condition)
                if not condition and name not in errors:
                    errors.append(name)
            old_messages, messages = original_row['messages'], row['messages']
            kept = info['message_index_map']
            check('index_map', len(kept) == len(messages) and kept == sorted(set(kept))
                  and all(0 <= i < len(old_messages) for i in kept))
            dropped = set(range(len(old_messages))) - set(kept)
            for index in dropped:
                m = old_messages[index]
                check('only_exact_duplicate_question_removed', index > 0 and m['role'] == 'user'
                      and old_messages[index-1]['role'] == 'user'
                      and m['content'].startswith('Solve the following task:')
                      and m['content'][len('Solve the following task:'):].strip() == old_messages[index-1]['content'].strip())
            check('mask_binding', row.get('message_loss_mask') == [original_row['message_loss_mask'][i] for i in kept])
            check('source_metadata', all(row.get(k) == original_row.get(k) for k in
                  ('id', 'source', 'tools', 'provenance', 'agent_role', 'parent_id', 'bundle_id', 'split_group_id')))
            for new_index, old_index in enumerate(kept):
                before, after = old_messages[old_index], messages[new_index]
                rules = {e['rule'] for e in info['events'] if e.get('message') == new_index}
                before_meta, after_meta = copy.deepcopy(before), copy.deepcopy(after)
                before_meta.pop('content'); after_meta.pop('content')
                for meta in (before_meta, after_meta):
                    if 'search_return_audit' in meta:
                        meta['search_return_audit'].pop('rendered_response_sha256', None)
                check('message_metadata_and_call_links', before_meta == after_meta)
                if normalize_training_text.is_observation(before):
                    expected = before['content']
                    if 'response_tool_label' in rules:
                        expected = re.sub(r'^(\s*<tool_response>\s*Tool:\s*)(search|visit_urls)(\s*\n)',
                                          lambda m: m[1] + {'search':'web_search','visit_urls':'web_extract'}[m[2]] + m[3], expected)
                    check('observation_body_unchanged', expected == after['content'])
                    if 'search_return_audit' in after:
                        check('response_hash', after['search_return_audit']['rendered_response_sha256']
                              == hashlib.sha256(after['content'].encode()).hexdigest())
                elif before['role'] in ('system', 'developer', 'user'):
                    expected = before['content']
                    if 'benchmark_user_wrapper' in rules:
                        expected = expected.split('\nQuestion:\n', 1)[1]
                    check('instructions_and_context_preserved', after['content'] == expected)
                else:
                    check('tool_call_xml_unchanged', re.findall(r'<tool_call>.*?</tool_call>', before['content'], re.S)
                          == re.findall(r'<tool_call>.*?</tool_call>', after['content'], re.S))
                    check('quoted_and_code_spans_unchanged', normalize_training_text.PROTECTED.findall(before['content'])
                          == normalize_training_text.PROTECTED.findall(after['content']))
                    if not normalize_training_text._thought_well_formed(before['content']) and not rules & {'empty_think_residue','duplicate_think_opening'}:
                        check('ambiguous_reasoning_preserved', before['content'] == after['content'])
                    expected = reasoning(before['content'], rules)
                    if new_index != len(messages)-1:
                        check('reasoning_only_authorized_edits', after['content'] == expected)
                    else:
                        old_head, old_body = final_parts(expected)
                        new_head, new_body = final_parts(after['content'])
                        if 'prediction_heading' in rules:
                            old_body = re.sub(r'\A(\s*)### Prediction[ \t]*\r?\n', r'\1', old_body, count=1)
                        if 'final_answer_prefix' in rules:
                            old_body = normalize_training_text._unprotected_sub(normalize_training_text.FINAL_PREFIX, r'\g<space>', old_body)
                        check('final_body_and_order_preserved', old_body == new_body and old_head == new_head)
                        if 'citation_source_mapping_unverified' in info['holds']:
                            check('unresolved_citations_preserved', re.findall(r'【[^】]+】',after['content'])
                                  == re.findall(r'【[^】]+】',before['content']))
            if row.get('supervision') is not None:
                try:
                    validate_plan(row['supervision'], messages)
                    check('supervision_binding', True)
                except ValueError:
                    check('supervision_binding', False)
            try:
                validate_messages(messages, tools)
                check('protocol', True)
            except ValueError:
                check('protocol', False)
            second = normalize_training_text.normalize(row, tool_schemas=tools)
            check('idempotent', digest(payload(second)) == digest(payload(row))
                  and second.get('message_loss_mask') == row.get('message_loss_mask')
                  and second['text_normalization']['holds'] == info['holds'])
            check('routing', (route == 'held') == bool(info['holds']))
            check('no_acceptance_fabricated', not row['training_ready'] and not row['harness_replayed'])
            scan = scanner.scan(row)
            check('safety_pass', scan['state'] == 'pass_rules' and scan['coverage_complete'])
            safety_file.write(json.dumps({'id':row['id'], 'content_sha256':digest(payload(row)), **scan}, ensure_ascii=False)+'\n')
            result = {'id':row['id'], 'source':row['source'], 'route':route, 'holds':info['holds'],
                      'changed':digest(payload(row)) != digest(payload(original_row)),
                      'events':info['events'], 'checks':checks, 'errors':errors}
            audit.write(json.dumps(result, ensure_ascii=False)+'\n')
            results.append(result)
            datasets[row['source']][route] += 1
            check_counts.update(k for k, ok in checks.items() if ok)
            if len(results) % 36 == 0:
                print(f'Verified {len(results)}/{len(original)} trajectories', flush=True)
    summary = {'rows':len(results), 'verified':sum(not r['errors'] for r in results),
               'changed':sum(r['changed'] for r in results), 'datasets':dict(datasets),
               'checks_passed_rows':dict(check_counts),
               'failures':[{'id':r['id'],'errors':r['errors']} for r in results if r['errors']],
               'held':[{'id':r['id'],'source':r['source'],'holds':r['holds']} for r in results if r['route']=='held'],
               'training_ready':False, 'harness_replayed':False}
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary['failures']:
        raise ValueError('trajectory_verification_failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--backfill',type=Path);p.add_argument('--exclude',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--per-dataset',type=int,default=36);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--verify-cleaned',type=Path);p.add_argument('--sample',type=Path)
    p.add_argument('--contract',type=Path,default=Path('data/processed/scientific-recovery-20261109/unified/tools.json'))
    args=p.parse_args()
    if args.verify_cleaned:
        if not args.sample:p.error('--verify-cleaned requires --sample')
        verify_cleaned(args)
    else:
        if not all((args.source,args.backfill,args.exclude)):p.error('Sampling requires --source, --backfill and --exclude')
        run(args)
