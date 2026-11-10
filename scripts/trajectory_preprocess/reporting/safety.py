"""Read-only per-dataset safety reports and deterministic, redacted audit samples.

No model requests, source edits, or changes to safety decisions. Audit selection
is stratified, not an unbiased estimate of population false-positive rates.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlsplit,parse_qsl

from client import save
from safety_context import build_context
from safety_rules import Scanner, decoded, URL
from .examples import AUDIT_EMAIL, safe_window


def rule_group(rule):
    if rule in {'benchmark.question_overlap', 'benchmark.current_question_answer'}:
        return 'question_answer_overlap'
    if rule.startswith('benchmark.'):
        return 'benchmark_resource'
    return rule.split('.', 1)[0]


def write_matrix(path, datasets, counts, state):
    rules=sorted({rule for _, outcome, rule in counts if outcome==state})
    with path.open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['dataset','unique_trajectories',*rules])
        for dataset, totals in sorted(datasets.items()):
            writer.writerow([dataset,totals[state],*[counts[dataset,state,rule] for rule in rules]])


def report(run,keyfile,per_stratum=2):
    manifest=json.loads((run/'manifest.json').read_text())
    scanner=Scanner(json.loads((run/'policy.json').read_text()),json.loads((run/'contract.json').read_text()),keyfile.read_text().strip().encode())
    if scanner.policy_hash!=manifest['policy_hash'] or scanner.fingerprint('key-id')[:16]!=manifest['key_id']:
        raise ValueError('Policy/code/key does not match scan')
    source=Path(manifest['input']['path']);origins={}
    with (source.parent/'origins.jsonl').open() as f:
        for line in f:
            origin=json.loads(line);key=scanner.fingerprint(str(origin['canonical_id']))
            meta={k:origin.get(k) for k in ('source','agent_role','source_artifact','source_line_zero_based')}
            if key in origins and origins[key]['source']!=meta['source']:
                raise ValueError('Cross-source aliases require explicit attribution')
            origins.setdefault(key,meta)
    db=sqlite3.connect('file:'+str(run/'state.sqlite')+'?mode=ro',uri=True,timeout=30)
    datasets=collections.defaultdict(collections.Counter)
    reasons=collections.Counter();causes=collections.Counter();group_causes=collections.Counter();role_counts=collections.Counter();reason_origins=collections.Counter();strata={};coverage=collections.Counter();rows=0
    # One read transaction binds the report to a consistent checkpoint.
    db.execute('BEGIN')
    checkpoint=json.loads(db.execute('SELECT state FROM checkpoint WHERE id=1').fetchone()[0])
    for (text,) in db.execute('SELECT result FROM decisions ORDER BY row_index'):
        result=json.loads(text);rows+=1;key=result['record_key'];meta=origins.get(key,{'source':'UNRESOLVED','agent_role':'unknown'})
        dataset=meta['source'];state=result['state'];counts=datasets[dataset]
        counts['scanned']+=1;counts[state]+=1
        role_counts[(dataset,meta['agent_role'],state)]+=1
        coverage['complete' if result['coverage_complete'] else 'incomplete']+=1
        rule_ids={f['rule_id'] for f in result['findings']}
        for rule in rule_ids:reasons[(dataset,state,rule)]+=1
        for rule in {f['rule_id'] for f in result['findings'] if f['disposition']==state}:
            causes[(dataset,state,rule)]+=1
        for group in {rule_group(f['rule_id']) for f in result['findings'] if f['disposition']==state}:
            group_causes[(dataset,state,group)]+=1
        for rule,origin,disposition in {(f['rule_id'],f.get('origin','coverage'),f['disposition']) for f in result['findings']}:
            reason_origins[(dataset,state,rule,origin,disposition)]+=1
        strata_keys=[(dataset,state,rule) for rule in rule_ids] if rule_ids else [(dataset,state,meta['agent_role'])]
        for stratum in strata_keys:
            rank=hashlib.sha256(('safety-audit-v1|'+key+'|'+str(stratum)).encode()).hexdigest()
            chosen=strata.setdefault(stratum,[])
            if len(chosen)<per_stratum or rank<chosen[-1][0]:
                chosen.append((rank,key,result,meta));chosen.sort(key=lambda p:p[0]);del chosen[per_stratum:]
    db.close()
    if rows!=checkpoint['next_row']:raise ValueError('Rows/checkpoint mismatch')
    destination=run/('report' if checkpoint['status']=='completed' else 'report-preview');destination.mkdir(exist_ok=True)
    columns=['dataset','scanned','pass_rules','pass_rate_percent','review','quarantine']
    dataset_rows=[]
    for name,c in sorted(datasets.items()):
        dataset_rows.append({'dataset':name,'scanned':c['scanned'],'pass_rules':c['pass_rules'],
                'pass_rate_percent':round(100*c['pass_rules']/c['scanned'],4),'review':c['review'],'quarantine':c['quarantine']})
    with (destination/'dataset_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(dataset_rows)
    with (destination/'agent_role_counts.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['dataset','agent_role','state','trajectories'])
        writer.writerows([*k,v] for k,v in sorted(role_counts.items()))
    reason_rows=[{'dataset':d,'state':s,'rule_id':r,'trajectories':n} for (d,s,r),n in sorted(reasons.items())]
    with (destination/'reasons_by_dataset.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','state','rule_id','trajectories']);writer.writeheader();writer.writerows(reason_rows)
    with (destination/'reasons_by_origin.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['dataset','state','rule_id','origin','finding_disposition','trajectories'])
        writer.writerows([*k,v] for k,v in sorted(reason_origins.items()))
    cause_rows=[{'dataset':d,'state':s,'rule_id':r,'trajectories':n} for (d,s,r),n in sorted(causes.items())]
    with (destination/'disposition_reasons_by_dataset.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','state','rule_id','trajectories']);writer.writeheader();writer.writerows(cause_rows)
    for state in ('quarantine','review'):
        write_matrix(destination/(state+'_rule_matrix.csv'),datasets,causes,state)
        write_matrix(destination/(state+'_category_matrix.csv'),datasets,group_causes,state)
    selected={};selected_strata=collections.defaultdict(list)
    for stratum,chosen in strata.items():
        for _,key,result,meta in chosen:
            selected[key]=(result,meta);selected_strata[key].append(stratum)
    samples=[]
    with source.open('rb') as f:
        for key,(result,meta) in sorted(selected.items(),key=lambda p:p[1][0]['source_row']):
            f.seek(result['offset_start']);item=json.loads(f.read(result['offset_end']-result['offset_start']))
            from client import digest
            if digest(item)!=result['content_hash']:raise ValueError('Audit source changed')
            views,actions,issues=build_context(item);lookup={}
            for v in views:
                lookup[scanner.fingerprint(v.pointer+str(v.derived))[:24]]=v.text
                value=decoded(v.text,scanner.policy['max_decode_layers'])
                if value!=v.text:lookup[scanner.fingerprint(v.pointer+'/decoded'+str(True))[:24]]=value
            excerpts=[]
            wanted={s[2] for s in selected_strata[key]};seen=set()
            for finding in result['findings']:
                rule=finding['rule_id']
                if rule not in wanted or rule in seen:continue
                seen.add(rule);text=lookup.get(finding.get('view_id'),'')
                a=finding.get('start',0);b=finding.get('end',0)
                # Hide the exact match for credentials/PII even if another scrub rule misses it.
                hit='[MATCH_WITHHELD]' if finding['category'] in ('credential','privacy') else scanner.scrub(text[a:b])[:200]
                excerpt=safe_window(text,max(0,a-220),a,scanner)+' «'+hit+'» '+safe_window(text,b,min(len(text),b+220),scanner)
                url_context=[]
                for match in URL.finditer(text):
                    if match.start()<b and match.end()>a:
                        try:
                            parsed=urlsplit(match[0])
                            url_context.append({'host':parsed.hostname,'query_parameter_names':sorted({k for k,v in parse_qsl(parsed.query)}),
                                                'has_userinfo':bool(parsed.username or parsed.password),'scheme':parsed.scheme})
                        except ValueError:pass
                excerpts.append({'rule_id':rule,'origin':finding.get('origin'),'pointer':finding.get('pointer'),
                                 'finding_id':finding['finding_id'],'match_length':b-a,'excerpt':excerpt,'details':finding.get('details'),
                                 'url_context':url_context})
            sample={'audit_id':'A'+str(len(samples)+1).zfill(3),'record_key':key,'source_row':result['source_row'],
                    'content_hash':result['content_hash'],'policy_hash':result['policy_hash'],**meta,'state':result['state'],
                    'rule_ids':sorted({x['rule_id'] for x in result['findings']}),'selected_strata':selected_strata[key],
                    'findings':excerpts,'schema_issues':issues,'call_counts':dict(collections.Counter(a['name'] for a in actions)),
                    'message_count':len(item['messages']),'assessment':'pending_context_review'}
            if not result['findings']:
                sample['task_excerpt']=scanner.scrub(next((v.text for v in views if v.kind=='task'),''))[:400]
                sample['final_excerpt']=scanner.scrub(next((v.text for v in reversed(views) if v.kind=='assistant'),''))[-500:]
            samples.append(json.loads(AUDIT_EMAIL.sub('[REDACTED_EMAIL]',scanner.scrub(json.dumps(sample,ensure_ascii=False)))))
    save(destination/'audit_samples.json',samples)
    totals=collections.Counter()
    for c in datasets.values():totals.update(c)
    if totals['scanned']!=sum(totals[k] for k in ('pass_rules','review','quarantine')):
        raise ValueError('Unexpected state or non-exclusive state counts')
    summary={'status':checkpoint['status'],'input':str(source),'rows':rows,'expected_unified_candidates':len(origins),
             'totals':dict(totals),'coverage':dict(coverage),'elapsed_seconds':checkpoint['elapsed_seconds'],
             'llm_calls':0,'api_cost':0,'policy_hash':manifest['policy_hash'],'datasets':dataset_rows,
             'reasons':reason_rows,'disposition_reasons':cause_rows,'sample_count':len(samples),'sampling':'lowest SHA256 ranks, up to two per dataset/state/rule; pass rows additionally stratified by agent role; not a prevalence estimate',
             'disposition_categories':[{'dataset':d,'state':s,'category':r,'trajectories':n} for (d,s,r),n in sorted(group_causes.items())],
             'counting':'reason counts are unique trajectories per rule and state; multiple rules overlap; category matrices deduplicate within each category and attribute only findings that caused the final safety state',
             'decisions_modified':False,'exported_training_data':False,'training_ready':False,
             'report_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(destination/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','rows','totals','sample_count','elapsed_seconds')},ensure_ascii=False),flush=True)
