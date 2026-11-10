"""Finalize an offline pilot and its dependency-aware acceptance ledger."""
import collections
import json
import sqlite3
from .io import read, save, rows, sha, csvout

def baseline_decisions(baseline):
    pilot=baseline/'final/decisions.jsonl'
    if pilot.exists():
        return {r['id']:{'source':r['source'],'state':r['state'],'reasons':r['reasons']} for r in rows(pilot)}
    outcomes={}
    for r in rows(baseline/'intake/held.jsonl'):
        outcomes[r['id']]={'source':r['source'],'state':'review','reasons':r['reasons']}
    for r in rows(baseline/'intake/duplicates.jsonl'):
        outcomes[r['id']]={'source':r['source'],'state':'duplicate','reasons':['exact_duplicate']}
    for name,state in [('held','review'),('duplicates','duplicate')]:
        for r in rows(baseline/'searchswarm'/f'{name}.jsonl'):
            outcomes[r['id']]={'source':'SearchSwarm','state':state,'reasons':r.get('reasons',['exact_duplicate'])}
    db=sqlite3.connect('file:'+str(baseline/'decisions.sqlite')+'?mode=ro',uri=True)
    for ident,source,state,reasons in db.execute('SELECT id,source,state,reasons FROM decisions'):
        outcomes[ident]={'source':source,'state':state,'reasons':json.loads(reasons)}
    db.close()
    for r in rows(baseline/'unified/exact_duplicates.jsonl'):
        outcomes[r['source_id']]={'source':r['source'],'state':'duplicate','reasons':['exact_duplicate']}
    return outcomes


def report(out,baseline):
    from client import digest
    from glasseek_shared.trajectory_release import validate_messages
    from sample_xml_audit import syntax_check
    old=baseline_decisions(baseline);decisions={};reason_counts=collections.Counter();diagnostics=collections.Counter()
    def reason(source,version,code):reason_counts[(source,version,code)]+=1
    for base,source_default in [('intake',None),('searchswarm','SearchSwarm')]:
        for name,state in [('held','review'),('duplicates','duplicate')]:
            for r in rows(out/base/f'{name}.jsonl'):
                ident=r['id'];src=r.get('source',source_default)
                decisions[ident]={'source':src,'state':state,'reasons':r.get('reasons',['exact_duplicate']),'role':r.get('agent_role','parent')}
    for r in rows(out/'unified/exact_duplicates.jsonl'):
        decisions[r['source_id']]={'source':r['source'],'state':'duplicate','reasons':['exact_duplicate'],'role':r['agent_role']}
    scan=sqlite3.connect('file:'+str(out/'safety/state.sqlite')+'?mode=ro',uri=True)
    results=scan.execute('SELECT result FROM decisions ORDER BY row_index')
    contract=read(out/'unified/tools.json');candidate_offsets={};count=0
    with (out/'unified/candidates.jsonl').open('rb') as f:
        while True:
            offset=f.tell();line=f.readline()
            if not line:break
            item=json.loads(line);r=json.loads(next(results)[0])
            assert r['source_row']==count and r['offset_start']==offset and r['offset_end']==f.tell()
            assert r['content_hash']==digest(item) and r['coverage_complete']
            src=item['source'];ident=item['id'];role=item.get('agent_role','parent')
            allowed={k:v for k,v in contract['schemas'].items() if role!='child' or k not in contract['delegation']['blocked_child_tools']}
            codes={x['rule_id'] for x in r['findings']};strict=True
            try:
                validate_messages(item['messages'],allowed)
                syntax_check(item['messages'])
            except ValueError as e:strict=False;codes.add('protocol.'+str(e))
            blocked={'target_call_response_alignment_required','unverified_call_response_association','tool_prompt_protocol_review_required'}
            codes.update('protocol.'+flag for flag in item['review_flags'] if flag in blocked)
            strict=strict and not blocked.intersection(item['review_flags'])
            if strict:diagnostics[(src,'batch_response_association_valid')]+=1
            state='quarantine' if r['state']=='quarantine' else 'pass' if strict and r['state']=='pass_rules' else 'review'
            decisions[ident]={'source':src,'role':role,'state':state,'reasons':sorted(codes),'delegation_calls':r.get('delegation_calls',0),'delegation_tasks':r.get('delegation_tasks',0)}
            candidate_offsets[ident]=(offset,f.tell(),digest(item));count+=1
    assert next(results,None) is None;scan.close()
    # Match the full-corpus baseline duplicate decisions, regardless of subset representatives.
    locked=collections.Counter()
    for ident,r in decisions.items():
        assert ident in old,(ident,r['source'])
        if old[ident]['state']=='duplicate':
            r['state']='duplicate';r['reasons']=['exact_duplicate'];locked[r['source']]+=1
    dependencies=collections.defaultdict(set);tasks=collections.defaultdict(set)
    for edge in rows(out/'unified/delegation_links.jsonl'):
        parent=edge.get('canonical_parent_id');child=edge.get('canonical_child_id')
        if parent in decisions:
            dependencies[parent].add(child);tasks[parent].add((edge.get('call_id'),edge.get('task_index',0)))
    held=set()
    for ident,r in decisions.items():
        if r.get('delegation_calls') and len(tasks[ident])<r['delegation_tasks']:held.add(ident)
    # Missing/held child records and cycles conservatively propagate to parents.
    remaining={p:len({c for c in cs if c in dependencies}) for p,cs in dependencies.items()};reverse=collections.defaultdict(set)
    for p,cs in dependencies.items():
        for c in cs:reverse[c].add(p)
    leaves=collections.deque(p for p,n in remaining.items() if n==0)
    while leaves:
        leaf=leaves.popleft();remaining.pop(leaf)
        for parent in reverse[leaf]:
            if parent in remaining:
                remaining[parent]-=1
                if remaining[parent]==0:leaves.append(parent)
    held.update(remaining)
    changed=True
    while changed:
        before=len(held)
        for p,cs in dependencies.items():
            if any(c not in decisions or decisions[c]['state']!='pass' or c in held for c in cs):held.add(p)
        changed=len(held)!=before
    for ident in held:
        if decisions[ident]['state']=='pass':
            decisions[ident]['state']='review';decisions[ident]['reasons'].append('dependency.child_missing_or_held_or_cycle')
    bysource=collections.defaultdict(collections.Counter);transitions=collections.Counter()
    final=out/'final';final.mkdir();reportdir=out/'report';reportdir.mkdir()
    with (final/'decisions.jsonl').open('w') as ledger,(final/'held.jsonl').open('w') as heldfile,(final/'quarantine.jsonl').open('w') as quarantine:
        for ident,r in decisions.items():
            src=r['source'];c=bysource[src];prior=old[ident];c['read']+=1;c['baseline_'+prior['state']]+=1;c['new_'+r['state']]+=1
            transitions[(src,prior['state'],r['state'])]+=1
            for code in set(prior['reasons']):reason(src,'baseline',code)
            for code in set(r['reasons']):reason(src,'new',code)
            row={'id':ident,**r,'baseline_state':prior['state']};ledger.write(json.dumps(row,ensure_ascii=False)+'\n')
            if r['state'] in ('review','quarantine'):
                target=heldfile if r['state']=='review' else quarantine
                target.write(json.dumps(row,ensure_ascii=False)+'\n')
    hashes=[];passcounts=collections.Counter();roles=collections.Counter()
    with (out/'unified/candidates.jsonl').open('rb') as source,(final/'offline_passed.jsonl').open('w') as dest,(final/'content_hashes.jsonl').open('w') as inventory:
        for ident,r in decisions.items():
            if r['state']!='pass':continue
            start,end,sourcehash=candidate_offsets[ident];source.seek(start);item=json.loads(source.read(end-start));assert digest(item)==sourcehash
            assert not any(code.startswith(('benchmark.','credential.')) for code in r['reasons'])
            item['offline_screening']={'protocol':'passed','safety':'pass_rules','runtime_replay':False,'semantic_quality':'not_evaluated','policy_hash':read(out/'safety/manifest.json')['policy_hash']}
            dest.write(json.dumps(item,ensure_ascii=False)+'\n');inventory.write(json.dumps({'id':ident,'source':r['source'],'content_sha256':digest(item)})+'\n')
            passcounts[r['source']]+=1;roles[(r['source'],r['role'])]+=1
    table=[]
    for src,c in sorted(bysource.items()):
        assert c['read']==sum(c['new_'+k] for k in ('pass','review','quarantine','duplicate'))
        assert c['read']==sum(c['baseline_'+k] for k in ('pass','review','quarantine','duplicate'))
        assert passcounts[src]==c['new_pass']
        intake_summary=read(out/'intake/cleaning_summary.json')
        swarm_counts=read(out/'searchswarm/summary.json')['counts']
        expected=swarm_counts['bundles']+swarm_counts['children_read'] if src=='SearchSwarm' else intake_summary['sources'][src]['scanned']
        assert c['read']==expected,(src,c['read'],expected)
        table.append({'dataset':src,'read':c['read'],'baseline_passed':c['baseline_pass'],'baseline_pass_rate':round(100*c['baseline_pass']/c['read'],4),
                      'new_passed':c['new_pass'],'new_pass_rate':round(100*c['new_pass']/c['read'],4),'delta_percentage_points':round(100*(c['new_pass']-c['baseline_pass'])/c['read'],4),
                      'new_held':c['new_review'],'new_quarantine':c['new_quarantine'],'new_duplicates':c['new_duplicate'],'fixed_baseline_duplicates':locked[src],
                      'batch_response_association_valid':diagnostics[(src,'batch_response_association_valid')]})
    csvout(reportdir/'by_dataset.csv',table)
    reasons=[{'dataset':s,'version':v,'reason':r,'trajectories':n} for (s,v,r),n in sorted(reason_counts.items())]
    csvout(reportdir/'reasons.csv',reasons)
    csvout(reportdir/'transitions.csv',[{'dataset':s,'baseline_state':b,'new_state':n,'rows':v} for (s,b,n),v in sorted(transitions.items())])
    totals={key:sum(r[key] for r in table) for key in ('read','baseline_passed','new_passed','new_held','new_quarantine','new_duplicates','batch_response_association_valid')}
    summary={'status':'completed','totals':totals,'by_dataset':table,'reasons':reasons,'role_pass_counts':[{'source':s,'role':r,'rows':n} for (s,r),n in sorted(roles.items())],
             'llm_calls':0,'live_tool_calls':0,'training_ready':False,'extract_return_policy':'Search/extract batch text or JSON is preserved. No mandatory query/page splitting or per-item historical status; call-batch association and safety gates still apply.',
             'search_batch_schema_upgrade':read(out/'manifest.json').get('search_batch_schema_upgrade',False),
             'baseline':str(baseline),'dedup_policy':'Full-baseline exact duplicates are fixed in both arms; new within-subset exact duplicates remain removed.',
             'limitations':['Baseline outcomes are projected onto the identical sampled source rows, not a second global run.','SearchSwarm is sampled by whole bundle; expanded child fraction can differ from 5%.','No semantic factual scoring, human approval or live replay.','Safety signals require context; reason counts overlap.']}
    save(reportdir/'summary.json',summary)
    for name in ('offline_passed.jsonl','decisions.jsonl','held.jsonl','quarantine.jsonl','content_hashes.jsonl'):
        p=final/name;hashes.append({'path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
    save(final/'manifest.json',{'status':'completed','version':'offline-adapter-pilot-v1','files':hashes,'rows':totals,'training_ready':False,'llm_calls':0})
    print(json.dumps({'totals':totals,'by_dataset':table},ensure_ascii=False),flush=True)

