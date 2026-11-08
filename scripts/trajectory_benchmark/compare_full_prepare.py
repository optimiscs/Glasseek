"""Compare completed offline runs by stable provenance IDs, including regressions."""
import argparse,collections,csv,json,re
from pathlib import Path

ID=re.compile(r'"id":\s*"([a-f0-9]+)"')
def rows(path):
    with path.open() as f:
        for line in f:yield json.loads(line)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--before',type=Path,required=True);ap.add_argument('--after',type=Path,required=True);args=ap.parse_args()
    before,after=args.before,args.after;out=after/'comparison';out.mkdir(exist_ok=True)
    old=json.loads((before/'cleaning_summary.json').read_text());new=json.loads((after/'cleaning_summary.json').read_text())
    assert old['status']==new['status']=='completed'
    assert old['counts']['scanned']==new['counts']['scanned']
    assert json.loads((before/'manifest.json').read_text())['inputs']==json.loads((after/'manifest.json').read_text())['inputs']
    old_held={r['id']:r for r in rows(before/'held.jsonl')}
    old_dups={r['id'] for r in rows(before/'duplicates.jsonl')}
    transitions=collections.defaultdict(collections.Counter);categories=collections.defaultdict(collections.Counter)
    reason_counts=collections.defaultdict(collections.Counter);format_errors=collections.defaultdict(collections.Counter)
    transitions_global=collections.Counter();examples={};category_counts=collections.Counter();new_linecounts=collections.Counter()
    category_reasons=collections.defaultdict(collections.Counter)
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"trajectory_preprocess"))
    from dispositions import categorize
    saved={name:(out/(name+'.jsonl')).open('w') for name in ('rescued','newly_held','still_unsupported')}
    def classify(ident):return 'held' if ident in old_held else 'duplicate' if ident in old_dups else 'eligible'
    def emit(name,row):saved[name].write(json.dumps(row,ensure_ascii=False)+'\n')
    for row in rows(after/'held.jsonl'):
        src=row['source'];prior=classify(row['id']);transitions[src][prior+'->held']+=1;transitions_global[prior+'->held']+=1;new_linecounts['held']+=1
        if prior=='eligible':emit('newly_held',row)
        if 'unsupported_tools' in row['categories']:emit('still_unsupported',row)
        for c in row['categories']:categories[src][c]+=1;category_counts[c]+=1
        for reason in row['reasons']:
            reason_counts[src][reason]+=1;examples.setdefault(src+' | '+reason,row)
            for c in categorize([reason],row.get('error','')):category_reasons[c][reason]+=1
        if row.get('error'):format_errors[src][re.sub(r': line .*','',row['error'])]+=1
    for row in rows(after/'duplicates.jsonl'):
        prior=classify(row['id']);transitions[row['source']][prior+'->duplicate']+=1;transitions_global[prior+'->duplicate']+=1;new_linecounts['duplicates']+=1
    for index,row in enumerate(rows(after/'clean.jsonl')):
        src=row['source'];prior=classify(row['id']);transitions[src][prior+'->eligible']+=1;transitions_global[prior+'->eligible']+=1;new_linecounts['clean']+=1
        assert not row['program_errors'] and not row['harness']['errors']
        if prior=='held':emit('rescued',{'id':row['id'],'source':src,'old_reasons':old_held[row['id']]['reasons'],'old_error':old_held[row['id']].get('error'),
                                       'new_clean_line_index':index,'provenance':row['provenance'],'adapter_flags':row['harness']['flags']})
    for f in saved.values():f.close()
    assert new_linecounts['clean']==new['counts'].get('eligible',0)
    assert new_linecounts['held']==new['counts'].get('held',0)
    assert new_linecounts['duplicates']==new['counts'].get('exact_duplicates',0)
    table=[]
    for src,n in new['sources'].items():
        o=old['sources'][src];population=n['scanned'];assert sum(transitions[src].values())==population
        table.append({'source':src,'scanned':population,'old_normalized':o.get('normalized',0),'new_normalized':n.get('normalized',0),
                      'old_eligible':o.get('eligible',0),'new_eligible':n.get('eligible',0),
                      'old_pass_percent':round(o.get('eligible',0)/population*100,3),'new_pass_percent':round(n.get('eligible',0)/population*100,3),
                      'eligible_delta':n.get('eligible',0)-o.get('eligible',0),'new_held':n.get('held',0),'new_duplicates':n.get('exact_duplicates',0),
                      'new_near_flagged':n.get('near_duplicate_flagged',0),'rescued':transitions[src]['held->eligible'],'newly_held':transitions[src]['eligible->held']})
    result={'before':str(before.resolve()),'after':str(after.resolve()),'counts':new['counts'],'elapsed_seconds':new['elapsed_seconds'],
            'by_dataset':table,'transitions':dict(transitions),'transitions_global':dict(transitions_global),
            'held_categories_overlapping':dict(categories),'category_totals_overlapping':dict(category_counts),
            'held_reasons':dict(reason_counts),'normalization_errors':dict(format_errors),'category_reason_counts':dict(category_reasons),
            'interpretation':'Mechanical eligibility only. quality_issues contains explicit structural defects, not factual correctness judgments. Incomplete episodes may be training prefixes. Category counts overlap.'}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'reason_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    with (out/'by_dataset.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
