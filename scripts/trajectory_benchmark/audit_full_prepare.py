"""Report offline intake outcomes; categories are triage labels, not quality verdicts."""
import argparse,collections,csv,json,re,sqlite3
from pathlib import Path

def lines(path):
    with path.open('rb') as f:
        return sum(block.count(b'\n') for block in iter(lambda:f.read(8*1024**2),b''))

def category(reason,error):
    if reason=='annotation_size_budget_exceeded':return 'processing_budget'
    if reason=='hierarchical_bundle_adapter_required':return 'adapter_or_tool_scope'
    if reason.startswith('unsupported_tool:'):return 'adapter_or_tool_scope'
    if reason=='normalization_error':
        if any(s in error for s in ('needs adapter','adapter required','expansion','Unsupported schema')):
            return 'adapter_or_tool_scope'
        if 'Empty trajectory' in error:return 'empty_trajectory'
        return 'structural_or_parser_review'
    return 'structural_or_parser_review'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);args=ap.parse_args()
    root=args.input;out=root/'audit';out.mkdir(exist_ok=True)
    summary=json.loads((root/'cleaning_summary.json').read_text())
    assert summary['status']=='completed',summary['status']
    manifest=json.loads((root/'manifest.json').read_text())
    assert summary['completed_files']==len(manifest['inputs'])
    bysource=collections.defaultdict(lambda:{'held_reasons':collections.Counter(),'normalization_errors':collections.Counter(),
                                            'category_combinations':collections.Counter(),'category_rows':collections.Counter()})
    examples={};combos=collections.Counter();cats=collections.Counter();held_sources=collections.Counter()
    for line in (root/'held.jsonl').open():
        row=json.loads(line);src=row['source'];held_sources[src]+=1
        info=bysource[src];error=row.get('error','');classes=set()
        for reason in row['reasons']:
            info['held_reasons'][reason]+=1;classes.add(category(reason,error))
            key=src+' | '+(re.sub(r': line .*','',error) if reason=='normalization_error' else reason)
            examples.setdefault(key,row)
        if error:info['normalization_errors'][re.sub(r': line .*','',error)]+=1
        combo=' + '.join(sorted(classes));info['category_combinations'][combo]+=1;combos[combo]+=1
        for c in classes:info['category_rows'][c]+=1;cats[c]+=1
    count_files={name:lines(root/(name+'.jsonl')) for name in ['clean','held','duplicates','near_duplicates','file_errors']}
    for name,key in [('clean','eligible'),('held','held'),('duplicates','exact_duplicates'),('near_duplicates','near_duplicate_flagged'),('file_errors','file_errors')]:
        assert count_files[name]==summary['counts'].get(key,0),(name,count_files[name],summary['counts'].get(key,0))
    assert count_files['clean']+count_files['held']+count_files['duplicates']==summary['counts']['scanned']
    expected=collections.Counter()
    for spec in manifest['inputs']:
        path=Path(spec['path'])
        expected[spec['source']]+=manifest['parquet_expected_rows'][str(path)] if path.suffix=='.parquet' else lines(path)
    rows=[]
    for src,counts in summary['sources'].items():
        assert expected[src]==counts['scanned'],(src,expected[src],counts['scanned'])
        assert held_sources[src]==counts.get('held',0)
        rows.append({'source':src,'scanned':counts['scanned'],'normalized':counts.get('normalized',0),
                     'eligible':counts.get('eligible',0),'pass_rate_percent':round(100*counts.get('eligible',0)/counts['scanned'],3),
                     'held':counts.get('held',0),'exact_duplicates':counts.get('exact_duplicates',0),
                     'near_duplicate_flagged':counts.get('near_duplicate_flagged',0)})
    db=sqlite3.connect('file:'+str((root/'dedup.sqlite').resolve())+'?mode=ro',uri=True)
    exact_entries=db.execute('SELECT count(*) FROM exact').fetchone()[0];db.close()
    assert exact_entries+summary['counts'].get('exact_duplicates',0)==summary['counts']['normalized']
    result={'status':summary['status'],'elapsed_seconds':summary['elapsed_seconds'],'completed_files':summary['completed_files'],
            'counts':summary['counts'],'by_source':rows,'held_detail':dict(bysource),'category_rows_overlapping':dict(cats),
            'category_combinations_exclusive':dict(combos),'file_line_counts':count_files,'expected_input_rows':dict(expected),
            'dedup_index_entries':exact_entries,
            'interpretation':['Pass means structurally eligible for later annotation; no semantic quality judgment or replay.',
                              'Held reasons overlap. Category combinations are exclusive per record.',
                              'Structural/parser flags require raw-record inspection; not all are demonstrated data errors.',
                              'Dedup applies to normalized records only; near duplicates are flagged, not removed.',
                              'SearchSwarm counts bundles, not expanded trajectories.']}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'reason_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    with (out/'by_dataset.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
