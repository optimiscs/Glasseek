"""Apply the user-requested universal XML adapter to the same selected records."""
import json,sys,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/trajectory_preprocess'))
import ingest,pipeline
from client import save
OUT=ROOT/'reports/trajectory-stratified-qwen38-20261107'
def main():
    before=OUT/'pre-xml-adapter';before.mkdir(exist_ok=True)
    for name in ['manifest.jsonl','clean.jsonl','sampling_summary.json','dedup_summary.json']:
        if not (before/name).exists():shutil.copy2(OUT/name,before/name)
    rows=[json.loads(l) for l in (before/'manifest.jsonl').open()]
    cfg=json.loads((ROOT/'scripts/trajectory_preprocess/config.qwen38.local.json').read_text());summary=[]
    for row in rows:
        for key in ['sample_path','program_errors','size_budget_passed','normalized_chars','normalization_error','dedup']:row.pop(key,None)
        try:
            if row['projection']!='full_record':raise ValueError('Hierarchical bundle requires explicit main/subagent expansion; metadata-only sampling projection')
            item=ingest.normalize(json.loads(Path(row['raw_path']).read_text()),row['source'],row['path'],row['row_index'])
            item['adapter_id']=item['id'];item['id']=row['id'];item['harness']=ingest.harness_plan(item,cfg['harness'])
            item['program_errors']=sorted(set(item['program_errors']+item['harness']['errors']));size=len(json.dumps(item,ensure_ascii=False))
            row.update(normalization_status='passed',normalized_chars=size,program_errors=item['program_errors'],size_budget_passed=size<=cfg['execution']['max_sample_chars'])
            row['eligible']=not item['program_errors'] and row['size_budget_passed']
            row['hold_category']='structural_or_harness_adapter' if item['program_errors'] else 'annotation_size_budget' if not row['size_budget_passed'] else None
            row['sample_path']=str(OUT/'normalized'/f"{row['id']}.json");save(row['sample_path'],item)
        except Exception as e:row.update(normalization_status='failed',eligible=False,hold_category='schema_adapter_unavailable',normalization_error=type(e).__name__+': '+str(e))
    (OUT/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    for source in dict.fromkeys(r['source'] for r in rows):
        rr=[r for r in rows if r['source']==source];summary.append({'source':source,'selected':len(rr),'normalized':sum(r['normalization_status']=='passed' for r in rr),'eligible':sum(r['eligible'] for r in rr)})
    save(OUT/'xml_adapter_summary.json',{'datasets':summary,'policy_hash_after_format_adapter':pipeline.policy_hash(),'quality_rules_changed':False,'sampling_changed':False})
    snapshot=OUT/'xml_source';snapshot.mkdir(exist_ok=True)
    for directory in [ROOT/'scripts/trajectory_preprocess',ROOT/'scripts/trajectory_preprocess_pilot']:
        for p in directory.glob('*.py'):shutil.copy2(p,snapshot/(directory.name+'-'+p.name))
    print(json.dumps(summary))
if __name__=='__main__':main()
