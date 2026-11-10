"""Frozen-rule exploratory stratified sampling, seeded and independent of quality verdicts."""
import argparse,collections,glob,hashlib,heapq,json,random,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/trajectory_preprocess'))
import ingest,pipeline
from client import digest,save,public_config
OUT=ROOT/'reports/trajectory-stratified-qwen38-20261107'
SEED=20261107

def rows(path,source):
    if source=='SearchSwarm':
        import pyarrow.parquet as pq
        # Its single row group contains 5.8 GB of nested subagent text. The frozen
        # normalizer does not support bundles; metadata projection preserves the
        # sampling denominator without silently dropping children into training.
        columns=['source','question','answer','num_subagents','messages.list.element.role']
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32,columns=columns,use_threads=False):
            yield from batch.to_pylist()
    else:yield from ingest.records(path)

def features(row):
    msgs=row.get('messages',row.get('trajectory',row.get('conversations',[])))
    if not isinstance(msgs,list):msgs=[]
    count=sum(m.get('role',m.get('from')) in ('assistant','gpt') for m in msgs)
    band=0 if count<=8 else 1 if count<=20 else 2 if count<=40 else 3
    question=row.get('question',(row.get('meta') or {}).get('question',''))
    if not question:
        question=next((m.get('content',m.get('value','')) for m in msgs if m.get('role',m.get('from')) in ('user','human')),'')
    language='zh' if re.search(r'[\u4e00-\u9fff]',str(question)) else 'other'
    names=set();chars=0
    for m in msgs:
        content=m.get('content',m.get('value','')) or ''
        if isinstance(content,str):
            chars+=len(content)
            for block in re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',content,re.S):
                try:names.add(json.loads(block).get('name','unknown'))
                except Exception:names.add('unparsed')
        for c in m.get('tool_calls') or []:
            if isinstance(c,dict):names.add(c.get('function',c).get('name','unknown'))
        if m.get('recipient'):names.add(m['recipient'])
    tool_mode='search_extract' if any(n in names for n in ('visit','web_extract','webextract')) else 'search_only' if any(n in names for n in ('search','web_search','websearch')) else 'other_or_unparsed'
    if row.get('num_subagents',0):tool_mode='hierarchical_bundle'
    return {'assistant_messages':count,'length_band':band,'language':language,'tool_mode':tool_mode,'observed_tool_names':sorted(str(n) for n in names),'text_chars':chars,'num_subagents':row.get('num_subagents',0)}

def main():
    global OUT, SEED
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--per-dataset',type=int,default=4)
    ap.add_argument('--seed',type=int,default=20261107)
    ap.add_argument('--config',type=Path,default=ROOT/'scripts/trajectory_preprocess/config.deepseek.flash.json')
    args=ap.parse_args();OUT=args.output.resolve();SEED=args.seed
    if args.per_dataset<1:raise ValueError('per-dataset must be positive')
    if (OUT/'manifest.jsonl').exists():raise ValueError('Use a fresh output directory; do not overwrite a frozen sample')
    OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    cfg=json.loads(args.config.read_text())
    specs=json.loads((ROOT/'scripts/trajectory_preprocess/inputs.datasets.json').read_text())
    if len(specs)*args.per_dataset>cfg['execution'].get('max_batch_samples',36):raise ValueError('Sampling exceeds configured batch limit')
    plan={'seed':SEED,'per_dataset':args.per_dataset,'datasets':len(specs),'target_units':len(specs)*args.per_dataset,
          'length_bands':['assistant messages 1-8','9-20','21-40','41+'],
          'allocation':'Round-robin over nonempty length bands; redistribute shortages. Four per source normally yields one per band.',
          'within_band':'seeded bottom-hash reservoir of 32, then diversify language and tool mode; no quality-label selection',
          'estimation_scope':'coverage-oriented exploratory sample; unweighted rates do not estimate full-corpus quality',
          'SearchSwarm_unit':'bundle metadata projection; unsupported by frozen adapter, not treated as low quality',
          'MiroVerse':'not locally available; excluded, not silently replaced',
          'frozen_policy_hash':pipeline.policy_hash(),'configured_policy_version':cfg.get('policy_version','offline-pilot-v3.1-general'),
          'augmentation_probe':'one fixed eligible parent per dataset, draft-only, followed by verification'}
    save(OUT/'sampling_plan.json',plan);save(OUT/'config.redacted.json',public_config(cfg))
    snapshot=OUT/'frozen_source';snapshot.mkdir(exist_ok=True)
    for directory in (ROOT/'scripts/trajectory_preprocess',ROOT/'scripts/trajectory_preprocess_pilot'):
        for p in directory.glob('*.py'):
            (snapshot/(directory.name+'-'+p.name)).write_bytes(p.read_bytes())
    selected=[];summary=[]
    for spec in specs:
        source=spec['source'];source_file=OUT/(source+'.sampled.jsonl');meta_file=OUT/(source+'.sampling.json')
        if source_file.exists() and meta_file.exists():
            selected.extend(json.loads(x) for x in source_file.read_text().splitlines());summary.append(json.loads(meta_file.read_text()));continue
        pools={i:[] for i in range(4)};counts=collections.Counter();scanned=0;errors=[];source_start=time.monotonic()
        for path in sorted(glob.glob(str(ROOT/spec['path']))):
            try:
                for index,row in enumerate(rows(path,source)):
                    f=features(row);band=f['length_band'];counts[band]+=1;scanned+=1
                    priority=int(digest([SEED,source,str(Path(path).relative_to(ROOT)),index]),16)
                    if len(pools[band])<32 or priority < -pools[band][0][0]:
                        entry={'source':source,'path':path,'row_index':index,'features':f,'priority':priority,'raw':row,
                               'projection':'bundle_metadata_without_contents' if source=='SearchSwarm' else 'full_record'}
                        if len(pools[band])==32:heapq.heapreplace(pools[band],(-priority,index,entry))
                        else:heapq.heappush(pools[band],(-priority,index,entry))
                    if scanned%10000==0:print(source,'scanned',scanned,flush=True)
            except Exception as e:errors.append({'path':path,'error':type(e).__name__+': '+str(e)[:300]})
        candidates={b:[v[2] for v in sorted(pool,reverse=True)] for b,pool in pools.items()}
        chosen=[];used=collections.Counter()
        # Round-robin yields two per band if each band has enough rows.
        while len(chosen)<args.per_dataset and any(candidates.values()):
            for b in range(4):
                if not candidates[b] or len(chosen)==args.per_dataset:continue
                entry=min(candidates[b],key=lambda x:(used[(x['features']['language'],x['features']['tool_mode'])],x['priority']))
                candidates[b].remove(entry);chosen.append(entry);used[(entry['features']['language'],entry['features']['tool_mode'])]+=1
        output=[]
        for entry in chosen:
            row=entry.pop('raw');entry.pop('priority');ident=digest([source,entry['path'],entry['row_index']])[:20]
            raw_path=OUT/'raw'/f'{ident}.json';save(raw_path,row)
            entry.update(id=ident,raw_path=str(raw_path),raw_projection_sha256=digest(row),normalization_status='failed',pipeline_decision='hold')
            try:
                if source=='SearchSwarm':raise ValueError('Hierarchical bundle requires explicit main/subagent expansion; metadata-only sampling projection')
                item=ingest.normalize(row,source,entry['path'],entry['row_index'])
                # The experiment ID is provenance-based; preserve adapter identity separately.
                item['adapter_id']=item['id'];item['id']=ident
                item['harness']=ingest.harness_plan(item,cfg['harness'])
                item['program_errors']=sorted(set(item['program_errors']+item['harness']['errors']))
                size=len(json.dumps(item,ensure_ascii=False));entry['normalized_chars']=size
                entry['normalization_status']='passed';entry['program_errors']=item['program_errors']
                entry['size_budget_passed']=size<=cfg['execution']['max_sample_chars']
                entry['eligible']=not item['program_errors'] and entry['size_budget_passed']
                item_path=OUT/'normalized'/f'{ident}.json';save(item_path,item);entry['sample_path']=str(item_path)
                if item['program_errors']:entry['hold_category']='structural_or_harness_adapter'
                elif not entry['size_budget_passed']:entry['hold_category']='annotation_size_budget'
                else:entry['hold_category']=None
            except Exception as e:entry.update(eligible=False,hold_category='schema_adapter_unavailable',normalization_error=type(e).__name__+': '+str(e))
            output.append(entry)
        source_file.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in output))
        info={'source':source,'scanned':scanned,'stratum_population_counts':dict(counts),'sampled':len(output),'normalization_passed':sum(x['normalization_status']=='passed' for x in output),'eligible':sum(x['eligible'] for x in output),'seconds':time.monotonic()-source_start,'read_errors':errors}
        save(meta_file,info);summary.append(info);selected.extend(output);print(json.dumps(info,ensure_ascii=False),flush=True)
    (OUT/'manifest.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in selected))
    (OUT/'clean.jsonl').write_text(''.join(Path(x['sample_path']).read_text().replace('\n','')+'\n' for x in selected if x['eligible']))
    save(OUT/'sampling_summary.json',{'datasets':summary,'total_sampled':len(selected),'eligible':sum(x['eligible'] for x in selected),'elapsed_seconds':time.monotonic()-started})
if __name__=='__main__':main()
