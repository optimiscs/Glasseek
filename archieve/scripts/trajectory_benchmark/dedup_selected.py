"""Run frozen dedup on normalized selected rows without replacement sampling."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/trajectory_preprocess'))
from ingest import Dedup
OUT=ROOT/'reports/trajectory-stratified-qwen38-20261107'
def main():
    manifest=[json.loads(l) for l in (OUT/'manifest.jsonl').open()]
    config=json.loads((ROOT/'scripts/trajectory_preprocess/config.qwen38.local.json').read_text())
    db=OUT/'selected-dedup.sqlite'
    if db.exists():db.unlink()
    dedup=Dedup(db,config['dedup']);items=[];counts={'checked':0,'exact_duplicates':0,'near_duplicate_flags':0}
    try:
        for row in manifest:
            if row['normalization_status']!='passed':continue
            path=Path(row['sample_path']);item=json.loads(path.read_text());d=dedup.add(item)
            item['dedup']=d;row['dedup']=d;counts['checked']+=1
            counts['near_duplicate_flags']+=bool(d['near_duplicates'])
            if d['exact_duplicate_of']:
                counts['exact_duplicates']+=1;row.update(eligible=False,hold_category='exact_duplicate')
            path.write_text(json.dumps(item,ensure_ascii=False,indent=2)+'\n')
            if row['eligible']:items.append(item)
    finally:dedup.close()
    (OUT/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest))
    (OUT/'clean.jsonl').write_text(''.join(json.dumps(i,ensure_ascii=False)+'\n' for i in items))
    counts['eligible']=len(items)
    (OUT/'dedup_summary.json').write_text(json.dumps(counts,indent=2)+'\n')
    parents=[min([i for i in items if i['source']==s],key=lambda i:i['id']) for s in sorted({i['source'] for i in items})]
    (OUT/'variant_parents.jsonl').write_text(''.join(json.dumps(i,ensure_ascii=False)+'\n' for i in parents))
    print(counts)
if __name__=='__main__':main()
