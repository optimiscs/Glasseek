"""Run one preselected parent/source, at most 16 concurrent API requests.

Each existing variants runner is sequential within its parent; separate output
folders prevent cache/summary races. Parent audits are copied for traceability.
"""
import concurrent.futures,json,subprocess,sys,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/trajectory_preprocess'))
from client import digest
OUT=ROOT/'reports/trajectory-stratified-qwen38-20261107'
def run(item):
    out=OUT/'variant_probes'/item['source'];out.mkdir(parents=True,exist_ok=True)
    inp=out/'parent.jsonl';inp.write_text(json.dumps(item,ensure_ascii=False)+'\n')
    source=OUT/'results'/(digest(item['id'])[:24]+'.json')
    if source.exists():
        (out/'results').mkdir(exist_ok=True);shutil.copy2(source,out/'results'/source.name)
    start=time.monotonic()
    # Both existing variant validators require located observation evidence.
    # Preserve the preselected parent; do not replace it based on model outcomes.
    if not any(step['observation'].strip() for step in item['steps']):
        summary={'parents':1,'rewrites':0,'counterfactuals':0,'failed':0,'precondition_holds':1,'reason':'No recorded observation; neither variant kind can satisfy mandatory evidence validation.','training_ready':0}
        (out/'variant_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        return {'source':item['source'],'parent_id':item['id'],'exit_code':0,'seconds':time.monotonic()-start,'summary':summary}
    with (out/'run.log').open('w') as log:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/trajectory_preprocess/pipeline.py'),'variants','--config',str(ROOT/'scripts/trajectory_preprocess/config.qwen38.local.json'),'--output',str(out),'--input',str(inp),'--limit','1'],stdout=log,stderr=subprocess.STDOUT)
    summary=json.loads((out/'variant_summary.json').read_text()) if (out/'variant_summary.json').exists() else {}
    return {'source':item['source'],'parent_id':item['id'],'exit_code':result.returncode,'seconds':time.monotonic()-start,'summary':summary}
def main():
    parents=[json.loads(l) for l in (OUT/'variant_parents.jsonl').open()];start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        rows=[]
        for row in pool.map(run,parents):rows.append(row);print(json.dumps(row),flush=True)
    (OUT/'variant_probe_summary.json').write_text(json.dumps({'wall_seconds':time.monotonic()-start,'sources':rows,'training_ready':False},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
