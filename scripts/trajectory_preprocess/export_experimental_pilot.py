"""Export explicitly requested experimental SFT without claiming release acceptance."""
import argparse,json,secrets
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from safety_rules import Scanner
from glasseek_shared.trajectory_release import payload,digest,runtime_schemas,file_hash
from glasseek_shared.experimental_release import VERSION,validate,verify

def run(source,output,authorization):
    output.mkdir(exist_ok=False,parents=True)
    policy=json.loads(Path('scripts/trajectory_preprocess/safety_policy.json').read_text())
    contract=json.loads(Path('data/processed/scientific-recovery-20261109/unified/tools.json').read_text())
    scanner=Scanner(policy,contract,secrets.token_bytes(32));tools=runtime_schemas();rows=[]
    with (output/'safety.jsonl').open('w') as audit:
        for line in source.open():
            r=json.loads(line)
            if r['text_normalization']['holds']:raise ValueError('held_candidate')
            safety=scanner.scan(r)
            if safety['state']!='pass_rules' or not safety['coverage_complete']:raise ValueError('safety_not_passed:'+r['id'])
            value=payload(r);sha=digest(value)
            acceptance={'content_sha256':sha,'safety':'pass_rules','quality':'prior_rubric_selected','review':'not_performed','replay':'not_performed','authorization':authorization}
            row={**value,'source':r['source'],'training_ready':False,'harness_replayed':False,'acceptance_status':'experimental_pilot','acceptance':json.dumps(acceptance),'content_sha256':sha}
            validate(row,tools)
            for k in ('messages','tools','provenance'):
                row[k]=json.dumps(row[k],ensure_ascii=False)
            row['supervision']=json.dumps(row['supervision']) if 'supervision' in row else None
            rows.append(row);audit.write(json.dumps({'id':r['id'],'content_sha256':sha,**safety})+'\n')
    pq.write_table(pa.Table.from_pylist(rows),output/'train.parquet')
    manifest={'version':VERSION,'acceptance_status':'experimental_pilot','rows':len(rows),'sha256':file_hash(output/'train.parquet'),'tool_schema_sha256':digest(tools),'source_sha256':file_hash(source),'authorization':authorization,'review':'not_performed','replay':'not_performed','training_ready':False}
    (output/'release.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));verify(output/'train.parquet');print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--authorization',required=True);a=p.parse_args();run(a.input,a.output,a.authorization)
