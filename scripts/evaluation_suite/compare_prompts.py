"""Compare paired predictions and native per-task grades without inventing scores."""
import argparse
import json
import math
from pathlib import Path
from collections import Counter
from report import load
from suite import dump


def percentile(values, q):
    values=sorted(values)
    return values[max(0,math.ceil(len(values)*q)-1)] if values else None


def cases(root, runs):
    result={}
    for run in runs:
        folders = set((root/'predictions').glob(run+'_*/suite_*'))
        manifest = load(root/run/'launch.json', {})
        for benchmark, identity in manifest.get('prediction_runs', {}).items():
            folders.add(root/'predictions'/identity/('suite_'+benchmark))
        for folder in sorted(folders):
            benchmark=folder.name.removeprefix('suite_')
            for path in folder.glob('conv/*/*.json'):
                row=load(path,{})
                if not row.get('question_id'):continue
                key=(benchmark,row['question_id'])
                if key in result:raise ValueError('duplicate task: '+str(key))
                grade=load(folder/'judge_official'/row['question_id']/(row['run_id']+'.json'),{})
                evaluation=grade.get('evaluation',{})
                metrics={m['key']:m['value'] for m in evaluation.get('metrics',[]) if isinstance(m.get('value'),(int,float)) and not isinstance(m['value'],bool)}
                result[key]={'status':row.get('status'),'correct':evaluation.get('correct'),
                    'grade_status':evaluation.get('status','pending'),'metrics':metrics,
                    'seconds':row.get('wall_seconds'),'tool_calls':sum((row.get('tool_calls') or {}).values())}
    return result


def compare(root, before, after):
    a,b=cases(root,before),cases(root,after);groups={};paired=[]
    for bench in sorted({k[0] for k in a}|{k[0] for k in b}):
        keys=sorted(k for k in a.keys() & b.keys() if k[0]==bench)
        pairs=[(a[k],b[k]) for k in keys]
        binary=[(x['correct'],y['correct']) for x,y in pairs if isinstance(x['correct'],bool) and isinstance(y['correct'],bool)]
        groups[bench]={'before_generated':sum(k[0]==bench for k in a),'after_generated':sum(k[0]==bench for k in b),'paired_generated':len(keys),
            'paired_binary_scored':len(binary),'before_accuracy_paired':sum(x for x,y in binary)/len(binary) if binary else None,
            'after_accuracy_paired':sum(y for x,y in binary)/len(binary) if binary else None,
            'wrong_to_right':sum(not x and y for x,y in binary),'right_to_wrong':sum(x and not y for x,y in binary)}
        for label,index in [('before',0),('after',1)]:
            rows=[p[index] for p in pairs];times=[r['seconds'] for r in rows if isinstance(r['seconds'],(int,float))]
            groups[bench][label]={'status':dict(Counter(r['status'] for r in rows)),'grade_status':dict(Counter(r['grade_status'] for r in rows)),
                'mean_seconds':sum(times)/len(times) if times else None,'p95_seconds':percentile(times,.95),
                'mean_tool_calls':sum(r['tool_calls'] for r in rows)/len(rows) if rows else None}
        paired.extend({'benchmark':k[0],'id':k[1],'before':a[k],'after':b[k]} for k in keys)
    return {'before_runs':before,'after_runs':after,'benchmarks':groups,'cases':paired,
            'notes':['Only common task IDs are paired; missing and unscored cases are reported separately.','Continuous metrics remain native per-task metrics; no blended score.','Single stochastic runs do not establish statistical significance; live web evidence can change.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=Path('Evaluation/output/suite5pct'));p.add_argument('--before',nargs='+',required=True);p.add_argument('--after',nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();dump(a.out,compare(a.output_root,a.before,a.after))

if __name__=='__main__':main()
