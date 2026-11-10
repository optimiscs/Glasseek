"""Reuse the evaluation launcher for isolated four-mode URL experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Evaluation/services'))
from eval_run import build


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--question-ids', nargs='+', help='Subset of original question IDs, preserving original order')
    parser.add_argument('--workers-per-mode', type=int, default=2, help='Concurrent questions per mode (default: 2)')
    parser.add_argument('--judge', action=argparse.BooleanOptionalAction, default=True,
                        help='Incrementally score completed answers (default: enabled)')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.workers_per_mode < 1:raise ValueError('--workers-per-mode must be positive')
    workers=1 if args.smoke else args.workers_per_mode
    if (out/'status.json').exists():
        raise FileExistsError('Use a new output directory; do not mix paused runs or prompt versions')
    original=ROOT/'Evaluation/data/runtime/archive/checkpoint_reeval/remote_snapshot/code/browsecomp_rl9b_non_timeout_34.jsonl'
    dataset='browsecomp_url_modes_smoke_v2' if args.smoke else 'browsecomp_rl9b_non_timeout_34'
    target=ROOT/'Evaluation/data/runtime'/f'{dataset}.jsonl'
    data=(json.dumps({'id':'url_smoke','type':'browsecomp','question':'Search the web for Example Domain example.com. Read its page using web_extract, then give the page title. Use any source references returned by tools.','answer':'Example Domain'})+'\n').encode() if args.smoke else original.read_bytes()
    if args.question_ids:
        if args.smoke:raise ValueError('--question-ids cannot be combined with --smoke')
        wanted=set(args.question_ids)
        selected=[line for line in data.decode().splitlines() if json.loads(line)['id'] in wanted]
        if {json.loads(line)['id'] for line in selected}!=wanted:raise ValueError('Unknown question ID')
        data=('\n'.join(selected)+'\n').encode()
        dataset='browsecomp_url_subset_'+hashlib.sha256(data).hexdigest()[:12]
        target=ROOT/'Evaluation/data/runtime'/f'{dataset}.jsonl'
    question_count=len(data.splitlines())
    if target.exists() and target.read_bytes()!=data:raise ValueError('Refusing to overwrite a different dataset')
    target.write_bytes(data)
    profile={'defaults':{'dataset':dataset,'workers_per_endpoint':workers,'max_rounds':5 if args.smoke else 200,'max_retries':1,
                         'timeout_seconds':240 if args.smoke else 3600,'context_length':131072,'max_tokens':16384,
                         'context_compression':True,'compression_threshold':0.5,'expected_questions':question_count,'repeats':1},
             'profiles':{}}
    stamp=time.strftime('%Y%m%d_%H%M%S')
    for mode in ('raw','hybrid','ref','evidence'):
        profile['profiles'][mode]={'model':'glasseek-base-9b','base_urls':'http://127.0.0.1:19109/v1',
                                   'run':f'url_{"smoke" if args.smoke else str(question_count)}_{mode}_{stamp}'}
    cfg=out/'profiles.json';cfg.write_text(json.dumps(profile,indent=2));os.environ['GLASSEEK_EVAL_PROFILES']=str(cfg)
    os.environ.pop('GLASSEEK_MAX_TOOL_CALLS',None);os.environ.pop('NO_TOTAL_TIMEOUT',None)
    jobs={};judge_jobs={};started=time.time();manifest={'started_at':started,'dataset_sha256':hashlib.sha256(data).hexdigest(),'dataset':dataset,
                                         'scope':f'4 x {question_count} end-to-end runs','question_ids':[json.loads(line)['id'] for line in data.splitlines()],
                                         'concurrency_per_mode':workers,'shared_endpoint':True,
                                         'comparison_note':'Concurrent modes share one GPU and tool pools; observed latencies include resource contention and current web results.', 'runs':{}}
    for mode in profile['profiles']:
        os.environ['GLASSEEK_URL_MODE']=mode
        argv,env=build(mode,'full')
        from glasseek_shared.hermes.url_prompt import prompt_module
        prompt = prompt_module(mode)
        (out/(mode+'.system-plugin.txt')).write_text(prompt['text'])
        env['GLASSEEK_URL_STATE_DIR']=str(out/'registry'/mode)
        env['PYTHONPATH']=str(ROOT/'Evaluation_backend')+os.pathsep+env.get('PYTHONPATH','')
        log=(out/(mode+'.log')).open('a')
        proc=subprocess.Popen(argv,cwd=ROOT/'Evaluation',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        jobs[mode]=(proc,log)
        manifest['runs'][mode]={'run':profile['profiles'][mode]['run'],'pid':proc.pid,'argv':argv,'url_state_dir':env['GLASSEEK_URL_STATE_DIR'],'status':'running',
                                'url_prompt':{k:v for k,v in prompt.items() if k!='text'}}
        if args.judge:
            judge_argv,judge_env=build(mode,'watch-judge')
            judge_argv += ['--parent-pid',str(proc.pid),'--viewer-model',profile['profiles'][mode]['model']]
            judge_log=(out/(mode+'.judge.log')).open('a')
            judge_proc=subprocess.Popen(judge_argv,cwd=ROOT/'Evaluation',env=judge_env,
                stdout=judge_log,stderr=subprocess.STDOUT,start_new_session=True)
            judge_jobs[mode]=(judge_proc,judge_log)
            manifest['runs'][mode]['judge']={'pid':judge_proc.pid,'status':'running','log':str(out/(mode+'.judge.log'))}
    def save():
        manifest['elapsed_seconds']=time.time()-started
        tmp=out/'status.json.tmp';tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(out/'status.json')
    save()
    while jobs or judge_jobs:
        for mode,(proc,log) in list(jobs.items()):
            code=proc.poll()
            if code is not None:
                manifest['runs'][mode].update(status='completed' if code==0 else 'failed',exit_code=code,finished_at=time.time())
                log.close();del jobs[mode];print(mode,code,flush=True)
        for mode,(proc,log) in list(judge_jobs.items()):
            code=proc.poll()
            if code is not None:
                manifest['runs'][mode]['judge'].update(status='completed' if code==0 else 'failed',exit_code=code,finished_at=time.time())
                log.close();del judge_jobs[mode];print(mode,'judge',code,flush=True)
        save()
        if jobs or judge_jobs:time.sleep(5)
    if any(r['exit_code'] or r.get('judge',{}).get('exit_code',0) for r in manifest['runs'].values()):raise SystemExit(1)
if __name__=='__main__':main()
