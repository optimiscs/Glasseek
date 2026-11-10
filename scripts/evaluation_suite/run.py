#!/usr/bin/env python3
"""Run Glasseek evaluation, live viewing and native scoring from a Python CLI."""
from resume_runs import prediction_run, prediction_root
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from suite import ROOT, dump, jsonl, read, sha, verify


def predictions(root, refs):
    wanted={r['opaque_id'] for r in refs};found={}
    for p in sorted(Path(root).glob('conv/*/*.json')):
        row=json.loads(p.read_text());key=row.get('question_id')
        if key not in wanted:raise ValueError('unknown prediction ID: '+str(key))
        if key in found:raise ValueError('duplicate prediction ID: '+key)
        found[key]=(p,row)
    return found


def prepare_score(release, output, predroot, benchmark, corpus_config=None, opaque_ids=None):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True,mode=0o700)
    refs=[r for r in read(release/'private/references.jsonl') if r['benchmark']==benchmark]
    found=predictions(predroot,refs); valid=[];failures=[]
    if opaque_ids is not None:
        refs=[r for r in refs if r["opaque_id"] in opaque_ids]
    for r in refs:
        hit=found.get(r['opaque_id'])
        if not hit:
            failures.append({'id':r['id'],'status':'missing_prediction'});continue
        path,row=hit;answer=row.get('model_response','')
        if row.get('question')!=str(r['question']).strip():raise ValueError('prediction question does not match assigned task')
        if row.get('status')!='success' or not row.get('final_assistant_valid',False) or not answer.strip():
            failures.append({'id':r['id'],'status':row.get('status'),'failure_type':row.get('failure_type')});continue
        valid.append(dict(r,prediction=answer,prediction_path=str(path),prediction_sha256=sha(path)))
    jsonl(output/'records.jsonl',valid);jsonl(output/'failures.jsonl',failures)
    if benchmark=='gaia':
        submissions=[]
        for r in valid:
            m=re.search(r'(?:^|\n)FINAL ANSWER:\s*(.+?)\s*$',r['prediction'],re.I|re.S)
            if not m:
                failures.append({'id':r['id'],'status':'answer_format_error'});continue
            submissions.append({'task_id':r['id'],'model_answer':m.group(1).strip()})
        jsonl(output/'submission.jsonl',submissions)
        dump(output/'reference.json',[{'task_id':r['id'],'answer':r['answer']} for r in refs])
    elif benchmark=='drbench':
        jsonl(output/'raw.jsonl',[{'id':int(r['id']),'prompt':r['question'],'article':r['prediction']} for r in valid])
        jsonl(output/'query.jsonl',[r['reference'] for r in valid])
    elif benchmark=='mind2web2':
        for r in valid:
            folder=output/'answers'/'glasseek'/r['id'];folder.mkdir(parents=True)
            (folder/'answer_1.md').write_text(r['prediction'])
    elif benchmark=='liveresearchbench':
        reports=[]
        for r in valid:
            path=output/('qid_'+r['id']+'_report.md');path.write_text(r['prediction'])
            reports.append({'model_name':'glasseek','query_id':r['id'],'query':r['question'],'report_file_path':str(path.resolve())})
        dump(output/'reports.json',{'metadata':{'use_realtime':False},'reports':reports})
        dump(output/'benchmark_data.json',{r['id']:{'question':r['question'],'checklists':[
            x['checklist_no_placeholder'] for x in r['reference']['checklists']]} for r in refs})
    elif benchmark=='widesearch':
        rows=[]
        for r in valid:
            spec=dict(r['reference']);spec['evaluation']=json.loads(spec['evaluation']) if isinstance(spec['evaluation'],str) else spec['evaluation'];rows.append(spec)
        jsonl(output/'queries.jsonl',rows)
    elif benchmark=='browsecomp_plus' and corpus_config:
        cfg=json.loads(Path(corpus_config).read_text());events=Path(cfg['docid_events'])
        if not events.is_file():raise ValueError('corpus backend did not produce docid events')
        expected={r['opaque_id'] for r in refs}
        namespace=Path(predroot).parent.name
        captured={r['opaque_id']:{'opaque_id':r['opaque_id'],'retrieved_docids':[],'tool_call_counts':{}} for r in valid}
        for event in read(events):
            key=event.get('opaque_id')
            if key not in captured or event.get('namespace')!=namespace:continue
            if event.get('corpus_revision')!=cfg['corpus_revision']:raise ValueError('corpus event revision mismatch')
            target=captured[key]
            target['retrieved_docids']=sorted(set(target['retrieved_docids'])|set(event['retrieved_docids']))
            for name,count in event['tool_call_counts'].items():target['tool_call_counts'][name]=target['tool_call_counts'].get(name,0)+count
        jsonl(output/'corpus-events.jsonl',captured.values())
    jsonl(output/'failures.jsonl',failures)
    dump(output/'manifest.json',{'benchmark':benchmark,'assigned':len(refs),'completed':len(valid),
        'failures':len(failures),'release_manifest_sha256':sha(release/'manifest.json'),
        'prediction_inputs':{r['prediction_path']:r['prediction_sha256'] for r in valid},
        'denominator_policy':'assigned is fixed; missing/failed/unscored reported separately, never excluded silently',
        'files':{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'scoring_status':'prepared_not_scored','protocol':next(r['protocol'] for r in json.loads((release/'summary.json').read_text()) if r['benchmark']==benchmark)})


def infer_command(args, benchmark, count):
    sys.path.insert(0,str(ROOT/'Evaluation/services'))
    from eval_run import build
    run=prediction_run(args,benchmark)
    cmd,env=build(args.profile,'full',{'dataset':'suite_'+benchmark,'run':run,
                    'workers_per_endpoint':args.workers,'ramp_start':args.workers,
                    'ramp_max':args.workers,'expected_questions':count,'repeats':1},python_entry=True)
    if args.eval_python:
        cmd[0]=str(args.eval_python);env['EVAL_PYTHON']=str(args.eval_python)
    i=cmd.index('--data-path');cmd[i+1]=str(args.suite/'inference'/(benchmark+'.jsonl'))
    cmd+=['--context-manager',args.context_manager,'--url-mode',args.url_mode,
          '--output-root',str(args.output_root/'predictions'),
          '--viewer' if args.viewer else '--no-viewer','--viewer-interval',str(args.viewer_interval)]
    if getattr(args, 'system_prompt_file', None):
        path = str(args.system_prompt_file.resolve())
        if '--custom-system-prompt-file' in cmd:
            cmd[cmd.index('--custom-system-prompt-file')+1] = path
        else:
            cmd += ['--enable-custom-system-prompt','true','--custom-system-prompt-file',path]
    if benchmark == 'hle' and getattr(args, 'research_prompt', 'legacy') != 'arex-native':
        prompt_path = ROOT/'Evaluation/config/prompts/hle-system.txt'
        if '--custom-system-prompt-file' in cmd:
            cmd[cmd.index('--custom-system-prompt-file')+1] = str(prompt_path)
        else:
            cmd += ['--enable-custom-system-prompt','true','--custom-system-prompt-file',str(prompt_path)]
    if getattr(args, 'research_prompt', 'legacy') == 'arex-native':
        if args.url_mode != 'raw':
            raise ValueError('AREX native currently requires raw URLs; opaque source-ID submission needs a separate adapter')
        if args.context_manager != 'hermes':
            raise ValueError('AREX native requires Hermes; CLM/fixed-corpus protocols are unchanged')
        from glasseek_shared.research.prompts import system_prompt
        # Both system and user templates switch together; never stack the old
        # research wrapper or official text-output instructions onto finish.
        prompt_path = args.output_root / args.run_id / 'prompts' / (benchmark + '.txt')
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(system_prompt(benchmark))
        if '--custom-system-prompt-file' in cmd:
            cmd[cmd.index('--custom-system-prompt-file')+1] = str(prompt_path)
        else:
            cmd += ['--enable-custom-system-prompt','true','--custom-system-prompt-file',str(prompt_path)]
        env.update(GLASSEEK_RESEARCH_PROTOCOL='arex-native', GLASSEEK_RESEARCH_DATASET='suite_' + benchmark,
                   GLASSEEK_RESEARCH_MAX_REVIEWS=str(args.max_reviews),
                   GLASSEEK_RESEARCH_CORRECTIONS=str(args.protocol_corrections),
                   GLASSEEK_RESEARCH_ACCEPT_CONFIDENCE=str(args.accept_confidence),
                   GLASSEEK_RESEARCH_MEDIUM_CONFIDENCE=str(args.medium_confidence))
        # The adapted no-subagent prompt does not expose delegation or context tools.
        cmd[cmd.index('--tool-whitelist')+1] = 'web,code_execution,vision'
    else:
        # A caller's shell must not accidentally mix protocols in a legacy run.
        env.pop('GLASSEEK_RESEARCH_PROTOCOL', None)
    env['GLASSEEK_ANSWER_PROMPT_REGISTRY']=str(args.suite/'prompts/registry.json')
    env['GLASSEEK_HARNESS_REVISION']='20261110-safety-eval-fixes'
    env['GLASSEEK_MAX_TOOL_CALLS']=str(args.max_tool_calls);env['NO_TOTAL_TIMEOUT']='1'
    # The suite intentionally removes the wall-clock timeout for every profile.
    if '--timeout-seconds' in cmd:
        index=cmd.index('--timeout-seconds');del cmd[index:index+2]
    env['GLASSEEK_VIEWER_AUTO_PUBLISH']='1'
    env['GLASSEEK_VIEWER_LABEL']=f'{benchmark} · {count} tasks · {args.context_manager} · {args.workers} workers'
    if benchmark=='browsecomp_plus':
        # Requires a separately attested fixed-corpus backend. Never silently run
        # BC+ on Serper/live pages and label it the original benchmark.
        if not args.corpus_config:raise ValueError('BC+ requires --corpus-config with pinned corpus, exclusive search/extract endpoints, and docid event capture')
        cfg=json.loads(args.corpus_config.read_text())
        if args.context_manager!='clm':raise ValueError('fixed-corpus bash isolation currently requires --context-manager clm')
        if cfg.get('managed_service') and args.phase!='plan' and not Path(cfg['index']).is_file():
            raise ValueError('fixed corpus index is not ready; no live-web substitution allowed')
        required=['corpus_revision','search_endpoint','fetch_endpoint','docid_events','live_web_disabled']
        if any(not cfg.get(k) for k in required) or cfg['live_web_disabled'] is not True:raise ValueError('incomplete fixed-corpus config')
        env.update({'SEARCH_SERVER_ENDPOINT':cfg['search_endpoint'],'GLASSEEK_SEARCH_POOL_ENDPOINT':cfg['search_endpoint'],
                    'GLASSEEK_FETCH_URL':cfg['fetch_endpoint'],'GLASSEEK_CORPUS_DOCID_EVENTS':cfg['docid_events'],
                    'GLASSEEK_FIXED_CORPUS_CONFIG':str(args.corpus_config),'GLASSEEK_CORPUS_RUN':run})
        j=cmd.index('--fetch-server-url');cmd[j+1]=cfg['fetch_endpoint']
        j=cmd.index('--fetch-watchdog-target');cmd[j+1]=''
        j=cmd.index('--tool-whitelist');cmd[j+1]='web'
    from resume_runs import verify_model
    verify_model(args, benchmark, cmd[cmd.index('--model')+1])
    return cmd,env


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,default=ROOT/'data/evaluation/suite5pct-v1/release')
    p.add_argument('--phase',choices=['plan','infer','prepare-score','score','all'],default='plan')
    p.add_argument('--benchmarks',default='all');p.add_argument('--workers',type=int,default=5)
    p.add_argument('--profile',default='base9b');p.add_argument('--context-manager',choices=['hermes','clm'],default='hermes')
    p.add_argument('--url-mode',choices=['raw','hybrid','ref','evidence'],default='raw')
    p.add_argument('--run-id',default='suite5pct_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--output-root',type=Path,default=ROOT/'Evaluation/output/suite5pct')
    p.add_argument('--corpus-config',type=Path);p.add_argument('--score-config',type=Path)
    p.add_argument('--system-prompt-file', type=Path, help='Override the research system prompt; HLE retains its official system prompt. User templates and tool schemas are unchanged.')
    p.add_argument('--resume-run-id', action='append', default=[], help='Adopt compatible legacy run identities and grading ledgers; repeat for separate GPU runs. Source producers must be stopped.')
    p.add_argument('--research-prompt', choices=['legacy','arex-native'], default='legacy', help='Switch system + user prompts and native finish as one protocol')
    p.add_argument('--max-reviews', type=int, default=2)
    p.add_argument('--protocol-corrections', type=int, default=3)
    p.add_argument('--accept-confidence', type=float, default=95)
    p.add_argument('--medium-confidence', type=float, default=90)
    p.add_argument('--affinity-endpoints', help='Comma-separated same-model endpoints; --workers is PER endpoint. Prefer the current dataset; backfill free slots across datasets with a shared per-endpoint limit.')
    p.add_argument('--score-schedule', choices=['trajectory','dataset','off'], default='trajectory', help='Default: score each completed trajectory during inference')
    p.add_argument('--per-benchmark',type=int,help='Prepare a fixed-seed stratified child release with this many tasks per benchmark')
    p.add_argument('--seed',type=int,help='Sampling seed; defaults to the frozen suite seed and does not seed model decoding')
    p.add_argument('--allow-blocked',action='store_true',help='Record blocked benchmarks and continue the others; never substitute protocols')
    p.add_argument('--eval-python',type=Path,help='Python interpreter for inference (default: existing Hermes runtime)')
    p.add_argument('--score-python',type=Path,help='Python interpreter for native graders (default: suite score-venv)')
    p.add_argument('--viewer',action=argparse.BooleanOptionalAction,default=True,help='Publish live trajectories automatically (default: enabled)')
    p.add_argument('--viewer-interval',type=float,default=20,help='Live trajectory refresh interval in seconds')
    p.add_argument('--max-tool-calls',type=int,default=120,help='Per-task tool-call limit; no wall-clock timeout')
    return p


def main(argv=None):
    p=build_parser()
    args=p.parse_args(argv);args.suite=args.suite.resolve();args.output_root=args.output_root.resolve()
    for name in ('eval_python','score_python'):
        value=getattr(args,name)
        if value is not None:
            # Preserve venv symlinks: resolving bin/python to the base binary
            # would silently lose the selected interpreter's site-packages.
            value=value.expanduser().absolute()
            if not os.access(value,os.X_OK):p.error(f'--{name.replace("_","-")} must be an executable interpreter')
            setattr(args,name,value)
    if args.system_prompt_file:
        args.system_prompt_file=args.system_prompt_file.expanduser().resolve()
        if not args.system_prompt_file.is_file() or not args.system_prompt_file.read_text().strip():p.error('--system-prompt-file must contain a nonempty prompt')
    if args.research_prompt == 'arex-native' and args.system_prompt_file:
        p.error('--research-prompt arex-native cannot be combined with a system-only override')
    if args.max_reviews < 0 or args.protocol_corrections < 1:
        p.error('review budget must be nonnegative; correction budget must be positive')
    if not 0 <= args.medium_confidence <= args.accept_confidence <= 100:
        p.error('confidence thresholds must satisfy 0 <= medium <= accept <= 100')
    if args.affinity_endpoints:
        from urllib.parse import urlsplit
        endpoints = [v.strip().rstrip('/') for v in args.affinity_endpoints.split(',')]
        if len(set(endpoints)) != len(endpoints) or any(urlsplit(e).scheme not in ('http','https') or not urlsplit(e).hostname or urlsplit(e).username for e in endpoints):
            p.error('--affinity-endpoints requires unique HTTP(S) URLs without credentials')
        args.affinity_endpoints = ','.join(endpoints)
    if args.score_config:args.score_config=args.score_config.resolve()
    if args.corpus_config:args.corpus_config=args.corpus_config.resolve()
    if args.workers<1:p.error('--workers must be positive')
    if args.max_tool_calls<1:p.error('--max-tool-calls must be positive')
    import math
    if not math.isfinite(args.viewer_interval) or args.viewer_interval<=0:p.error('--viewer-interval must be finite and positive')
    if not re.fullmatch(r'[\w.-]+',args.run_id):p.error('unsafe run-id')
    verify(args.suite)
    frozen_seed=json.loads((args.suite/'manifest.json').read_text()).get('seed',20261108)
    if args.seed is None:args.seed=frozen_seed
    elif args.per_benchmark is None and args.seed!=frozen_seed:
        p.error('--seed differs from the frozen --suite; changing it requires preparing a new sampled release')
    if args.per_benchmark is not None:
        if args.per_benchmark<1:p.error('--per-benchmark must be positive')
        from smoke import prepare
        args.suite=prepare(args.suite,args.per_benchmark,args.seed)
    summary=json.loads((args.suite/'summary.json').read_text());all_names=[r['benchmark'] for r in summary]
    names=all_names if args.benchmarks=='all' else args.benchmarks.split(',')
    if not names or any(n not in all_names for n in names) or len(set(names))!=len(names):p.error('invalid benchmark list')
    from resume_runs import prepare as prepare_resume
    resume_sources = prepare_resume(args, names)
    counts={r['benchmark']:r['selected'] for r in summary};plans=[];blocked=[]
    if args.phase in ('plan','infer','all'):
        for name in names:
            try:
                if not counts[name]:raise ValueError('no uncontaminated selected tasks')
                cmd,env=infer_command(args,name,counts[name]);plans.append((name,cmd,env))
                print(json.dumps({'benchmark':name,'tasks':counts[name],'command':cmd},ensure_ascii=False))
            except ValueError as e:blocked.append({'benchmark':name,'reason':str(e)})
        if blocked:print(json.dumps({'blocked':blocked},ensure_ascii=False))
        if args.phase=='plan':return
        if blocked and not args.allow_blocked:raise SystemExit('preflight blocked; select explicit available --benchmarks or configure missing backend, or use --allow-blocked')
    record_root=args.output_root/args.run_id;record_root.mkdir(parents=True,exist_ok=True)
    if args.phase in ('infer','all'):
        from preflight import probe_vision
        checked={}
        for _,cmd,env in plans:
            if 'vision' not in cmd[cmd.index('--tool-whitelist')+1].split(','):continue
            endpoints=args.affinity_endpoints or (cmd[cmd.index('--base-urls')+1] if '--base-urls' in cmd else '')
            if not endpoints:raise ValueError('vision preflight needs explicit endpoints')
            for endpoint in endpoints.split(','):
                if endpoint not in checked:
                    checked[endpoint]=probe_vision(endpoint,cmd[cmd.index('--model')+1],env.get('HERMES_API_KEY','EMPTY'))
        dump(record_root/'capability-preflight.json',checked)
    launch_manifest={'suite_sha256':sha(args.suite/'manifest.json'),'benchmarks':names,'workers':args.workers,
        'harness_revision':'20261110-safety-eval-fixes',
        'context_manager':args.context_manager,'url_mode':args.url_mode,'profile':args.profile,
        'runtime_sha256':{p:sha(ROOT/p) for p in ['Evaluation/config/eval_profiles.json','Evaluation/services/eval_run.py',
             'Evaluation/services/native_eval.py','Evaluation/scripts/fetch_watchdog.py','Evaluation/browse_comp_eval/run.py',
             'Evaluation/services/corpus.py','Evaluation/services/viewer_publish.py','Evaluation/benchmark_scores.py',
             'RL_training/glasseek_training/web_tools.py','Evaluation_backend/tools/environments/local.py',
             'Shared/glasseek_shared/fixed_corpus.py','Shared/glasseek_shared/fixed_corpus_sandbox.py',
             'Shared/glasseek_shared/context/official/web.py',
             '../Quest/evaluation/official_dataset/run.py','../Quest/evaluation/official_dataset/score_browsecomp_plus.py',
             'Evaluation/browse_comp_eval/hermes_client.py','Evaluation/services/clm_worker.py','Evaluation/config/prompts/hle-system.txt',
             'Evaluation_backend/plugins/browse_comp_guard/__init__.py','Evaluation_backend/tools/quest_page_retrieval.py','Evaluation_backend/run_agent.py',
             'Shared/glasseek_shared/context/official/run.py','Shared/glasseek_shared/hermes/url_registry.py']},
        'corpus_config_sha256':sha(args.corpus_config) if args.corpus_config else None}
    launch_manifest['research_protocol'] = {'name': args.research_prompt,
        'max_reviews': args.max_reviews, 'protocol_corrections': args.protocol_corrections,
        'accept_confidence': args.accept_confidence, 'medium_confidence': args.medium_confidence,
        'sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'Shared/glasseek_shared/research').glob('*.py'))}}
    launch_manifest['affinity_endpoints'] = args.affinity_endpoints
    launch_manifest['prediction_runs'] = args.prediction_runs
    launch_manifest['resume_sources'] = resume_sources
    for path in ('Evaluation_backend/agent/conversation_loop.py', 'Evaluation_backend/agent/chat_completion_helpers.py',
                 'Evaluation_backend/agent/turn_finalizer.py', 'Evaluation_backend/tools/web_tools.py',
                 'Shared/glasseek_shared/hermes/fetch_provider.py', 'Shared/glasseek_shared/endpoint_slots.py',
                 'Evaluation/browse_comp_eval/runner.py', 'Evaluation/browse_comp_eval/endpoint_registry.py'):
        launch_manifest['runtime_sha256'][path] = sha(ROOT/path)
    launch_manifest['system_prompt_override']={'path':str(args.system_prompt_file),'sha256':sha(args.system_prompt_file),'hle_policy':'official_unchanged'} if args.system_prompt_file else None
    launch_manifest.update(per_benchmark=args.per_benchmark,seed=args.seed,score_config_sha256=sha(args.score_config) if args.score_config else None)
    if args.corpus_config:
        corpus=json.loads(args.corpus_config.read_text())
        launch_manifest['corpus_index_manifest_sha256']=sha(Path(corpus['index']).with_suffix('.manifest.json'))
    launch_manifest.update(eval_python=str(args.eval_python) if args.eval_python else None,
        score_python=str(args.score_python) if args.score_python else None,
        viewer=args.viewer,viewer_interval=args.viewer_interval,max_tool_calls=args.max_tool_calls,score_schedule=args.score_schedule)
    launch_manifest['suite_code_sha256']={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    identity=record_root/'launch.json'
    if identity.exists() and json.loads(identity.read_text())!=launch_manifest:raise ValueError('cannot resume with changed suite/runtime settings')
    dump(identity,launch_manifest)
    if args.phase in ('infer','all'):
        from resume_runs import adopt_grading_ledger
        adopt_grading_ledger(args, resume_sources)
        from orchestrate import execute
        execute(args,plans,blocked,names,counts)
        return
    if args.phase in ('prepare-score','score','all'):
        for name in names:
            packet=record_root/'score-inputs'/name
            if not packet.exists():prepare_score(args.suite,packet,prediction_root(args,name),name,args.corpus_config)
            if args.phase!='prepare-score':
                cmd=[str(args.score_python) if args.score_python else os.environ.get('SUITE_SCORE_PYTHON',str(args.suite.parent/'score-venv/bin/python')),
                     str(Path(__file__).with_name('score.py')),'--suite',str(args.suite),'--packet',str(packet),
                     '--out',str(record_root/'scores'/name)]
                if args.score_config:cmd+=['--config',str(args.score_config)]
                subprocess.run(cmd,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})


if __name__=='__main__':
    import signal
    # Shell background jobs can inherit SIGINT=SIG_IGN. Restore an orderly
    # interrupt so the coordinator stops producers and preserves checkpoints.
    signal.signal(signal.SIGINT,signal.default_int_handler)
    signal.signal(signal.SIGTERM,signal.default_int_handler)
    main()
