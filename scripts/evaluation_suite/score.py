#!/usr/bin/env python3
"""Score explicit packets with frozen author prompts, schemas and metric implementations.

Never routes report/table/attribution tasks through the short-answer grader.
Judge endpoints and keys are supplied separately; no downgrade on API errors.
"""
import argparse
import ast
import asyncio
import contextlib
import dataclasses
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Literal

from suite import ROOT, dump, jsonl, read, sha, verify


def verify_packet(folder):
    packet=json.loads((Path(folder)/'manifest.json').read_text())
    for rel,h in packet['files'].items():
        if sha(Path(folder)/rel)!=h:raise ValueError('score input changed: '+rel)
    return packet


def literal(source, name):
    for node in ast.parse(Path(source).read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            v=node.value
            if isinstance(v,ast.Call) and isinstance(v.func,ast.Attribute) and v.func.attr=='strip':v=v.func.value
            return ast.literal_eval(v).strip()
    raise ValueError('missing author constant '+name)


def source_import(path, name):
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


async def short_answers(name, records, source, cfg, out):
    from openai import AsyncOpenAI
    client=AsyncOpenAI(api_key=os.environ[cfg.get('key_env','OPENAI_API_KEY')],base_url=cfg.get('base_url') or None,timeout=300,max_retries=1)
    if name=='browsecomp':
        template=literal(source/'browsecomp_eval.py','GRADER_TEMPLATE');schema=None
    else:
        code=source/'hle_eval/run_judge_results.py';template=literal(code,'JUDGE_PROMPT')
        from pydantic import BaseModel
        node=next(n for n in ast.parse(code.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ExtractedAnswer')
        namespace={'BaseModel':BaseModel,'Literal':Literal};exec(compile(ast.Module(body=[node],type_ignores=[]),str(code),'exec'),namespace)
        schema=namespace['ExtractedAnswer']
    sem=asyncio.Semaphore(cfg.get('workers',4));results=[]
    async def one(r):
        async with sem:
            prompt=template.format(question=r['question'],response=r['prediction'],correct_answer=r['answer'])
            try:
                if schema:
                    response=await client.beta.chat.completions.parse(model=cfg['model'],max_completion_tokens=4096,
                        messages=[{'role':'user','content':prompt}],response_format=schema)
                    verdict=response.choices[0].message.parsed
                    if verdict is None:raise ValueError('structured judge refusal/empty')
                    data=verdict.model_dump();correct=data['correct']=='yes'
                else:
                    response=await client.chat.completions.create(model=cfg['model'],messages=[{'role':'user','content':prompt}],max_completion_tokens=4096)
                    text=response.choices[0].message.content or '';match=re.search(r'correct: (yes|no)',text)
                    if not match:raise ValueError('unparseable judge verdict')
                    data={'raw':text,'correct':match.group(1)};correct=match.group(1)=='yes'
                result={'id':r['id'],'status':'scored','correct':correct,'judge':data,
                        'usage':response.usage.model_dump() if response.usage else None}
            except Exception as e:result={'id':r['id'],'status':'unscored','error_type':type(e).__name__}
            dump(out/(r['opaque_id']+'.json'),result);results.append(result)
    await asyncio.gather(*(one(r) for r in records));await client.close()
    return results


def main():
    sys.dont_write_bytecode=True
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--packet',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path)
    a=p.parse_args();verify(a.suite)
    a.suite=a.suite.resolve();a.packet=a.packet.resolve();a.out=a.out.resolve()
    packet=verify_packet(a.packet);name=packet['benchmark']
    if packet['release_manifest_sha256']!=sha(a.suite/'manifest.json'):raise ValueError('wrong release')
    for path,h in packet['prediction_inputs'].items():
        if sha(path)!=h:raise ValueError('prediction changed after packet preparation')
    records=list(read(a.packet/'records.jsonl'))
    if not records:raise ValueError('no completed predictions; no judge calls made')
    config=json.loads(a.config.read_text()) if a.config else {}
    defaults={'browsecomp':{'model':'gpt-4o-2024-08-06'},'hle':{'model':'o3-mini-2025-01-31'},
              'liveresearchbench':{'provider':'openai','model':'gpt-5-mini'},'widesearch':{'native_config':'default_eval_config'},
              'mind2web2':{},'browsecomp_plus':{'model':'gpt-4.1'}}
    cfg={**defaults.get(name,{}),**config.get(name,{})}
    cfg['usage_log']=str(a.out/'api_usage.jsonl')
    if cfg.get('credential_file'):
        from judge_transport import credentials
        credentials(cfg)
    if cfg.get('transport'):
        from judge_transport import install
        install(cfg)
    if cfg.get('key_env') and os.environ.get(cfg['key_env']):os.environ['OPENAI_API_KEY']=os.environ[cfg['key_env']]
    if cfg.get('base_url'):os.environ['OPENAI_BASE_URL']=cfg['base_url']
    base=a.suite.parent;authors=base/'evaluators';quest=ROOT.parent/'Quest'
    a.out.mkdir(parents=True,exist_ok=False,mode=0o700)
    observation={'benchmark':name,'assigned':packet['assigned'],'completed':packet['completed'],
                 'protocol':packet['protocol'],'packet_sha256':sha(a.packet/'manifest.json'),
                 'config':{k:v for k,v in cfg.items() if k not in ('api_key','key')},'status':'running'}
    observation['runtime_package_versions']={pkg:importlib.metadata.version(pkg) for pkg in ['openai','pydantic','numpy','pandas','datasets']}
    dump(a.out/'receipt.json',observation)
    try:
        if name in ('browsecomp','hle'):
            source=authors/('simple-evals' if name=='browsecomp' else 'hle')
            results=asyncio.run(short_answers(name,records,source,cfg,a.out))
            jsonl(a.out/'scores.jsonl',results)
            observation.update(scored=sum(r['status']=='scored' for r in results),
                verified_correct_over_assigned=sum(r.get('correct',False) for r in results)/packet['assigned'],
                correction='BrowseComp regex captures group(1), correcting pinned upstream group(0) aggregation bug' if name=='browsecomp' else None)
            if any(r['status']!='scored' for r in results):raise ValueError('judge errors remain unscored')
            if name=='hle':
                # Import just the pinned author's metric functions, avoiding its
                # CLI's floating full-test download and hardcoded denominator.
                import math
                import numpy as np
                tree=ast.parse((source/'hle_eval/run_judge_results.py').read_text())
                nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('calib_err','dump_metrics')]
                scope={'np':np,'math':math};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<hle-native-metrics>','exec'),scope)
                if len(results)>=100:
                    preds={r['id']:{'judge_response':r['judge']} for r in results}
                    with (a.out/'author-metrics.log').open('w') as log,contextlib.redirect_stdout(log):scope['dump_metrics'](preds,packet['assigned'])
                else:observation['native_calibration_status']='undefined_below_100_by_pinned_author_implementation'
        elif name=='gaia':
            cmd=[sys.executable,str(quest/'evaluation/official_dataset/gaia.py'),'--submission',str(a.packet/'submission.jsonl'),
                 '--reference',str(a.packet/'reference.json'),'--source',str(authors/'gaia-leaderboard'),'--out',str(a.out/'author')]
            subprocess.run(cmd,check=True)
        elif name=='drbench':
            cmd=[sys.executable,str(quest/'evaluation/official_dataset/run.py'),'drbench','--raw',str(a.packet/'raw.jsonl'),
                '--query',str(a.packet/'query.jsonl'),'--sources-dir',str(authors),'--runs-dir',str(a.out),
                '--out',str(a.out/'author'),'--workers',str(cfg.get('workers',1))]
            if cfg.get('api_config'):cmd+=['--api-config',cfg['api_config']]
            cmd+=['--fact-stat-runner',str(Path(__file__).with_name('native_fixes.py'))]
            observation['runtime_fixes']=['native-edge-cases-v1:fact-zero-denominator']
            subprocess.run(cmd,check=True)
        elif name=='liveresearchbench':
            sys.path.insert(0,str(authors/'liveresearchbench'))
            os.chdir(authors/'liveresearchbench')
            from liveresearchbench.batch_evaluator import BatchEvaluator
            # Pass frozen full checklists through the author's supported argument;
            # avoids the old 80-task HF default and any runtime date substitution.
            refs=json.loads((a.packet/'benchmark_data.json').read_text())
            from liveresearchbench.common.reference_reports import get_reference_report_path
            missing_depth=[r['id'] for r in records if not get_reference_report_path(r['id'])]
            evaluator=BatchEvaluator(provider=cfg['provider'],model=cfg['model'],output_dir=str(a.out),max_concurrent=cfg.get('workers',4))
            criteria=['presentation','consistency','citation','coverage']
            path=asyncio.run(evaluator.grade_json_file(str(a.packet/'reports.json'),criteria,benchmark_data=refs))
            detailed=sorted(Path(path).glob('detailed_results_*.json'))
            if len(detailed)!=1:raise ValueError('ambiguous/missing native LRB detailed results')
            data=json.loads(detailed[0].read_text())
            if any(any(c+'_grading_results' not in r or r[c+'_grading_results'].get('error') for c in criteria) for r in data['reports']):
                raise ValueError('native LRB criterion missing/failed; no complete score')
            depth_reports=[r for r in data['reports'] if r['query_id'] not in missing_depth]
            if depth_reports:
                depth_input=a.out/'depth-input.json';dump(depth_input,{'metadata':{'use_realtime':False},'reports':depth_reports})
                depth_evaluator=BatchEvaluator(provider=cfg['provider'],model=cfg['model'],output_dir=str(a.out/'depth'),max_concurrent=cfg.get('workers',4))
                asyncio.run(depth_evaluator.grade_json_file(str(depth_input),['depth'],benchmark_data=refs))
            observation['depth_coverage']={'assigned':packet['assigned'],'native_reference_available':len(depth_reports),'unscored_missing_reference':missing_depth}
        elif name=='mind2web2':
            source=authors/'mind2web2'
            blockers={};errors={}
            for r in records:
                task_script=base/'sources/Mind2Web-2/evaluation_scripts/2025_10_23'/(r['id']+'.py')
                # Official task dependencies are audited before executing that
                # task; never replace Maps directions with an LLM estimate.
                if 'GoogleMapsTool(' in task_script.read_text() and not os.getenv('GOOGLE_MAPS_API_KEY'):
                    blockers[r['id']]={'status':'blocked','reason':'GOOGLE_MAPS_API_KEY missing; official task requires Google Maps'}
                    continue
                cmd=[sys.executable,str(source/'run_eval.py'),'--task_id',r['id'],'--agent_name','glasseek',
                    '--answer_folder',str(a.packet/'answers'),'--eval_scripts_root',str(base/'sources/Mind2Web-2/evaluation_scripts'),
                    '--eval_version','2025_10_23','--eval_results_root',str(a.out/'author'),'--cache_root',str(a.out/'cache')]
                if cfg.get('transport'):
                    cmd=[sys.executable,str(Path(__file__).with_name('judge_transport.py')),'--config',str(a.config.resolve()),'--benchmark',name,'--usage-log',str(a.out/'api_usage.jsonl'),'--script',str(source/'run_eval.py')]+cmd[2:]
                result=subprocess.run(cmd,cwd=source,check=False)
                if result.returncode:errors[r['id']]={'status':'error','exit_code':result.returncode}
            dump(a.out/'task-blockers.json',blockers)
            sys.path.insert(0,str(ROOT/'Evaluation'))
            from benchmark_scores import task_scores
            native=task_scores(a.out,name,records)
            observation.update(scored=sum(r['status']=='scored' for r in native.values()),task_blockers=blockers,
                task_errors={**errors,**{k:{'status':v['status'],'reason':'native result missing'} for k,v in native.items() if v['status']=='pending'}})
            if observation['task_errors']:raise ValueError('native Mind2Web2 task errors remain unscored')
        elif name=='widesearch':
            source=authors/'widesearch';sys.path.insert(0,str(source));os.chdir(source)
            from src.evaluation.data_loader import WideSearchDataLoader,WideSearchResponse
            from src.evaluation import evaluation as wide_evaluation
            from native_fixes import fix_widesearch
            observation['runtime_fixes']=[fix_widesearch(wide_evaluation)]
            from src.utils.config import model_config
            native=model_config[cfg['native_config']]
            native['base_url']=os.environ.get(cfg.get('azure_endpoint_env','AZURE_OPENAI_ENDPOINT'),native['base_url'])
            native['api_key']=os.environ.get(cfg.get('key_env','OPENAI_API_KEY'),native['api_key'])
            native['model_name']=os.environ.get(cfg.get('azure_deployment_env','AZURE_OPENAI_DEPLOYMENT'),native['model_name'])
            if cfg.get('transport'):
                from openai import OpenAI
                from src.evaluation import metric_utils
                client=OpenAI(api_key=os.environ[cfg.get('key_env','OPENAI_API_KEY')],base_url=cfg['base_url'])
                # Only replace endpoint transport. Native alignment, cell prompts,
                # parsing, entity metrics and column criteria are used verbatim.
                def completion(messages,model_config_name=None,tools=None):
                    messages=[{'role':'user','content':messages}] if isinstance(messages,str) else messages
                    return client.chat.completions.create(model=cfg['model'],messages=messages,max_tokens=10240).choices[0].message
                metric_utils.llm_completion=completion
            elif not native['base_url'] or not native['api_key']:raise ValueError('native WideSearch evaluator needs configured Azure endpoint/key')
            loader=WideSearchDataLoader(str(a.packet/'queries.jsonl'),str(base/'sources/WideSearch/widesearch_gold'))
            results=[]
            for r in records:
                query=loader.load_query_by_instance_id(r['id'])
                result=wide_evaluation.evaluate_single_query(query,WideSearchResponse(instance_id=r['id'],response=r['prediction']),
                         str(a.out/(r['id']+'.csv')),cfg['native_config'])
                results.append(dataclasses.asdict(result))
            jsonl(a.out/'native_metrics.jsonl',results)
            if any(str(r.get('msg','')).startswith('evaluator error:') for r in results):
                raise ValueError('native WideSearch errors remain unscored')
        elif name=='browsecomp_plus':
            # Corpus retrieval evidence is mandatory. URL visits are not docids.
            if not (a.packet/'corpus-events.jsonl').is_file():raise ValueError('missing authenticated corpus docid events; BC+ remains unscored')
            events={r['opaque_id']:r for r in read(a.packet/'corpus-events.jsonl')}
            errors={}
            for r in records:
                ev=events[r['opaque_id']]
                record={'query_id':r['id'],'tool_call_counts':ev['tool_call_counts'],'status':'completed',
                        'retrieved_docids':ev['retrieved_docids'],'result':[{'type':'output_text','output':r['prediction']}]}
                task=a.out/(r['opaque_id']+'.task.json');rr=a.out/(r['opaque_id']+'.record.json')
                dump(task,{'filename':r['id'],'question':r['question']});dump(rr,record)
                reference=a.out/(r['opaque_id']+'.reference.jsonl');jsonl(reference,[r['reference']])
                cmd=[sys.executable,str(quest/'evaluation/official_dataset/score_browsecomp_plus.py'),
                    '--source',str(authors/'browsecomp-plus'),'--task',str(task),'--record',str(rr),'--reference',str(reference),
                    '--qrels',str(authors/'browsecomp-plus/topics-qrels/qrel_evidence.txt'),
                    '--out',str(a.out/(r['opaque_id']+'.author')),'--model',cfg['model']]
                if cfg.get('transport'):
                    cmd+=['--transport-config',str(a.config),'--transport-runner',str(Path(__file__).with_name('judge_transport.py')),
                          '--usage-log',str(a.out/'api_usage.jsonl'),'--max-output-tokens','4096']
                result=subprocess.run(cmd,check=False)
                if result.returncode:errors[r['id']]=result.returncode
            observation['task_errors']=errors
            observation['native_calibration_status']='undefined_below_100' if len(records)<100 else 'see_author_outputs'
            if errors:raise ValueError('some BC+ native judges failed; other tasks retained')
        else:raise ValueError('no native grader for '+name)
        observation['status']='native_evaluation_finished_requires_completeness_audit'
        if name=='mind2web2' and observation.get('task_blockers'):observation['status']='native_evaluation_partial_dependency_blocked'
        if name=='drbench':
            fact_status=a.out/'author/fact/fact_result.status.json'
            if fact_status.exists():
                observation['fact_status']=json.loads(fact_status.read_text())
                if observation['fact_status']['status']!='defined':observation['status']='native_evaluation_partial_no_verifiable_citations'
        if name=='liveresearchbench' and observation['depth_coverage']['unscored_missing_reference']:
            observation['status']='native_evaluation_partial_missing_depth_reference'
    except Exception as e:
        observation.update(status='unscored_error',error_type=type(e).__name__)
        dump(a.out/'receipt.json',observation);raise
    dump(a.out/'receipt.json',observation)


if __name__=='__main__':main()
