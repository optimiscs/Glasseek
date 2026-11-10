"""Read native evaluation artifacts for the viewer without changing any scores.

Metrics are task-local unless scope explicitly says run. Missing/error results
are unscored; continuous metrics never imply an invented pass threshold.
"""
import json
import math
import re
from pathlib import Path

LABELS = {'browsecomp': 'BrowseComp', 'browsecomp_plus': 'BrowseComp-Plus',
          'gaia': 'GAIA', 'hle': 'HLE', 'drbench': 'DRBench',
          'liveresearchbench': 'LiveResearchBench', 'mind2web2': 'Mind2Web2', 'widesearch': 'WideSearch'}
BINARY = {'browsecomp', 'browsecomp_plus', 'gaia', 'hle'}
WIDE = {'score': '整表完全匹配', 'precision_by_row': '行精确率', 'recall_by_row': '行召回率',
        'f1_by_row': '行 F1', 'precision_by_item': '单元格精确率',
        'recall_by_item': '单元格召回率', 'f1_by_item': '单元格 F1'}
LRB = {'presentation': '表达', 'consistency': '一致性', 'citation': '引用', 'coverage': '覆盖', 'depth': '深度对比'}


def load(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError, ValueError):return default


def rows(path):
    try:
        # A running scorer may still be appending its final line.
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except (OSError, ValueError):return []


def metric(key, label, value, maximum=None, **extra):
    if isinstance(value, float) and not math.isfinite(value):value = None
    return dict(key=key, label=label, value=value, maximum=maximum, **extra)


def task_scores(folder, benchmark, references):
    """Map official task IDs to scores; never attach a run mean to a task."""
    folder = Path(folder);receipt = load(folder / 'receipt.json', {})
    result = {str(r['id']): {'benchmark': benchmark, 'label': LABELS[benchmark],
        'kind': 'binary' if benchmark in BINARY else 'dimensions', 'status': 'pending',
        'correct': None, 'metrics': [], 'details': None, 'source': str(folder)} for r in references}
    def assign(key, metrics, details, correct=None, status='scored'):
        if str(key) in result:
            result[str(key)].update(metrics=metrics, details=details, correct=correct, status=status)
    if benchmark in ('browsecomp', 'hle', 'gaia'):
        path = folder / ('author/scores.jsonl' if benchmark == 'gaia' else 'scores.jsonl')
        for row in rows(path):
            correct = row.get('correct');key = row.get('id', row.get('task_id'))
            valid = isinstance(correct, bool) and row.get('status', 'scored') == 'scored'
            metrics = [metric('correct', '答案正确', correct)] if valid else []
            judge = row.get('judge') or {}
            if benchmark == 'hle' and valid and isinstance(judge.get('confidence'), (int, float)):
                metrics.append(metric('confidence', '评审置信度', judge['confidence'], 100))
            assign(key, metrics, judge or row, correct if valid else None, 'scored' if valid else 'error')
    elif benchmark == 'browsecomp_plus':
        for reference in references:
            receipt_path = folder / (reference.get('opaque_id', str(reference['id'])) + '.author') / 'receipt.json'
            task_receipt = load(receipt_path, {})
            if not task_receipt:continue
            correct = task_receipt.get('correct')
            valid = task_receipt.get('status') == 'scored' and isinstance(correct, bool)
            details = load(task_receipt.get('judge_result_path') or '', task_receipt)
            metrics = [metric('correct', '答案正确', correct)] if valid else []
            if valid:
                recall=details.get('retrieval',{}).get('recall')
                if isinstance(recall,(int,float)):metrics.append(metric('evidence_recall','证据召回率',recall,1))
                for key,value in details.get('citations',{}).get('metrics',{}).items():
                    if isinstance(value,(int,float)):metrics.append(metric('citation_'+key,'引用 · '+key,value,1))
            assign(reference['id'], metrics,
                   details, correct if valid else None, 'scored' if valid else 'error')
    elif benchmark == 'widesearch':
        for row in rows(folder / 'native_metrics.jsonl'):
            # The author's successful msg contains the per-cell comparison
            # table. Only its explicit exception marker denotes judge failure.
            valid = not str(row.get('msg', '')).startswith('evaluator error:')
            assign(row['instance_id'], [metric(k, label, row.get(k), 1) for k, label in WIDE.items()] if valid else [],
                   row, status='scored' if valid else 'error')
    elif benchmark == 'liveresearchbench':
        merged = {}
        for path in sorted(folder.rglob('detailed_results_*.json')):
            for row in load(path, {}).get('reports', []):
                merged.setdefault(str(row['query_id']), {}).update({k: v for k, v in row.items() if k.endswith('_grading_results')})
        for key, details in merged.items():
            values = [];complete = set()
            for dim, label in LRB.items():
                item = details.get(dim + '_grading_results', {})
                if not item or item.get('error'):continue
                if dim == 'depth':
                    winners = {'evaluated_model': '被评模型胜', 'reference_model': '参考模型胜', 'tie': '平局'}
                    for n in (1, 2):
                        winner_key = f'winner_comparison_{n}'
                        value = item.get(winner_key)
                        values.append(metric(winner_key, f'深度对比 {n}', winners.get(value, value)))
                    if all(item.get(f'winner_comparison_{n}') in winners for n in (1, 2)):complete.add(dim)
                    continue
                elif dim in ('presentation', 'coverage'):
                    summary = item.get('summary', {});value = summary.get('average_pass_rate', summary.get('percentage_addressed'));maximum = 100
                else:value = item.get('score');maximum = 100
                values.append(metric(dim, label, value, maximum))
                if value is not None:complete.add(dim)
            assign(key, values, details, status='scored' if complete == set(LRB) else 'partial')
    elif benchmark == 'mind2web2':
        blockers = load(folder / 'task-blockers.json', {})
        for reference in references:
            key = str(reference['id'])
            evidence = [load(p, {}) for p in sorted((folder / 'author/glasseek' / key).glob('answer_*/results/*.json'))]
            valid = [r for r in evidence if isinstance(r.get('final_score'), (int, float))
                     and not isinstance(r['final_score'], bool) and 0 <= r['final_score'] <= 1
                     and 'eval_breakdown' in r and r.get('answer_name') == 'answer_1.md']
            if valid:
                # The author's summary status "failed" is a legitimate zero
                # score when a real result exists. Exceptions produce no result.
                task = valid[-1];value = task['final_score']
                assign(key, [metric('score', '原生得分', value, 1),
                    metric('success', '原生成功判定', value == 1)], task)
            elif key in blockers:result[key].update(status='blocked', details=blockers[key])
    elif benchmark == 'drbench':
        for row in rows(folder / 'author/race/raw_results.jsonl'):
            values = [metric('RACE', 'RACE', row.get('overall_score'), 1)]
            for k, value in row.items():
                if k.endswith('_score') and k != 'overall_score' and isinstance(value, (int, float)):
                    values.append(metric(k, k, value, 100))
            assign(row.get('id', row.get('query_id')), values, row, status='partial')
        # FACT author output is run-wide. It must not become a per-task score.
    for value in result.values():
        if value['status'] == 'pending' and receipt.get('status') == 'unscored_error':
            value.update(status='error', details={'error_type': receipt.get('error_type')})
        elif value['status'] == 'pending' and receipt.get('status', '').startswith('native_evaluation_'):
            value.update(status='unscored', details={'reason': 'native task result missing'})
    return result


def enrich_package(package, output, data_path, dataset, run):
    """Enrich suite packages in the normal automatic publisher's refresh loop."""
    if not dataset.startswith('suite_') or dataset[6:] not in LABELS:return package
    benchmark = dataset[6:];suffix = '_' + benchmark
    if not run.endswith(suffix):return package
    root = Path(output).parents[2] / run[:-len(suffix)]
    folder = root / 'scores' / benchmark
    attempts = sorted(folder.glob('attempt-*'))
    folder = attempts[-1] if attempts else folder / 'attempt-001'
    refs = [r for r in rows(Path(data_path).parent.parent / 'private/references.jsonl') if r['benchmark'] == benchmark]
    refs_by_opaque = {r['opaque_id']: r for r in refs}
    scores = task_scores(folder, benchmark, refs)
    for case in package['cases']:
        reference = refs_by_opaque.get(case['id'])
        if not reference:continue
        evaluation = scores[str(reference['id'])]
        case.update(field=LABELS[benchmark], fieldZh=LABELS[benchmark], evaluation=evaluation,
                    judge=f'{LABELS[benchmark]} · 原生评分协议')
        case['correct'] = evaluation['correct']
        if not case.get('inProgress') and case.get('evaluationStatus') in ('judging', 'correct', 'incorrect', 'judge_error', 'native_scored', 'native_partial', 'unscored'):
            case['evaluationStatus'] = {'scored': ('correct' if evaluation['correct'] else 'incorrect') if evaluation['kind'] == 'binary' else 'native_scored',
                'partial': 'native_partial', 'error': 'judge_error', 'blocked': 'unscored', 'unscored': 'unscored', 'pending': 'judging'}[evaluation['status']]
        if evaluation['status'] == 'error':case['evaluationError'] = '原生评分未完成或报错；详见评分详情'
        if evaluation['kind'] == 'dimensions':
            case['scoredDimensions'] = {m['key']: m['value'] for m in evaluation['metrics']} or None
        if benchmark in BINARY and reference.get('answer') is not None:case['groundTruth'] = str(reference['answer'])
    run_data = package['run'];values = [c['evaluation'] for c in package['cases'] if c.get('evaluation')]
    scored = sum(x['status'] == 'scored' for x in values);partial = sum(x['status'] == 'partial' for x in values)
    metrics = []
    if benchmark in BINARY:
        judged = [x for x in values if isinstance(x['correct'], bool)]
        metrics.append(metric('accuracy', '已评分准确率', sum(x['correct'] for x in judged) / len(judged) if judged else None, 1, count=len(judged), scope='run'))
        if benchmark=='browsecomp_plus':
            values=[m['value'] for x in judged for m in x['metrics'] if m['key']=='evidence_recall']
            metrics.append(metric('evidence_recall','证据召回率',sum(values)/len(values) if values else None,1,count=len(values),scope='run'))
    else:
        for key in dict.fromkeys(m['key'] for x in values for m in x['metrics']):
            ms = [m for x in values if x['status'] in ('scored', 'partial') for m in x['metrics'] if m['key'] == key and isinstance(m['value'], (int, float)) and not isinstance(m['value'], bool)]
            if ms:metrics.append(dict(ms[0], value=sum(m['value'] for m in ms) / len(ms), count=len(ms), scope='run'))
    if benchmark == 'liveresearchbench':
        for path in sorted(folder.rglob('summary_*.json')):
            depth = load(path, {}).get('overall_results', {}).get('depth')
            if depth:metrics.append(metric('depth_win_rate', '深度胜率（排除平局）', depth.get('win_rate'), 100, scope='run', count=depth.get('decisive_games', 0)))
    if benchmark == 'drbench':
        fact_status = load(folder / 'author/fact/fact_result.status.json', {})
        if fact_status:
            metrics.append(metric('FACT.status', 'FACT 可评估性', fact_status.get('status'), scope='run'))
            metrics.append(metric('FACT.citation_coverage', '包含正文引用的报告比例', fact_status.get('citation_coverage'), 1, scope='run', count=fact_status.get('reports')))
        fact = folder / 'author/fact/fact_result.txt'
        if fact.is_file():
            for key, value in re.findall(r'^([a-z_]+):\s*([0-9.]+)', fact.read_text(), re.M):
                metrics.append(metric('FACT.' + key, 'FACT · ' + key, float(value), scope='run'))
    run_data.update(benchmark=f'{LABELS[benchmark]} · {run_data.get("requestedCases", len(refs))} 题',
        judgeScored=scored, judgePassed=sum(x['correct'] is True for x in values),
        evaluation={'benchmark': benchmark, 'label': LABELS[benchmark], 'kind': 'binary' if benchmark in BINARY else 'dimensions',
            'scored': scored, 'partial': partial, 'assigned': run_data.get('requestedCases', len(refs)), 'metrics': metrics})
    counts = {}
    for case in package['cases']:counts[case['evaluationStatus']] = counts.get(case['evaluationStatus'], 0) + 1
    run_data.update(statusCounts=counts, errorCount=sum(counts.get(k, 0) for k in ('inference_error', 'tool_error', 'judge_error', 'user_terminated')),
        wiki=f'{LABELS[benchmark]}；完整评分 {scored}，部分评分 {partial}；各指标保留原生量纲，缺失评分不记作零分。')
    return package
