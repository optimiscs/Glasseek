"""Account for native CLM runs without counting paused attempts as answers."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import re
import sqlite3

ROOT = Path(__file__).resolve().parents[2]


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def load_cases(run, dataset):
    folder = ROOT / 'Evaluation/output/preds' / run / dataset
    cases = []
    for path in sorted((folder / 'conv').glob('*/*.json')):
        case = read(path)
        if not case:
            continue
        work = Path(case['workspace_dir'])
        sidecar = read(work / 'official-clm-result.json', {})
        usage = read(work / 'official-clm/agent/usage.json', sidecar.get('usage', {}))
        judge = read(folder / 'judge_official' / case['question_id'] / (case['run_id'] + '.json'), {})
        operator_interrupted = (not case.get('timed_out') and
            (case.get('status') == 'interrupted' or case.get('returncode') in (-2, -15)
             or case.get('error') == 'evaluation interrupted'))
        calls = sidecar.get('web_calls') or read(work/'official-clm/workspace/.clm/web-calls.json') or {k: v for k, v in case.get('tool_calls', {}).items()
                                           if k in ('web_search', 'web_extract')}
        cached = max(usage.get('cached_tokens', 0), usage.get('cache_read_input_tokens', 0))
        cases.append(dict(question_id=case['question_id'], run_id=case['run_id'],
            status=case['status'], operator_interrupted=operator_interrupted,
            score=judge.get('score') if judge.get('status') == 'judged' else None,
            judge_status=judge.get('status'), error=case.get('error', ''),
            answer=case.get('model_response', ''), submitted_file_answer=bool(sidecar.get('answer')),
            completion_protocol=sidecar.get('completion_protocol', 'legacy-file-echo'),
            wall_seconds=(case.get('wall_seconds') or 0)+read(work/'continuation.json', {}).get('previous_wall_seconds', 0),
            input_tokens=usage.get('prompt_tokens'), cached_input_tokens=cached if usage else None,
            output_tokens=usage.get('completion_tokens'),
            model_calls=usage.get('n_lm_calls'), bash=usage.get('n_bash'),
            web_search=calls.get('web_search', 0), web_extract=calls.get('web_extract', 0),
            context_edits=usage.get('n_ctx_syncs_real'), context_syncs=usage.get('n_ctx_syncs'),
            rejected_edits=usage.get('n_ctx_rejected'), rollbacks=usage.get('n_retry_on_limit'),
            usage_available=bool(usage), source=str(path)))
    return cases


def aggregate(rows):
    measured = [r for r in rows if r['usage_available']]
    scored = [r for r in rows if r['score'] is not None]
    sums = {key: sum(r.get(key) or 0 for r in measured) for key in
            ('input_tokens', 'cached_input_tokens', 'output_tokens', 'model_calls', 'bash',
             'context_edits', 'context_syncs', 'rejected_edits', 'rollbacks')}
    sums.update(web_search=sum(r['web_search'] for r in rows), web_extract=sum(r['web_extract'] for r in rows))
    n = len(rows)
    correct = sum(r['score'] == 1 for r in scored)
    success = sum(r['status'] == 'success' for r in rows)
    return dict(cases=n, success=success, failed=sum(r['status'] == 'failed' for r in rows),
        operator_interrupted=sum(r['operator_interrupted'] for r in rows),
        scored=len(scored), correct=correct, incorrect=len(scored)-correct,
        accuracy_scored=correct/len(scored) if scored else None,
        submission_rate=success/n if n else None,
        usage_cases=len(measured), **sums,
        cache_hit_fraction=sums['cached_input_tokens']/sums['input_tokens'] if sums['input_tokens'] else None,
        wall_seconds_sum=sum(r['wall_seconds'] or 0 for r in rows),
        errors=dict(Counter(r['error'] for r in rows if r['error'])))


def live_counters(folder, finished):
    """Successful API responses only; never add these to final usage a second time."""
    result = dict(cases=0, input_tokens=0, output_tokens=0, cached_input_tokens=0, model_calls=0,
                  observed_applied_edits=0, observed_large_edits=0)
    for work in (folder / 'workspace').glob('*'):
        if work.name in finished:
            continue
        request = read(work / 'attempt_1/official-clm-request.json', {})
        if not request:
            continue
        profile = request['session_id'].removeprefix('clm-')
        db_path = ROOT / 'Evaluation/data/runtime/hermes/profiles' / profile / 'state.db'
        if not db_path.exists():
            continue
        try:
            with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=1) as db:
                data = db.execute('SELECT SUM(input_tokens),SUM(output_tokens),SUM(cache_read_tokens),'
                    'SUM(cache_write_tokens),SUM(api_call_count) FROM sessions').fetchone()
                inputs, outputs, cache, writes, calls = [x or 0 for x in data]
                result['cases'] += 1
                result['input_tokens'] += inputs + cache + writes
                result['output_tokens'] += outputs
                result['cached_input_tokens'] += cache
                result['model_calls'] += calls
                notes = db.execute("SELECT content FROM messages WHERE role='tool' AND "
                    "COALESCE(inherited_from_previous_session,0)=0 AND content LIKE '%LIVE_CTX_MAIN.txt: edit applied%'").fetchall()
                for (note,) in notes:
                    result['observed_applied_edits'] += 1
                    match = re.search(r'context ~(\d+)->(\d+) tokens', note)
                    if match and abs(int(match[1])-int(match[2])) >= max(32, int(match[1])*.01):
                        result['observed_large_edits'] += 1
        except sqlite3.Error:
            pass
    return result


def build_report(output):
    output = Path(output)
    state = read(output / 'status.json')
    run = state['runs']['clm']['run']
    dataset = state['dataset']
    rows = load_cases(run, dataset)
    previous = read(output / 'previous-round-snapshot.json')['rows']
    old_terminal = [r for r in previous if not r['operator_interrupted']]
    new_terminal = [r for r in rows if not r['operator_interrupted']]
    old_by_id = {r['question_id']: r for r in old_terminal}
    new_by_id = {r['question_id']: r for r in new_terminal}
    ids = sorted(old_by_id.keys() & new_by_id.keys())
    paired_scored = [q for q in ids if old_by_id[q]['score'] is not None and new_by_id[q]['score'] is not None]
    pairs = [dict(question_id=q, previous=old_by_id[q], current=new_by_id[q],
        input_token_delta=(new_by_id[q]['input_tokens']-old_by_id[q]['input_tokens'])
            if new_by_id[q]['input_tokens'] is not None and old_by_id[q]['input_tokens'] is not None else None,
        wall_seconds_delta=(new_by_id[q]['wall_seconds']-old_by_id[q]['wall_seconds'])
            if new_by_id[q]['wall_seconds'] is not None and old_by_id[q]['wall_seconds'] is not None else None)
        for q in ids]
    folder = ROOT / 'Evaluation/output/preds' / run / dataset
    active = live_counters(folder, {r['run_id'] for r in rows})
    completed = aggregate(new_terminal)
    report = dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), run=run, status=state['status'],
        workers=state.get('workers'), max_tool_calls=state.get('max_tool_calls'), timeout_seconds=state.get('timeout_seconds'),
        total=34, policy_phases=state.get('policy_phases', []), previous_run=read(output / 'previous-round-snapshot.json')['run'],
        previous_all_recorded=aggregate(previous), previous_naturally_terminal=aggregate(old_terminal),
        current_all_recorded=aggregate(rows), current_naturally_terminal=completed,
        current_live_observed=active,
        current_observed_tokens=dict(input_tokens=completed['input_tokens']+active['input_tokens'],
            output_tokens=completed['output_tokens']+active['output_tokens'],
            cached_input_tokens=completed['cached_input_tokens']+active['cached_input_tokens']),
        common_terminal=dict(ids=ids, previous=aggregate([old_by_id[q] for q in ids]),
            current=aggregate([new_by_id[q] for q in ids])),
        common_scored=dict(ids=paired_scored, previous=aggregate([old_by_id[q] for q in paired_scored]),
            current=aggregate([new_by_id[q] for q in paired_scored])),
        pairs=pairs, rows=rows, source_drift=state.get('source_drift', []),
        semantics=['Input tokens include cached tokens; do not add cache reads a second time.',
            *(['Current run contains the original c12/3600-second phase and a c5/no-total-timeout continuation with a cumulative 120 tool-round budget.',
               'Resumed cases use cumulative native token counts and cumulative execution wall time; paused wall time is excluded.'] if state.get('policy_phases') else []),
            'bash counts model-facing calls; web counts underlying bridge calls. They overlap.',
            'Exact context-edit counts come from official n_ctx_syncs_real for finished cases. Live observations are separate lower bounds.',
            'Previous operator-interrupted attempts are excluded from natural-terminal and paired comparisons.',
            'Previous missing/invalid submissions have no answer score; they are not silently assigned a judge score of zero.',
            'Previous run was paused and changed recording code mid-run. Search results and server load also vary; comparison is descriptive.'])
    atomic(output / 'comparison.json', report)
    fmt = lambda value: '—' if value is None else f'{value:,.0f}'
    pct = lambda value: '—' if value is None else f'{value:.1%}'
    old, new = report['previous_naturally_terminal'], completed
    lines = ['# CLM + Hermes 34-task evaluation', '', f'Status: **{state["status"]}**. Updated: {report["at_utc"]}.', '',
        f'Run: `{run}`. Workers: {state.get("workers", 12)}. Previous natural terminal cases: {old["cases"]}; current: {new["cases"]}/34.', '',
        '| Metric | Previous natural terminal | Current natural terminal |', '|---|---:|---:|']
    for label, key in [('Cases','cases'),('Successful submissions','success'),('Scored','scored'),('Correct','correct'),
        ('Input tokens (includes cache)','input_tokens'),('Cached input tokens','cached_input_tokens'),
        ('Output tokens','output_tokens'),('Model calls','model_calls'),('bash calls','bash'),
        ('web_search','web_search'),('web_extract','web_extract'),('Real context edits','context_edits'),
        ('Rollback retries','rollbacks')]:
        lines.append(f'| {label} | {fmt(old[key])} | {fmt(new[key])} |')
    lines += [f'| Accuracy among scored answers | {pct(old["accuracy_scored"])} | {pct(new["accuracy_scored"])} |', '',
        f'Common naturally ended tasks: {len(ids)}. Common scored tasks: {len(paired_scored)}.', '',
        '| Task | Previous status / score | Current status / score | Input token delta | Wall seconds delta |', '|---|---|---|---:|---:|']
    for pair in pairs:
        a,b=pair['previous'],pair['current']
        lines.append(f'| {pair["question_id"]} | {a["status"]} / {a["score"]} | {b["status"]} / {b["score"]} | {fmt(pair["input_token_delta"])} | {fmt(pair["wall_seconds_delta"])} |')
    lines += ['', '## Accounting limits', ''] + ['- '+x for x in report['semantics']]
    (output / 'comparison.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.output)
    print(json.dumps({'status': report['status'], 'current': report['current_naturally_terminal'],
                      'common': len(report['common_terminal']['ids'])}, ensure_ascii=False))
