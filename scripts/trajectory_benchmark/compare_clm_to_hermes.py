"""Historical no-CLM baselines, paired only on scored identical questions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from clm_comparison import atomic, read, ROOT


def compressions(path):
    if not Path(path).exists():return None
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
        return db.execute("SELECT COUNT(*) FROM sessions WHERE end_reason='compression'").fetchone()[0]


def freeze_baselines(output):
    destination=output/'no-clm-baselines.json'
    if destination.exists():return read(destination)
    source=ROOT/'reports/url-modes-34-20261107/live-v3-full-c8/final-analysis/report.json'
    original=read(source);first=[]
    for r in original['rows']:
        if r['mode']!='raw':continue
        first.append(dict(question_id=r['question_id'],question_sha256=hashlib.sha256(r['question'].encode()).hexdigest(),
            score=r['judge']['score'],input_tokens=r['input_tokens_including_cache'],cached=r['cache_read_tokens'],
            output_tokens=r['output_tokens'],model_calls=r['api_calls'],
            web_search=r['counts'].get('web_search_calls',0),web_extract=r['counts'].get('web_extract_calls',0),
            wall_seconds=r['duration_seconds'],context_operations=compressions(r['db_path']),
            retrieval_errors=r['retrieval_error_count']))
    run='url_34_raw_20261107_221522';folder=ROOT/'Evaluation/output/preds'/run/'browsecomp_rl9b_non_timeout_34';second=[]
    for path in sorted((folder/'conv').glob('*/*.json')):
        case=read(path);judge=read(folder/'judge_official'/case['question_id']/(case['run_id']+'.json'),{})
        if case['status']!='success' or judge.get('status')!='judged':continue
        db_path=ROOT/'Evaluation/data/runtime/hermes/profiles'/case['profile']/'state.db'
        with sqlite3.connect(f'file:{db_path}?mode=ro',uri=True) as db:
            values=db.execute('SELECT SUM(input_tokens),SUM(output_tokens),SUM(cache_read_tokens),'
                              'SUM(cache_write_tokens),SUM(api_call_count) FROM sessions').fetchone()
        second.append(dict(question_id=case['question_id'],question_sha256=hashlib.sha256(case['question'].encode()).hexdigest(),
            score=judge['score'],input_tokens=sum(values[i] or 0 for i in (0,2,3)),cached=values[2] or 0,
            output_tokens=values[1] or 0,model_calls=values[4] or 0,
            web_search=case['tool_calls'].get('web_search',0),web_extract=case['tool_calls'].get('web_extract',0),
            wall_seconds=case['wall_seconds'],context_operations=compressions(db_path),retrieval_errors=None))
    data=dict(first=dict(run='url_34_raw_20261107_170244',rows=first),
        repeat=dict(run=run,rows=second),source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    atomic(destination,data);return data


def summary(rows):
    out={k:sum(r[k] for r in rows) for k in ('score','input_tokens','cached','output_tokens','model_calls',
                                           'web_search','web_extract','wall_seconds')}
    out.update(n=len(rows),accuracy=out['score']/len(rows) if rows else None,
        mean_seconds=out['wall_seconds']/len(rows) if rows else None,
        uncached=out['input_tokens']-out['cached'],
        cache_hit_fraction=out['cached']/out['input_tokens'] if out['input_tokens'] else None,
        context_operations=sum(r.get('context_operations') or 0 for r in rows))
    return out


def update(output,baselines):
    current=read(output/'comparison.json');new={}
    dataset={r['id']:r for r in [json.loads(s) for s in (output/'dataset.jsonl').read_text().splitlines() if s.strip()]}
    for row in current['rows']:
        if row['status']!='success' or row['score'] is None:continue
        new[row['question_id']]=dict(question_id=row['question_id'],score=row['score'],
            input_tokens=row['input_tokens'],cached=row['cached_input_tokens'],output_tokens=row['output_tokens'],
            model_calls=row['model_calls'],web_search=row['web_search'],web_extract=row['web_extract'],
            wall_seconds=row['wall_seconds'],context_operations=row['context_edits'])
    old={name:{r['question_id']:r for r in data['rows']} for name,data in baselines.items() if name in ('first','repeat')}
    ids=sorted(new.keys()&old['first'].keys()&old['repeat'].keys())
    for q in ids:
        expected=hashlib.sha256(dataset[q]['question'].encode()).hexdigest()
        assert all(old[name][q]['question_sha256']==expected for name in old)
    summaries={name:summary([rows[q] for q in ids]) for name,rows in old.items()}
    summaries['clm']=summary([new[q] for q in ids])
    clean=[q for q in ids if old['first'][q]['retrieval_errors']==0]
    report=dict(at_utc=current['at_utc'],status=current['status'],run=current['run'],common_ids=ids,
        paired=summaries,paired_rows=[dict(question_id=q,first=old['first'][q],repeat=old['repeat'][q],clm=new[q]) for q in ids],
        baseline_overall={name:summary(data['rows']) for name,data in baselines.items() if name in ('first','repeat')},
        first_round_without_retrieval_error=dict(ids=clean,hermes=summary([old['first'][q] for q in clean]),clm=summary([new[q] for q in clean])),
        limits=['Baselines use the same model, raw URLs, identical questions, 128K context and 16384 output budget.',
            'Historical URL modes each used concurrency 2 and shared a max-seqs-8 server; current CLM uses concurrency 12.',
            'The first raw run had a retrieval outage. Counts include only identical, jointly scored tasks.',
            'Hermes allowed parallel native web tools per model response; official CLM uses a single bash action.',
            'Prompt, tool error propagation, web results and load also differ. This is not an isolated causal estimate of context editing.',
            'Traditional summary compressions and CLM mirror edits are different operations.'])
    atomic(output/'comparison-vs-hermes.json',report)
    labels={'first':'Hermes first run','repeat':'Hermes repeat','clm':'CLM + Hermes'}
    lines=['# CLM versus pre-CLM Hermes','',f"Status: {current['status']}. Common scored tasks: {len(ids)}. Updated: {current['at_utc']}.",'',
        '| Metric | Hermes first run | Hermes repeat | CLM + Hermes |','|---|---:|---:|---:|']
    for title,key in [('Correct','score'),('Model calls','model_calls'),('web_search','web_search'),('web_extract','web_extract'),
        ('Input including cache','input_tokens'),('Uncached input','uncached'),('Output','output_tokens'),
        ('Mean seconds','mean_seconds'),('Context operations','context_operations')]:
        values=[summaries[name][key] for name in labels]
        lines.append('| '+title+' | '+' | '.join('—' if x is None else f'{x:,.2f}' if key=='mean_seconds' else f'{x:,.0f}' for x in values)+' |')
    lines+=['','| Question | Hermes first | Hermes repeat | CLM |','|---|---:|---:|---:|']
    lines+=['| '+q+' | '+' | '.join(str(int(rows[q]['score'])) for rows in (old['first'],old['repeat'],new))+' |' for q in ids]
    lines+=['','## Limits','']+['- '+s for s in report['limits']]
    (output/'comparison-vs-hermes.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    baselines=freeze_baselines(args.output)
    while True:
        report=update(args.output,baselines)
        print(json.dumps({'status':report['status'],'paired':report['paired']},ensure_ascii=False),flush=True)
        if not args.watch or report['status']!='running':break
        time.sleep(20)
