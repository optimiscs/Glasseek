"""Judge completed checkpoint trajectories promptly; keep failures separate."""
import argparse
import fcntl
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time

E = Path(os.environ.get('GLASSEEK_EVAL_ROOT', '/root/autodl-tmp/Glasseek/Evaluation'))
sys.path.insert(0, str(E))
from browse_comp_eval.judge import (BrowseCompJudge, JudgeClient, JudgeClientConfig,
    JudgeRunConfig, load_prediction_record, prediction_fingerprint, read_json_object)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--dataset', default='browsecomp_rl9b_non_timeout_34')
    parser.add_argument('--interval', type=float, default=3)
    parser.add_argument('--handoff-lock', type=Path)
    parser.add_argument('--handoff-stop', type=Path)
    args = parser.parse_args()
    lock_file = None
    if args.handoff_lock:
        lock_file = args.handoff_lock.open('a')
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = E / 'output/preds' / args.run / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    conv = root / 'conv'
    output = root / 'judge_official'
    output.mkdir(exist_ok=True)
    key = json.loads(Path(os.environ.get('REEVAL_CREDENTIALS', '/root/autodl-tmp/Glasseek/checkpoint-reeval/private/eval-credentials.json')).read_text())['judge']
    cfg = JudgeClientConfig(api_key=key, base_url='https://api.deepseek.com/v1', model='deepseek-flash', provider='openai', api_version='', timeout_seconds=60, max_retries=3, max_tokens=2048, temperature=0, top_p=1, token_param='max_tokens', use_json_schema=False, judge_template='browsecomp')
    run = JudgeRunConfig(conv_dir=conv, output_dir=output, summary_path=root/'judge_official_summary.jsonl', metrics_path=root/'judge_official_metrics.json', num_workers=4, force=False, include_failed=False)
    judge = BrowseCompJudge(JudgeClient(cfg), run)
    active = {}
    retry_after = {}
    def evaluate(rec, fingerprint):
        try:
            if rec.pred_status != 'success':
                result = judge._base_result(rec)
                result.update(status='inference_error', score=None, error=str(rec.raw.get('error') or rec.pred_status))
            elif not rec.pred.strip():
                result = judge._base_result(rec)
                result.update(status='inference_error', score=None, error='empty prediction')
            else:
                scored = judge.client.judge(rec)
                result = judge._base_result(rec)
                result.update(status='judged', score=scored['score'], judge_label=scored.get('judge_label'), judge_reason=scored['judge_reason'], judge_raw_response=scored['judge_raw_response'], judge_response=scored['judge_response'])
        except Exception as exc:
            result = judge._judge_failed_result(rec, exc)
        current = load_prediction_record(rec.path)
        if current is None or prediction_fingerprint(current) != fingerprint:
            return 'changed'
        judge._write_result(rec, result)
        return result['status']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while True:
            if args.handoff_stop and args.handoff_stop.exists():
                print('Final judge requested handoff; finishing in-flight scores', flush=True)
                break
            for future, (keyid, fp) in list(active.items()):
                if future.done():
                    try: status = future.result()
                    except Exception as exc:
                        print(f'worker error {keyid}: {exc}', flush=True); status = 'judge_failed'
                    if status == 'judge_failed': retry_after[(keyid, fp)] = time.monotonic()+60
                    print(f'{args.run} {keyid} {status}', flush=True)
                    del active[future]
            busy = {v for v in active.values()}
            if conv.exists():
                for path in sorted(conv.glob('*/*.json')):
                    if len(active) >= 4: break
                    rec = load_prediction_record(path)
                    if not rec or rec.pred_status not in ('success', 'failed', 'policy_violation') or rec.raw.get('attempt_consumed') is False: continue
                    fp = prediction_fingerprint(rec)
                    ident = (rec.run_id, fp)
                    prior = read_json_object(output / rec.question_id / (rec.run_id + '.json'))
                    if prior and prior.get('pred_fingerprint') == fp and prior.get('status') in ('judged', 'inference_error'): continue
                    if ident in busy or time.monotonic() < retry_after.get(ident, 0): continue
                    active[pool.submit(evaluate, rec, fp)] = ident
            time.sleep(args.interval)

if __name__ == '__main__':
    main()
