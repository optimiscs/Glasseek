"""Judge completed checkpoint trajectories promptly; keep failures separate."""
import argparse
import fcntl
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time

E = Path(os.environ.get('GLASSEEK_EVAL_ROOT', str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(E))
from browse_comp_eval.judge import (BrowseCompJudge, JudgeClient, JudgeClientConfig,
    JudgeRunConfig, load_prediction_record, prediction_fingerprint, read_json_object)
from browse_comp_eval.judge_policy import RecordJudgePolicy


def load_api_key():
    if os.environ.get('JUDGE_API_KEY'):
        return os.environ['JUDGE_API_KEY']
    explicit = os.environ.get('REEVAL_CREDENTIALS') or os.environ.get('GLASSEEK_API_CONFIG')
    quest = Path(os.environ.get('QUEST_ROOT', E.parent.parent / 'Quest'))
    paths = [Path(explicit)] if explicit else [
        E.parent / 'checkpoint-reeval/private/eval-credentials.json',
        quest / 'workspace/strict-reproduction/.secrets/api_config.json']
    for path in paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        key = data.get('judge') or data.get('common', {}).get('JUDGE_OPENAI_API_KEY')
        if key:
            return key
    raise RuntimeError('Judge credentials unavailable: configure JUDGE_API_KEY or a credential file')


def owner_alive(pid):
    if not pid:
        return True
    try:
        os.kill(pid, 0)
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
    except (OSError, IndexError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--dataset', default='browsecomp_rl9b_non_timeout_34')
    parser.add_argument('--interval', type=float, default=3)
    parser.add_argument('--handoff-lock', type=Path)
    parser.add_argument('--handoff-stop', type=Path)
    parser.add_argument('--config', type=Path, default=E / 'config/async_judge.json')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--retry-seconds', type=float, default=60)
    parser.add_argument('--parent-pid', type=int, default=0, help='Drain completed predictions and exit after evaluation ends')
    parser.add_argument('--viewer-model', help='Publish final scores to the existing viewer before exiting')
    args = parser.parse_args()
    if args.workers < 1 or args.retry_seconds < 0:
        parser.error('workers must be positive and retry-seconds nonnegative')
    policy = RecordJudgePolicy(args.workers, args.retry_seconds)
    lock_file = None
    if args.handoff_lock:
        args.handoff_lock.parent.mkdir(parents=True, exist_ok=True)
        lock_file = args.handoff_lock.open('a')
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = E / 'output/preds' / args.run / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    conv = root / 'conv'
    output = root / 'judge_official'
    output.mkdir(exist_ok=True)
    key = load_api_key()
    settings = json.loads(args.config.read_text())
    for field, variable in (('base_url', 'JUDGE_BASE_URL'), ('model', 'JUDGE_MODEL')):
        if os.environ.get(variable):
            settings[field] = os.environ[variable]
    cfg = JudgeClientConfig(api_key=key, **settings)
    run = JudgeRunConfig(conv_dir=conv, output_dir=output, summary_path=root/'judge_official_summary.jsonl', metrics_path=root/'judge_official_metrics.json', num_workers=policy.workers, force=False, include_failed=False)
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=policy.workers) as pool:
        while True:
            if args.handoff_stop and args.handoff_stop.exists():
                print('Final judge requested handoff; finishing in-flight scores', flush=True)
                break
            for future, (keyid, fp) in list(active.items()):
                if future.done():
                    try: status = future.result()
                    except Exception as exc:
                        print(f'worker error {keyid}: {exc}', flush=True); status = 'judge_failed'
                    if status == 'judge_failed': retry_after[(keyid, fp)] = time.monotonic()+policy.retry_seconds
                    print(f'{args.run} {keyid} {status}', flush=True)
                    del active[future]
            busy = {v for v in active.values()}
            owner_exited = not owner_alive(args.parent_pid)
            unfinished = False
            if conv.exists():
                for path in sorted(conv.rglob('*.json')):
                    if len(active) >= policy.workers: break
                    rec = load_prediction_record(path)
                    if not rec or not policy.terminal(rec.pred_status, rec.raw): continue
                    fp = prediction_fingerprint(rec)
                    ident = (rec.run_id, fp)
                    prior = read_json_object(output / rec.question_id / (rec.run_id + '.json'))
                    if not prior or prior.get('pred_fingerprint') != fp or prior.get('status') not in ('judged', 'inference_error'):
                        unfinished = True
                    if not policy.pending(ident, prior, busy, retry_after, time.monotonic()): continue
                    active[pool.submit(evaluate, rec, fp)] = ident
            if owner_exited and not active and not unfinished:
                print('Evaluation ended; all terminal predictions scored', flush=True)
                if args.viewer_model:
                    import subprocess
                    subprocess.run([sys.executable, str(E / 'services/viewer_publish.py'),
                        '--run', args.run, '--dataset', args.dataset, '--model', args.viewer_model,
                        '--label', f'{args.viewer_model} · {args.dataset}', '--once'], check=True)
                break
            time.sleep(args.interval)

if __name__ == '__main__':
    main()
