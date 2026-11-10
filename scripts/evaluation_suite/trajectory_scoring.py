"""Incremental native grading, with durable per-prediction deduplication.

A claim is written before calling a paid grader. Interrupted claims are not
silently retried: they remain available for inspection, preventing duplicate bills.
"""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from report import load, publish_scores
from suite import dump, read, sha


class TrajectoryScorer:
    def __init__(self, args, environments):
        self.args = args
        self.environments = environments
        self.root = args.output_root / args.run_id / 'trajectory-scores'
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / 'worker.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.jobs = {}
        self.refs = {r['opaque_id']: r for r in read(args.suite / 'private/references.jsonl')}

    def tick(self):
        for future in list(self.jobs):
            if future.done():
                future.result()
                del self.jobs[future]
        # Bound the queue so new results do not create unbounded pending requests.
        for name, env in self.environments.items():
            pred = self.args.output_root / 'predictions' / (self.args.run_id+'_'+name) / ('suite_'+name)
            for path in sorted(pred.glob('conv/*/*.json')):
                row = load(path, {})
                if row.get('status') != 'success' or not row.get('final_assistant_valid') or not str(row.get('model_response', '')).strip():
                    continue
                key = row.get('question_id')
                if key not in self.refs or self.refs[key]['benchmark'] != name:
                    continue
                previous = load(pred/'judge_official'/key/(str(row.get('run_id',''))+'.json'), {})
                if previous.get('prediction') == row.get('model_response') and (previous.get('status') == 'judged' or previous.get('evaluation', {}).get('status') == 'scored'):
                    continue
                identity = hashlib.sha256((key+'\0'+sha(path)).encode()).hexdigest()
                task = self.root / name / identity
                if task.exists():
                    continue
                if len(self.jobs) >= 2:
                    return
                task.mkdir(parents=True)
                dump(task/'state.json', {'state':'claimed', 'opaque_id':key, 'prediction_sha256':sha(path)})
                self.jobs[self.pool.submit(self.score, name, key, task, pred, env)] = task

    def score(self, name, key, task, pred, env):
        from run import prepare_score
        start = time.monotonic()
        state = load(task/'state.json', {})
        try:
            packet = task/'packet'; output = task/'native'
            prepare_score(self.args.suite, packet, pred, name, self.args.corpus_config, opaque_ids={key})
            if not list(read(packet/'records.jsonl')):
                raise ValueError('prediction no longer complete')
            cmd = [str(self.args.score_python or os.environ.get('SUITE_SCORE_PYTHON', self.args.suite.parent/'score-venv/bin/python')),
                   str(Path(__file__).with_name('score.py')), '--suite', str(self.args.suite), '--packet', str(packet), '--out', str(output)]
            if self.args.score_config:
                cmd += ['--config', str(self.args.score_config)]
            with (task/'score.log').open('a') as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
            receipt = load(output/'receipt.json', {})
            status = 'scored' if result.returncode == 0 else 'score_error'
            if result.returncode == 0 and str(receipt.get('status','')).startswith('native_evaluation_partial'):
                status = 'partially_scored'
            state.update(state=status, exit_code=result.returncode, score_dir=str(output), receipt_status=receipt.get('status'))
            if self.args.viewer:
                publish_scores(self.args, name, output, env, opaque_ids={key})
        except Exception as exc:
            state.update(state='score_error', error=type(exc).__name__+': '+str(exc)[:300])
        finally:
            state['seconds'] = time.monotonic()-start
            dump(task/'state.json', state)

    def drain(self):
        while True:
            self.tick()
            if not self.jobs:
                break
            time.sleep(1)

    def close(self):
        self.pool.shutdown(wait=True)
        self.lock.close()


def main():
    # Attach to an existing run without restarting or changing inference.
    from run import build_parser
    args = build_parser().parse_args()
    args.suite = args.suite.absolute(); args.output_root = args.output_root.absolute()
    names = args.benchmarks.split(',')
    if args.benchmarks == 'all':
        names = [r['benchmark'] for r in load(args.suite/'summary.json', [])]
    env = {**os.environ, 'EVAL_PYTHON':str(args.eval_python or 'python')}
    engine = TrajectoryScorer(args, {name:env for name in names})
    try:
        while True:
            engine.tick()
            status = load(args.output_root/args.run_id/'status.json', {})
            if all(status.get(n, {}).get('state') in ('inferred','inference_error','scored','partially_scored','score_error','blocked') for n in names):
                engine.drain(); break
            time.sleep(2)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
