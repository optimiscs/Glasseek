"""Python evaluation entrypoint with the existing fetch watchdog and runner."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

EVALUATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATION))
sys.path.insert(0, str(EVALUATION / 'scripts'))
sys.path.insert(0, str(EVALUATION.parent / 'Shared'))
from fetch_watchdog import active_run_ids, mark_failed, probe


def terminate(process, grace=10):
    """Stop only this evaluation's isolated process group, including workers."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
        os.killpg(process.pid, signal.SIGCONT)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def supervise(command, args, runner):
    targets = [x.strip() for x in args.fetch_watchdog_target.split(',') if x.strip()]
    probe(args.fetch_server_url, 30, targets)
    print('Fetch watchdog preflight passed.', flush=True)
    process = subprocess.Popen(command, cwd=EVALUATION, start_new_session=True)
    handlers = {}
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        handlers[sig] = signal.signal(sig, interrupted)
    # Same health/functional intervals and failure thresholds as run_eval.sh.
    last_health = last_functional = time.monotonic()
    failures = {'health': 0, 'functional': 0}
    reason = None
    try:
        while process.poll() is None:
            now = time.monotonic()
            checks = []
            if now - last_health >= 10:
                checks.append(('health', 3, [], 3)); last_health = now
            if targets and now - last_functional >= 300:
                checks.append(('functional', 30, targets, 2)); last_functional = now
            for kind, timeout, urls, limit in checks:
                try:
                    probe(args.fetch_server_url, timeout, urls)
                    failures[kind] = 0
                except Exception as exc:
                    failures[kind] += 1
                    print(f'Fetch watchdog {kind} failed ({failures[kind]}/{limit}): {exc}', flush=True)
                    if failures[kind] >= limit:
                        reason = f'{kind}: {exc}'
                        break
            if reason:
                try:
                    os.killpg(process.pid, signal.SIGSTOP)
                    run_ids = active_run_ids(process.pid)
                except ProcessLookupError:
                    run_ids = []
                terminate(process)
                with tempfile.TemporaryDirectory(prefix='glasseek-fetch-watchdog-') as directory:
                    path = Path(directory) / 'run-ids.txt'
                    path.write_text('\n'.join(run_ids))
                    mark_failed(SimpleNamespace(run_ids_file=str(path), data_path=runner.data_path,
                        output_root=runner.output_root, project_root=str(EVALUATION),
                        save_name=runner.save_name, model=runner.model, dataset=runner.dataset,
                        fetch_server_url=args.fetch_server_url, reason=reason,
                        numbered_output=runner.numbered_output))
                return 75
            time.sleep(1)
        return process.returncode
    finally:
        terminate(process)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main(argv=None):
    from browse_comp_eval.run import build_parser
    from browse_comp_eval.dataset import load_dataset
    from glasseek_shared.hermes.url_registry import add_url_argument
    from glasseek_shared.hermes.url_prompt import configure_url_arguments
    p = argparse.ArgumentParser(add_help=False)
    add_url_argument(p)
    p.add_argument('--expected-questions', type=int)
    p.add_argument('--fetch-server-url', required=True)
    p.add_argument('--fetch-watchdog-target', default='https://example.com')
    p.add_argument('--prepare-only', action='store_true')
    args, extra = p.parse_known_args(argv)
    configure_url_arguments(args)
    runner = build_parser().parse_args(extra)
    if args.expected_questions is not None:
        actual = len(load_dataset(Path(runner.data_path)))
        if actual != args.expected_questions:
            p.error(f'expected {args.expected_questions} questions, found {actual}')
    for executable in (sys.executable, runner.hermes_bin):
        if not os.access(executable, os.X_OK):
            p.error(f'executable unavailable: {executable}')
    # The tool implementation reads these shared variables, as in run_eval.sh.
    os.environ['FETCH_SERVER_BASE_URL'] = os.environ['FETCH_SERVER_URL'] = args.fetch_server_url.rstrip('/')
    if args.prepare_only:
        print('Preparation complete; evaluation not started.')
        return 0
    if os.environ.get('GLASSEEK_FIXED_CORPUS_CONFIG'):
        from corpus import ensure
        ensure(os.environ['GLASSEEK_FIXED_CORPUS_CONFIG'],sys.executable)
    command = [sys.executable, '-m', 'browse_comp_eval.run', *extra]
    if runner.dry_run:
        return subprocess.run(command, cwd=EVALUATION, check=False).returncode
    return supervise(command, args, runner)


if __name__ == '__main__':
    sys.exit(main())
