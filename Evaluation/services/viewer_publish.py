"""Publish any Hermes evaluation using the existing Quest exporter/importer."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
from urllib.parse import urlparse

EVALUATION = Path(__file__).resolve().parents[1]

# Reuse Quest's transactional importer for this run only. A site-wide import
# scans unrelated historical packages and can take longer than a refresh cycle.
IMPORT_RUN_CODE = """
import json, sys
from pathlib import Path
from quest_viewer.config import Settings
from quest_viewer.db.session import create_session_factory
from quest_viewer.ingestion.importer import import_run
registry, run_id = Path(sys.argv[1]), sys.argv[2]
entries = json.loads(registry.read_text())['datasets']
position, metadata = next((i, e) for i, e in enumerate(entries) if e['id'] == run_id)
engine, factory = create_session_factory(Settings())
try:
    changed = import_run(factory, run_id, registry.parent.parent / metadata['data'], metadata, position + 2)
    print(json.dumps({'run': run_id, 'imported': changed}), flush=True)
finally:
    engine.dispose()
"""


def ensure_web_service(quest, lock_dir):
    """Reuse the shared local viewer, starting its existing entrypoint if absent."""
    address = urlparse(os.environ.get('GLASSEEK_VIEWER_BASE_URL', 'http://127.0.0.1:6080/trace-viewer-zh/'))
    if address.hostname not in ('127.0.0.1', 'localhost') or address.scheme != 'http':
        return  # A configured remote viewer owns its own service lifecycle.
    port = address.port or 80
    entrypoint = quest / 'web/serve.py'
    if not entrypoint.is_file():
        return
    with (lock_dir / 'viewer-web.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        def listening():
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=0.5):
                    return True
            except OSError:
                return False
        if listening():
            return
        with (lock_dir / 'viewer-web.log').open('a') as log:
            process = subprocess.Popen([sys.executable, str(entrypoint), '--port', str(port)],
                cwd=quest / 'web', stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        (lock_dir / 'viewer-web.pid').write_text(str(process.pid))
        for _ in range(20):
            if listening():
                return
            if process.poll() is not None:
                break
            threading.Event().wait(0.2)
        raise OSError('Viewer web service did not start; see ' + str(lock_dir / 'viewer-web.log'))


def atomic_json(path, data):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.index-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'dataset', 'model', 'label'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--interval', type=float, default=20)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--parent-pid', type=int, default=0, help='Stop after the owning evaluation exits')
    args = parser.parse_args()
    if args.interval <= 0 or args.run in ('', '.', '..') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in args.run):
        parser.error('run must be a safe identifier and interval must be positive')
    quest = Path(os.environ.get('QUEST_ROOT', EVALUATION.parent.parent / 'Quest'))
    registry = quest / 'web/trace-viewer/datasets/index.json'
    lock_dir = Path(os.environ.get('GLASSEEK_RUNTIME_DIR', EVALUATION / 'data/runtime')) / 'tool-services'
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = (lock_dir / (args.run + '.publisher.lock')).open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('A publisher already owns this run; reusing its live viewer.', flush=True)
        return
    output = registry.parent / args.run / 'traces.json'
    env = dict(os.environ, GLASSEEK_VIEWER_RUN=args.run, GLASSEEK_VIEWER_WORKSPACE=args.run,
               GLASSEEK_VIEWER_DATASET=args.dataset, GLASSEEK_VIEWER_MODEL=args.model, GLASSEEK_VIEWER_LABEL=args.label)
    previous = None
    stop = threading.Event()
    if args.parent_pid:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stop.set())
    def owner_alive():
        if not args.parent_pid:return True
        try:
            os.kill(args.parent_pid, 0)
            return Path(f'/proc/{args.parent_pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
        except (OSError, IndexError):
            return False
    while True:
        final_refresh = stop.is_set() or not owner_alive()
        try:
            ensure_web_service(quest, lock_dir)
            subprocess.run([sys.executable, str(EVALUATION / 'quest_viewer_checkpoint_adapter.py'),
                            '--output', str(output), '--once'], env=env, check=True, timeout=25)
            if output.is_file():
                digest = hashlib.sha256(output.read_bytes()).hexdigest()
                if digest != previous:
                    # The four modes share one registry and import database.
                    with (lock_dir / 'viewer-registry.lock').open('a') as shared_lock:
                        fcntl.flock(shared_lock, fcntl.LOCK_EX)
                        data = json.loads(registry.read_text()) if registry.exists() else {'datasets': []}
                        package = json.loads(output.read_text())
                        count = package.get('run', {}).get('requestedCases')
                        benchmark = f'{args.dataset} · {count} 题' if count is not None else args.dataset
                        if not any(entry.get('id') == args.run for entry in data['datasets']):
                            data['datasets'].insert(0, dict(id=args.run, label=args.label, description=benchmark+' · 实时轨迹与工具统计',
                                folder='datasets/' + args.run, data=f'datasets/{args.run}/traces.json', benchmark=benchmark, visible=True))
                            atomic_json(registry, data)
                        subprocess.run([str(quest / 'workspace/viewer/venv/bin/python'), '-c', IMPORT_RUN_CODE,
                                        str(registry), args.run],
                                       cwd=quest / 'web', check=True, timeout=25)
                    previous = digest
            else:
                print('Waiting for the first saved trajectory', flush=True)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            if args.once:
                raise
            print(f'Publisher retry after {type(exc).__name__}: {exc}', flush=True)
        if args.once or final_refresh:
            return
        stop.wait(args.interval)


if __name__ == '__main__':
    main()
