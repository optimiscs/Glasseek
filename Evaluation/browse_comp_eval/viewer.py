"""Manage live trajectory publishing as part of an evaluation's lifecycle."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from urllib.parse import quote


class EvaluationViewer:
    def __init__(self, *, run, dataset, model, output, data_path, count, workers,
                 enabled=True, interval=20, parent_pid=None):
        self.process = None
        self.enabled = enabled
        self.interval = float(interval)
        self.output = Path(output).resolve()
        evaluation = Path(__file__).resolve().parents[1]
        mode = os.environ.get('GLASSEEK_URL_MODE', 'raw')
        label = os.environ.get('GLASSEEK_VIEWER_LABEL') or f'{model} · {mode} · {count} 题 · 并发 {workers}'
        self.url = os.environ.get('GLASSEEK_VIEWER_BASE_URL', 'http://127.0.0.1:6080/trace-viewer-zh/').rstrip('/') + '/?dataset=' + quote(run, safe='')
        self.command = [sys.executable, str(evaluation / 'services/viewer_publish.py'),
                        '--run', run, '--dataset', dataset, '--model', model, '--label', label,
                        '--interval', str(interval), '--parent-pid', str(parent_pid or os.getpid())]
        self.env = dict(os.environ, GLASSEEK_VIEWER_OUTPUT_DIR=str(self.output),
                        GLASSEEK_VIEWER_DATA_PATH=str(Path(data_path).resolve()),
                        GLASSEEK_VIEWER_LIMIT=str(count))

    def start(self):
        if not self.enabled:
            return self
        try:
            self.output.mkdir(parents=True, exist_ok=True)
            with (self.output / 'viewer.log').open('a') as log:
                self.process = subprocess.Popen(self.command, env=self.env, stdin=subprocess.DEVNULL,
                                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (self.output / 'viewer.json').write_text(json.dumps({
                'url': self.url, 'publisher_pid': self.process.pid, 'automatic': True,
                'log': str(self.output / 'viewer.log'), 'refresh_seconds': self.interval}, indent=2))
            print('Trajectory viewer: ' + self.url, flush=True)
        except OSError as exc:
            print(f'Trajectory viewer could not start: {exc}', file=sys.stderr)
        return self

    def close(self):
        if self.process is None or self.process.poll() is not None:
            return
        try:
            self.process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            # Export/import have their own timeouts. Bound shutdown even if a
            # subprocess becomes stuck, including its child exporter/importer.
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
            print('Trajectory viewer final refresh timed out; see viewer.log', file=sys.stderr)
