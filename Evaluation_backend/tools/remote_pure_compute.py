"""Run the native pure-compute namespace sandbox on an explicitly trusted SSH host."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
import time
from pathlib import Path

_CAP = 50_000
_MAX_CODE = 1_000_000


def execute(code: str, host: str, timeout: int = 300) -> str:
    """Fail closed: SSH transport only; execute through the native isolation launcher."""
    started = time.monotonic()
    if host != 'Lab3090' or not code.strip() or len(code.encode('utf-8')) > _MAX_CODE:
        return json.dumps({'status': 'error', 'error': 'invalid remote sandbox configuration or code'})
    timeout = max(1, min(int(timeout), 300))
    launcher = Path(__file__).with_name('execute_code_fs_sandbox.py').read_bytes()
    encoded_launcher = base64.b64encode(launcher).decode('ascii')
    encoded_code = base64.b64encode(code.encode('utf-8')).decode('ascii')
    # Values contain base64 only. The program is never interpreted by a shell.
    payload = f'HELPER={encoded_launcher}\nCODE={encoded_code}\nLIMIT={timeout}\n' + r'''set -eu
umask 077
D=$(mktemp -d /tmp/glasseek-hermes-sandbox.XXXXXXXX)
trap 'rm -rf -- "$D"' EXIT
printf %s "$HELPER" | base64 -d > "$D/helper.py"
printf %s "$CODE" | base64 -d > "$D/script.py"
printf %s '' > "$D/hermes_tools.py"
ulimit -v 2097152
ulimit -t 120
ulimit -f 8192
ulimit -n 128
env -i PATH=/usr/bin:/bin HERMES_SANDBOX_NO_FORK=1 timeout --signal=TERM --kill-after=3 "$LIMIT" \
  unshare --user --map-root-user --mount --pid --ipc --uts --net \
  --fork --kill-child --propagation private /usr/bin/python3 "$D/helper.py" --staging "$D"
'''
    cmd = ['/usr/bin/ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
           '-o', 'StrictHostKeyChecking=yes', host, 'bash -s']
    chunks = {'out': bytearray(), 'err': bytearray()}
    totals = {'out': 0, 'err': 0}
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def drain(pipe, key):
            try:
                while True:
                    data = pipe.read(8192)
                    if not data:
                        break
                    totals[key] += len(data)
                    chunks[key].extend(data[:max(0, _CAP - len(chunks[key]))])
            finally:
                pipe.close()

        readers = [threading.Thread(target=drain, args=(proc.stdout, 'out'), daemon=True),
                   threading.Thread(target=drain, args=(proc.stderr, 'err'), daemon=True)]
        for thread in readers:
            thread.start()
        write_errors = []

        def send():
            try:
                proc.stdin.write(payload.encode('utf-8'))
            except (BrokenPipeError, OSError) as exc:
                write_errors.append(exc)
            finally:
                proc.stdin.close()

        writer = threading.Thread(target=send, daemon=True)
        writer.start()
        try:
            rc = proc.wait(timeout=timeout + 20)
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        writer.join(timeout=3)
        if write_errors:
            raise RuntimeError('remote sandbox transport closed during input')
        for thread in readers:
            thread.join(timeout=3)
        from agent.redact import redact_sensitive_text
        output = redact_sensitive_text(chunks['out'].decode('utf-8', errors='replace'))
        error = redact_sensitive_text(chunks['err'].decode('utf-8', errors='replace'))
        if totals['out'] > _CAP:
            output += '\n[output truncated]'
        result = {'status': 'success' if rc == 0 else ('timeout' if rc == 124 else 'error'),
                  'output': output, 'tool_calls_made': 0,
                  'duration_seconds': round(time.monotonic() - started, 2)}
        if rc:
            result['error'] = f'remote sandbox exited {rc}: {error[:1000]}'
        return json.dumps(result, ensure_ascii=False)
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        return json.dumps({'status': 'error', 'error': str(exc), 'tool_calls_made': 0,
                           'duration_seconds': round(time.monotonic() - started, 2)}, ensure_ascii=False)
