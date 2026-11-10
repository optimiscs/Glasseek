"""Pinned scientific containers on an explicitly trusted SSH host.

Every call has its own cgroup, private tmpfs, no network or host bind mounts.
The SSH-side supervisor owns deadlines, output limits and container cleanup.
"""
import base64
import hashlib
import json
import shlex
import subprocess
import time
import uuid
from pathlib import Path

_MAX_CODE = 1_000_000
HERE = Path(__file__).resolve().parent


def execute(code: str, host: str, timeout: int = 300) -> str:
    started = time.monotonic()
    if host != 'Lab3090' or not code.strip() or len(code.encode()) > _MAX_CODE:
        return json.dumps({'status': 'error', 'error': 'invalid remote sandbox configuration or code'})
    job_id = 'glasseek-compute-' + uuid.uuid4().hex
    ssh = ['/usr/bin/ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
           '-o', 'StrictHostKeyChecking=yes', host]
    process = None
    try:
        timeout = max(1, min(int(timeout), 300))
        manifest_bytes = (HERE / 'scientific_runtime.json').read_bytes()
        manifest = json.loads(manifest_bytes)
        worker = (HERE / 'scientific_container_worker.py').read_bytes()
        request = {'job_id': job_id, 'image_id': manifest['image_id'],
                   'code': code, 'timeout': timeout, 'limits': manifest['limits']}
        launcher = 'import base64;exec(compile(base64.b64decode(' + repr(base64.b64encode(worker).decode()) + "),'<glasseek-supervisor>','exec'))"
        command = '/usr/bin/python3 -c ' + shlex.quote(launcher)
        process = subprocess.Popen([*ssh, command], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate(json.dumps(request).encode(), timeout=timeout+90)
        if process.returncode:
            raise RuntimeError('remote supervisor failed: '+error.decode(errors='replace')[:1000])
        result = json.loads(output)
        if result.get('job_id') != job_id or result.get('image_id') != request['image_id']:
            raise RuntimeError('Remote receipt identity mismatch')
        if not result.get('cleanup_verified'):
            raise RuntimeError('Remote cleanup was not verified')
        result['runtime_manifest_sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
        result['supervisor_sha256'] = hashlib.sha256(worker).hexdigest()
        from agent.redact import redact_sensitive_text
        for key in ('output', 'error'):
            if key in result:
                result[key] = redact_sensitive_text(result[key])
        return json.dumps(result, ensure_ascii=False)
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        return json.dumps({'status': 'error', 'error': str(exc), 'job_id': job_id,
                           'tool_calls_made': 0, 'duration_seconds': round(time.monotonic()-started, 2)})
    finally:
        # Also handle cancellation, broken SSH and malformed/missing receipts.
        # The remote supervisor independently retains its own deadline.
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
            try:
                subprocess.run([*ssh, 'docker rm -f -v '+job_id],
                               capture_output=True, timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                pass
