"""Trusted SSH-side supervisor. No host mounts or Docker socket in the child.

Input is one JSON object on stdin; output is a bounded execution receipt.
This file is shipped by the transport, not installed/imported by dataset code.
"""
import fcntl
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time


def docker(*args, **kwargs):
    return subprocess.run(['docker', *args], capture_output=True, timeout=20, **kwargs)


def run(request):
    name = request['job_id']
    image = request['image_id']
    if not re.fullmatch(r'glasseek-compute-[0-9a-f]{32}', name):
        raise ValueError('Invalid job identity')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
        raise ValueError('Unpinned image')
    limits = request['limits']
    timeout = min(max(int(request['timeout']), 1), 300)
    started = time.monotonic()
    result = {'job_id': name, 'image_id': image, 'limits': limits,
              'status': 'error', 'output': '', 'tool_calls_made': 0}
    # Per-user slots bound aggregate load from this runtime, including calls
    # from other SSH clients. The lock is held until container removal completes.
    locks = Path.home() / '.cache/glasseek-compute-slots'
    locks.mkdir(parents=True, exist_ok=True, mode=0o700)
    slot = None
    for i in range(limits['max_concurrent_jobs']):
        handle = (locks / str(i)).open('a')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            slot = handle
            break
        except BlockingIOError:
            handle.close()
    if slot is None:
        return {**result, 'status': 'capacity_exceeded', 'cleanup_verified': True}
    process = None
    try:
        created = docker('create', '--name', name, '--pull=never', '-i',
            '--label', 'glasseek.compute=isolated', '--network=none', '--read-only',
            '--user=65534:65534', '--cap-drop=ALL', '--security-opt=no-new-privileges:true',
            '--cpus='+str(limits['cpus']), '--memory='+str(limits['memory_bytes']),
            '--memory-swap='+str(limits['memory_bytes']), '--pids-limit='+str(limits['pids']),
            '--tmpfs=/tmp:rw,noexec,nosuid,nodev,size='+str(limits['tmp_bytes'])+',mode=1777',
            '--shm-size='+str(limits['shm_bytes']), '--ulimit=nofile=128:128',
            '--ulimit=core=0:0', '--ulimit=fsize='+str(limits['file_bytes'])+':'+str(limits['file_bytes']),
            '--log-driver=none', image)
        if created.returncode:
            raise RuntimeError(created.stderr.decode(errors='replace')[:2000])
        process = subprocess.Popen(['docker', 'start', '-ai', name], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.stdin.write(request['code'].encode())
        process.stdin.close()
        captured = {'stdout': bytearray(), 'stderr': bytearray()}
        total = 0
        reason = None
        with selectors.DefaultSelector() as selector:
            for key, pipe in [('stdout', process.stdout), ('stderr', process.stderr)]:
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, key)
            while selector.get_map():
                if time.monotonic() - started > timeout:
                    reason = 'timeout'
                    break
                for key, _ in selector.select(.1):
                    data = os.read(key.fileobj.fileno(), 8192)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    total += len(data)
                    room = max(0, limits['output_bytes'] - sum(map(len, captured.values())))
                    captured[key.data].extend(data[:room])
                    if total > limits['output_bytes']:
                        reason = 'output_limit'
                        break
                if reason:
                    break
        if reason:
            # Detach first: an output-flooded attach client can otherwise block
            # Docker's stream teardown while the supervisor waits for kill/rm.
            process.kill()
            process.wait(timeout=10)
            process.stdout.close()
            process.stderr.close()
            docker('kill', name)
        process.wait(timeout=10)
        state = docker('inspect', '--format={{json .State}}', name)
        if state.returncode:
            raise RuntimeError('Container state unavailable')
        state = json.loads(state.stdout)
        result.update(status=reason or ('memory_limit' if state['OOMKilled'] else
                      'success' if state['ExitCode'] == 0 else 'error'),
                      exit_code=state['ExitCode'], oom_killed=state['OOMKilled'],
                      output=captured['stdout'].decode(errors='replace'),
                      error=captured['stderr'].decode(errors='replace'), output_bytes=total)
    except Exception as exc:
        result.update(status='error', error=str(exc))
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            process.stdout.close()
            process.stderr.close()
        try:
            docker('rm', '-f', '-v', name)
            absent = docker('inspect', name)
            result['cleanup_verified'] = absent.returncode != 0 and b'No such' in absent.stderr
        except (OSError, subprocess.TimeoutExpired):
            result['cleanup_verified'] = False
        if not result['cleanup_verified']:
            result['status'] = 'cleanup_error'
        slot.close()
        result['duration_seconds'] = round(time.monotonic()-started, 3)
    return result


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise RuntimeError('Supervisor interrupted: '+str(signum))
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupted)
    print(json.dumps(run(json.load(sys.stdin))), flush=True)
