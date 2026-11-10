"""Process-shared endpoint leases. Kernel locks, not PID/timeouts, own capacity."""
import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def guard(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'guard.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _path(root, endpoint, index):
    key = hashlib.sha256(endpoint.encode()).hexdigest()
    return Path(root) / f'{key}-{index}.slot'


def _occupied(root, endpoint, capacity):
    owners = {}
    for i in range(capacity):
        with _path(root, endpoint, i).open('a+') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.seek(0)
                dataset = json.load(handle)['dataset']
                owners[dataset] = owners.get(dataset, 0) + 1
    return owners


def snapshot(root, endpoints, capacity):
    with guard(root):
        return {e: _occupied(root, e, capacity) for e in endpoints}


def try_acquire(root, endpoint, dataset, capacity):
    """Return an open locked file, or None. Every dataset shares the same cap.

    Quotas are read under the acquisition mutex, so a delayed endpoint-registry
    reload cannot exceed a newly assigned dataset quota. Existing leases survive
    all quota changes and remain bound until the trajectory returns.
    """
    with guard(root):
        plan = json.loads((Path(root) / 'allocation.json').read_text())
        if plan['capacity'] != capacity:
            raise ValueError('endpoint slot capacity differs from coordinator')
        quota = plan['quotas'].get(endpoint, {}).get(dataset, 0)
        owners = _occupied(root, endpoint, capacity)
        if owners.get(dataset, 0) >= quota or sum(owners.values()) >= capacity:
            return None
        for i in range(capacity):
            handle = _path(root, endpoint, i).open('a+')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            handle.seek(0)
            handle.truncate()
            json.dump({'dataset': dataset}, handle)
            handle.flush()
            return handle
    return None
