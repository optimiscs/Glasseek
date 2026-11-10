"""Dataset affinity with shared per-endpoint slots and cross-dataset backfill.

Reuse the existing registry/runner. A dataset has one queue and one writer;
only unclaimed tasks are distributed by the endpoint pool's least-active policy.
"""
import json
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def set_option(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


class AffinityScheduler:
    def __init__(self, endpoints, plans, counts, root, workers):
        if len(endpoints) != len(set(endpoints)) or not endpoints or workers < 1:
            raise ValueError('affinity endpoints must be unique; capacity must be positive')
        self.endpoints = list(endpoints)
        self.plans = {name: (list(cmd), dict(env)) for name, cmd, env in plans}
        self.counts = counts
        self.root = Path(root)
        self.workers = workers
        self.slots = self.root / 'endpoint-slots'
        self.initial = {e: [] for e in endpoints}
        loads = dict.fromkeys(endpoints, 0)
        for name in sorted(self.plans, key=lambda n: (-counts[n], n)):
            endpoint = min(endpoints, key=lambda e: (loads[e], endpoints.index(e)))
            self.initial[endpoint].append(name)
            loads[endpoint] += counts[name]
        self.preferred = {e: (q[0] if q else None) for e, q in self.initial.items()}
        self.active = set()
        self.done = set()
        self.quotas = {e: {} for e in endpoints}
        self.events = []
        # A run has one coordinator; registries/results must not have two writers.
        import fcntl
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / 'affinity.lock').open('a+')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError('an affinity coordinator already owns this run')
        for name in self.plans:
            (self.root / 'scheduling' / (name + '.json')).unlink(missing_ok=True)

    def close(self):
        self.lock.close()

    def remaining(self, name):
        if name in self.done:
            return 0
        if name in self.active:
            try:
                progress = json.loads((self.root / 'scheduling' / (name + '.json')).read_text())
                return progress['unscheduled'] + progress['in_flight']
            except (OSError, ValueError, KeyError):
                pass
        return self.counts[name]

    def unavailable(self):
        health = {e: [] for e in self.endpoints}
        for name in self.active:
            try:
                value = json.loads((self.root / 'scheduling' / (name + '.json')).read_text())
                for route in value.get('endpoint_health', []):
                    if route.get('base_url') in health:
                        health[route['base_url']].append(route.get('state'))
            except (OSError, ValueError):
                pass
        return {e for e, states in health.items() if states and all(s in ('unhealthy', 'recovering') for s in states)}

    def registry(self, name):
        cmd, _ = self.plans[name]
        model = cmd[cmd.index('--model') + 1]
        path = self.root / 'endpoint-registries' / (name + '.json')
        # Zero quota suspends NEW claims, never retires/cancels active routes.
        atomic_json(path, {'version': 1, 'models': {model: [
            {'base_url': e, 'worker_limit': self.quotas[e].get(name, 0)}
            for e in self.endpoints]}})
        return path

    def ready(self):
        from glasseek_shared.endpoint_slots import guard, _occupied
        ready = []
        unavailable = self.unavailable()
        with guard(self.slots):
            occupied = {e: _occupied(self.slots, e, self.workers) for e in self.endpoints}
            quotas = {e: dict(counts) for e, counts in occupied.items()}
            remaining = {n: max(0, self.remaining(n) - sum(v.get(n, 0) for v in occupied.values()))
                         for n in self.plans}
            for endpoint in self.endpoints:
                if endpoint in unavailable:
                    continue
                free = self.workers - sum(quotas[endpoint].values())
                order = list(dict.fromkeys([self.preferred[endpoint], *self.initial[endpoint],
                    *(n for n in self.plans if n not in self.active), *self.plans]))
                first = None
                for name in order:
                    if not free:
                        break
                    if not name or not remaining[name]:
                        continue
                    take = min(free, remaining[name])
                    quotas[endpoint][name] = quotas[endpoint].get(name, 0) + take
                    remaining[name] -= take
                    free -= take
                    first = first or name
                if first:
                    self.preferred[endpoint] = first
            if quotas != self.quotas:
                self.events.append({'event': 'allocate_slots', 'quotas': {e: dict(q) for e, q in quotas.items()}})
            self.quotas = quotas
            atomic_json(self.slots / 'allocation.json', {'capacity': self.workers, 'quotas': quotas})
        for name in self.plans:
            if name in self.done:
                continue
            if name not in self.active and not any(q.get(name, 0) for q in quotas.values()):
                continue
            registry = self.registry(name)
            if name in self.active:
                continue
            self.active.add(name)
            cmd, env = self.plans[name]
            cmd, env = list(cmd), dict(env)
            for flag in ('--base-urls', '--base-url', '--key-base-urls', '--endpoint-registry',
                         '--endpoint-registry-refresh-interval', '--ramp-start', '--ramp-max',
                         '--ramp-step', '--ramp-window', '--ramp-interval', '--ramp-metrics-url'):
                while flag in cmd:
                    at = cmd.index(flag)
                    del cmd[at:at+2]
            if '--ramp-registry' in cmd:
                cmd.remove('--ramp-registry')
            set_option(cmd, '--endpoint-registry', registry)
            set_option(cmd, '--endpoint-registry-refresh-interval', 0.5)
            set_option(cmd, '--workers-per-endpoint', self.workers)
            set_option(cmd, '--num-workers', self.workers)
            if '--endpoint-watchdog' not in cmd:
                cmd.append('--endpoint-watchdog')
            env.update(GLASSEEK_SCHEDULING_STATE=str(self.root / 'scheduling' / (name + '.json')),
                       GLASSEEK_ENDPOINT_SLOTS=str(self.slots), GLASSEEK_SLOT_DATASET=name,
                       GLASSEEK_ENDPOINT_CAPACITY=str(self.workers))
            env['NO_PROXY'] = env['no_proxy'] = '*'
            ready.append((name, cmd, env))
        self.save(occupied)
        return ready

    def save(self, occupied):
        atomic_json(self.root / 'affinity.json', {'initial_datasets': self.initial,
            'preferred': self.preferred, 'quotas': self.quotas, 'in_flight': occupied,
            'active': sorted(self.active), 'finished': sorted(self.done),
            'workers_per_endpoint': self.workers, 'events': self.events})

    def finished(self, name):
        from glasseek_shared.endpoint_slots import guard, _occupied
        self.active.remove(name)
        self.done.add(name)
        with guard(self.slots):
            occupied = {e: _occupied(self.slots, e, self.workers) for e in self.endpoints}
            for quota in self.quotas.values():
                quota.pop(name, None)
            atomic_json(self.slots / 'allocation.json', {'capacity': self.workers, 'quotas': self.quotas})
        self.save(occupied)

    @property
    def pending(self):
        return bool(set(self.plans) - self.active - self.done)
