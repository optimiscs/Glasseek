"""Dataset affinity with hot-added idle endpoints; one owner per GPU at a time.

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
        if len(endpoints) != len(set(endpoints)) or not endpoints:
            raise ValueError('affinity endpoints must be nonempty and unique')
        self.endpoints = list(endpoints)
        self.plans = {name: (list(cmd), dict(env)) for name, cmd, env in plans}
        self.root = Path(root)
        self.workers = workers
        self.queues = {endpoint: [] for endpoint in endpoints}
        self.owner = {endpoint: None for endpoint in endpoints}
        self.active = {}
        loads = dict.fromkeys(endpoints, 0)
        for name in sorted(self.plans, key=lambda n: (-counts[n], n)):
            endpoint = min(endpoints, key=lambda e: (loads[e], endpoints.index(e)))
            self.queues[endpoint].append(name)
            loads[endpoint] += counts[name]
        self.initial = {e: list(q) for e, q in self.queues.items()}
        self.events = []
        self.save()

    def save(self):
        atomic_json(self.root / 'affinity.json', {'initial_datasets': self.initial,
            'owners': self.owner, 'queued': self.queues, 'active': self.active,
            'workers_per_endpoint': self.workers, 'events': self.events})

    def registry(self, name):
        cmd, _ = self.plans[name]
        model = cmd[cmd.index('--model') + 1]
        endpoints = self.active[name]
        path = self.root / 'endpoint-registries' / (name + '.json')
        atomic_json(path, {'version': 1, 'models': {model: [{'base_url': e} for e in endpoints]}})
        return path

    def ready(self):
        ready = []
        for endpoint in self.endpoints:
            if self.owner[endpoint] is not None:
                continue
            queue = self.queues[endpoint]
            if not queue:
                # Move a whole not-yet-started dataset before splitting an active
                # one: this preserves the largest possible common prompt prefix.
                queue = max(self.queues.values(), key=len)
            if queue:
                name = queue.pop(0)
                self.owner[endpoint] = name
                self.active[name] = [endpoint]
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
                set_option(cmd, '--endpoint-registry', self.registry(name))
                set_option(cmd, '--endpoint-registry-refresh-interval', 1)
                set_option(cmd, '--workers-per-endpoint', self.workers)
                set_option(cmd, '--num-workers', self.workers)
                if '--endpoint-watchdog' not in cmd:
                    cmd.append('--endpoint-watchdog')
                env['GLASSEEK_SCHEDULING_STATE'] = str(self.root / 'scheduling' / (name + '.json'))
                env['NO_PROXY'] = env['no_proxy'] = '*'
                ready.append((name, cmd, env))
                self.events.append({'event': 'start_dataset', 'dataset': name, 'endpoint': endpoint})
                continue
            candidates = []
            for name in self.active:
                path = self.root / 'scheduling' / (name + '.json')
                try:
                    progress = json.loads(path.read_text())
                    pending = progress['unscheduled']
                except (OSError, ValueError, KeyError):
                    continue
                if pending > 0:
                    candidates.append((pending / len(self.active[name]), name))
            if candidates:
                _, name = max(candidates)
                self.owner[endpoint] = name
                self.active[name].append(endpoint)
                self.registry(name)  # additive only: never cancel an in-flight task
                self.events.append({'event': 'join_pending_queue', 'dataset': name, 'endpoint': endpoint})
        self.save()
        return ready

    def finished(self, name):
        for endpoint in self.active.pop(name):
            self.owner[endpoint] = None
        self.events.append({'event': 'finished_dataset', 'dataset': name})
        self.save()

    @property
    def pending(self):
        return any(self.queues.values())
