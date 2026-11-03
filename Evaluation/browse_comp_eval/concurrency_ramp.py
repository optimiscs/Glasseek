"""Optional vLLM preemption-aware limit for the evaluator's agent queue."""
from __future__ import annotations

import json
import math
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


class ConcurrencyRamp:
    def __init__(self, url: str, log_path: Path, *, start: int = 2, maximum: int = 16,
                 step: int = 2, window: float = 90, interval: float = 5,
                 timeout: float = 3, endpoint_source=None, allow_preemption: bool = False):
        if not (1 <= start <= maximum and step >= 1 and all(math.isfinite(x) and x > 0 for x in (window, interval, timeout))):
            raise ValueError('invalid concurrency ramp settings')
        self.urls = [item.strip() for item in url.split(',') if item.strip()]
        if (not self.urls and endpoint_source is None) or len(set(self.urls)) != len(self.urls) or (self.urls and endpoint_source is not None):
            raise ValueError('metrics URLs must be nonempty and distinct')
        self.endpoint_source = endpoint_source
        self.allow_preemption = allow_preemption
        self._ready_urls = None
        self.log_path = log_path
        self.start, self.maximum, self.step = start, maximum, step
        self.window, self.interval, self.timeout = window, interval, timeout
        self._current_limit = 0  # Do not start work until a baseline is available.
        self._previous = None
        self._stable_since = None
        self._frozen = False
        self._stop = threading.Event()
        self._thread = None

    def _urls(self):
        if self.endpoint_source is None:
            return self.urls
        state = self.endpoint_source()
        if state.get('last_error') or not state.get('last_successful_read'):
            raise ValueError('endpoint registry snapshot unavailable')
        urls = []
        for entry in state.get('last_valid_endpoint_snapshot', []):
            parsed = urlsplit(str(entry['base_url']))
            if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.query or parsed.fragment:
                raise ValueError('unsupported registry endpoint URL')
            path = parsed.path.rstrip('/')
            if path.endswith('/v1'):
                path = path[:-3]
            urls.append(urlunsplit((parsed.scheme, parsed.netloc, path + '/metrics', '', '')))
        if not urls or len(set(urls)) != len(urls):
            raise ValueError('registry has no endpoints or duplicate metrics URLs')
        return urls

    @property
    def current_limit(self):
        if self.endpoint_source is not None:
            try:
                if frozenset(self._urls()) != self._ready_urls:
                    return 0
            except Exception:
                return 0
        return self._current_limit

    def _record(self, event: str, **fields):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'time': time.time(), 'event': event, 'limit': self.current_limit,
                                     'frozen': self._frozen, **fields}) + '\n')

    def _read(self, url):
        with urllib.request.urlopen(url, timeout=self.timeout) as response:
            text = response.read(2_000_000).decode('utf-8')
        metrics = {}
        for line in text.splitlines():
            if line.startswith('vllm:'):
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0].split('{', 1)[0]
                    if name in ('vllm:num_preemptions_total', 'vllm:kv_cache_usage_perc',
                                'vllm:num_requests_running', 'vllm:num_requests_waiting'):
                        value = float(parts[1])
                        if not math.isfinite(value) or value < 0:
                            raise ValueError('invalid metric ' + name)
                        metrics.setdefault(name, []).append(value)
        values = metrics.get('vllm:num_preemptions_total')
        if not values:
            raise ValueError('vllm:num_preemptions_total missing')
        return sum(values), {name: (max(vals) if name.endswith('kv_cache_usage_perc') else sum(vals))
                             for name, vals in metrics.items()}

    def _sample(self):
        now = time.monotonic()
        try:
            urls = self._urls()
            samples = {url: self._read(url) for url in urls}
            values = {url: sample[0] for url, sample in samples.items()}
            metrics = {url: sample[1] for url, sample in samples.items()}
        except Exception as exc:
            self._stable_since = None
            if self.endpoint_source is not None:
                self._ready_urls = None
            self._record('metrics_error', error=f'{type(exc).__name__}: {exc}')
            return
        previous = self._previous
        common = set(values) & set(previous or {})
        if previous is None:
            self._current_limit = self.start
            self._stable_since = now
            event = 'baseline'
        elif any(values[url] > previous[url] for url in common):
            if self.allow_preemption:
                event = 'preemption_observed'
            else:
                self._current_limit = max(1, self._current_limit - self.step)
                self._frozen = True
                self._stable_since = None
                event = 'preemption_backoff'
        elif any(values[url] < previous[url] for url in common):
            # A restarted server loses its counters; freeze rather than infer safety.
            self._frozen = True
            self._stable_since = None
            event = 'counter_reset'
        elif set(values) != set(previous):
            self._stable_since = now
            event = 'endpoints_changed'
        elif self._stable_since is None:
            self._stable_since = now
            event = 'metrics_recovered'
        elif not self._frozen and now - self._stable_since >= self.window and self.current_limit < self.maximum:
            self._current_limit = min(self.maximum, self.current_limit + self.step)
            self._stable_since = now
            event = 'ramp_up'
        else:
            event = 'sample'
        self._previous = values
        self._ready_urls = frozenset(urls)
        self._record(event, preemptions=values, previous_preemptions=previous, metrics=metrics)

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, name='concurrency-ramp', daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join(timeout=self.timeout + 1)
