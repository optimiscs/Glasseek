import json
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Evaluation'))
from browse_comp_eval.concurrency_ramp import ConcurrencyRamp
from browse_comp_eval.runner import BrowseCompRunner, RunnerConfig


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('condition did not become true')


class MetricsServer:
    def __init__(self):
        self.counter = 0
        self.fail = False
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(503 if owner.fail else 200)
                self.end_headers()
                self.wfile.write(f'vllm:num_preemptions_total{{engine="0"}} {owner.counter}\n'.encode())
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server.server_port}/metrics'


class RampTest(unittest.TestCase):
    def test_registry_hot_add_remove_and_monitor_each_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp, MetricsServer() as first, MetricsServer() as second:
            state = {'last_successful_read': 'now', 'last_error': '', 'last_valid_endpoint_snapshot': [
                {'base_url': first.url.removesuffix('/metrics') + '/v1'}]}
            ramp = ConcurrencyRamp('', Path(tmp)/'ramp.jsonl', maximum=6, window=.3, interval=.02,
                                   endpoint_source=lambda: dict(state))
            with ramp:
                wait_for(lambda: ramp.current_limit == 2)
                state['last_valid_endpoint_snapshot'] = [
                    {'base_url': first.url.removesuffix('/metrics') + '/v1'},
                    {'base_url': second.url.removesuffix('/metrics') + '/v1'}]
                self.assertEqual(ramp.current_limit, 0)
                wait_for(lambda: ramp.current_limit == 2)
                second.fail = True
                wait_for(lambda: ramp.current_limit == 0)
                second.fail = False
                wait_for(lambda: ramp.current_limit == 2)
                wait_for(lambda: ramp.current_limit == 4)
                second.counter = 1
                wait_for(lambda: ramp.current_limit == 2)
                state['last_valid_endpoint_snapshot'] = [{'base_url': first.url.removesuffix('/metrics') + '/v1'}]
                self.assertEqual(ramp.current_limit, 0)
                wait_for(lambda: ramp.current_limit == 2)
                time.sleep(.35)
                self.assertEqual(ramp.current_limit, 2)
            events = [json.loads(line)['event'] for line in (Path(tmp)/'ramp.jsonl').read_text().splitlines()]
            self.assertIn('endpoints_changed', events)
            self.assertIn('metrics_error', events)
            self.assertIn('preemption_backoff', events)

    def test_two_endpoints_back_off_when_either_preempts(self):
        with tempfile.TemporaryDirectory() as tmp, MetricsServer() as first, MetricsServer() as second:
            ramp = ConcurrencyRamp(f'{first.url},{second.url}', Path(tmp)/'ramp.jsonl', maximum=4, window=.15, interval=.02)
            with ramp:
                wait_for(lambda: ramp.current_limit == 4)
                second.counter = 1
                wait_for(lambda: ramp.current_limit == 2)
                time.sleep(.2)
                self.assertEqual(ramp.current_limit, 2)
            events = [json.loads(line) for line in (Path(tmp)/'ramp.jsonl').read_text().splitlines()]
            backoff = next(event for event in events if event['event'] == 'preemption_backoff')
            self.assertEqual(backoff['preemptions'][second.url], 1)
            self.assertEqual(backoff['preemptions'][first.url], 0)

    def test_preemption_can_be_recorded_without_backoff(self):
        with tempfile.TemporaryDirectory() as tmp, MetricsServer() as server:
            path = Path(tmp) / 'ramp.jsonl'
            ramp = ConcurrencyRamp(server.url, path, start=16, maximum=16,
                                   interval=.02, allow_preemption=True)
            with ramp:
                wait_for(lambda: ramp.current_limit == 16)
                server.counter = 2
                wait_for(lambda: any(json.loads(line)['event'] == 'preemption_observed'
                                     for line in path.read_text().splitlines()))
                self.assertEqual(ramp.current_limit, 16)
                self.assertFalse(ramp._frozen)

    def test_real_runner_ramps_then_drains_without_cancelling(self):
        with tempfile.TemporaryDirectory() as tmp, MetricsServer() as server:
            root = Path(tmp)
            window = float(os.environ.get('RAMP_TEST_WINDOW', '.25'))
            ramp = ConcurrencyRamp(server.url, root / 'ramp.jsonl', maximum=4, window=window, interval=.1 if window > 1 else .03)
            config = RunnerConfig(root/'out', root/'work', root/'summary', 16, 1, False, set(), set(), concurrency_ramp=ramp)
            lock = threading.Lock()
            started = []
            finished = []
            gates = [threading.Event() for _ in range(6)]
            class Client:
                def scheduling_capacity(self): return 16
            class Runner(BrowseCompRunner):
                def _should_run(self, spec): return True
                def _run_one(self, spec):
                    with lock: started.append(spec.index)
                    if not gates[spec.index].wait(window + 15): raise TimeoutError('test gate')
                    with lock: finished.append(spec.index)
                    return {'status': 'success'}
            runner = Runner(Client(), config)
            specs = [SimpleNamespace(index=i, run_id=str(i)) for i in range(6)]
            result = []
            thread = threading.Thread(target=lambda: result.append(runner.run(specs)))
            thread.start()
            try:
                wait_for(lambda: len(started) == 2)
                self.assertEqual(ramp.current_limit, 2)
                wait_for(lambda: len(started) == 4, timeout=window + 10)
                self.assertEqual(ramp.current_limit, 4)
                server.counter = 1
                wait_for(lambda: ramp.current_limit == 2)
                self.assertEqual(len(finished), 0)
                gates[0].set(); gates[1].set()
                wait_for(lambda: len(finished) == 2)
                time.sleep(.2)
                self.assertEqual(len(started), 4)
                gates[2].set()
                wait_for(lambda: len(started) == 5)
                for gate in gates: gate.set()
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(result[0]['success'], 6)
                events = [json.loads(line)['event'] for line in (root/'ramp.jsonl').read_text().splitlines()]
                self.assertIn('ramp_up', events)
                self.assertIn('preemption_backoff', events)
            finally:
                for gate in gates: gate.set()
                thread.join(6)

    def test_outage_restarts_stability_window_and_reset_freezes(self):
        with tempfile.TemporaryDirectory() as tmp, MetricsServer() as server:
            ramp = ConcurrencyRamp(server.url, Path(tmp)/'ramp.jsonl', maximum=4, window=.2, interval=.02)
            server.fail = True
            with ramp:
                time.sleep(.08)
                self.assertEqual(ramp.current_limit, 0)
                server.fail = False
                wait_for(lambda: ramp.current_limit == 2)
                server.fail = True
                time.sleep(.12)
                server.fail = False
                time.sleep(.1)
                self.assertEqual(ramp.current_limit, 2)
                wait_for(lambda: ramp.current_limit == 4)
                server.counter = 1
                wait_for(lambda: ramp.current_limit == 2)
                server.counter = 0
                time.sleep(.3)
                self.assertEqual(ramp.current_limit, 2)
            events = [json.loads(line)['event'] for line in (Path(tmp)/'ramp.jsonl').read_text().splitlines()]
            self.assertIn('metrics_error', events)
            self.assertIn('counter_reset', events)


if __name__ == '__main__':
    unittest.main()
