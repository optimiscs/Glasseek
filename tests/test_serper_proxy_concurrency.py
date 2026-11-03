import asyncio
import importlib.util
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'Evaluation/services/serper_proxy.py'
spec = importlib.util.spec_from_file_location('glasseek_serper_proxy_test', PATH)
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)

class Request:
    async def json(self):
        return {'q': 'test'}

class ProxyConcurrencyTest(unittest.IsolatedAsyncioTestCase):
    async def test_blocking_upstream_calls_are_concurrent_and_bounded(self):
        active = peak = 0
        lock = threading.Lock()
        def upstream(payload):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(.1)
            with lock:
                active -= 1
            return 200, b'{}'
        proxy.slots = asyncio.Semaphore(16)
        with patch.object(proxy, 'upstream_search', upstream):
            responses = await asyncio.gather(*(proxy.search(Request()) for _ in range(20)))
        self.assertEqual(len(responses), 20)
        self.assertEqual(peak, 16)

if __name__ == '__main__':
    unittest.main()
