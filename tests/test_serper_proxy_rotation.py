import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / 'Evaluation/services/serper_proxy.py'
spec = importlib.util.spec_from_file_location('serper_proxy_test', path)
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)


class SerperRotationTests(unittest.TestCase):
    def test_exhausted_key_rotates_and_stays_on_working_key(self):
        seen = []

        class Connection:
            def __init__(self, *args, **kwargs): pass
            def request(self, method, path, body, headers):
                self.key = headers['X-API-KEY']
                seen.append(self.key)
            def getresponse(self):
                class Response:
                    status = 400 if self.key == 'exhausted' else 200
                    def read(inner):
                        return b'{"message":"Not enough credits"}' if inner.status == 400 else b'{"organic":[]}'
                return Response()
            def close(self): pass

        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory) / 'keys.json'
            pool.write_text(json.dumps({'serper': ['exhausted', 'working']}))
            with patch.dict('os.environ', {'SERPER_KEY_POOL_FILE': str(pool)}), patch.object(proxy.http.client, 'HTTPSConnection', Connection):
                proxy.key_index = 0
                self.assertEqual(proxy.upstream_search({'q': 'test'})[0], 200)
                self.assertEqual(proxy.upstream_search({'q': 'test'})[0], 200)
        self.assertEqual(seen, ['exhausted', 'working', 'working'])


if __name__ == '__main__':
    unittest.main()
