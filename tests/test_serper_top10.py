import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Evaluation_backend'))
from plugins.web.search_server import handle_search_server
from plugins.web.search_server.provider import _payload


class SerperTop10Tests(unittest.TestCase):
    def test_enabled_requests_and_returns_ten(self):
        with patch.dict(os.environ, {'GLASSEEK_SEARCH_TOP10': '1'}), \
             patch('tools.web_tools.web_search_tool', return_value='{}') as search:
            handle_search_server({'query': 'test', 'limit': 3})
            self.assertEqual(search.call_args.kwargs['limit'], 10)
            self.assertEqual(_payload('test')['num'], 10)

    def test_default_is_unchanged(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch('tools.web_tools.web_search_tool', return_value='{}') as search:
            handle_search_server({'query': 'test'})
            self.assertEqual(search.call_args.kwargs['limit'], 5)
            self.assertNotIn('num', _payload('test'))


if __name__ == '__main__':
    unittest.main()
