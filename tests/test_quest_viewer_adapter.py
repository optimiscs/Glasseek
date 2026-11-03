import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Evaluation"))
import quest_viewer_adapter as adapter


class QuestViewerAdapterTests(unittest.TestCase):
    def test_partial_tool_and_first_token_metrics_and_zero_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); log = root / 'agent.log'; config = root / 'config.yaml'
            config.write_text('model:\n  context_length: 131072\n')
            log.write_text('INFO [s1] agent.conversation_loop: API call #1: model=m provider=custom in=100 out=20 total=120 latency=2.0s cache=0/100 (0%) ttft=0.125s\n'
                           'INFO [s1] agent.conversation_loop: API call #2: model=m provider=custom in=200 out=30 total=230 latency=3.0s ttft=0.375s\n')
            steps = [{'tool': {'name': 'web_search'}, 'tool_seconds': 2.0},
                     {'tool': {'name': 'web_extract'}, 'tool_seconds': None}]
            data = adapter.session_statistics({'session_id': 's1'}, log, config, steps)
            self.assertEqual((data['ttftSeconds'], data['ttftSamples']), (0.25, 2))
            self.assertEqual((data['toolSeconds'], data['toolSamples'], data['toolCalls']), (2.0, 1, 2))
            self.assertEqual(data['usage']['cachedInputTokens'], 0)
            self.assertEqual(data['usage']['uncachedInputTokens'], 100)

    def test_long_run_resolves_hashed_profile_and_disambiguates_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            profiles = Path(directory)
            run = 'glasseek_sft9b_quest128_gpu_embedding_c10_browsecomp'
            case = 'row0006'
            first = profiles / adapter.profile_name(run, case + '__r1', 1)
            second = profiles / adapter.profile_name(run, case + '__r1', 2)
            for path, sid in ((first, 'session-first'), (second, 'session-second')):
                (path / 'logs').mkdir(parents=True)
                (path / 'logs/agent.log').write_text(f'INFO [{sid}] API call\n')
            self.assertNotIn(case, first.name)
            session = {'session_id': 'session-second', 'messages': []}
            self.assertEqual(adapter.find_profile(run, case, session, {'attempt': 2}, profiles), second)
            self.assertEqual(adapter.find_profile(run, case, {'session_id': 'unknown'}, None, profiles),
                             profiles / '__missing_profile__')

    def test_web_extract_top_level_results_and_pending(self):
        args = {'urls': ['https://example.org/a']}
        output = '<untrusted_tool_result source="web_extract">\n\n' + json.dumps({'results': [
            {'url': 'https://example.org/a', 'title': 'Page', 'content': 'Actual content', 'error': None},
            {'url': 'https://example.org/b', 'error': 'timeout'}]}) + '\n</untrusted_tool_result>'
        _, tool = adapter.tool_view('web_extract', args, output)
        self.assertEqual(tool['pageCount'], 2)
        self.assertEqual(tool['pages'][0]['summary'], 'Actual content')
        self.assertEqual(tool['pages'][1]['summary'], '读取失败：timeout')
        _, pending = adapter.tool_view('web_extract', args, '')
        self.assertEqual(pending['pageCount'], 0)
        self.assertEqual(pending['pages'][0]['summary'], '正在读取，尚未返回结果。')
        _, failed = adapter.tool_view('web_extract', args, '{"success":false,"error":"offline"}')
        self.assertEqual(failed['pages'][0]['summary'], '读取失败：offline')

    def test_statistics_use_only_recorded_model_calls_and_cache_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root/'agent.log'
            config = root/'config.yaml'
            config.write_text('model:\n  context_length: 131072\n')
            log.write_text('2026-01-01 12:00:03,000 INFO [s1] agent.conversation_loop: API call #1: model=m provider=custom in=100 out=20 total=120 latency=2.0s cache=25/100 (25%)\n'
                           '2026-01-01 12:00:05,000 INFO [s1] agent.conversation_loop: API call #2: model=m provider=custom in=200 out=30 total=230 latency=3.0s\n'
                           '2026-01-01 12:00:06,000 INFO [other] agent.conversation_loop: API call #1: model=m provider=custom in=999 out=999 total=1998 latency=9.0s\n')
            data = adapter.session_statistics({'session_id': 's1', 'messages': []}, log, config)
            self.assertEqual(data['usage'], {'inputTokens': 300, 'outputTokens': 50, 'totalTokens': 350,
                'usageSamples': 2, 'cachedInputTokens': 25, 'uncachedInputTokens': 75,
                'cacheMeasuredInputTokens': 100, 'cacheSamples': 1})
            self.assertEqual(data['context'], {'usedTokens': 200, 'capacityTokens': 131072})
            self.assertEqual(data['modelSeconds'], 5.0)
            data = adapter.session_statistics({'session_id': 'other', 'messages': []}, log, config)
            self.assertNotIn('cachedInputTokens', data['usage'])

    def test_parallel_calls_show_shared_thought_once(self):
        messages = [{"role": "assistant", "reasoning": "same reasoning", "tool_calls": [
            {"id": "a", "function": {"name": "web_search", "arguments": '{"query":"first"}'}},
            {"id": "b", "function": {"name": "web_search", "arguments": '{"query":"second"}'}},
        ]}]
        steps = adapter.build_steps(messages)
        self.assertIn('same reasoning', steps[0]['thought'])
        self.assertIn('web_search({"query":"first"})', steps[0]['thought'])
        self.assertIn('web_search({"query":"second"})', steps[1]['thought'])
        self.assertNotIn('same reasoning', steps[1]['thought'])
        self.assertEqual([s['toolCallId'] for s in steps], ['a', 'b'])

    def test_search_call_is_matched_with_its_result(self):
        messages = [
            {"role": "assistant", "reasoning": "Find evidence", "tool_calls": [
                {"id": "a", "function": {"name": "web_search", "arguments": '{"query":"example"}'}}
            ]},
            {"role": "tool", "tool_call_id": "a", "content": '<untrusted>\n' + json.dumps({"data": {"web": [{"title": "Example", "url": "https://example.org/a", "description": "snippet"}]}}) + '\n</untrusted>'},
        ]
        step = adapter.build_steps(messages)[0]
        self.assertEqual(step["kind"], "search")
        self.assertEqual(step["tool"]["queries"][0]["query"], "example")
        self.assertEqual(step["tool"]["queries"][0]["results"][0]["domain"], "example.org")
        self.assertEqual(step["tool"]["resultCount"], 1)
        self.assertIsNone(step["tool_seconds"])

    def test_failed_or_missing_tool_results_are_preserved(self):
        messages = [{"role": "assistant", "tool_calls": [
            {"id": "a", "function": {"name": "execute_code", "arguments": '{"code":"print(1)"}'}},
            {"id": "b", "function": {"name": "web_search", "arguments": '{"query":"q"}'}}
        ]}, {"role": "tool", "tool_call_id": "a", "content": "refused"}]
        steps = adapter.build_steps(messages)
        self.assertEqual([s["kind"] for s in steps], ["compute", "search"])
        self.assertEqual(steps[0]["tool"]["stdout"], "refused")
        self.assertEqual(steps[1]["tool"]["resultCount"], 0)

    def test_atomic_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "traces.json"
            adapter.atomic_json(path, {"run": {"name": "测试"}, "cases": []})
            self.assertEqual(adapter.read_json(path)["run"]["name"], "测试")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_compaction_preserves_parent_without_repeating_inherited_messages(self):
        parent = {"session_id": "one", "session_start": "2026-01-01T00:00:00", "messages": [{"role": "assistant", "content": "before"}]}
        child = {"session_id": "two", "parent_session_id": "one", "messages": [
            {"role": "assistant", "content": "before", "_inherited_from_previous_session": True},
            {"role": "assistant", "content": "after"},
        ]}
        merged = adapter.merge_session_chain([child, parent])
        self.assertEqual([m["content"] for m in merged["messages"]], ["before", "after"])
        self.assertEqual(merged["session_start"], parent["session_start"])

    def test_recorded_timing_is_matched_and_ambiguity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "state.db"
            log = root / "agent.log"
            origin = datetime(2026, 1, 1, 12).timestamp()
            calls = [{"id": "call1", "function": {"name": "web_search", "arguments": '{"query":"q"}'}}]
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, timestamp REAL, tool_call_id TEXT, tool_name TEXT, content TEXT, tool_calls TEXT, inherited_from_previous_session INTEGER)")
                conn.execute("INSERT INTO messages VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)", ("s1", "assistant", origin + 3.2, None, None, "", json.dumps(calls), 0))
                conn.execute("INSERT INTO messages VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)", ("s1", "tool", origin + 5.2, "call1", "web_search", "abc", None, 0))
            def stamp(offset):
                return datetime.fromtimestamp(origin + offset).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
            api = f"{stamp(3)} INFO [s1] agent.conversation_loop: API call #1: model=m latency=2.0s\n"
            tool = f"{stamp(5)} INFO agent.tool_executor: tool web_search completed (1.50s, 3 chars)\n"
            log.write_text(api + tool)
            messages = [{"role": "assistant", "_viewer_session_id": "s1", "tool_calls": calls}, {"role": "tool", "tool_call_id": "call1", "content": "abc"}]
            steps = adapter.build_steps(messages)
            session = {"session_start": datetime.fromtimestamp(origin).isoformat()}
            self.assertTrue(adapter.enrich_timing(session, steps, db, log))
            self.assertEqual((steps[0]["modelStart"], steps[0]["model_seconds"]), (1.0, 2.0))
            self.assertEqual((steps[0]["toolStart"], steps[0]["tool_seconds"]), (3.5, 1.5))
            earlier_failure = f"{stamp(4)} INFO agent.tool_executor: tool web_search failed (0.50s)\n"
            log.write_text(api + earlier_failure + tool)
            self.assertTrue(adapter.enrich_timing(session, adapter.build_steps(messages), db, log))
            log.write_text(api + tool + tool)
            self.assertFalse(adapter.enrich_timing(session, adapter.build_steps(messages), db, log))

    def test_missing_parallel_result_does_not_hide_completed_call_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); db = root/'state.db'; log = root/'agent.log'
            origin = datetime(2026, 1, 1, 12).timestamp()
            calls = [{'id': name, 'function': {'name': 'web_extract', 'arguments': '{}'}} for name in ('pending', 'done')]
            with sqlite3.connect(db) as conn:
                conn.execute('CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, timestamp REAL, tool_call_id TEXT, tool_name TEXT, content TEXT, tool_calls TEXT, inherited_from_previous_session INTEGER)')
                conn.execute('INSERT INTO messages VALUES (NULL,?,?,?,?,?,?,?,?)', ('s', 'assistant', origin+3.2, None, None, '', json.dumps(calls), 0))
                conn.execute('INSERT INTO messages VALUES (NULL,?,?,?,?,?,?,?,?)', ('s', 'tool', origin+5.2, 'done', 'web_extract', 'abc', None, 0))
            def stamp(offset): return datetime.fromtimestamp(origin+offset).strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            log.write_text(f'{stamp(3)} INFO [s] agent.conversation_loop: API call #1: model=m latency=2.0s\n' +
                           f'{stamp(5)} INFO agent.tool_executor: tool web_extract completed (1.50s, 3 chars)\n')
            messages = [{'role': 'assistant', '_viewer_session_id': 's', 'tool_calls': calls}, {'role': 'tool', 'tool_call_id': 'done', 'content': 'abc'}]
            steps = adapter.build_steps(messages)
            self.assertFalse(adapter.enrich_timing({'session_start': datetime.fromtimestamp(origin).isoformat()}, steps, db, log))
            self.assertIsNone(steps[0]['tool_seconds'])
            self.assertEqual(steps[1]['tool_seconds'], 1.5)

    def test_identified_parallel_spans_and_single_error_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); db = root/'state.db'; log = root/'agent.log'
            origin = datetime(2026, 1, 1, 12).timestamp()
            calls = [{'id': name, 'function': {'name': 'web_search', 'arguments': '{}'}} for name in ('a', 'b')]
            with sqlite3.connect(db) as conn:
                conn.execute('CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, timestamp REAL, tool_call_id TEXT, tool_name TEXT, content TEXT, tool_calls TEXT, inherited_from_previous_session INTEGER)')
                conn.execute('INSERT INTO messages VALUES (NULL,?,?,?,?,?,?,?,?)', ('s', 'assistant', origin+3.2, None, None, '', json.dumps(calls), 0))
                for name in ('a', 'b'):
                    conn.execute('INSERT INTO messages VALUES (NULL,?,?,?,?,?,?,?,?)', ('s', 'tool', origin+6.2, name, 'web_search', 'same', None, 0))
            def stamp(offset): return datetime.fromtimestamp(origin+offset).strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            api = f'{stamp(3)} INFO [s] agent.conversation_loop: API call #1: model=m latency=2.0s\n'
            def event(offset, cid, seconds):
                value = json.dumps({'session_id':'s', 'tool_call_id':cid, 'tool_name':'web_search', 'duration_seconds':seconds})
                return f'{stamp(offset)} INFO agent.tool_executor: glasseek_tool_timing {value}\n'
            log.write_text(api + event(5, 'a', 1.1) + event(6, 'b', 1.8))
            messages = [{'role':'assistant', '_viewer_session_id':'s', 'tool_calls':calls}] + [{'role':'tool', 'tool_call_id':name, 'content':'same'} for name in ('a','b')]
            steps = adapter.build_steps(messages)
            self.assertTrue(adapter.enrich_timing({'session_start':datetime.fromtimestamp(origin).isoformat()}, steps, db, log))
            self.assertEqual([s['tool_seconds'] for s in steps], [1.1, 1.8])
            # A warning is accepted only if its preview matches the result.
            log.write_text(api + f'{stamp(6)} WARNING agent.tool_executor: Tool web_search returned error (1.80s): same\n')
            steps = adapter.build_steps(messages)
            self.assertFalse(adapter.enrich_timing({'session_start':datetime.fromtimestamp(origin).isoformat()}, steps, db, log))
            self.assertIsNone(steps[0]['tool_seconds'])
            with sqlite3.connect(db) as conn:
                conn.execute('UPDATE messages SET tool_calls=? WHERE role="assistant"', (json.dumps(calls[:1]),))
                conn.execute('DELETE FROM messages WHERE tool_call_id="b"')
            steps = adapter.build_steps([{'role':'assistant', '_viewer_session_id':'s', 'tool_calls':calls[:1]}, messages[1]])
            self.assertTrue(adapter.enrich_timing({'session_start':datetime.fromtimestamp(origin).isoformat()}, steps, db, log))
            self.assertEqual(steps[0]['tool_seconds'], 1.8)

    def test_single_page_receives_recorded_tool_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); db = root/'state.db'; log = root/'agent.log'
            origin = datetime(2026, 1, 1, 12).timestamp()
            calls = [{'id': 'read', 'function': {'name': 'web_extract', 'arguments': '{"urls":["https://example.org"]}'}}]
            output = json.dumps({'results': [{'url': 'https://example.org', 'content': 'content'}]})
            with sqlite3.connect(db) as conn:
                conn.execute('CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, timestamp REAL, tool_call_id TEXT, tool_name TEXT, content TEXT, tool_calls TEXT, inherited_from_previous_session INTEGER)')
                conn.execute('INSERT INTO messages VALUES (NULL,?,?,?,?,?,?,?,?)', ('s', 'assistant', origin+3.2, None, None, '', json.dumps(calls), 0))
                conn.execute('INSERT INTO messages VALUES (NULL,?,?,?,?,?,?,?,?)', ('s', 'tool', origin+5.2, 'read', 'web_extract', output, None, 0))
            def stamp(offset): return datetime.fromtimestamp(origin+offset).strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            log.write_text(f'{stamp(3)} INFO [s] agent.conversation_loop: API call #1: model=m latency=2.0s\n' +
                           f'{stamp(5)} INFO agent.tool_executor: tool web_extract completed (1.50s, {len(output)} chars)\n')
            steps = adapter.build_steps([{'role': 'assistant', '_viewer_session_id': 's', 'tool_calls': calls}, {'role': 'tool', 'tool_call_id': 'read', 'content': output}])
            self.assertTrue(adapter.enrich_timing({'session_start': datetime.fromtimestamp(origin).isoformat()}, steps, db, log))
            self.assertEqual(steps[0]['tool']['pages'][0]['readerSeconds'], 1.5)


if __name__ == "__main__":
    unittest.main()
