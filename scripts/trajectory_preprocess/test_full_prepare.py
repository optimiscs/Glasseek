import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import full_prepare as fp


class FullPrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {'max_sample_chars': 250000,
                       'dedup': {'near_question_hamming_distance': 3, 'max_bucket_candidates': 200},
                       'harness': {'search_name': 'web_search', 'extract_name': 'web_extract', 'extract_max_urls': 5}}

    def tearDown(self):
        self.temp.cleanup()

    def row(self, number):
        return {'question': 'Question ' + str(number), 'messages': [
            {'role': 'assistant', 'content': '<tool_call>{"name":"search","arguments":{"query":"q"}}</tool_call>'},
            {'role': 'tool', 'content': 'Evidence'},
            {'role': 'assistant', 'content': 'Answer ' + str(number)}]}

    def spec(self, name, rows):
        path = self.root / (name + '.jsonl')
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return {'path': str(path), 'source': name}

    def test_restart_matches_uninterrupted_and_deduplicates_across_sources(self):
        specs = [self.spec('A', [self.row(i) for i in range(6)]),
                 self.spec('B', [self.row(0), {'unknown': 'schema'}])]
        baseline = self.root / 'baseline'
        expected = fp.run(specs, self.config, baseline, checkpoint_every=3)
        resumed = self.root / 'resumed'
        original = fp.Dedup.add
        calls = 0

        def fail_after_uncommitted_row(instance, item, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 5:
                raise RuntimeError('simulated interruption')
            return original(instance, item, **kwargs)

        with patch.object(fp.Dedup, 'add', fail_after_uncommitted_row):
            with self.assertRaises(RuntimeError):
                fp.run(specs, self.config, resumed, checkpoint_every=3)
        actual = fp.run(specs, self.config, resumed, checkpoint_every=3)
        self.assertEqual(expected['counts'], actual['counts'])
        self.assertEqual(actual['counts']['eligible'], 6)
        self.assertEqual(actual['counts']['exact_duplicates'], 1)
        for name in fp.OUTPUTS:
            self.assertEqual((baseline / (name + '.jsonl')).read_bytes(),
                             (resumed / (name + '.jsonl')).read_bytes())
        # Re-running a completed directory is idempotent.
        self.assertEqual(fp.run(specs, self.config, resumed)['counts'], actual['counts'])

    def test_structural_failures_do_not_enter_clean_and_changed_rules_refused(self):
        bad = self.row(1)
        bad['messages'] = bad['messages'][:1]
        spec = self.spec('A', [bad, self.row(2)])
        out = self.root / 'out'
        result = fp.run([spec], self.config, out)
        self.assertEqual(result['counts']['held'], 1)
        clean = [json.loads(line) for line in (out / 'clean.jsonl').read_text().splitlines()]
        self.assertEqual(len(clean), 1)
        self.assertFalse(clean[0]['training_ready'])
        self.assertEqual(clean[0]['program_errors'], [])
        changed = copy.deepcopy(self.config)
        changed['max_sample_chars'] = 10
        with self.assertRaises(ValueError):
            fp.run([spec], changed, out)

    def test_corrupt_file_is_visible_not_success(self):
        spec = self.spec('broken', [self.row(1)])
        with Path(spec['path']).open('a') as file:
            file.write('{')
        result = fp.run([spec], self.config, self.root / 'out')
        self.assertEqual(result['status'], 'completed_with_file_errors')
        self.assertEqual(result['counts']['file_errors'], 1)
        self.assertEqual(result['completed_files'], 0)

    def test_real_parquet_read_and_metadata_projection(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = self.root / 'rows.parquet'
        pq.write_table(pa.Table.from_pylist([self.row(1), self.row(2)]), path)
        bundle = self.root / 'bundle.parquet'
        pq.write_table(pa.Table.from_pylist([
            {'question': 'Q', 'num_subagents': 1, 'subagents': ['large nested content']}]), bundle)
        specs = [{'path': str(path), 'source': 'test'},
                 {'path': str(bundle), 'source': 'SearchSwarm'}]
        out = self.root / 'parquet-out'
        result = fp.run(specs, self.config, out)
        self.assertEqual(result['counts']['scanned'], 3)
        self.assertEqual(result['counts']['eligible'], 2)
        self.assertEqual(result['counts']['held'], 1)
        self.assertEqual(result['completed_files'], 2)
        self.assertEqual(result['status'], 'completed')
        manifest = json.loads((out / 'manifest.json').read_text())
        self.assertEqual(manifest['parquet_expected_rows'][str(path)], 2)

    def test_missing_parquet_dependency_fails_before_processing(self):
        path = self.root / 'dummy.parquet'
        path.write_bytes(b'')
        out = self.root / 'missing-dependency'
        import sys
        with patch.dict(sys.modules, {'pyarrow': None, 'pyarrow.parquet': None}):
            with self.assertRaisesRegex(RuntimeError, 'require pyarrow'):
                fp.run([{'path': str(path), 'source': 'test'}], self.config, out)
        self.assertFalse((out / 'manifest.json').exists())
        self.assertFalse((out / 'clean.jsonl').exists())

    def test_parallel_output_and_resume_are_byte_identical_to_serial(self):
        rows = [self.row(i) for i in range(30)]
        rows += [self.row(0), {'unknown': 'schema'}]
        rows[4]['messages'] = rows[4]['messages'][:1]
        specs = [self.spec('A', rows), self.spec('B', [self.row(1), self.row(40)])]
        serial, parallel = self.root / 'serial', self.root / 'parallel'
        baseline = fp.run(specs, self.config, serial)
        fp.run(specs, self.config, parallel, workers=2, stop_after=11)
        result = fp.run(specs, self.config, parallel, workers=3)
        self.assertEqual(result['counts'], baseline['counts'])
        for name in fp.OUTPUTS:
            self.assertEqual((serial / (name + '.jsonl')).read_bytes(),
                             (parallel / (name + '.jsonl')).read_bytes())


if __name__ == '__main__':
    unittest.main()
