import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Evaluation'))
from browse_comp_eval.answer_prompts import load_answer_prompt
from browse_comp_eval.dataset import DatasetItem, RunSpec
from browse_comp_eval.hermes_client import build_query

class AnswerPromptTests(unittest.TestCase):
    def test_browsecomp_format_used_for_quest_subset(self):
        item = DatasetItem('row', 'Who is it?', 'PRIVATE REFERENCE', 'Music', 1)
        prompt = build_query(RunSpec('row', 'row__r1', 1, item), dataset='quest_qwen27b_128')
        for label in ('Explanation:', 'Exact Answer:', 'Confidence:'):
            self.assertIn(label, prompt)
        self.assertNotIn('FINAL ANSWER:', prompt)
        self.assertNotIn('PRIVATE REFERENCE', prompt)

    def test_dataset_registry_exact_fallback_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in ('default', 'general', 'specific'):
                (d / (name + '.txt')).write_text(name)
            p = d / 'registry.json'
            p.write_text(json.dumps({'default': 'default.txt', 'datasets': {'set*': 'general.txt', 'set-one': 'specific.txt'}}))
            self.assertEqual(load_answer_prompt('set-one', p), ('set-one', 'specific'))
            self.assertEqual(load_answer_prompt('set-two', p), ('set*', 'general'))
            self.assertEqual(load_answer_prompt('unmapped', p), ('default', 'default'))
            p.write_text(json.dumps({'default': 'default.txt', 'datasets': {'set*': 'general.txt', '*two': 'specific.txt'}}))
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                load_answer_prompt('set-two', p)

if __name__ == '__main__': unittest.main()
