import json
from pathlib import Path
import sys
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from suite import ROOT
from native_fixes import fact_statistics,fix_widesearch,clean_author_checkout


class NativeFixTests(unittest.TestCase):
    def test_clean_source_copy_ignores_bytecode_but_rejects_modified_code(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);source=p/'source';source.mkdir()
            def git(*args):return subprocess.run(['git','-C',str(source),*args],check=True,capture_output=True)
            git('init');(source/'main.py').write_text('print(1)\n');git('add','main.py')
            git('-c','user.name=Test','-c','user.email=test@example.org','commit','-m','fixture')
            (source/'__pycache__').mkdir();(source/'__pycache__/main.pyc').write_bytes(b'cache')
            clean_author_checkout(source,p/'clean')
            assert not (p/'clean/__pycache__').exists()
            assert (p/'clean/main.py').read_text()=='print(1)\n'
            (source/'main.py').write_text('print(2)\n')
            with self.assertRaises(subprocess.CalledProcessError):clean_author_checkout(source,p/'bad')

    def setUp(self):
        original_path = sys.path[:]
        original_modules = {k:v for k,v in sys.modules.items() if k == 'src' or k.startswith('src.')}
        def restore():
            sys.path[:] = original_path
            for key in list(sys.modules):
                if key == 'src' or key.startswith('src.'):
                    sys.modules.pop(key)
            sys.modules.update(original_modules)
        self.addCleanup(restore)

    def test_fact_zero_denominator_is_undefined_not_zero_precision(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);source=ROOT/'data/evaluation/suite5pct-v1/evaluators/deep-research-bench/utils/stat.py'
            (p/'input.jsonl').write_text(json.dumps({'id':1,'citations':[]})+'\n')
            status=fact_statistics(source,p/'input.jsonl',p/'result.txt')
            self.assertEqual(status['status'],'undefined_no_citations')
            self.assertIn('valid_rate: undefined',(p/'result.txt').read_text())
            self.assertEqual(status['citation_coverage'],0)

    def test_fact_defined_ratios_still_execute_pinned_author(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);source=ROOT/'data/evaluation/suite5pct-v1/evaluators/deep-research-bench/utils/stat.py'
            row={'id':1,'citations':[{}],'citations_deduped':{'u':{'validate_error':None,
                 'validate_res':[{'result':'supported'},{'result':'unsupported'},{'result':'unknown'}]}}}
            (p/'input.jsonl').write_text(json.dumps(row)+'\n')
            with patch.object(sys,'argv',[]):status=fact_statistics(source,p/'input.jsonl',p/'result.txt')
            self.assertEqual(status['status'],'defined')
            self.assertIn('valid_rate: 0.5',(p/'result.txt').read_text())

    def test_wide_empty_and_nonempty_intersections_use_native_metrics(self):
        import pandas as pd
        source=ROOT/'data/evaluation/suite5pct-v1/evaluators/widesearch'
        sys.path.insert(0,str(source))
        from src.evaluation import evaluation
        from src.evaluation.data_loader import WideSearchQuery
        fix_widesearch(evaluation)
        spec={'required':['key','value'],'unique_columns':['key'],
              'eval_pipeline':{'key':{'metric':['exact_match']},'value':{'metric':['number_near'],'criterion':0.0}}}
        for predicted,expected in [('other',0.0),('known',1.0)]:
            query=WideSearchQuery(instance_id='offline',query='q',evaluation=spec,
                                 answer=pd.DataFrame([{'key':'known','value':'10'}]),language='en')
            response=SimpleNamespace(instance_id='offline',extract_dataframe=lambda:pd.DataFrame([{'key':predicted,'value':'10'}]))
            with patch.object(evaluation,'primary_key_preprocess',return_value={}):
                result=evaluation.evaluate_single_query(query,response,None,'default_eval_config')
            self.assertFalse(result.msg.startswith('evaluator error:'))
            self.assertEqual(result.f1_by_row,expected)


if __name__=='__main__':unittest.main()
