import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Evaluation'))
from benchmark_scores import task_scores, enrich_package


def write(path, value, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(x) for x in value) if lines else json.dumps(value))


class BenchmarkScoresTests(unittest.TestCase):
    def test_wide_scores_are_task_local_and_zero_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);refs=[{'id':'a'}, {'id':'b'}, {'id':'c'}, {'id':'d'}]
            write(p/'native_metrics.jsonl',[{'instance_id':'a','score':0,'f1_by_row':0.1,'msg':'cell match table\n0 1.0 0.0'},
                {'instance_id':'b','score':1,'f1_by_row':0.9,'msg':''},
                {'instance_id':'c','score':0,'msg':'evaluator error: traceback'}],True)
            result=task_scores(p,'widesearch',refs)
            self.assertEqual(result['a']['metrics'][0]['value'],0)
            self.assertEqual(result['b']['metrics'][0]['value'],1)
            self.assertEqual(result['d']['status'],'pending')
            self.assertEqual(result['c']['status'],'error');self.assertEqual(result['c']['metrics'],[])
            self.assertIsNone(result['a']['correct'])
            self.assertEqual(result['a']['status'],'scored')

    def test_failed_native_placeholder_zero_is_unscored(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            write(p/'author/glasseek/summary.json',{'tasks':{'a':{'avg_score':0,'pass':False}}})
            write(p/'author/glasseek/a/summary.json',[{'score':0,'status':'failed'}])
            write(p/'receipt.json',{'status':'unscored_error'})
            result=task_scores(p,'mind2web2',[{'id':'a'}])['a']
            self.assertEqual(result['status'],'error');self.assertEqual(result['metrics'],[])
            self.assertIsNone(result['correct'])
            write(p/'author/glasseek/a/answer_1/results/native.json',
                  {'answer_name':'answer_1.md','final_score':0.0,'eval_breakdown':[]})
            result=task_scores(p,'mind2web2',[{'id':'a'}])['a']
            self.assertEqual(result['status'],'scored');self.assertEqual(result['metrics'][0]['value'],0)

    def test_binary_correctness_does_not_use_truthiness(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            write(p/'scores.jsonl',[{'id':'a','correct':False,'status':'scored'},
                {'id':'b','correct':False,'status':'unscored','error_type':'Timeout'}],True)
            result=task_scores(p,'hle',[{'id':'a'},{'id':'b'}])
            self.assertIs(result['a']['correct'],False)
            self.assertEqual(result['b']['status'],'error');self.assertIsNone(result['b']['correct'])

    def test_lrb_keeps_per_task_dimensions_and_missing_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            write(p/'main/detailed_results_1.json',{'reports':[{'query_id':'a',
                'presentation_grading_results':{'summary':{'average_pass_rate':60}},
                'coverage_grading_results':{'summary':{'percentage_addressed':40}},
                'citation_grading_results':{'score':0},'consistency_grading_results':{'score':90}}]})
            result=task_scores(p,'liveresearchbench',[{'id':'a'}])['a']
            self.assertEqual(result['status'],'partial')
            self.assertEqual({m['key']:m['value'] for m in result['metrics']},
                             {'presentation':60,'consistency':90,'citation':0,'coverage':40})
            write(p/'depth/detailed_results_2.json',{'reports':[{'query_id':'a',
                'depth_grading_results':{'winner_comparison_1':'reference_model','winner_comparison_2':'tie'}}]})
            result=task_scores(p,'liveresearchbench',[{'id':'a'}])['a']
            self.assertEqual(result['status'],'scored')
            self.assertEqual(result['metrics'][-2]['value'],'参考模型胜')
            self.assertEqual(result['metrics'][-1]['value'],'平局')

    def test_package_metadata_and_aggregate_use_valid_tasks_only(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);output=base/'predictions/r_widesearch/suite_widesearch'
            path=base/'suite/inference/widesearch.jsonl'
            refs=[{'id':'a','opaque_id':'t1','benchmark':'widesearch'}, {'id':'b','opaque_id':'t2','benchmark':'widesearch'}]
            write(path.parent.parent/'private/references.jsonl',refs,True)
            write(base/'r/scores/widesearch/attempt-001/native_metrics.jsonl',
                  [{'instance_id':'a','score':0,'f1_by_row':0.4,'msg':''}],True)
            package={'run':{'requestedCases':2},'cases':[{'id':'t1','evaluationStatus':'judging','correct':None},
                       {'id':'t2','evaluationStatus':'judging','correct':None}]}
            result=enrich_package(package,output,path,'suite_widesearch','r_widesearch')
            self.assertEqual(result['cases'][0]['evaluationStatus'],'native_scored')
            self.assertEqual(result['cases'][1]['evaluationStatus'],'judging')
            self.assertEqual(result['run']['evaluation']['scored'],1)
            self.assertEqual(result['run']['evaluation']['metrics'][0]['count'],1)
            self.assertIsNone(result['cases'][0]['correct'])


if __name__=='__main__':unittest.main()
