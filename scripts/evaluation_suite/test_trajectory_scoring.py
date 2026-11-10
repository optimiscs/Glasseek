import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from suite import dump, jsonl
from trajectory_scoring import TrajectoryScorer


def fixture(tmp_path):
    suite=tmp_path/'release';suite.mkdir()
    jsonl(suite/'private/references.jsonl',[{'opaque_id':'a','id':'1','benchmark':'gaia'}])
    args=SimpleNamespace(suite=suite,output_root=tmp_path/'output',run_id='run',corpus_config=None,score_python=None,score_config=None,viewer=True)
    pred=args.output_root/'predictions/run_gaia/suite_gaia/conv/a/result.json'
    dump(pred,{'question_id':'a','status':'success','final_assistant_valid':True,'model_response':'answer'})
    return args,pred


def test_completed_task_grades_before_dataset_finishes_and_deduplicates(tmp_path):
    args,pred=fixture(tmp_path)
    with patch.object(TrajectoryScorer,'score') as score:
        engine=TrajectoryScorer(args,{'gaia':{}})
        engine.drain();engine.tick();engine.close()
        assert score.call_count==1
        engine=TrajectoryScorer(args,{'gaia':{}})
        engine.drain();engine.close()
        assert score.call_count==1


def test_incomplete_failed_and_partial_files_never_grade(tmp_path):
    args,pred=fixture(tmp_path)
    with patch.object(TrajectoryScorer,'score') as score:
        engine=TrajectoryScorer(args,{'gaia':{}})
        for row in [{'status':'running'},{'status':'success','final_assistant_valid':False},{'status':'error','final_assistant_valid':True,'model_response':'x'}]:
            dump(pred,dict(row,question_id='a'));engine.tick()
        pred.write_text('{');engine.tick();engine.close()
        score.assert_not_called()


def test_default_schedule_and_explicit_opt_out():
    from run import build_parser
    assert build_parser().parse_args([]).score_schedule=='trajectory'
    assert build_parser().parse_args(['--score-schedule','off']).score_schedule=='off'


def test_publication_only_updates_selected_task(tmp_path):
    import report
    args,pred=fixture(tmp_path)
    jsonl(args.suite/'private/references.jsonl',[{'opaque_id':'a','id':'1','benchmark':'gaia'},{'opaque_id':'b','id':'2','benchmark':'gaia'}])
    data=json.loads(pred.read_text());data['run_id']='r';dump(pred,data)
    second=pred.parent.parent/'b/result.json';dump(second,dict(data,question_id='b'))
    root=pred.parents[2]
    old=root/'judge_official/b/r.json';dump(old,{'status':'judged','score':1})
    score=tmp_path/'native';jsonl(score/'author/scores.jsonl',[{'task_id':'1','correct':True}])
    with patch('subprocess.run'):
        report.publish_scores(args,'gaia',score,{'EVAL_PYTHON':'python'},opaque_ids={'a'})
    assert json.loads(old.read_text())=={'status':'judged','score':1}
    assert json.loads((root/'judge_official/a/r.json').read_text())['score']==1


def test_packet_contains_only_completed_selected_task(tmp_path):
    from run import prepare_score
    args,pred=fixture(tmp_path)
    refs=[{'opaque_id':'a','id':'1','benchmark':'gaia','question':'Q','answer':'42'}, {'opaque_id':'b','id':'2','benchmark':'gaia','question':'other','answer':'9'}]
    jsonl(args.suite/'private/references.jsonl',refs)
    dump(args.suite/'manifest.json',{})
    dump(args.suite/'summary.json',[{'benchmark':'gaia','protocol':'native'}])
    dump(pred,{'question_id':'a','question':'Q','status':'success','final_assistant_valid':True,'model_response':'FINAL ANSWER: 42'})
    packet=tmp_path/'packet'
    prepare_score(args.suite,packet,pred.parents[2],'gaia',opaque_ids={'a'})
    assert json.loads((packet/'submission.jsonl').read_text())=={'task_id':'1','model_answer':'42'}
    assert (packet/'failures.jsonl').read_text()==''
    assert json.loads((packet/'manifest.json').read_text())['assigned']==1


def test_existing_grade_not_resubmitted(tmp_path):
    args,pred=fixture(tmp_path)
    dump(pred.parents[2]/'judge_official/a/.json',{'prediction':'answer','status':'judged','score':0})
    with patch.object(TrajectoryScorer,'score') as score:
        engine=TrajectoryScorer(args,{'gaia':{}});engine.drain();engine.close()
        score.assert_not_called()


def test_question_check_matches_loader_strip_but_rejects_content_changes(tmp_path):
    import pytest
    from run import prepare_score
    args,pred=fixture(tmp_path)
    jsonl(args.suite/'private/references.jsonl',[{'opaque_id':'a','id':'1','benchmark':'gaia','question':' Q\n\n','answer':'42'}])
    dump(args.suite/'manifest.json',{});dump(args.suite/'summary.json',[{'benchmark':'gaia','protocol':'native'}])
    row={'question_id':'a','question':'Q','status':'success','final_assistant_valid':True,'model_response':'FINAL ANSWER: 42'}
    dump(pred,row)
    prepare_score(args.suite,tmp_path/'ok',pred.parents[2],'gaia',opaque_ids={'a'})
    dump(pred,dict(row,question='different Q'))
    with pytest.raises(ValueError,match='question does not match'):
        prepare_score(args.suite,tmp_path/'bad',pred.parents[2],'gaia',opaque_ids={'a'})
