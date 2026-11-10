import json
from pathlib import Path
import pytest
from trajectory_review import pipeline_view as view


@pytest.fixture
def run(tmp_path,monkeypatch):
    monkeypatch.setenv('GLASSEEK_CURATION_RUN',str(tmp_path));view.CACHE.clear()
    item={'id':'x','source':'DS','question':'<script>untrusted</script>','steps':[
        {'step_id':1,'assistant':'answer','calls':[],'observation':''}]}
    (tmp_path/'clean.jsonl').write_text(json.dumps(item)+'\n')
    (tmp_path/'gpu0/results').mkdir(parents=True)
    return tmp_path,item


def test_execution_success_is_not_quality_pass_or_training_release(run):
    path,item=run
    (path/'gpu0/calls.jsonl').write_text(json.dumps({'id':'x','stage':'review','status':'success','seconds':2})+'\n{"partial":')
    s=view.snapshot()
    assert s['rows'][0]['states']['review']=='done'
    assert s['rows'][0]['states']['safety']=='done'
    assert s['summary']['candidate']==s['summary']['training_ready']==0
    assert s['stages'][4]['requests']==1


def test_candidates_keep_human_and_replay_pending_and_merges_are_not_conflicts(run):
    path,item=run
    result={'status':'scored','offline_quality_candidate':True,'automatic_acceptance_passed':True,
            'selection':{'eligible_for_ranking':True},'human_review':'pending','training_ready':False,
            'eligibility':{'status':'eligible','safety':{'state':'pass_rules','coverage_complete':True}},
            'merged_behavior_annotations':[{'kind':'program_conversion_collision'}]}
    (path/'gpu0/results/x.json').write_text(json.dumps({'sample':item,'result':result}))
    s=view.snapshot();row=s['rows'][0]
    assert s['summary']['candidate']==1 and s['summary']['training_ready']==0
    assert row['states']['human']==row['states']['replay']=='pending'
    assert row['verification_merges']==1 and row['conflicts']==0


def test_explicit_conflict_remains_held_and_all_stage_counts_cover_batch(run):
    path,item=run
    result={'status':'scored','offline_quality_candidate':False,'audit':{'items':[{'status':'conflict'}]},
            'eligibility':{'status':'eligible'}}
    (path/'gpu0/results/x.json').write_text(json.dumps({'sample':item,'result':result}))
    s=view.snapshot()
    assert s['rows'][0]['states']['review']=='hold'
    assert s['rows'][0]['conflicts']==1
    assert all(sum(stage['counts'].values())==1 for stage in s['stages'])


def test_details_cannot_select_files_outside_frozen_input(run):
    with pytest.raises(KeyError):view.detail('../../etc/passwd')
    assert view.detail('x')['question']=='<script>untrusted</script>'


def test_percentiles_include_long_failed_requests():
    assert view.percentile([1,2,100],.95)==90.2
    assert view.percentile([],.95) is None


def test_later_execution_error_does_not_reclassify_passed_quality_as_bad_data(run):
    path,item=run
    result={'status':'annotation_error','error':'segment_contribution: API HTTP 400',
            'raw_behaviors':{'steps':[]},'offline_quality_candidate':False,
            'quality_issues':[],'evidence_gaps':[],'annotation_issues':[],
            'eligibility':{'status':'eligible'}}
    (path/'gpu0/results/x.json').write_text(json.dumps({'sample':item,'result':result}))
    row=view.snapshot()['rows'][0]
    assert row['states']['quality']=='pass'
    assert row['states']['segment_contribution']=='error'
    assert row['outcome']=='annotation_error'
