import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from resume_runs import prepare, prediction_root, prediction_run, adopt_grading_ledger, verify_model
from suite import sha


def setup(tmp_path):
    suite=tmp_path/'suite';suite.mkdir();(suite/'manifest.json').write_text('{}')
    prompt=tmp_path/'prompt.txt';prompt.write_text('Compact prompt')
    old=tmp_path/'old';old.mkdir()
    manifest={'suite_sha256':sha(suite/'manifest.json'),'context_manager':'hermes','url_mode':'raw',
        'max_tool_calls':120,'system_prompt_override':{'sha256':sha(prompt)},'score_config_sha256':None,
        'benchmarks':['browsecomp']}
    (old/'launch.json').write_text(json.dumps(manifest))
    args=SimpleNamespace(suite=suite,output_root=tmp_path,run_id='new',resume_run_id=['old'],
        context_manager='hermes',url_mode='raw',max_tool_calls=120,system_prompt_file=prompt,
        score_config=None,research_prompt='legacy')
    return args,old


def test_resume_preserves_prediction_identity_and_grade_claims(tmp_path):
    args,old=setup(tmp_path)
    sources=prepare(args,['browsecomp','hle'])
    assert prediction_run(args,'browsecomp')=='old_browsecomp'
    assert prediction_run(args,'hle')=='new_hle'
    assert prediction_root(args,'browsecomp')==tmp_path/'predictions/old_browsecomp/suite_browsecomp'
    task=old/'trajectory-scores/browsecomp/hash';task.mkdir(parents=True)
    (task/'state.json').write_text('{"state":"claimed"}')
    adopt_grading_ledger(args,sources)
    new=tmp_path/'new/trajectory-scores/browsecomp/hash/state.json'
    assert new.read_text()=='{"state":"claimed"}'
    new.write_text('{"state":"scored"}')
    adopt_grading_ledger(args,sources)
    assert json.loads(new.read_text())['state']=='scored'


@pytest.mark.parametrize('field,value',[('research_prompt','arex-native'),('url_mode','ref'),('max_tool_calls',12)])
def test_resume_rejects_changed_protocol(tmp_path,field,value):
    args,_=setup(tmp_path);setattr(args,field,value)
    with pytest.raises(ValueError):prepare(args,['browsecomp'])


def test_resume_rejects_prompt_and_model_changes(tmp_path):
    args,_=setup(tmp_path)
    args.system_prompt_file.write_text('different prompt')
    with pytest.raises(ValueError,match='system prompt'):prepare(args,['browsecomp'])
    args.system_prompt_file.write_text('Compact prompt');prepare(args,['browsecomp'])
    pred=prediction_root(args,'browsecomp')/'conv/id';pred.mkdir(parents=True)
    (pred/'id__r1.json').write_text('{"model":"other-model"}')
    with pytest.raises(ValueError,match='model'):verify_model(args,'browsecomp','base9b')


def test_duplicate_sources_are_rejected(tmp_path):
    args,_=setup(tmp_path);args.resume_run_id=['old','old']
    with pytest.raises(ValueError,match='multiple'):prepare(args,['browsecomp'])


def test_prompt_comparison_follows_adopted_identity(tmp_path):
    from compare_prompts import cases
    (tmp_path/'new').mkdir()
    (tmp_path/'new/launch.json').write_text('{"prediction_runs":{"browsecomp":"old_browsecomp"}}')
    pred=tmp_path/'predictions/old_browsecomp/suite_browsecomp/conv/id';pred.mkdir(parents=True)
    (pred/'id__r1.json').write_text('{"question_id":"id","run_id":"id__r1","status":"success"}')
    assert set(cases(tmp_path,['new']))=={('browsecomp','id')}
