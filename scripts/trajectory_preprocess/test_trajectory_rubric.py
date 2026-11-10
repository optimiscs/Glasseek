import pytest
from trajectory_rubric import RULES, GATES, score

def judgment():
    return {'difficulty':{'refs':[],'reason':'Direct factual lookup.','score':'20.25','confidence':'90.50'},'criteria':{k:{'level':'strong'} for k in RULES},
            'gates':{k:{'level':'pass'} for k in GATES}}

def test_hard_gate_cannot_be_offset_by_quality_points():
    value=judgment()
    value['gates']['integrity']['level']='fail'
    result=score(value)
    assert result['score']==100
    assert result['status']=='quality_hold'
    assert result['training_ready'] is False

def test_unknown_is_not_low_quality_or_rankable():
    value=judgment()
    value['criteria']['evidence_quality']['level']='unknown'
    result=score(value)
    assert result['score'] is None
    assert result['status']=='review'

def test_program_weights_and_gate_uncertainty():
    value=judgment()
    value['criteria']['efficiency']['level']='absent'
    assert score(value)['score']==95
    value['gates']['critical_support']['level']='unknown'
    assert score(value)['status']=='review'


def test_conflicts_not_repeated_verification_require_review():
    value=judgment()
    value['criteria']['answer_grounding']['level']='absent'
    result=score(value)
    assert result['label_conflicts']==['critical_support:answer_grounding']
    assert result['status']=='review'


def test_full_prompt_hides_reference_and_retains_observation_and_instruction():
    from test_curation import fixture
    from trajectory_rubric import request
    item,_,_=fixture()
    item['reference_answer']='SECRET_REFERENCE_CANARY'
    item['instruction_messages']=[{'role':'user','before_step':2,'content':'late requirement'}]
    prompt,_,_=request(item)
    assert 'SECRET_REFERENCE_CANARY' not in prompt
    assert 'Recorded result' in prompt and 'late requirement' in prompt
    assert '七、待评价轨迹' in prompt and '420' in prompt
    assert '多次验证不等于冲突' in prompt


def test_safety_routes_before_any_single_call():
    from test_curation import fixture
    from trajectory_rubric import screen
    item,canonical,scanner=fixture()
    scanner.scan=lambda row:{'state':'quarantine','coverage_complete':True}
    class Client:
        def call(self,*args,**kwargs):
            pytest.fail('Quarantined data must not reach LLM')
    result=screen(item,canonical,scanner,Client())
    assert result['decision']=='quarantine' and result['score'] is None


def test_single_call_and_unknown_evidence_is_annotation_error():
    from test_curation import fixture
    from trajectory_rubric import screen
    item,canonical,scanner=fixture()
    class Client:
        calls=0
        def call(self,stage,ident,prompt,system,validator,*,schema):
            self.calls+=1
            assert stage=='whole_trajectory'
            value=judgment()
            for group in (value['criteria'],value['gates']):
                for row in group.values():
                    row.update(reason='Observed support',refs=['S999:V0:0'])
            return validator(value)
    client=Client()
    result=screen(item,canonical,scanner,client)
    assert client.calls==1
    assert result['status']=='annotation_error'
    assert result['training_ready'] is False


def test_assistant_assertions_cannot_prove_factual_pass():
    from test_curation import fixture
    from trajectory_rubric import request
    item,_,_=fixture()
    _,_,validate=request(item)
    value=judgment()
    for group in (value['criteria'],value['gates']):
        for row in group.values():
            row.update(reason='assertion',refs=['B2:0'])
    with pytest.raises(ValueError,match='Tool evidence'):
        validate(value)


def test_compressed_full_pool_links_exact_original_rows(tmp_path):
    import gzip,json
    from rubric_batch import connect,prepare,unpack
    source=tmp_path/'source.jsonl.gz'
    items=[{'id':'one','source':'DF'},{'id':'two','source':'DF'}]
    with gzip.open(source,'wt') as f:
        for item in items:
            f.write(json.dumps(item)+chr(10))
    canonical=tmp_path/'passed.jsonl'
    canonical.write_text(json.dumps({'id':'two','source':'DF','safety':{'state':'pass_rules'},
                                    'source_artifact':str(source),'source_line_zero_based':1})+chr(10))
    db=connect(tmp_path/'batch.sqlite')
    prepare(db,canonical)
    assert unpack(db.execute('SELECT sample FROM inputs').fetchone()[0])==items[1]
    db.close()


def test_config_rejects_external_and_generation_retries():
    from rubric_batch import validate_config
    config={'local_only':True,'providers':{'gpu0':{'url':'http://127.0.0.1:19340/v1',
            'model':'qwen3.5-9b-local','transport_attempts':1,'max_concurrent_requests':16}},
            'generation':{'attempts':1},'stages':{'whole_trajectory':{'provider':'default'}}}
    validate_config(config)
    config['providers']['gpu0']['max_concurrent_requests']=64
    validate_config(config)
    for invalid in (0,65,True,1.5):
        config['providers']['gpu0']['max_concurrent_requests']=invalid
        with pytest.raises(ValueError):validate_config(config)
    config['providers']['gpu0']['max_concurrent_requests']=64
    config['generation']['attempts']=2
    with pytest.raises(ValueError):validate_config(config)
    config['generation']['attempts']=1
    config['providers']['gpu0']['url']='https://external.invalid/v1'
    with pytest.raises(ValueError):validate_config(config)


def test_wire_schema_blocks_invented_ids_and_assistant_only_factual_passes():
    import copy
    import jsonschema
    from test_curation import fixture
    from trajectory_rubric import request
    item,_,_=fixture()
    _,schema,_=request(item)
    valid=judgment()
    for group in (valid['criteria'],valid['gates']):
        for row in group.values():
            row.update(reason='Located supporting observation',refs=['S1:V0:0'])
    jsonschema.Draft202012Validator(schema).validate(valid)
    bad=copy.deepcopy(valid)
    bad['criteria']['efficiency']['refs']=['S999:V0:0']
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(bad)
    bad=copy.deepcopy(valid)
    bad['gates']['critical_support']['refs']=['B2:0']
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(bad)


@pytest.mark.parametrize('numeric',[False,True])
def test_difficulty_schema_and_independent_quality(numeric):
    import copy,jsonschema
    from test_curation import fixture
    from trajectory_rubric import request
    item,_,_=fixture()
    prompt,schema,validate=request(item,numeric=numeric)
    value=judgment()
    for group in (value['criteria'],value['gates']):
        for row in group.values():row.update(reason='Recorded support',refs=['S1:V0:0'])
    if numeric:
        for row in value['criteria'].values():
            row.pop('level');row['score']='95.25'
    jsonschema.Draft202012Validator(schema).validate(value)
    for level in ('0.00','29.99','65.25','100.00',None):
        candidate=copy.deepcopy(value);candidate['difficulty']['score']=level
        result=score(validate(candidate),numeric=numeric)
        assert result['status']=='rankable'
        assert result['score']==(95.25 if numeric else 100)
        assert result['difficulty']['score']==level
        assert 'difficulty' in result['rubric_version']
    bad=copy.deepcopy(value);del bad['difficulty']
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(bad,schema)
    bad=copy.deepcopy(value);bad['difficulty']['refs']=['S999:V0:0']
    with pytest.raises(ValueError,match='difficulty evidence'):validate(bad)
    assert '实际轮次多' in prompt and '不参与六维质量加权' in prompt
    assert 'difficulty' in schema['required']


def test_difficulty_decimal_rejects_invalid_format():
    from test_curation import fixture
    from trajectory_rubric import request
    item,_,_=fixture()
    _,_,validate=request(item)
    for field in ('score','confidence'):
        for invalid in (0,50.25,'50','50.2','100.01','-0.01','101.00'):
            value=judgment();value['difficulty'][field]=invalid
            with pytest.raises(ValueError,match='difficulty decimal'):validate(value)
