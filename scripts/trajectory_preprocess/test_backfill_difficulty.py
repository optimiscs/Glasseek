import copy
import jsonschema
from test_curation import fixture
from test_trajectory_rubric import judgment
from trajectory_rubric import request,difficulty_request,merge_difficulty,score


def test_same_prompt_schema_and_preserved_quality():
    item,_,_=fixture()
    full,schema,validate=request(item,numeric=True)
    prompt,partial,check=difficulty_request(item)
    rules=full[full.index('五补充、独立题目难度'):full.index('六、输出格式')]
    assert rules in prompt
    assert partial['properties']['difficulty']==schema['properties']['difficulty']
    assert partial['$defs']==schema['$defs']
    value=judgment()
    for group in (value['criteria'],value['gates']):
        for row in group.values():row.update(refs=['S1:V0:0'],reason='Recorded support')
    for row in value['criteria'].values():row.pop('level');row['score']='95.25'
    value=validate(value)
    original={'id':item['id'],'judgment':value,**score(value,numeric=True)}
    original.pop('difficulty');original['judgment'].pop('difficulty')
    before=copy.deepcopy(original)
    annotation={'difficulty':{'refs':[],'reason':'A direct lookup','score':'25.50','confidence':'85.25'}}
    jsonschema.validate(annotation,partial)
    result=merge_difficulty(item,original,check(annotation))
    assert original==before
    for key in ('score','points','status','failed_gates','training_ready'):
        assert result[key]==original[key]
    assert result['judgment']['criteria']==original['judgment']['criteria']
    assert result['judgment']['gates']==original['judgment']['gates']
    assert result['difficulty']==result['judgment']['difficulty']==annotation['difficulty']
