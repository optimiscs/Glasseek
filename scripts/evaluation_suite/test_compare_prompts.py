from unittest.mock import patch
from compare_prompts import compare


def test_pairs_only_common_ids_and_never_turns_missing_grade_into_wrong(tmp_path):
    def item(correct):return {'correct':correct,'status':'success','grade_status':'pending' if correct is None else 'scored','seconds':2,'tool_calls':3,'metrics':{}}
    before={('hle','a'):item(False),('hle','b'):item(True),('hle','c'):item(True)}
    after={('hle','a'):item(True),('hle','b'):item(None)}
    with patch('compare_prompts.cases',side_effect=[before,after]):r=compare(tmp_path,['old'],['new'])['benchmarks']['hle']
    assert r['paired_generated']==2
    assert r['paired_binary_scored']==1
    assert r['wrong_to_right']==1 and r['right_to_wrong']==0
    assert r['after_accuracy_paired']==1
