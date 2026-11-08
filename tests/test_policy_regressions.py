import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'Shared'), str(ROOT / 'RL_training'), str(ROOT / 'Evaluation')]
from glasseek_shared.screening import policy
from glasseek_training.curriculum_policy import build_sampler
from glasseek_training.turn_sampler import TurnSampler
from browse_comp_eval.judge_policy import RecordJudgePolicy, SummaryJudgePolicy


def test_screening_preserves_policy_differences():
    top, strong = policy('top10'), policy('strong')
    for p in (top, strong):
        assert p.teacher_answer('<think>Final Answer: wrong</think>\nFinal Answer: Paris') == 'Paris'
        assert p.teacher_answer('Final Answer: A\nFinal Answer: B') is None
        assert p.teacher_answer('Final Answer: unknown') is None
    # This is an existing distinction, not silently unified during refactoring.
    assert top.teacher_answer('Final Answer: ｕｎｋｎｏｗｎ') is None
    assert strong.teacher_answer('Final Answer: ｕｎｋｎｏｗｎ') == 'ｕｎｋｎｏｗｎ'
    assert (top.judge_tokens, top.judge_thinking) == (4096, True)
    assert (strong.judge_tokens, strong.judge_thinking) == (1024, False)
    rows = [{'id': 1}]
    assert top.initially_pending('q', rows, {}, set())
    assert not strong.initially_pending('q', rows, {}, set())
    assert strong.initially_pending('q', rows, {'q': ()}, set())
    assert not strong.initially_pending('q', rows, {'q': ()}, {1})


def test_candidate_validation_rejects_equivalent_duplicates():
    candidates = [{'rank': 1, 'answer': 'Paris', 'aliases': []},
                  {'rank': 2, 'answer': 'ＰＡＲＩＳ', 'aliases': []}]
    with pytest.raises(ValueError, match='duplicate_candidate'):
        policy('top10').parse_candidates(json.dumps({'candidates': candidates}))
    candidates[1]['answer'] = 'London'
    assert policy('top10').parse_candidates(json.dumps({'candidates': candidates})) == candidates
    candidates[0]['rank'] = True
    with pytest.raises(ValueError, match='invalid_candidate'):
        policy('top10').parse_candidates(json.dumps({'candidates': candidates}))


@pytest.mark.parametrize('size', [1, 2, 7, 17])
def test_turn_sampler_matches_pre_refactor_order_and_padding(size):
    from torch.utils.data import Sampler
    namespace = {'Sampler': Sampler}
    source = (ROOT / 'tests/fixtures/turn-sampler-reference.txt').read_text()
    exec(compile(source, '<frozen-turn-sampler>', 'exec'), namespace)
    messages = [[{'role': 'assistant'}] * (i % 3 + 1) for i in range(size)]
    for rank in (0, 1):
        original = namespace['TurnSampler'](messages, 4, 2, rank)
        actual = TurnSampler(messages, 4, 2, rank)
        for epoch in (0, 1):
            original.set_epoch(epoch); actual.set_epoch(epoch)
            assert list(actual) == list(original)
            assert actual.pad_indices == original.pad_indices


def test_turn_policy_validates_data_and_world_size(tmp_path):
    class Dataset:
        messages = [[{'role': 'assistant'}]] * 3
        def __len__(self): return len(self.messages)
    train = tmp_path / 'train.parquet'
    train.write_bytes(b'fixture')
    manifest = {'output_sha256': {'train': hashlib.sha256(b'fixture').hexdigest()},
                'max_length': 65536, 'counts': {'train': 3}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    assert len(build_sampler('turns', Dataset(), 4, 2, 0, 1, str(train))) == 2
    with pytest.raises(ValueError, match='two ranks'):
        build_sampler('turns', Dataset(), 4, 4, 0, 1, str(train))
    train.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='does not match manifest'):
        build_sampler('turns', Dataset(), 4, 2, 0, 1, str(train))


def test_judge_policy_preserves_status_fingerprint_and_retry_gates():
    records = RecordJudgePolicy()
    assert not records.terminal('interrupted', {})
    assert not records.terminal('failed', {'attempt_consumed': False})
    assert records.terminal('failed', {})
    identity = ('run', 'new-fingerprint')
    assert records.pending(identity, {'status':'judged', 'pred_fingerprint':'old'}, set(), {}, 0)
    assert not records.pending(identity, {'status':'judged', 'pred_fingerprint':identity[1]}, set(), {}, 0)
    assert not records.pending(identity, None, {identity}, {}, 0)
    assert not records.pending(identity, None, set(), {identity: 60}, 59)
    assert records.pending(identity, None, set(), {identity: 60}, 60)
    summary = SummaryJudgePolicy(60)
    assert not summary.due(None, None, 100, 0)
    assert not summary.due('same', 'same', 59, 0)
    assert summary.due('new', 'same', 1, 0)
    assert summary.due('same', 'same', 60, 0)
