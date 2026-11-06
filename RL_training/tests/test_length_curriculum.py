import json
import hashlib

import pytest

from glasseek_training.length_curriculum import DistributedLengthCurriculumSampler, LengthCurriculumSampler, load_training_lengths


def test_curriculum_selects_before_sort_and_changes_each_epoch():
    lengths = [8, 2, 6, 2, 4, 7, 1, 5, 3, 9]
    sampler = LengthCurriculumSampler(list(range(len(lengths))), lengths, batch_size=4, seed=0)
    first = list(sampler)
    assert len(sampler) == len(first) == 8
    assert len(set(first)) == 8
    assert [lengths[i] for i in first] == sorted(lengths[i] for i in first)
    assert first == list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first != second
    assert [lengths[i] for i in second] == sorted(lengths[i] for i in second)
    # The longest are not systematically dropped by taking a sorted prefix.
    assert 9 in first or 9 in second


def test_length_audit_requires_matching_parquet_and_complete_order(tmp_path):
    parquet = tmp_path / 'train.parquet'
    parquet.write_bytes(b'train data')
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'output_sha256': {'train': hashlib.sha256(b'train data').hexdigest()},
        'counts': {'train': 2}, 'max_length': 10,
    }))
    audit = tmp_path / 'selection.jsonl'
    rows = [dict(source_line=1, tokens=8, exact=True, split='train'),
            dict(source_line=2, tokens=3, exact=True, split='train')]
    audit.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert load_training_lengths(audit, parquet, 2) == [8, 3]
    parquet.write_bytes(b'other data')
    with pytest.raises(ValueError, match='does not match'):
        load_training_lengths(audit, parquet, 2)


def test_distributed_resume_reconstructs_each_global_batch():
    lengths = list(range(103, 0, -1))
    original = LengthCurriculumSampler(lengths, lengths, batch_size=16, seed=7)
    expected = list(original)
    ranks = [list(DistributedLengthCurriculumSampler(lengths, lengths, 16, 4, rank, seed=7, start_batch=3))
             for rank in range(4)]
    for batch in range(3, 6):
        local = (batch - 3) * 4
        reconstructed = sum((part[local:local + 4] for part in ranks), [])
        assert reconstructed == expected[batch * 16:(batch + 1) * 16]
