import pytest
import torch

from glasseek_training.ddp_checkpoint import checked_optimizer_state


def test_optimizer_conversion_rejects_reordered_or_incompatible_parameters():
    a, b = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(3))
    state = {'param_groups': [{'params': [0, 1]}],
             'state': {0: {'exp_avg': torch.ones(2), 'exp_avg_sq': torch.ones(2), 'step': torch.tensor(80.)}}}
    result = checked_optimizer_state(state, ['a', 'b'], [('a', a), ('b', b)])
    assert torch.equal(result['state'][0]['exp_avg'], torch.ones(2))
    with pytest.raises(ValueError, match='names or order'):
        checked_optimizer_state(state, ['b', 'a'], [('a', a), ('b', b)])
    with pytest.raises(ValueError, match='shape mismatch'):
        checked_optimizer_state(state, ['a', 'b'], [('a', b), ('b', a)])
