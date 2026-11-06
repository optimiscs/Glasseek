"""Check the real verl registry API and the recipe's worker adapter without CUDA imports."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
import torch

from glasseek_training.loss import cm_gspo_loss


def test_registered_loss_receives_global_denominator_and_rejects_reweighting():
    root = os.environ.get('VERL_SOURCE')
    if not root:
        pytest.skip('Set VERL_SOURCE for the upstream registry test')
    tree = ast.parse((Path(root)/'verl/trainer/ppo/core_algos.py').read_text())
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('register_policy_loss', 'get_policy_loss_fn')]
    namespace = {'Callable': Callable, 'PolicyLossFn': Callable, 'POLICY_LOSS_REGISTRY': {},
                 'cm_gspo_loss': cm_gspo_loss}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), 'verl_registry', 'exec'), namespace)
    adapter = Path(__file__).resolve().parents[1]/'glasseek_training/verl_loss.py'
    funcs = [node for node in ast.parse(adapter.read_text()).body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(adapter), 'exec'), namespace)
    fn = namespace['get_policy_loss_fn']('cm_gspo')
    config = SimpleNamespace(clip_ratio_low=.0003, clip_ratio_high=.0003, clip_ratio=.0003,
                             global_batch_info={'global_batch_size': 128, 'dp_size': 8})
    old = torch.full((2, 3), -2.)
    current = old.clone().requires_grad_()
    advantage, mask = torch.ones_like(old), torch.ones_like(old)
    actual, _ = fn(old, current, advantage, mask, config=config)
    expected, _ = cm_gspo_loss(old, current, advantage, mask, rollout_count=128, dp_size=8)
    torch.testing.assert_close(actual, expected)
    assert actual.item() == pytest.approx(-2*8/128)
    with pytest.raises(ValueError, match='importance weights'):
        fn(old, current, advantage, mask, config=config, rollout_is_weights=mask)
    config.clip_ratio_high = .0004
    with pytest.raises(ValueError, match='symmetric'):
        fn(old, current, advantage, mask, config=config)
    config.clip_ratio_high = .0003
    config.global_batch_info = {}
    with pytest.raises(ValueError, match='denominator'):
        fn(old, current, advantage, mask, config=config)
