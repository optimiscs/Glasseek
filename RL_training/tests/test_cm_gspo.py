import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch

from glasseek_training.cm_gspo import Rollout, Segment, group_advantages, pack_rollouts


def segment(n, context=10, tool=False):
    return Segment([context], list(range(20, 20 + n)) + ([99] if tool else []),
                   [1] * n + ([0] if tool else []), [-2.0] * n + ([0.] if tool else []))


def test_weights_contexts_and_padding():
    rollouts = [Rollout("q", 1., [segment(1, 10, True), segment(3, 88)]),
                Rollout("q", 0., [segment(2, 11)])]
    data, info = pack_rollouts(rollouts, group_size=2, pad_token_id=0, batch_multiple=4)
    a = group_advantages(rollouts, 2)
    assert info == dict(rollout_count=2, segment_count=3, padded_rows=4)
    assert data["prompts"].flatten().tolist() == [10, 88, 11, 10]
    assert data["advantages"][0, 0] == pytest.approx(a[0].item() / 4)
    assert data["advantages"][1, 0] == pytest.approx(a[0].item() * 3 / 4)
    assert data["advantages"][2, 0] == a[1]
    assert data["advantages"][3].eq(0).all()
    assert data["response_mask"][0].tolist() == [1, 0, 0]
    assert data["attention_mask"][0].tolist() == [1, 1, 1, 0]
    assert data["position_ids"][0].tolist() == [0, 1, 2, 0]


def test_constant_and_incomplete_groups():
    rows = [Rollout("q", 1., [segment(2)]), Rollout("q", 1., [segment(9)])]
    assert group_advantages(rows, 2).eq(0).all()
    with pytest.raises(ValueError, match="Expected 8"):
        group_advantages(rows, 8)


def test_no_truncation_or_empty_segments():
    with pytest.raises(ValueError, match="exceeds model context"):
        pack_rollouts([Rollout("q", 1., [segment(4)]), Rollout("q", 0., [segment(1)])],
                      group_size=2, pad_token_id=0, max_model_len=4)
    with pytest.raises(ValueError, match="execution token"):
        Segment([1], [2], [0], [0.]).validate()


@pytest.fixture
def upstream_gspo():
    """Run the actual pinned upstream loss without importing GPU/Ray dependencies.

    Only extract the two pure numerical functions; do NOT copy/reimplement GSPO
    in this recipe. Set VERL_SOURCE to a checkout of the pinned commit.
    """
    root = os.environ.get("VERL_SOURCE")
    if not root:
        pytest.skip("Set VERL_SOURCE for upstream loss compatibility tests")
    tree = ast.parse((Path(root) / "verl/trainer/ppo/core_algos.py").read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("agg_loss", "compute_policy_loss_gspo"):
            node.decorator_list = []
            selected.append(node)
    class ActorConfig(SimpleNamespace):
        pass
    namespace = dict(torch=torch, Optional=Optional, Any=Any, ActorConfig=ActorConfig,
                     verl_F=SimpleNamespace(masked_sum=lambda v, m: (v*m).sum(),
                                            masked_mean=lambda v, m: (v*m).sum()/m.sum()))
    exec(compile(ast.Module(body=selected, type_ignores=[]), "upstream_gspo", "exec"), namespace)
    def loss(data, current, dp_size=1):
        config = ActorConfig(clip_ratio_low=.0003, clip_ratio_high=.0004, clip_ratio=.2,
                             global_batch_info={"global_batch_size": 2, "dp_size": dp_size})
        return namespace["compute_policy_loss_gspo"](
            data["old_log_probs"], current, data["advantages"], data["response_mask"], config=config)[0]
    return loss


def test_upstream_gradient_token_weights_and_microbatch_invariance(upstream_gspo):
    rows = [Rollout("q", 1., [segment(1, tool=True), segment(3)]),
            Rollout("q", 0., [segment(2)])]
    data, _ = pack_rollouts(rows, group_size=2, pad_token_id=0, batch_multiple=4)
    current = data["old_log_probs"].clone().requires_grad_()
    loss = upstream_gspo(data, current)
    loss.backward()
    a = group_advantages(rows, 2)
    # Eq.(4) token-local gradient, Eq.(6) normalization: -A_i / (G * L_i).
    expected = torch.zeros_like(current)
    expected[0, 0] = -a[0] / 8
    expected[1, :3] = -a[0] / 8
    expected[2, :2] = -a[1] / 4
    torch.testing.assert_close(current.grad, expected)
    # Simulate two ranks, multiple microbatches, then DDP averaging.
    split_current = data["old_log_probs"].clone().requires_grad_()
    distributed = sum(upstream_gspo({k: v[i:i+1] for k, v in data.items()},
                                    split_current[i:i+1], dp_size=2) for i in range(4)) / 2
    distributed.backward()
    torch.testing.assert_close(distributed, loss)
    torch.testing.assert_close(split_current.grad, current.grad)


@pytest.mark.parametrize("positive,shift", [(True, .01), (False, -.01)])
def test_upstream_clipping_blocks_improving_outside_bound(upstream_gspo, positive, shift):
    rows = [Rollout("q", 1., [segment(2)]), Rollout("q", 0., [segment(2)])]
    data, _ = pack_rollouts(rows, group_size=2, pad_token_id=0)
    current = data["old_log_probs"].clone()
    index = 0 if positive else 1
    current[index] += shift
    current.requires_grad_()
    upstream_gspo(data, current).backward()
    assert current.grad[index].eq(0).all()
    assert current.grad[1-index].abs().sum() > 0


def test_upstream_cap_changes_negative_advantage_gradient(upstream_gspo):
    """Regression: upstream GSPO is not exactly Eq.(4) for log-ratio > 10."""
    from glasseek_training.loss import cm_gspo_loss
    rows = [Rollout("q", 1., [segment(2)]), Rollout("q", 0., [segment(2)])]
    data, _ = pack_rollouts(rows, group_size=2, pad_token_id=0)
    old = data['old_log_probs'] - 15  # old=-17, current=-6: both valid log probabilities
    current = (old + 11).requires_grad_()
    data['old_log_probs'] = old
    upstream_gspo(data, current).backward()
    assert current.grad[1].eq(0).all()  # the problematic cap
    current.grad = None
    loss, _ = cm_gspo_loss(old, current, data['advantages'], data['response_mask'], rollout_count=2)
    loss.backward()
    expected = -data['advantages'][1, 0] * torch.exp(torch.tensor(11.)) / 4
    torch.testing.assert_close(current.grad[1], expected.expand(2))
    assert current.grad[0].eq(0).all()  # positive advantage is legitimately clipped
