"""Paper-derived analytic oracles; no dependency on verl's loss implementation."""
import math

import pytest
import torch

from glasseek_training.cm_gspo import Rollout, Segment, group_advantages, pack_rollouts
from glasseek_training.loss import cm_gspo_loss, subtrajectory_token_ratio


def make_batch():
    def seg(n, context):
        # Observations occur between generated tokens, not just as a trailing pad.
        return Segment([context, context + 1], [50] + list(range(60, 60+n)),
                       [0] + [1]*n, [0.] + [-3.]*n)
    rows = [Rollout('a', 1., [seg(1, 1), seg(3, 8), seg(2, 10)]),
            Rollout('a', 0., [seg(4, 20)]),
            Rollout('b', 0., [seg(2, 30), seg(1, 40)]),
            Rollout('b', 1., [seg(3, 50)])]
    data, info = pack_rollouts(rows, group_size=2, pad_token_id=0, batch_multiple=8)
    return rows, data, info


def test_ratio_value_and_token_local_jacobian():
    old = torch.tensor([[-3., -5., -2.]], dtype=torch.float64, requires_grad=True)
    current = torch.tensor([[-2.9, 123., -2.3]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 0, 1]])
    ratio, _ = subtrajectory_token_ratio(current, old, mask)
    expected = math.exp(-.1)
    torch.testing.assert_close(ratio, torch.full_like(ratio, expected))
    jac = torch.autograd.functional.jacobian(
        lambda x: subtrajectory_token_ratio(x, old, mask)[0], current).reshape(3, 3)
    torch.testing.assert_close(jac, torch.diag(torch.tensor([expected, 0., expected], dtype=torch.float64)))
    ratio.sum().backward()
    assert old.grad is None  # behavior policy is held fixed


def test_off_policy_value_and_gradient_against_equations_3_to_6():
    rows, data, info = make_batch()
    # Segment ratios straddle both clipping edges, including non-clipped
    # worsening updates for positive and negative advantages.
    shifts = torch.tensor([.0001, .001, -.01, .01, -.001, .0001, -.002, 0.], dtype=torch.float64)
    current = data['old_log_probs'].double() + shifts[:, None]
    current.requires_grad_()
    loss, _ = cm_gspo_loss(data['old_log_probs'], current, data['advantages'], data['response_mask'],
                           rollout_count=info['rollout_count'])
    expected_value = 0.
    expected_grad = torch.zeros_like(current)
    advantage = group_advantages(rows, 2)
    index = 0
    for i, rollout in enumerate(rows):
        total_length = sum(s.length for s in rollout.segments)
        for s in rollout.segments:
            a_cm = float(advantage[i]) * s.length / total_length
            ratio = math.exp(float(shifts[index]))
            clipped = min(max(ratio, 1-.0003), 1+.0003)
            expected_value -= min(ratio*a_cm, clipped*a_cm)/len(rows)
            active = (a_cm > 0 and ratio < 1+.0003) or (a_cm < 0 and ratio > 1-.0003)
            if active:
                expected_grad[index, data['response_mask'][index].bool()] = (
                    -float(advantage[i])*ratio/(len(rows)*total_length))
            index += 1
    assert loss.item() == pytest.approx(expected_value, abs=2e-8)
    loss.backward()
    torch.testing.assert_close(current.grad, expected_grad, atol=1e-8, rtol=1e-6)


def test_masked_nonfinite_logits_and_padding_do_not_enter_loss_or_lengths():
    _, data, info = make_batch()
    mask = data['response_mask'].bool()
    old = data['old_log_probs'].clone()
    current = old.clone()
    reference, _ = cm_gspo_loss(old, current, data['advantages'], mask, rollout_count=4)
    current[~mask] = float('nan')
    old[~mask] = float('inf')
    current.requires_grad_()
    loss, metrics = cm_gspo_loss(old, current, data['advantages'], mask, rollout_count=4)
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert torch.isfinite(current.grad).all()
    assert current.grad[~mask].eq(0).all()
    assert not mask[-1].any()  # dispatch dummy is genuinely fully masked
    assert all(math.isfinite(v) for v in metrics.values())
    fake_loss, fake_metrics = cm_gspo_loss(old[-1:], current[-1:], data['advantages'][-1:], mask[-1:],
                                         rollout_count=info['rollout_count'], dp_size=2)
    assert fake_loss.item() == 0
    assert all(math.isfinite(v) for v in fake_metrics.values())


def test_nonfinite_execution_likelihoods_fail_instead_of_capping():
    with pytest.raises(FloatingPointError, match='likelihood'):
        cm_gspo_loss(torch.zeros(1, 1), torch.full((1, 1), float('nan')),
                     torch.ones(1, 1), torch.ones(1, 1), rollout_count=1)
    with pytest.raises(FloatingPointError, match='overflow'):
        cm_gspo_loss(torch.full((1, 1), -1000.), torch.full((1, 1), -1.),
                     -torch.ones(1, 1), torch.ones(1, 1), rollout_count=1)


def test_permutation_and_arbitrary_microbatch_sums_do_not_change_gradients():
    _, data, info = make_batch()
    current = data['old_log_probs'].clone().requires_grad_()
    args = (data['old_log_probs'], current, data['advantages'], data['response_mask'])
    reference, _ = cm_gspo_loss(*args, rollout_count=info['rollout_count'])
    reference.backward()
    expected_grad = current.grad.clone()
    current.grad = None
    # No division by microbatch count; shards contain unequal segment lengths.
    pieces = [[6], [3, 7, 0], [2, 4], [5, 1]]
    result = sum(cm_gspo_loss(*(x[indices] for x in args), rollout_count=4, dp_size=2)[0]
                 for indices in pieces)/2
    result.backward()
    torch.testing.assert_close(result, reference)
    torch.testing.assert_close(current.grad, expected_grad)


def test_fixed_summary_context_may_still_influence_execution_gradient():
    # Masking summary ACTION likelihoods must not detach their context embeddings.
    # A tiny causal policy conditions execution logits on one recorded summary ID.
    context_embedding = torch.tensor([.2, -.1, .3], requires_grad=True)
    summary_action_logprob = torch.tensor([[-2.]], requires_grad=True)
    execution_logprob = context_embedding.log_softmax(0)[1].reshape(1, 1)
    old = execution_logprob.detach().clone()
    loss, _ = cm_gspo_loss(old, execution_logprob, torch.ones(1, 1), torch.ones(1, 1), rollout_count=1)
    loss.backward()
    assert summary_action_logprob.grad is None
    assert context_embedding.grad.abs().sum() > 0


def test_detach_factor_does_not_numerically_cancel_the_segment_ratio():
    current = torch.full((1, 2), -100000000., requires_grad=True)
    old = current.detach().clone()
    old[0, 0] -= 8  # fp32-representable increment; mean log-ratio = 4
    ratio, mean = subtrajectory_token_ratio(current, old, torch.ones_like(old))
    torch.testing.assert_close(mean, torch.tensor([4.]))
    torch.testing.assert_close(ratio, torch.full_like(current, math.exp(4.)))
