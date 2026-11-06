"""Exact execution-policy surrogate in Glasseek equations (3)-(6).

A row is one conditioned sub-trajectory. Its advantage already includes
L_segment / L_rollout. `rollout_count` is global across DATA-parallel ranks.
No additional microbatch mean, token-global mean, or ratio cap is applied.
"""
import math
import torch


def subtrajectory_token_ratio(log_prob, old_log_prob, response_mask):
    """Eq.(3) in value, Eq.(4) in gradient; masked likelihoods never enter either.

    Mask BEFORE arithmetic: multiplying NaN/Inf by zero is not masking. Old
    likelihoods and the sequence ratio are fixed in the backward pass.
    """
    if log_prob.ndim != 2 or log_prob.shape != old_log_prob.shape or log_prob.shape != response_mask.shape:
        raise ValueError("Expected aligned [segment, token] likelihoods and mask")
    mask = response_mask.bool()
    # Keep float64 inputs for numerical tests; upgrade low-precision model output.
    dtype = torch.promote_types(log_prob.dtype, torch.float32)
    current = torch.where(mask, log_prob.to(dtype), 0.0)
    old = torch.where(mask, old_log_prob.detach().to(dtype), 0.0)
    if not torch.isfinite(current).all() or not torch.isfinite(old).all():
        raise FloatingPointError("Non-finite execution-token likelihood")
    length = mask.sum(-1).clamp_min(1)
    mean_log_ratio = (current - old).sum(-1) / length
    log_ratio = mean_log_ratio.detach().unsqueeze(-1) + (current - current.detach())
    # No max=10 cap: it would zero the negative-advantage penalty gradient.
    ratio = torch.exp(log_ratio)
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("CM-GSPO ratio overflow; refusing to silently change the objective")
    return ratio, mean_log_ratio.detach()


def cm_gspo_loss(old_log_prob, log_prob, advantages, response_mask, *,
                 rollout_count: int, dp_size: int = 1, clip_epsilon: float = 0.0003):
    """Return -Eq.(6), scaled by DP size to compensate for gradient averaging.

    Each rank sums its microbatch losses. DDP/FSDP averages these rank gradients,
    yielding a single global division by P*G original rollouts for P prompts.
    Fully masked dispatch-padding rows contribute exactly zero, even alone in a
    microbatch. Summaries may influence execution as context; their own action
    likelihoods are absent from this surrogate.
    """
    if rollout_count <= 0 or dp_size <= 0:
        raise ValueError("Need positive global rollout_count and data-parallel size")
    if not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be finite and between zero and one")
    if advantages.shape != log_prob.shape:
        raise ValueError("Advantages must align with likelihoods")
    ratio, mean_log_ratio = subtrajectory_token_ratio(log_prob, old_log_prob, response_mask)
    mask = response_mask.bool()
    advantage = torch.where(mask, advantages.detach().to(ratio.dtype), 0.0)
    if not torch.isfinite(advantage).all():
        raise FloatingPointError("Non-finite execution-token advantage")
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage
    per_token_loss = -torch.minimum(unclipped, clipped)
    lengths = mask.sum(-1).clamp_min(1)
    per_segment_loss = torch.where(mask, per_token_loss, 0.0).sum(-1) / lengths
    loss = per_segment_loss.sum() * (dp_size / rollout_count)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite CM-GSPO loss")
    with torch.no_grad():
        count = mask.sum().clamp_min(1)
        clipfrac = ((clipped < unclipped) & mask).sum() / count
        kl = -(mean_log_ratio * mask.sum(-1)).sum() / count
    return loss, {"actor/pg_clipfrac": clipfrac.item(), "actor/ppo_kl": kl.item()}
