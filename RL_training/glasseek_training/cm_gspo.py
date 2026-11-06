"""Paper equations (3)-(6): recorded contexts, rollout advantages, segment weights.

The optimizer uses the registered CM-GSPO loss on each segment, with global_batch_size
set to the number of ORIGINAL rollouts (not the number of segment rows).
"""
from dataclasses import dataclass, field
import math
from typing import Any

import torch


@dataclass
class Segment:
    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)

    @property
    def length(self) -> int:
        return sum(self.response_mask)

    def validate(self) -> None:
        if not self.prompt_ids or not self.response_ids or self.length <= 0:
            raise ValueError("An execution segment needs a context and at least one execution token")
        if len({len(self.response_ids), len(self.response_mask), len(self.log_probs)}) != 1:
            raise ValueError("Response IDs, masks and behavior log probabilities must align")
        if any(m not in (0, 1) for m in self.response_mask):
            raise ValueError("Execution mask must be binary")
        if any(not math.isfinite(p) for p in self.log_probs):
            raise ValueError("Non-finite behavior log probability")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return dict(prompt_ids=self.prompt_ids, response_ids=self.response_ids,
                    response_mask=self.response_mask, log_probs=self.log_probs)


@dataclass
class Rollout:
    prompt_id: str
    reward: float
    segments: list[Segment]


def group_advantages(rollouts: list[Rollout], group_size: int, epsilon: float = 1e-6) -> torch.Tensor:
    """Normalize terminal rewards once per prompt, BEFORE expanding segments.

    Use sample std (correction=1), matching verl's GRPO convention. Constant
    reward groups produce exactly zero, including all-failure groups.
    """
    if group_size < 2 or not rollouts:
        raise ValueError("CM-GSPO requires nonempty groups with at least two rollouts")
    groups: dict[str, list[int]] = {}
    for i, rollout in enumerate(rollouts):
        if not math.isfinite(rollout.reward) or not rollout.segments:
            raise ValueError("Each original rollout needs a finite reward and execution segments")
        groups.setdefault(rollout.prompt_id, []).append(i)
    rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32)
    result = torch.zeros_like(rewards)
    for indices in groups.values():
        if len(indices) != group_size:
            raise ValueError(f"Expected {group_size} rollouts per prompt, got {len(indices)}")
        values = rewards[indices]
        result[indices] = (values - values.mean()) / (values.std(correction=1) + epsilon)
    return result


def pack_rollouts(rollouts: list[Rollout], *, group_size: int, pad_token_id: int,
                  batch_multiple: int = 1, max_model_len: int = 131072) -> tuple[dict, dict]:
    """Pack one row per conditioned segment, padding only for distributed dispatch.

    Duplicate rows used for dispatch padding have ZERO execution mask and advantage. They are not
    new rollouts and never change the objective's global denominator.
    """
    if batch_multiple < 1:
        raise ValueError("batch_multiple must be positive")
    advantages = group_advantages(rollouts, group_size)
    rows = []
    for i, rollout in enumerate(rollouts):
        for segment in rollout.segments:
            segment.validate()
            if len(segment.prompt_ids) + len(segment.response_ids) > max_model_len:
                raise ValueError("Recorded segment exceeds model context; never truncate training contexts")
        total = sum(s.length for s in rollout.segments)
        for segment in rollout.segments:
            rows.append((segment, advantages[i].item() * segment.length / total))
    real_rows = len(rows)
    while len(rows) % batch_multiple:
        rows.append((rows[0][0], 0.0))
    p = max(len(s.prompt_ids) for s, _ in rows)
    r = max(len(s.response_ids) for s, _ in rows)
    n = len(rows)
    prompts = torch.full((n, p), pad_token_id, dtype=torch.long)
    responses = torch.full((n, r), pad_token_id, dtype=torch.long)
    attention = torch.zeros((n, p + r), dtype=torch.long)
    mask = torch.zeros((n, r), dtype=torch.long)
    old = torch.zeros((n, r), dtype=torch.float32)
    adv = torch.zeros_like(old)
    for i, (segment, value) in enumerate(rows):
        plen, rlen = len(segment.prompt_ids), len(segment.response_ids)
        prompts[i, -plen:] = torch.tensor(segment.prompt_ids)
        responses[i, :rlen] = torch.tensor(segment.response_ids)
        attention[i, p - plen:p + rlen] = 1
        if i < real_rows:
            mask[i, :rlen] = torch.tensor(segment.response_mask)
            old[i, :rlen] = torch.tensor(segment.log_probs)
        adv[i, :rlen] = value * mask[i, :rlen]
    positions = (attention.cumsum(-1) - 1).clamp_min(0) * attention
    tensors = dict(prompts=prompts, responses=responses, input_ids=torch.cat((prompts, responses), -1),
                   attention_mask=attention, position_ids=positions, response_mask=mask,
                   old_log_probs=old, advantages=adv)
    return tensors, dict(rollout_count=len(rollouts), segment_count=real_rows, padded_rows=n)
