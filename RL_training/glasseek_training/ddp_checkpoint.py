"""Utilities for inspecting world-size-one FSDP LoRA checkpoints.

This module deliberately does not guess a mapping from optimizer indices to
parameter names. Callers must obtain and compare names from both model layouts.
"""
import copy

import torch
from torch.distributed.tensor import DTensor


def materialize_single_rank(value):
    if isinstance(value, DTensor):
        if value.device_mesh.mesh.numel() != 1:
            raise ValueError('Only world-size-one DTensors can be materialized locally')
        local = value.to_local().detach().cpu().clone()
        if tuple(local.shape) != tuple(value.shape):
            raise ValueError('Checkpoint tensor is not fully present on its only rank')
        return local
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: materialize_single_rank(item) for key, item in value.items()}
    if isinstance(value, list):
        return [materialize_single_rank(item) for item in value]
    if isinstance(value, tuple):
        return tuple(materialize_single_rank(item) for item in value)
    return copy.deepcopy(value)


def checked_optimizer_state(state, source_names, target_named_parameters):
    """Convert state only after verifying exact parameter identity and shapes."""
    target = list(target_named_parameters)
    target_names = [name for name, _ in target]
    if list(source_names) != target_names:
        raise ValueError('Optimizer parameter names or order differ; refusing positional restore')
    groups = state['param_groups']
    if len(groups) != 1 or len(groups[0]['params']) != len(target):
        raise ValueError('Optimizer parameter groups differ')
    converted = materialize_single_rank(state)
    for index, (name, parameter) in zip(groups[0]['params'], target):
        for key, value in converted['state'].get(index, {}).items():
            if key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq') and value.shape != parameter.shape:
                raise ValueError(f'Optimizer tensor shape mismatch at {name}: {key}')
    return converted
