"""Explicit curriculum policies; validation and sampling remain policy-specific."""
import hashlib
import json
import os
from pathlib import Path

from .length_curriculum import LengthCurriculumSampler, load_training_lengths
from .turn_sampler import TurnSampler


def build_sampler(policy, dataset, global_batch, world, rank, sp, train_path, audit=None):
    if policy == 'length':
        if world != 1 or sp != 1:
            raise ValueError('Length curriculum supports single-GPU, SP=1 training only')
        if not isinstance(train_path, str):
            raise ValueError('Length curriculum expects one training parquet')
        lengths = load_training_lengths(audit or os.environ['SFT_LENGTH_AUDIT'], train_path, len(dataset))
        return LengthCurriculumSampler(dataset, lengths, global_batch, seed=0)
    if policy != 'turns':
        raise ValueError(f'Unknown curriculum policy: {policy}')
    if sp != 1:
        raise ValueError('Turn curriculum requires SP=1')
    path = Path(train_path)
    manifest = json.loads((path.parent / 'manifest.json').read_text())
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != manifest['output_sha256']['train']:
        raise ValueError('Training parquet does not match manifest')
    if manifest['max_length'] != 65536 or manifest['counts']['train'] != len(dataset):
        raise ValueError('Unexpected 64K training dataset')
    return TurnSampler(dataset.messages, global_batch, world, rank)
