"""Deterministic length-ascending SFT sampling for a single GPU."""
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DistributedSampler


def load_training_lengths(audit_path, parquet_path, dataset_size):
    """Match the saved token-length audit to its verified training parquet."""
    audit_path, parquet_path = Path(audit_path), Path(parquet_path)
    manifest = json.loads((audit_path.parent / 'manifest.json').read_text())
    digest = hashlib.sha256()
    with parquet_path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != manifest['output_sha256']['train']:
        raise ValueError('Training parquet does not match the length audit manifest')
    records = [json.loads(line) for line in audit_path.open()]
    train = [row for row in records if row['split'] == 'train']
    if len(train) != dataset_size or len(train) != manifest['counts']['train']:
        raise ValueError('Training length count does not match the dataset')
    if any(not row['exact'] or row['tokens'] <= 0 or row['tokens'] > manifest['max_length'] for row in train):
        raise ValueError('Training length audit contains an invalid length')
    if any(a['source_line'] >= b['source_line'] for a, b in zip(train, train[1:])):
        raise ValueError('Training length audit is not in source order')
    return [row['tokens'] for row in train]


class LengthCurriculumSampler(DistributedSampler):
    """Sort selected trajectories by length every epoch; randomly omit the tail batch."""

    def __init__(self, dataset, lengths, batch_size, seed=0):
        if len(dataset) != len(lengths):
            raise ValueError('Expected one length for each training trajectory')
        if batch_size <= 0:
            raise ValueError('Batch size must be positive')
        super().__init__(dataset, num_replicas=1, rank=0, shuffle=True, seed=seed, drop_last=True)
        self.lengths = tuple(lengths)
        self.num_samples = len(dataset) // batch_size * batch_size
        self.total_size = self.num_samples
        if self.num_samples == 0:
            raise ValueError('Training dataset has no complete batch')

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        candidates = torch.randperm(len(self.dataset), generator=generator).tolist()[:self.num_samples]
        # Stable sorting retains the randomized order of equal-length trajectories.
        candidates.sort(key=self.lengths.__getitem__)
        return iter(candidates)


class DistributedLengthCurriculumSampler(LengthCurriculumSampler):
    """Partition each original global batch while preserving its membership.

    start_batch applies only to the initial epoch after a checkpoint; the caller
    must clear it before advancing to the next epoch.
    """

    def __init__(self, dataset, lengths, batch_size, world_size, rank, seed=0, start_batch=0):
        super().__init__(dataset, lengths, batch_size, seed=seed)
        if world_size <= 0 or not 0 <= rank < world_size or batch_size % world_size:
            raise ValueError('Global batch size must divide evenly among ranks')
        if not 0 <= start_batch <= self.num_samples // batch_size:
            raise ValueError('Invalid resume batch')
        self.global_batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.start_batch = start_batch
        self.local_batch_size = batch_size // world_size

    def __iter__(self):
        indices = list(super().__iter__())
        result = []
        for start in range(self.start_batch * self.global_batch_size, len(indices), self.global_batch_size):
            offset = start + self.rank * self.local_batch_size
            result.extend(indices[offset:offset + self.local_batch_size])
        return iter(result)

    def __len__(self):
        return (self.num_samples // self.global_batch_size - self.start_batch) * self.local_batch_size
