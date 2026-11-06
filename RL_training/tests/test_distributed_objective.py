"""Real two-process Gloo/DDP gradient averaging, independent of GPU availability."""
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from glasseek_training.loss import cm_gspo_loss
from test_paper_objective import make_batch


class TokenLogProbs(torch.nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.log_probs = torch.nn.Parameter(initial.clone())

    def forward(self, indices):
        return self.log_probs[indices]


def ddp_rank(rank, init_file, output_file):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init_file,
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        _, data, info = make_batch()
        model = DistributedDataParallel(TokenLogProbs(data['old_log_probs']))
        # Rank 0: three segments of one rollout + another rollout.
        # Rank 1: two other rollouts + a fully masked dummy.
        rank_indices = list(range(rank*4, (rank+1)*4))
        total_loss = torch.tensor(0.)
        for micro in [rank_indices[:2], rank_indices[2:]]:
            current = model(micro)
            loss, _ = cm_gspo_loss(data['old_log_probs'][micro], current,
                                   data['advantages'][micro], data['response_mask'][micro],
                                   rollout_count=info['rollout_count'], dp_size=2)
            loss.backward()  # actual DDP reduction on each microbatch
            total_loss += loss.detach()
        dist.all_reduce(total_loss)
        total_loss /= 2
        if rank == 0:
            torch.save({'grad': model.module.log_probs.grad, 'loss': total_loss}, output_file)
    finally:
        dist.destroy_process_group()


def test_two_process_ddp_matches_single_process_objective(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip('PyTorch build lacks Gloo')
    init_file, output_file = str(tmp_path/'gloo_init'), str(tmp_path/'result.pt')
    mp.spawn(ddp_rank, args=(init_file, output_file), nprocs=2, join=True)
    actual = torch.load(output_file, weights_only=True)
    _, data, info = make_batch()
    current = data['old_log_probs'].clone().requires_grad_()
    expected, _ = cm_gspo_loss(data['old_log_probs'], current, data['advantages'], data['response_mask'],
                              rollout_count=info['rollout_count'])
    expected.backward()
    torch.testing.assert_close(actual['grad'], current.grad)
    torch.testing.assert_close(actual['loss'], expected, atol=1e-7, rtol=1e-6)
