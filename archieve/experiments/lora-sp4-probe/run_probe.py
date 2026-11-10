"""Run unmodified verl SFT update, suppress checkpoint writes, report GPU peaks."""
import os, time
import torch
from verl.trainer import sft_trainer
from verl.workers.engine_workers import TrainingWorker
from verl.utils.checkpoint import CheckpointHandler

CheckpointHandler.save_checkpoint = lambda self, *args, **kwargs: None
original_train_batch = TrainingWorker.train_batch

def measured(self, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.monotonic()
    out = original_train_batch(self, *args, **kwargs)
    torch.cuda.synchronize()
    print('PROBE_METRICS', {'rank': torch.distributed.get_rank(), 'seconds': round(time.monotonic()-t0, 3),
          'max_allocated_gib': round(torch.cuda.max_memory_allocated()/2**30, 3),
          'max_reserved_gib': round(torch.cuda.max_memory_reserved()/2**30, 3)}, flush=True)
    return out
TrainingWorker.train_batch = measured
sft_trainer.main()
