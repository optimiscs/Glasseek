"""Instrument the verl training worker without checkpoint I/O."""
import os, time, torch
from pathlib import Path
from verl.trainer import sft_trainer
from verl.workers.engine_workers import TrainingWorker
from verl.utils.checkpoint import CheckpointHandler
CheckpointHandler.save_checkpoint = lambda self, *a, **kw: None
if os.environ.get('GRAD_CAPTURE_DIR'):
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
    import verl.workers.engine.fsdp.transformer_impl as fsdp_impl
    original_get_peft_model = fsdp_impl.get_peft_model
    def seeded_get_peft_model(*args, **kwargs):
        torch.manual_seed(8675309)
        torch.cuda.manual_seed_all(8675309)
        return original_get_peft_model(*args, **kwargs)
    fsdp_impl.get_peft_model = seeded_get_peft_model
    original_step = FSDPEngine.optimizer_step
    def save_gradients(self):
        if not getattr(self, '_bench_saved', False):
            path = Path(os.environ['GRAD_CAPTURE_DIR'])
            path.mkdir(parents=True, exist_ok=True)
            gradients = {}
            weights = {}
            for name, param in self.module.named_parameters():
                if param.grad is not None:
                    grad = param.grad.detach()
                    if hasattr(grad, 'to_local'):
                        grad = grad.to_local()
                    gradients[name] = grad.float().cpu().clone()
                    weight = param.detach()
                    if hasattr(weight, 'to_local'):
                        weight = weight.to_local()
                    weights[name] = weight.cpu().clone()
            torch.save(gradients, path / f"rank-{torch.distributed.get_rank()}.pt")
            torch.save(weights, path / f"weights-rank-{torch.distributed.get_rank()}.pt")
            self._bench_saved = True
        return original_step(self)
    FSDPEngine.optimizer_step = save_gradients
original = TrainingWorker.train_batch

def measured(self, data, *args, **kwargs):
    try:
        mask = data['loss_mask']
        if mask.is_nested:
            mask = mask.values()
        print('SUPERVISED_TOKENS', {'rank': torch.distributed.get_rank(), 'count': int(mask.sum().item())}, flush=True)
    except Exception as exc:
        print('SUPERVISED_TOKENS_UNAVAILABLE', repr(exc), flush=True)
    torch.cuda.synchronize()
    start=time.monotonic()
    result=original(self, data, *args, **kwargs)
    torch.cuda.synchronize()
    print('PROBE_METRICS', {'rank': torch.distributed.get_rank(), 'seconds': round(time.monotonic()-start, 3), 'max_allocated_gib': round(torch.cuda.max_memory_allocated()/2**30, 3), 'max_reserved_gib': round(torch.cuda.max_memory_reserved()/2**30, 3)}, flush=True)
    return result
TrainingWorker.train_batch=measured
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
sft_trainer.main()
