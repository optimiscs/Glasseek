"""Evaluate saved curriculum checkpoints with the trainer's validation path."""
import os
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from verl.trainer import sft_trainer
from verl.utils import tensordict_utils as tu

from glasseek_training.curriculum_sft import CurriculumSFTTrainer


class HistoricalValidationTrainer(CurriculumSFTTrainer):
    def fit(self):
        if self.rank != 0 or self.engine.get_data_parallel_size() != 1:
            raise RuntimeError('Historical validation supports only one GPU')
        if self.val_dataloader is None:
            raise RuntimeError('No validation data configured')
        checkpoints = Path(self.config.trainer.default_local_dir)
        steps = [int(s) for s in os.environ['SFT_VALIDATION_STEPS'].split(',')]
        output = Path(os.environ['SFT_VALIDATION_TENSORBOARD_DIR'])
        meta = {
            'use_remove_padding': self.config.model.use_remove_padding,
            'use_dynamic_bsz': self.config.data.use_dynamic_bsz,
            'max_token_len_per_gpu': self.config.data.max_token_len_per_gpu,
            'micro_batch_size_per_gpu': self.config.data.micro_batch_size_per_gpu,
            'temperature': 1.0,
            'global_batch_size': self.global_batch_size,
            'pad_mode': self.config.data.pad_mode,
            'pad_token_id': self.model_config.tokenizer.pad_token_id,
        }
        with SummaryWriter(log_dir=str(output)) as writer:
            for step in steps:
                print(f'Historical validation: loading step {step}', flush=True)
                self.engine.load_checkpoint(str(checkpoints / f'global_step_{step}'))
                losses = []
                for batch in self.val_dataloader:
                    batch = tu.get_tensordict(tensor_dict=batch, non_tensor_dict=meta)
                    result = self.training_client.infer_batch(batch)
                    losses.append(tu.get(result, 'metrics')['loss'])
                if not losses:
                    raise RuntimeError(f'Validation loader returned no batches at step {step}')
                loss = torch.mean(torch.tensor(losses, device=self.device_name)).item()
                if not torch.isfinite(torch.tensor(loss)):
                    raise ValueError(f'Non-finite validation loss at step {step}: {loss}')
                writer.add_scalar('val/loss', loss, step)
                writer.flush()
                print(f'Historical validation complete: step={step} val/loss={loss} batches={len(losses)}', flush=True)


if __name__ == '__main__':
    sft_trainer.SFTTrainer = HistoricalValidationTrainer
    sft_trainer.main()
