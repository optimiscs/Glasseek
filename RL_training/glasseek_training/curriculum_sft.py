"""Launch verl SFT with deterministic length-ascending training epochs."""
import os
from pathlib import Path

from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer import sft_trainer
from verl.utils.device import get_device_name

from glasseek_training.length_curriculum import LengthCurriculumSampler, load_training_lengths


class CurriculumSFTTrainer(sft_trainer.SFTTrainer):
    def _build_dataloader(self):
        super()._build_dataloader()
        if self.engine.get_data_parallel_size() != 1 or self.engine.ulysses_sequence_parallel_size != 1:
            raise ValueError('This curriculum launcher supports single-GPU, SP=1 training only')
        audit = Path(os.environ['SFT_LENGTH_AUDIT'])
        train_path = self.config.data.train_files
        if not isinstance(train_path, str):
            raise ValueError('The curriculum launcher expects one training parquet')
        lengths = load_training_lengths(audit, train_path, len(self.train_dataset))
        self.train_sampler = LengthCurriculumSampler(self.train_dataset, lengths, self.global_batch_size, seed=0)
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.data.num_workers,
            pin_memory=False,
            drop_last=True,
            pin_memory_device=get_device_name(),
        )
        print(f'Length curriculum: {len(lengths)} trajectories; {len(self.train_sampler)} per epoch, '
              f'ascending token length, seed={self.train_sampler.seed}', flush=True)


if __name__ == '__main__':
    sft_trainer.SFTTrainer = CurriculumSFTTrainer
    sft_trainer.main()
