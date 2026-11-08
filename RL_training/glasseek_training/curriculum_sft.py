"""One curriculum trainer, with explicit length or turn-order sampling policies."""
import os
from pathlib import Path

from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer import sft_trainer
from verl.utils.device import get_device_name

from glasseek_training.curriculum_policy import build_sampler


class CurriculumSFTTrainer(sft_trainer.SFTTrainer):
    def _build_dataloader(self):
        super()._build_dataloader()
        policy = os.environ.get('SFT_CURRICULUM_POLICY', 'length')
        self.train_sampler = build_sampler(policy, self.train_dataset, self.global_batch_size,
            self.engine.get_data_parallel_size(), self.engine.get_data_parallel_rank(),
            self.engine.ulysses_sequence_parallel_size, self.config.data.train_files)
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
        print(f'Curriculum policy={policy}: trajectories={len(self.train_dataset)}, '
              f'local samples={len(self.train_sampler)}', flush=True)


if __name__ == '__main__':
    if os.environ.get('SFT_CURRICULUM_POLICY') == 'turns' and os.environ.get('TURN_AUDIT_DIR'):
        import json
        from glasseek_training.sft_dataset import GlasseekSFTDataset
        original_getitem = GlasseekSFTDataset.__getitem__
        def audited_getitem(self, index):
            result = original_getitem(self, index)
            name = os.environ.get('TURN_AUDIT_PHASE', 'run') + '-rank' + os.environ.get('RANK', 'unknown') + '.jsonl'
            with open(os.path.join(os.environ['TURN_AUDIT_DIR'], name), 'a') as output:
                output.write(json.dumps({'index': int(index)}) + '\n')
            return result
        GlasseekSFTDataset.__getitem__ = audited_getitem
    sft_trainer.SFTTrainer = CurriculumSFTTrainer
    sft_trainer.main()
