"""Two-rank SFT curriculum ordered by assistant message count."""
import hashlib
import json
import math
from pathlib import Path
from torch.utils.data import Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer import sft_trainer
from verl.utils.device import get_device_name

class TurnSampler(Sampler):
    def __init__(self, messages, global_batch, world, rank):
        if world != 2 or global_batch % world or not 0 <= rank < world:
            raise ValueError("Expected two ranks and a divisible global batch")
        self.world, self.rank, self.global_batch = world, rank, global_batch
        self.local_batch = global_batch // world
        self.rounds = [sum(m.get("role") == "assistant" for m in row) for row in messages]
        if not self.rounds or min(self.rounds) < 1:
            raise ValueError("Every trajectory must contain an assistant message")
        order = sorted(range(len(messages)), key=lambda i: (self.rounds[i], i))
        pad = (-len(order)) % global_batch
        self.pad_indices = [order[i % len(order)] for i in range(pad)]
        self.order = sorted(order + self.pad_indices, key=lambda i: (self.rounds[i], i))
        self.epoch = 0
    def set_epoch(self, epoch):
        self.epoch = epoch
    def __len__(self):
        return len(self.order) // self.world
    def __iter__(self):
        for start in range(0, len(self.order), self.global_batch):
            offset = start + self.rank * self.local_batch
            yield from self.order[offset:offset + self.local_batch]

class TurnCurriculumTrainer(sft_trainer.SFTTrainer):
    def _build_dataloader(self):
        super()._build_dataloader()
        world = self.engine.get_data_parallel_size()
        rank = self.engine.get_data_parallel_rank()
        if self.engine.ulysses_sequence_parallel_size != 1:
            raise ValueError("Turn curriculum requires SP=1")
        path = Path(self.config.data.train_files)
        manifest = json.loads((path.parent / "manifest.json").read_text())
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != manifest["output_sha256"]["train"]:
            raise ValueError("Training parquet does not match manifest")
        if manifest["max_length"] != 65536 or manifest["counts"]["train"] != len(self.train_dataset):
            raise ValueError("Unexpected 64K training dataset")
        self.train_sampler = TurnSampler(self.train_dataset.messages, self.global_batch_size, world, rank)
        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp, sampler=self.train_sampler,
            collate_fn=self.collate_fn, num_workers=self.config.data.num_workers,
            pin_memory=False, drop_last=True, pin_memory_device=get_device_name())
        print(f"Turn curriculum: rank={rank}, trajectories={len(self.train_dataset)}, "
              f"assistant_rounds={min(self.train_sampler.rounds)}..{max(self.train_sampler.rounds)}, "
              f"padded_repetitions={len(self.train_sampler.pad_indices)}, "
              f"padding_dataset_index={self.train_sampler.order[-1]}, manifest_sha256={digest.hexdigest()}", flush=True)

if __name__ == "__main__" and __import__("os").environ.get("TURN_AUDIT_DIR"):
    import os
    from glasseek_training.sft_dataset import GlasseekSFTDataset
    _original_getitem = GlasseekSFTDataset.__getitem__
    def _audited_getitem(self, index):
        result = _original_getitem(self, index)
        name = os.environ.get("TURN_AUDIT_PHASE", "run") + "-rank" + os.environ.get("RANK", "unknown") + ".jsonl"
        with open(os.path.join(os.environ["TURN_AUDIT_DIR"], name), "a") as out:
            out.write(json.dumps({"index": int(index)}) + "\n")
        return result
    GlasseekSFTDataset.__getitem__ = _audited_getitem

if __name__ == "__main__":
    sft_trainer.SFTTrainer = TurnCurriculumTrainer
    sft_trainer.main()
