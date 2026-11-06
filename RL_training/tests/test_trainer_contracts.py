"""Exercise production trainer methods with RPC boundaries replaced by local doubles."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from glasseek_training.cm_gspo import Rollout, Segment, pack_rollouts


def trainer_class():
    source = Path(__file__).resolve().parents[1] / 'glasseek_training/trainer.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CMTrainer')
    node.bases = [ast.Name(id='object', ctx=ast.Load())]
    ast.fix_missing_locations(node)
    class Proto:
        @staticmethod
        def from_dict(tensors):
            obj = Proto()
            obj.tensors, obj.meta_info = tensors, {}
            return obj
        def to_tensordict(self):
            return {**self.tensors, **self.meta_info}
    namespace = dict(np=np, OmegaConf=OmegaConf, DataProto=Proto, pack_rollouts=pack_rollouts,
                     Rollout=Rollout, Segment=Segment, left_right_2_no_padding=lambda x: x,
                     tu=SimpleNamespace(assign_non_tensor=lambda td, **kw: td.update(kw),
                                        get=lambda td, key: td[key]),
                     Tracking=lambda **kw: SimpleNamespace(log=lambda *a, **kw: None))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['CMTrainer']


def test_actor_rpc_receives_original_rollout_denominator():
    captured = []
    trainer = trainer_class()()
    trainer.config = OmegaConf.create({'actor_rollout_ref': {
        'actor': {'data_loader_seed': 42}, 'rollout': {'n': 8, 'temperature': 1.}},
        'cm_gspo': {'context': {'max_model_len': 128}}})
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    def update(data):
        captured.append(data)
        return {'metrics': {'loss': [0.]}}
    trainer.actor_rollout_wg = SimpleNamespace(world_size=32, update_actor=update)
    trainer._get_dp_size = lambda wg, role: 8
    rollouts = []
    for prompt in range(16):
        for sample in range(8):
            parts = [Segment([11], [12], [1], [-1.])] * (2 if sample == 0 else 1)
            rollouts.append(Rollout(str(prompt), float(sample == 0), parts))
    metrics = trainer._update_cm(rollouts)
    assert len(captured) == 1
    assert captured[0]['global_batch_size'] == 128
    assert captured[0]['mini_batch_size'] == 144
    assert captured[0]['epochs'] == 1
    assert metrics['cm/segment_count'] == 144


def test_resume_stops_at_100_optimizer_steps_across_dataloader_epochs():
    trainer = trainer_class()()
    trainer.config = OmegaConf.create({'trainer': {
        'project_name': 'test', 'experiment_name': 'test', 'logger': ['console'],
        'val_before_train': False, 'val_only': False, 'save_freq': 10, 'test_freq': -1}})
    trainer.total_training_steps = 100
    trainer.train_dataloader = [0, 1]
    trainer._load_checkpoint = lambda: setattr(trainer, 'global_steps', 97)
    weights, updates, saves = [], [], []
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda step: weights.append(step))
    rollout = Rollout('q', 1., [Segment([1], [2], [1], [-1.])])
    trainer._generate_cm = lambda batch: ([rollout], SimpleNamespace(non_tensor_batch={
        'cm_compactions': np.array([0]), 'cm_stop_reason': np.array(['answer'])}))
    def update(rows):
        updates.append(trainer.global_steps)
        return {}
    trainer._update_cm = update
    trainer._save_checkpoint = lambda: saves.append(trainer.global_steps)
    trainer._shutdown_dump_executor = lambda: setattr(trainer, 'cleaned', True)
    trainer.fit()
    assert trainer.global_steps == 100
    assert updates == [97, 98, 99]
    assert weights == [97, 98, 99, 100]
    assert saves == [100]
    assert trainer.cleaned
