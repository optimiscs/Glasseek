import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'Evaluation/services'), str(ROOT / 'RL_training')]
import model_server
from glasseek_training import lora_probe, sft_launch, verify_weights


@pytest.mark.parametrize('profile,length,truncation,offload', [
    ('pro6000', 327680, 'error', False), ('128k', 131072, 'right', False),
    ('offload', 327680, 'error', True), ('ddp', 262144, 'error', False)])
def test_lora_preserves_original_variants(profile, length, truncation, offload):
    cfg = lora_probe.configuration(profile)
    assert (cfg['max_length'], cfg['truncation'], cfg['activation_offload']) == (length, truncation, offload)
    assert (cfg['lora_rank'], cfg['lora_alpha'], cfg['lr']) == (16, 32, 2e-5)


@pytest.mark.parametrize('profile,nproc,module', [
    ('128k', 1, 'verl.trainer.sft_trainer'), ('64k', 2, 'verl.trainer.sft_trainer'),
    ('turns', 2, 'glasseek_training.curriculum_sft'),
    ('clean2899', 2, 'glasseek_training.curriculum_sft'),
    ('length', 1, 'glasseek_training.curriculum_sft')])
def test_sft_profiles_preserve_trainer_and_override_order(profile, nproc, module, monkeypatch):
    for key in ('NPROC_PER_NODE', 'SFT_TRAIN_MODULE', 'NNODES'):
        monkeypatch.delenv(key, raising=False)
    args, env = sft_launch.build(profile, ['optim.lr=0.0001'])
    assert f'--nproc_per_node={nproc}' in args and module in args
    assert args[-1] == 'optim.lr=0.0001'
    assert str(ROOT / 'RL_training') in env['PYTHONPATH']
    if profile in ('turns', 'clean2899'):
        assert 'engine.use_no_sync_for_gradient_accumulation=false' in args


def test_vllm_profiles_and_environment_override(monkeypatch):
    monkeypatch.delenv('EVAL_PER_GPU_CONCURRENCY', raising=False)
    for profile, limit in [('rl9b', 16), ('step39', 32), ('step76', 8)]:
        args, env = model_server.build(profile, 1, 19000)
        assert args[args.index('--max-num-seqs') + 1] == str(limit)
        assert env['CUDA_VISIBLE_DEVICES'] == '1'
        assert ('--default-chat-template-kwargs' in args) == (profile == 'step39')
    monkeypatch.setenv('EVAL_PER_GPU_CONCURRENCY', '6')
    args, _ = model_server.build('step39', 0, 19001)
    assert args[args.index('--max-num-seqs') + 1] == '6'


def test_weight_verification_matches_and_rejects_corruption(tmp_path):
    import torch
    from safetensors.torch import save_file
    tensor = torch.arange(6).reshape(2, 3)
    target = str(tmp_path / 'model.safetensors')
    save_file({'weight': tensor}, target)
    for rank in range(2):
        torch.save({'weight': tensor}, tmp_path / f'model_world_size_2_rank_{rank}.pt')
    assert verify_weights.verify(tmp_path, target)['tensor_shards_checked'] == 2
    torch.save({'weight': tensor + 1}, tmp_path / 'model_world_size_2_rank_1.pt')
    with pytest.raises(ValueError, match='Weight mismatch'):
        verify_weights.verify(tmp_path, target)


def test_weight_shard_boundaries_and_dtype():
    import torch
    from torch.distributed.tensor import Shard
    merged = torch.arange(5)
    for rank, local in enumerate((merged[:3], merged[3:])):
        tensor = SimpleNamespace(placements=(Shard(0),), to_local=lambda local=local: local)
        assert verify_weights.compare_tensor(merged, tensor, rank, 2)
    assert not verify_weights.compare_tensor(merged, merged.float(), 0, 1)
