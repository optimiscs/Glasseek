"""Build SFT commands from profiles; keep trainer algorithms in their modules."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Shared"))
from glasseek_shared.hermes.url_registry import add_url_argument, VERSION


def build(profile, extra=()):
    profiles = json.loads((ROOT / 'RL_training/configs/sft_launch_profiles.json').read_text())
    generic = profile == 'default'
    cfg = {'python': sys.executable, 'nproc': 4, 'module': 'verl.trainer.sft_trainer',
           'config': 'sft', 'overrides': [], 'env': {}} if generic else profiles[profile]
    env = os.environ.copy()
    if generic and not env.get('MODEL_PATH'):
        raise ValueError('Set MODEL_PATH to a base model or local checkpoint')
    python = env.get('SFT_PYTHON', cfg['python'])
    nproc = env.get('NPROC_PER_NODE', str(cfg['nproc']))
    module = env.get('SFT_TRAIN_MODULE', cfg['module'])
    paths = [str(ROOT / 'RL_training'), str(ROOT / 'Shared')]
    if not generic:
        paths.insert(0, str(ROOT / '.cache/experiments/pro6000-lora/flash-wheel'))
        defaults = {'TRITON_CACHE_DIR': '/root/autodl-tmp/pro6000-triton-cache',
            'TORCHINDUCTOR_CACHE_DIR': '/root/autodl-tmp/pro6000-torchinductor-cache',
            'TMPDIR': '/root/autodl-tmp', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
            'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '8'}
        for key, value in dict(defaults, **cfg['env']).items():
            env.setdefault(key, value.replace('{root}', str(ROOT)))
    env['PYTHONPATH'] = os.pathsep.join(paths + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    argv = [python, '-m', 'torch.distributed.run']
    if generic or int(env.get('NNODES', '1')) != 1:
        for name, value in [('nnodes', env.get('NNODES', '1')), ('nproc_per_node', nproc),
                ('node_rank', env.get('NODE_RANK', '0')), ('master_addr', env.get('MASTER_ADDR', '127.0.0.1')),
                ('master_port', env.get('MASTER_PORT', '29500'))]:
            argv += [f'--{name}={value}']
    else:
        argv += ['--standalone', f'--nproc_per_node={nproc}']
    argv += ['-m', module, f'--config-path={ROOT}/RL_training/configs',
             '--config-name=' + env.get('SFT_CONFIG_NAME', cfg['config'])]
    argv += [arg.replace('{root}', str(ROOT)) for arg in cfg['overrides']]
    argv += [f'trainer.nnodes={env.get("NNODES", "1")}', f'trainer.n_gpus_per_node={nproc}']
    argv += list(extra)
    if env.get('GLASSEEK_URL_MODE', 'raw') != 'raw':
        argv += ['+data.url_mode=' + env['GLASSEEK_URL_MODE'], '+data.url_protocol=' + VERSION]
    return argv, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='default')
    parser.add_argument('--print-command', action='store_true')
    add_url_argument(parser)
    from glasseek_shared.context.config import add_arguments, configure
    add_arguments(parser)
    args, extra = parser.parse_known_args()
    from glasseek_shared.hermes.url_prompt import configure_url_arguments
    configure_url_arguments(args)
    configure(args)
    argv, env = build(args.profile, extra)
    if args.print_command:
        print(json.dumps(argv))
        return
    subprocess.run([argv[0], '-c', 'from glasseek_training.validation import require_verl_version; require_verl_version()'], env=env, check=True)
    os.chdir(ROOT)
    os.execvpe(argv[0], argv, env)


if __name__ == '__main__':
    main()
