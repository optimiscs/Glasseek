"""Start vLLM replicas from named model profiles; no GPU work in dry runs."""
import argparse
import json
import os
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]


def build(profile, gpu, port, overrides=None):
    if gpu < 0 or not 1 <= port <= 65535:
        raise ValueError('Invalid GPU index or port')
    document = json.loads(Path(os.environ.get('GLASSEEK_MODEL_PROFILES', SOURCE / 'config/model_profiles.json')).read_text())
    cfg = dict(document['defaults'], **document['profiles'][profile])
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    evaluation = Path(os.environ.get('GLASSEEK_EVAL_ROOT', SOURCE))
    runtime = Path(os.environ.get('GLASSEEK_RUNTIME_DIR', evaluation / 'data/runtime'))
    quest = Path(os.environ.get('QUEST_ROOT', SOURCE.parent.parent / 'Quest'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4')
    env['PATH'] = cfg.get('path_prefix', str(runtime / 'venv/bin')) + os.pathsep + env.get('PATH', '')
    if cfg.get('cuda_home'):
        env['CUDA_HOME'] = cfg['cuda_home']
    env.setdefault('HF_HOME', str(runtime / 'cache/huggingface'))
    for key, prefix in [('VLLM_CACHE_ROOT', 'vllm'), ('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor')]:
        env.setdefault(key, str(runtime / 'cache' / f'{prefix}-{profile}-gpu{gpu}'))
    if cfg.get('sequence_env') and not (overrides or {}).get('max_num_seqs'):
        cfg['max_num_seqs'] = int(env.get(cfg['sequence_env'], cfg['max_num_seqs']))
    model = cfg['model_path'].format(root=SOURCE.parent, runtime=runtime)
    argv = [env.get('VLLM_BIN', str(quest / 'workspace/env/bin/vllm')), 'serve', model,
            '--served-model-name', cfg['served_model_name'], '--host', '127.0.0.1', '--port', str(port)]
    for name in ('tensor_parallel_size', 'max_model_len', 'gpu_memory_utilization', 'kv_cache_dtype', 'max_num_seqs'):
        argv += ['--' + name.replace('_', '-'), str(cfg[name])]
    argv += ['--reasoning-parser', 'qwen3', '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder', '--enable-prompt-tokens-details']
    if cfg.get('thinking') is not None:
        argv += ['--default-chat-template-kwargs', json.dumps({'enable_thinking': cfg['thinking']})]
    return argv, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile')
    parser.add_argument('gpu', type=int)
    parser.add_argument('port', type=int)
    parser.add_argument('--model-path')
    parser.add_argument('--served-model-name')
    parser.add_argument('--max-num-seqs', type=int)
    parser.add_argument('--max-model-len', type=int)
    parser.add_argument('--print-command', action='store_true')
    args = parser.parse_args()
    argv, env = build(args.profile, args.gpu, args.port, vars(args))
    if args.print_command:
        print(json.dumps(argv))
    else:
        os.execvpe(argv[0], argv, env)


if __name__ == '__main__':
    main()
