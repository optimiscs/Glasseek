"""Launch evaluation/judging with named profiles and shared tool routing."""
import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from search_pool import client_env, load_config as search_config
from tool_services import environment, load_config as tool_config

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE.parent / "Shared"))
from glasseek_shared.hermes.url_registry import add_url_argument


def build(profile, mode, overrides=None, *, python_entry=False):
    config = json.loads(Path(os.environ.get('GLASSEEK_EVAL_PROFILES', SOURCE / 'config/eval_profiles.json')).read_text())
    preset = config['profiles'][profile]
    settings = dict(config['defaults'])
    settings.update(config.get('protocols', {}).get(preset.get('protocol'), {}))
    settings.update(preset)
    settings.update(overrides or {})
    evaluation = Path(os.environ.get('GLASSEEK_EVAL_ROOT', SOURCE))
    runtime = Path(os.environ.get('GLASSEEK_RUNTIME_DIR', evaluation / 'data/runtime'))
    env = dict(os.environ, **environment(tool_config()))
    env['PYTHONPATH'] = str(SOURCE.parent / 'Shared') + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.update(client_env(search_config()))
    defaults = {'HERMES_HOME': str(runtime / 'hermes'), 'EVAL_PYTHON': str(runtime / 'venv/bin/python'),
        'HERMES_BIN': str(runtime / 'venv/bin/hermes'), 'HERMES_API_KEY': 'local',
        'HERMES_REMOTE_PURE_COMPUTE': 'Lab3090', 'GLASSEEK_SEARCH_TOP10': '1',
        'GLASSEEK_FETCH_DOH_SAFETY': '1', 'GLASSEEK_ALLOW_KV_PREEMPTION': '1',
        'GLASSEEK_RESUME_INTERRUPTED': '1'}
    for key, value in defaults.items():
        env.setdefault(key, value)
    if settings.get('max_tool_calls') is not None:
        env['GLASSEEK_MAX_TOOL_CALLS'] = str(settings['max_tool_calls'])
    if settings.get('timeout_seconds') is None:
        env['NO_TOTAL_TIMEOUT'] = '1'
    if mode == 'publish':
        argv = [env['EVAL_PYTHON'], str(SOURCE / 'services/viewer_publish.py'),
                '--run', settings['run'], '--dataset', settings['dataset'], '--model', settings['model'],
                '--label', f"{settings['model']} · BrowseComp 128 · {settings['workers_per_endpoint']} 并发"]
    elif mode == 'watch-judge':
        argv = [env['EVAL_PYTHON'], str(SOURCE / 'services/async_judge.py'),
                '--run', settings['run'], '--dataset', settings['dataset'], '--workers', '4',
                '--handoff-lock', str(runtime / 'tool-services' / (settings['run'] + '.judge.lock'))]
    elif mode == 'judge':
        cfg = json.loads((SOURCE / 'config/async_judge.json').read_text())
        argv = ['bash', str(evaluation / 'scripts/judge.sh'), '--save-name', settings['run'],
            '--dataset', settings['dataset'], '--judge-provider', cfg['provider'],
            '--judge-base-url', env.get('JUDGE_BASE_URL') or cfg['base_url'],
            '--judge-model', env.get('JUDGE_MODEL') or cfg['model'], '--prompt-version', 'official',
            '--token-param', cfg['token_param'], '--no-json-schema', '--temperature', str(cfg['temperature'])]
        env['GLASSEEK_JUDGE_TERMINAL_ONLY'] = '1'
    else:
        endpoint_args = ['--base-urls', settings['base_urls']] if settings.get('base_urls') else [
            '--endpoint-registry', str(runtime / settings['registry']), '--endpoint-registry-refresh-interval', '5',
            '--ramp-registry',
            '--ramp-start', str(settings['ramp_start']), '--ramp-max', str(settings['ramp_max']),
            '--ramp-step', '2', '--ramp-window', '90', '--ramp-interval', '5']
        entry = [env['EVAL_PYTHON'], str(SOURCE / 'services/native_eval.py')] if python_entry else [
            'bash', str(evaluation / 'scripts/run_eval.sh')]
        argv = [*entry, '--model', settings['model'],
            *endpoint_args, '--workers-per-endpoint', str(settings['workers_per_endpoint']),
            '--dataset', settings['dataset'], '--data-path', str(runtime / (settings['dataset'] + '.jsonl')),
            '--save-name', settings['run'], '--numbered-output', '--tool-whitelist', 'web,code_execution,delegation,vision',
            '--max-rounds', str(settings['max_rounds']), '--max-retries', str(settings['max_retries']),
            '--context-length', str(settings['context_length']),
            '--fetch-server-url', env['GLASSEEK_FETCH_URL'], '--fetch-watchdog-target', 'https://example.com']
        for name in ('timeout_seconds', 'expected_questions', 'repeats', 'max_tokens', 'context_compression', 'compression_threshold'):
            value = settings.get(name)
            if value is not None:
                argv += ['--' + name.replace('_', '-'), str(value).lower() if isinstance(value, bool) else str(value)]
        if settings.get('system_prompt'):
            argv += ['--enable-custom-system-prompt', 'true', '--custom-system-prompt-file',
                     settings['system_prompt'].format(root=SOURCE.parent, runtime=runtime)]
        if mode == 'smoke':
            argv += ['--limit', '1']
        if env.get('PREPARE_ONLY') == '1':
            argv += ['--prepare-only']
        if python_entry:
            # Match the existing launcher's worker/environment defaults without
            # invoking a shell. num-workers also bounds multi-endpoint profiles.
            argv += ['--num-workers', str(settings['workers_per_endpoint']),
                     '--hermes-bin', env['HERMES_BIN'], '--provider', 'custom',
                     '--api-key-env', 'HERMES_API_KEY', '--ignore-rules', '--endpoint-watchdog']
            env.setdefault('HERMES_SESSION_STORAGE_ROOT', str(Path(env['HERMES_HOME']) / 'sessions'))
            env.setdefault('PYTHONUNBUFFERED', '1')
            env['FETCH_SERVER_BASE_URL'] = env['FETCH_SERVER_URL'] = env['GLASSEEK_FETCH_URL']
            for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'BROWSER_CDP_URL'):
                env.pop(key, None)
            hosts = [urlsplit(url.strip()).hostname for url in settings.get('base_urls', '').split(',') if url.strip()]
            env['NO_PROXY'] = env['no_proxy'] = ','.join(dict.fromkeys(
                host for host in [*hosts, 'localhost', '127.0.0.1', '::1'] if host))
    return argv, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile')
    parser.add_argument('mode', choices=('full', 'smoke', 'judge', 'publish', 'watch-judge'))
    parser.add_argument('--run-name', help='Shared run identity for evaluation, publishing and judging')
    parser.add_argument('--print-command', action='store_true')
    add_url_argument(parser)
    args, extra = parser.parse_known_args()
    from glasseek_shared.hermes.url_prompt import configure_url_arguments
    configure_url_arguments(args)
    argv, env = build(args.profile, args.mode, {'run': args.run_name} if args.run_name else None)
    argv += extra
    if args.print_command:
        print(json.dumps(argv))
        return
    if args.mode == 'judge' and not env.get('JUDGE_API_KEY'):
        quest = Path(env.get('QUEST_ROOT', SOURCE.parent.parent / 'Quest'))
        credentials = Path(env.get('GLASSEEK_API_CONFIG', quest / 'workspace/strict-reproduction/.secrets/api_config.json'))
        env['JUDGE_API_KEY'] = json.loads(credentials.read_text())['common']['JUDGE_OPENAI_API_KEY']
    os.chdir(Path(env.get('GLASSEEK_EVAL_ROOT', SOURCE)))
    os.execvpe(argv[0], argv, env)


if __name__ == '__main__':
    main()
