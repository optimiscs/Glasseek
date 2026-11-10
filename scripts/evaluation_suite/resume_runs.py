"""Adopt compatible run identities without copying/replaying completed trajectories."""
import json
from pathlib import Path
import re
import shutil
from suite import sha


def prediction_run(args, benchmark):
    mapping = getattr(args, 'prediction_runs', None)
    if mapping is None:
        try:
            mapping = json.loads((args.output_root/args.run_id/'launch.json').read_text()).get('prediction_runs', {})
        except (OSError, ValueError):
            mapping = {}
    return mapping.get(benchmark, args.run_id + '_' + benchmark)


def prediction_root(args, benchmark):
    return args.output_root / 'predictions' / prediction_run(args, benchmark) / ('suite_' + benchmark)


def prepare(args, names):
    args.prediction_runs = {}
    sources = []
    requested = getattr(args, 'resume_run_id', []) or []
    for run_id in requested:
        if not re.fullmatch(r'[\w.-]+', run_id) or run_id in ('.', '..') or run_id == args.run_id:
            raise ValueError('resume source must be a different safe run ID')
        source = args.output_root / run_id
        manifest_path = source / 'launch.json'
        manifest = json.loads(manifest_path.read_text())
        expected = {'suite_sha256': sha(args.suite / 'manifest.json'),
                    'context_manager': args.context_manager, 'url_mode': args.url_mode,
                    'max_tool_calls': args.max_tool_calls}
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise ValueError(f'incompatible resume {run_id}: {field}')
        old_protocol = (manifest.get('research_protocol') or {}).get('name', 'legacy')
        if old_protocol != args.research_prompt or old_protocol != 'legacy':
            raise ValueError('cross-run adoption currently supports matching legacy prompts only')
        old_prompt = manifest.get('system_prompt_override')
        new_hash = sha(args.system_prompt_file) if args.system_prompt_file else None
        if (old_prompt or {}).get('sha256') != new_hash:
            raise ValueError(f'incompatible resume {run_id}: system prompt')
        if manifest.get('score_config_sha256') != (sha(args.score_config) if args.score_config else None):
            raise ValueError(f'incompatible resume {run_id}: grading configuration')
        benchmarks = [n for n in names if n in manifest['benchmarks']]
        for name in benchmarks:
            if name in args.prediction_runs:
                raise ValueError(f'multiple resume sources supply {name}')
            args.prediction_runs[name] = manifest.get('prediction_runs', {}).get(name, run_id + '_' + name)
        sources.append({'run_id': run_id, 'manifest_sha256': sha(manifest_path), 'benchmarks': benchmarks})
    return sources


def verify_model(args, benchmark, model):
    if benchmark not in getattr(args, 'prediction_runs', {}):
        return
    for path in prediction_root(args, benchmark).glob('conv/*/*.json'):
        record = json.loads(path.read_text())
        if record.get('model') != model:
            raise ValueError(f'incompatible resume model in {path}')
    # Do not create two inference writers for the same save-name. Read-only
    # inspection of procfs; stopped producers must be explicitly terminated too.
    save_name = prediction_run(args, benchmark)
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = (proc / 'cmdline').read_bytes().decode().split('\0')
            if '--save-name' in argv and argv[argv.index('--save-name') + 1] == save_name:
                raise ValueError(f'resume source still has an active producer: PID {proc.name}')
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue


def adopt_grading_ledger(args, sources):
    """Keep paid-call claims and receipts, including uncertain/error claims."""
    destination = args.output_root / args.run_id / 'trajectory-scores'
    for source in sources:
        for name in source['benchmarks']:
            old = args.output_root / source['run_id'] / 'trajectory-scores' / name
            if not old.exists():
                continue
            new = destination / name
            new.mkdir(parents=True, exist_ok=True)
            for task in old.iterdir():
                if task.is_dir() and not (new / task.name).exists():
                    shutil.copytree(task, new / task.name)
