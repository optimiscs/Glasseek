"""Keep historical implementations out of active entrypoints and test discovery."""
import ast
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_analysis_directory_is_retired():
    for name in ('analysis', 'reports', 'experiments',
                 'scripts/trajectory_benchmark', 'scripts/trajectory_preprocess_pilot'):
        assert not os.path.lexists(ROOT / name), name


def test_active_scripts_do_not_import_archived_implementations():
    retired = {'archieve', 'run_v2', 'run_v3', 'policy_v4',
               'trajectory_benchmark', 'trajectory_preprocess_pilot'}
    violations = []
    for path in (ROOT / 'scripts').rglob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
            if any(retired.intersection(name.split('.')) for name in names):
                violations.append(str(path.relative_to(ROOT)))
        if re.search(r'_v\d+(?:_|\.)', path.name):
            violations.append(str(path.relative_to(ROOT)))
    assert not violations, violations


def test_training_outputs_do_not_recreate_historical_directories():
    import json
    profiles = json.loads((ROOT / 'RL_training/configs/sft_launch_profiles.json').read_text())
    for profile in profiles.values():
        for value in profile.get('env', {}).values():
            assert not value.startswith(('{root}/reports/', '{root}/experiments/')), value


def test_runnable_sources_do_not_depend_on_local_analysis():
    locations = ['scripts', 'Evaluation/services', 'Evaluation/config',
                 'RL_training/glasseek_training', 'RL_training/scripts',
                 'Data_curation/multi_agent_mpcs_v5_publish/browsecomp_v2',
                 'Data_curation/multi_agent_mpcs_v5_publish/start_v5p1.sh']
    suffixes = {'.py', '.sh', '.json', '.yaml', '.yml', '.toml'}
    violations = []
    for location in locations:
        base = ROOT / location
        candidates = [base] if base.is_file() else base.rglob('*')
        for path in candidates:
            if path.is_symlink() or not path.is_file() or path.suffix not in suffixes:
                continue
            # Remote checkpoint-reeval deployment paths were never part of the
            # local analysis archive; retain that explicit remote contract.
            content = path.read_text().replace('checkpoint-reeval/analysis/', 'remote-deployment/')
            if re.search(r"(?<![\w-])analysis/|['\"]analysis['\"]\s*/", content):
                violations.append(str(path.relative_to(ROOT)))
    assert not violations, violations
