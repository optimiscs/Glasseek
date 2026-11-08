"""Prevent reintroducing the retired local analysis tree into runnable code."""
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_analysis_directory_is_retired():
    assert not os.path.lexists(ROOT / 'analysis')


def test_runnable_sources_do_not_depend_on_local_analysis():
    locations = ['experiments', 'Evaluation/services', 'Evaluation/config',
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
