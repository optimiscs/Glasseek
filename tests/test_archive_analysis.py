"""Exercise migration against actual files, links, and the rollback journal."""
import importlib.util
import hashlib
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location('archive_analysis', Path(__file__).resolve().parents[1] / 'scripts/maintenance/archive_analysis.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


@pytest.fixture
def fixture(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    contents = {'run.py': 'print(42)\n', 'metrics.json': '{"score": 1}',
                'questions.jsonl': '{}\n', 'wheels/pkg.whl': 'wheel',
                'checkpoints/step1/weights.pt': 'weights', 'hermes/.env': 'PRIVATE=fixture'}
    for name, content in contents.items():
        path = tmp_path / 'analysis/demo' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(['git', 'add', 'analysis/demo/run.py'], cwd=tmp_path, check=True)
    output = tmp_path / 'reports/inventory'
    plan = archive.inventory(tmp_path, output)
    return tmp_path, output, plan, contents


def test_phases_preserve_identity_links_and_rollback(fixture):
    root, output, plan, contents = fixture
    assert archive.apply(plan, output, list(archive.TARGETS)) == 5
    assert archive.verify(plan)['failures'] == []
    assert archive.apply(plan, output, ['experiments']) == 1
    assert (root / 'analysis/demo').is_symlink()
    assert archive.verify(plan)['failures'] == []
    for name, content in contents.items():
        assert (root / 'analysis/demo' / name).read_text() == content
    assert archive.apply(plan, output, list(archive.TARGETS) + ['experiments']) == 0
    archive.rollback(root, output / 'migration-journal.jsonl')
    assert not (root / 'analysis/demo').is_symlink()
    assert archive.verify(plan)['failures'] == []
    assert not (root / 'analysis/demo/metrics.json').is_symlink()


def test_refuses_changed_source(fixture):
    root, output, plan, _ = fixture
    (root / 'analysis/demo/metrics.json').write_text('changed content')
    with pytest.raises(ValueError, match='changed since inventory'):
        archive.apply(plan, output, ['results'])


def test_refuses_destination_conflict(fixture):
    root, output, plan, _ = fixture
    target = root / 'reports/experiments/demo/metrics.json'
    target.parent.mkdir(parents=True)
    target.write_text('existing')
    with pytest.raises(ValueError, match='Conflicting'):
        archive.apply(plan, output, ['results'])
    assert target.read_text() == 'existing'


def test_readers_allowed_writers_blocked(fixture, monkeypatch):
    _, output, plan, _ = fixture
    monkeypatch.setattr(archive, 'active_paths', lambda root: [{'access': 'writer', 'pid': 1}])
    with pytest.raises(RuntimeError, match='Active analysis paths'):
        archive.apply(plan, output, ['cache'])
    monkeypatch.setattr(archive, 'active_paths', lambda root: [{'access': 'reader', 'pid': 1}])
    assert archive.apply(plan, output, ['cache']) == 1


def test_link_failure_restores_source(fixture, monkeypatch):
    root, output, plan, _ = fixture
    def fail(*args, **kwargs):
        raise OSError('link failure')
    monkeypatch.setattr(Path, 'symlink_to', fail)
    with pytest.raises(OSError, match='link failure'):
        archive.apply(plan, output, ['results'])
    assert (root / 'analysis/demo/metrics.json').read_text() == '{"score": 1}'
    assert not (root / 'reports/experiments/demo/metrics.json').exists()


def test_verify_explicit_post_archive_edits(fixture):
    root, _, plan, _ = fixture
    row = next(row for row in plan['files'] if row['path'].endswith('run.py'))
    path = root / row['path']
    path.write_text('print(43)\n')
    assert len(archive.verify(plan)['failures']) == 1
    change = {'path': row['path'], 'original_sha256': row['sha256'],
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    assert archive.verify(plan, [change])['failures'] == []
    path.write_text('unexpected later edit')
    assert len(archive.verify(plan, [change])['failures']) == 1
    change['original_sha256'] = 'incorrect'
    with pytest.raises(ValueError, match='does not match'):
        archive.verify(plan, [change])


def test_preserves_journal_inventory_and_blocks_live_rollback(fixture, monkeypatch):
    root, output, plan, _ = fixture
    archive.apply(plan, output, ['cache'])
    with pytest.raises(ValueError, match='already has a migration journal'):
        archive.inventory(root, output)
    monkeypatch.setattr(archive, 'active_paths', lambda root: [{'access': 'writer', 'pid': 1}])
    with pytest.raises(RuntimeError, match='before rollback'):
        archive.rollback(root, output / 'migration-journal.jsonl')


def test_retire_keeps_canonical_assets_and_verification(fixture):
    root, output, plan, contents = fixture
    archive.apply(plan, output, list(archive.TARGETS) + ['experiments'])
    assert archive.retire(plan, output) == 1
    assert not (root / 'analysis').exists()
    assert archive.verify(plan, canonical=True)['failures'] == []
    for name, content in contents.items():
        assert (root / 'experiments/demo' / name).read_text() == content
    assert archive.retire(plan, output) == 0
    with pytest.raises(RuntimeError, match='retired'):
        archive.rollback(root, output / 'migration-journal.jsonl')


def test_retire_refuses_unexpected_files_and_changed_archive(fixture):
    root, output, plan, _ = fixture
    archive.apply(plan, output, list(archive.TARGETS) + ['experiments'])
    unexpected = root / 'analysis/keep.txt'
    unexpected.write_text('new user file')
    with pytest.raises(ValueError, match='Unexpected analysis entry'):
        archive.retire(plan, output)
    assert (root / 'analysis/demo').is_symlink()
    unexpected.unlink()
    (root / 'experiments/demo/run.py').write_text('unexpected edit')
    with pytest.raises(ValueError, match='Canonical archive verification failed'):
        archive.retire(plan, output)
    assert (root / 'analysis/demo').is_symlink()
