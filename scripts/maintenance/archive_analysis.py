"""Inventory and reversibly archive analysis assets without copying large files.

Inventory is read-only except for the requested report directory. Apply uses an
explicit saved plan, same-filesystem renames, compatibility links, and a journal.
It never deletes assets or follows directory symlinks while scanning.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
CACHE_NAMES = {'__pycache__', 'site', 'deps', 'test-deps', 'wheels', 'flash-wheel',
               'cuda12', 'Hy-MT2-1.8B'}
RUNTIME_NAMES = {'hermes', 'profiles', 'remote_snapshot', 'session-audit', 'remote'}
RESULT_DIRS = {'translated', 'tensorboard', 'tensorboard_comparison', 'remote-output',
               'remote-audit', 'prompt-resume-audit', 'error-analysis-four'}
CODE_SUFFIXES = {'.py', '.sh', '.md', '.yaml', '.yml', '.toml', '.sql'}
MODEL_NAMES = {'checkpoints', 'adapters'}
TARGETS = {'cache': '.cache/experiments', 'checkpoints': 'checkpoints/archive',
           'runtime': 'Evaluation/data/runtime/archive', 'results': 'reports/experiments',
           'inputs': 'data/experiments'}


def classification(path, directory=False):
    name = path.name
    if directory:
        if name in CACHE_NAMES or name.startswith(('vllm-cache', 'triton-cache')):
            return 'cache'
        if name in MODEL_NAMES or name.endswith('-hf'):
            return 'checkpoints'
        if name in RUNTIME_NAMES:
            return 'runtime'
        if name in RESULT_DIRS or name.startswith(('grads-', 'seeded-grads-', 'verified-grads-')):
            return 'results'
        return None
    if path.suffix in CODE_SUFFIXES or name.endswith('.py.disabled'):
        return None
    if path.suffix == '.txt' and ('prompt' in name or name in ('requirements.txt',)):
        return None
    if name.endswith('-registry.json') or name in ('samples.json', 'representative.json'):
        return None
    if path.suffix == '.parquet' or name in ('questions.jsonl', 'browsecomp128.jsonl', 'browsecomp_rl9b_non_timeout_34.jsonl'):
        return 'inputs'
    return 'results'


def snapshot(path):
    st = path.lstat()
    return {'device': st.st_dev, 'inode': st.st_ino, 'size': st.st_size,
            'mtime_ns': st.st_mtime_ns, 'mode': stat.S_IMODE(st.st_mode)}


def active_paths(root):
    """Only record paths/PIDs, never process environments or command contents."""
    prefixes = tuple(str(root / name) + '/' for name in ('analysis', 'experiments'))
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            paths = [(os.readlink(proc / 'cwd'), 'cwd')]
            for fd in (proc / 'fd').iterdir():
                try:
                    path = os.readlink(fd)
                    flags = next(line.split()[1] for line in (proc / 'fdinfo' / fd.name).read_text().splitlines()
                                 if line.startswith('flags:'))
                    # Readers retain their inode across rename and can reopen via
                    # compatibility links. Writers must finish before archival.
                    access = 'reader' if int(flags, 8) & os.O_ACCMODE == os.O_RDONLY else 'writer'
                    paths.append((path, access))
                except (OSError, StopIteration):
                    continue
            paths += [(arg.decode(errors='replace'), 'argv') for arg in (proc / 'cmdline').read_bytes().split(b'\0')
                      if b'\n' not in arg]
            for path, access in set(paths):
                if path.startswith(prefixes):
                    found.append({'pid': int(proc.name), 'path': path, 'access': access})
        except (OSError, PermissionError):
            continue
    return found


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def inventory(root, output):
    root, output = root.resolve(), output.resolve()
    if (output / 'migration-journal.jsonl').exists():
        raise ValueError('Inventory already has a migration journal; choose a new report directory')
    output.mkdir(parents=True, exist_ok=True)
    tracked = set(subprocess.run(['git', 'ls-files', '-z', 'analysis'], cwd=root,
                  capture_output=True, check=True).stdout.decode().split('\0'))
    files, actions, groups = [], [], []
    duplicates = collections.defaultdict(list)
    for group in sorted((root / 'analysis').iterdir()):
        if not group.is_dir() or group.is_symlink():
            continue
        group_files = []
        for base, dirs, names in os.walk(group, followlinks=False):
            for name in names:
                path = Path(base) / name
                if path.is_symlink():
                    continue
                rel = path.relative_to(root).as_posix()
                row = {'path': rel, 'experiment': group.name, 'tracked': rel in tracked, **snapshot(path)}
                if row['tracked'] and row['size'] <= 2 * 1024**2:
                    row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                    if path.suffix in CODE_SUFFIXES:
                        duplicates[row['sha256']].append(rel)
                files.append(row); group_files.append(row)
        def visit(directory):
            for path in sorted(directory.iterdir()):
                if path.is_symlink():
                    continue
                kind = classification(path, path.is_dir())
                if kind:
                    suffix = path.relative_to(root / 'analysis')
                    source = path.relative_to(root).as_posix()
                    selected = [f for f in group_files if f['path'] == source or f['path'].startswith(source + '/')]
                    actions.append({'category': kind, 'source': source,
                        'destination': (Path(TARGETS[kind]) / suffix).as_posix(),
                        'bytes': sum(f['size'] for f in selected), 'files': len(selected),
                        'directory': path.is_dir(), 'identity': snapshot(path)})
                elif path.is_dir():
                    visit(path)
        visit(group)
        actions.append({'category': 'experiments', 'source': group.relative_to(root).as_posix(),
                        'destination': f'experiments/{group.name}', 'directory': True,
                        'identity': snapshot(group), 'bytes': 0, 'files': 0})
        groups.append({'experiment': group.name, 'files': len(group_files),
                       'bytes': sum(f['size'] for f in group_files),
                       'tracked_files': sum(f['tracked'] for f in group_files)})
    # Assign every file to its destination, including unchanged experiment code.
    for row in files:
        asset = next((a for a in actions if a['category'] != 'experiments' and
                      (row['path'] == a['source'] or row['path'].startswith(a['source'] + '/'))), None)
        row['category'] = asset['category'] if asset else 'experiments'
        row['destination'] = (asset['destination'] + row['path'][len(asset['source']):]
                              if asset else row['path'].replace('analysis/', 'experiments/', 1))
    plan = {'root': str(root), 'created_at_epoch': time.time(), 'groups': groups,
            'files': files, 'actions': actions, 'active_paths': active_paths(root),
            'duplicate_code': [paths for paths in duplicates.values() if len(paths) > 1]}
    (output / 'inventory.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    write_csv(output / 'files.csv', files, ['path', 'category', 'destination', 'size', 'tracked', 'sha256'])
    write_csv(output / 'directories.csv', groups, ['experiment', 'files', 'bytes', 'tracked_files'])
    write_csv(output / 'migration-plan.csv', actions, ['category', 'source', 'destination', 'files', 'bytes'])
    return plan


def journal_append(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def move_one(root, action, journal):
    source, destination = root / action['source'], root / action['destination']
    # Only the inventory's fixed destination roots are accepted.
    if '..' in Path(action['source']).parts or '..' in Path(action['destination']).parts:
        raise ValueError('Parent traversal in plan')
    if not action['source'].startswith('analysis/') or not any(
            action['destination'].startswith(prefix + '/') for prefix in (*TARGETS.values(), 'experiments')):
        raise ValueError('Unexpected migration path')
    if source.is_symlink() and source.resolve() == destination.resolve() and destination.exists():
        return False
    if source.is_symlink() or destination.exists() or destination.is_symlink():
        raise ValueError(f'Conflicting destination or source: {source}')
    current = snapshot(source)
    expected = action['identity']
    checks = ('device', 'inode') if action['directory'] else ('device', 'inode', 'size', 'mtime_ns')
    if any(current[k] != expected[k] for k in checks):
        raise ValueError(f'Source changed since inventory: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise ValueError('Cross-filesystem move refused; no implicit large copy')
    record = {'source': action['source'], 'destination': action['destination'], 'category': action['category']}
    journal_append(journal, dict(record, event='intent'))
    source.rename(destination)
    try:
        source.symlink_to(os.path.relpath(destination, source.parent), target_is_directory=action['directory'])
    except BaseException:
        destination.rename(source)
        raise
    journal_append(journal, dict(record, event='complete'))
    return True


def apply(plan, output, categories):
    root = Path(plan['root'])
    active = [item for item in active_paths(root) if item['access'] != 'reader']
    if active:
        raise RuntimeError('Active analysis paths; stop those runs before migration: ' + str(active))
    moved = 0
    for action in plan['actions']:
        if action['category'] in categories:
            moved += move_one(root, action, output / 'migration-journal.jsonl')
    return moved


def verify(plan, changes=None, canonical=False):
    root = Path(plan['root'])
    failures = []
    changes = {row['path']: row for row in (changes or [])}
    original = {row['path']: row for row in plan['files']}
    for path, change in changes.items():
        if path not in original or not change.get('original_sha256') or change['original_sha256'] != original[path].get('sha256'):
            raise ValueError(f'Change does not match inventoried source: {path}')
    for row in plan['files']:
        path = root / row['destination' if canonical else 'path']
        if not path.is_file():
            failures.append({'path': row['path'], 'error': 'missing'}); continue
        if row['path'] in changes:
            if hashlib.sha256(path.read_bytes()).hexdigest() != changes[row['path']]['sha256']:
                failures.append({'path': row['path'], 'error': 'declared source edit hash changed'})
            continue
        st = path.stat()
        if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) != (row['device'], row['inode'], row['size'], row['mtime_ns']):
            failures.append({'path': row['path'], 'error': 'identity or content metadata changed'})
        elif row.get('sha256') and hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            failures.append({'path': row['path'], 'error': 'content hash changed'})
    return {'verified_files': len(plan['files']) - len(failures), 'failures': failures}


def retire(plan, output, changes=None):
    """Remove only inventoried top-level aliases after verifying canonical files."""
    root = Path(plan['root'])
    active = [item for item in active_paths(root) if item['access'] != 'reader']
    if active:
        raise RuntimeError('Active experiment paths; stop those runs before retiring aliases: ' + str(active))
    report = verify(plan, changes, canonical=True)
    if report['failures']:
        raise ValueError('Canonical archive verification failed: ' + str(report['failures']))
    directory = root / 'analysis'
    if not directory.exists():
        return 0
    expected = {row['source']: root / row['destination'] for row in plan['actions']
                if row['category'] == 'experiments'}
    links = []
    for path in directory.iterdir():
        if path.name == 'README.md' and path.is_file() and not path.is_symlink():
            continue
        destination = expected.get(path.relative_to(root).as_posix())
        if destination is None or not path.is_symlink() or path.resolve() != destination.resolve():
            raise ValueError(f'Unexpected analysis entry; left intact: {path}')
        links.append(path)
    record = {'removed_aliases': [p.relative_to(root).as_posix() for p in links],
              'verified_files': report['verified_files'], 'status': 'intent'}
    record_path = output / 'analysis-retirement.json'
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    for path in links:
        path.unlink()
    (directory / 'README.md').unlink(missing_ok=True)
    directory.rmdir()
    record['status'] = 'complete'
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    return len(links)


def rollback(root, journal):
    if (journal.parent / 'analysis-retirement.json').exists():
        raise RuntimeError('Aliases have been retired; the original layout rollback is no longer applicable')
    active = [item for item in active_paths(root) if item['access'] != 'reader']
    if active:
        raise RuntimeError('Active analysis paths; stop those runs before rollback: ' + str(active))
    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    for row in reversed(entries):
        if row['event'] != 'complete':
            continue
        source, destination = root / row['source'], root / row['destination']
        if source.is_symlink() and source.resolve() == destination.resolve():
            source.unlink(); destination.rename(source)
            journal_append(journal, dict(row, event='rolled_back'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inventory', 'apply', 'verify', 'rollback', 'retire'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--category', action='append', choices=(*TARGETS, 'experiments'))
    parser.add_argument('--changes', type=Path, help='Explicit reviewed source edits after archival (verify/retire)')
    parser.add_argument('--canonical', action='store_true', help='Verify final destinations without compatibility paths')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == 'inventory':
        plan = inventory(args.root, output)
        print(json.dumps({'groups': len(plan['groups']), 'files': len(plan['files']),
              'bytes': sum(f['size'] for f in plan['files']), 'actions': len(plan['actions']),
              'active_paths': plan['active_paths'], 'duplicate_code': plan['duplicate_code']}, ensure_ascii=False))
        return
    plan = json.loads((output / 'inventory.json').read_text())
    if args.command == 'apply':
        if not args.category:
            parser.error('apply requires explicit --category selections')
        print(json.dumps({'moved': apply(plan, output, args.category)}))
    elif args.command == 'rollback':
        rollback(Path(plan['root']), output / 'migration-journal.jsonl')
    elif args.command == 'retire':
        changes = json.loads(args.changes.read_text()) if args.changes else None
        print(json.dumps({'retired_aliases': retire(plan, output, changes)}))
    else:
        changes = json.loads(args.changes.read_text()) if args.changes else None
        report = verify(plan, changes, canonical=args.canonical)
        report['declared_source_edits'] = len(changes or [])
        name = 'retirement-verification.json' if args.canonical else ('final-verification.json' if changes else 'verification.json')
        (output / name).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False))
        if report['failures']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
