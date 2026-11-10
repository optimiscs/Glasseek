"""Audited runtime fixes for pinned author evaluators; frozen sources stay intact."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
sys.dont_write_bytecode = True

VERSION = 'native-edge-cases-v1'


def fix_widesearch(module):
    """An empty DataFrame.apply(axis=1) is not a tuple Series on pandas 2.x."""
    path = Path(module.__file__);tree = ast.parse(path.read_text())
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'evaluate_single_query')
    class Fix(ast.NodeTransformer):
        count = 0
        def visit_Assign(self, node):
            self.generic_visit(node)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'metric_info_series':
                # Keep every nonempty result untouched. Only give a typed empty
                # Series to the author's existing row/item calculations.
                guard = ast.parse('if df_inner.empty:\n    metric_info_series = pd.Series(index=df_inner.index, dtype=object)').body[0]
                self.count += 1
                return [node, guard]
            return node
    fix = Fix();function = fix.visit(function)
    if fix.count != 2:raise ValueError('WideSearch source drift: expected numeric and LLM metric assignments')
    code = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(code, str(path), 'exec'), module.__dict__)
    return {'id': VERSION + ':widesearch-empty-intersection', 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def fact_statistics(source, input_path, output_path):
    """Delegate defined ratios to upstream; represent zero denominators as null."""
    records = [json.loads(line) for line in Path(input_path).read_text().splitlines() if line.strip()]
    n = 0;citations = 0;supported = 0
    for row in records:
        if not isinstance(row.get('citations'), list):raise ValueError('FACT extraction failed; cannot treat as no citations')
        if not row['citations']:continue
        n += 1
        for item in row.get('citations_deduped', {}).values():
            if item.get('validate_error') is not None:continue
            for verdict in item.get('validate_res', []):
                if verdict['result'] != 'unknown':
                    citations += 1;supported += verdict['result'] == 'supported'
    status = {'reports': len(records), 'reports_with_citations': n, 'verified_citations': citations,
              'supported_citations': supported, 'citation_coverage': n / len(records) if records else None,
              'status': 'defined' if n and citations else 'undefined_no_citations' if not n else 'undefined_no_verifiable_citations',
              'runtime_fix': VERSION, 'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest()}
    Path(output_path).with_suffix('.status.json').write_text(json.dumps(status, indent=2))
    if n and citations:
        # Upstream uses the generic package name `utils`. Run in a fresh
        # interpreter so other evaluators' modules cannot shadow that package.
        subprocess.run([sys.executable, '-c',
            'import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); '
            'source=sys.argv.pop(1); sys.argv[0]=source; '
            'runpy.run_path(source,run_name="__main__")',
            str(Path(source).resolve().parent.parent), str(Path(source).resolve()),
            '--input_path', str(Path(input_path).resolve()),
            '--output_path', str(Path(output_path).resolve())], check=True)
    else:
        values = {'total_citations': citations / n if n else None,
                  'total_valid_citations': supported / n if n else None, 'valid_rate': None}
        Path(output_path).write_text(''.join(f'{k}: {v if v is not None else "undefined"}\n' for k, v in values.items()))
        print(json.dumps(status), flush=True)
    return status


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('action', choices=['fact-stat'])
    p.add_argument('--source', type=Path, required=True);p.add_argument('--input_path', type=Path, required=True)
    p.add_argument('--output_path', type=Path, required=True)
    a = p.parse_args();fact_statistics(a.source, a.input_path, a.output_path)
