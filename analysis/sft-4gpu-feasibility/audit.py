"""Read-only hardware/model audit and token counts for every complete local SFT row.

Uses the selected base tokenizer and the pinned verl per-turn rendering helpers.
No model weights are loaded and no CUDA context or training job is started.
"""
import ast
import collections
import hashlib
import json
import math
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path('/root/autodl-tmp/Glasseek')
MODEL = Path('/root/autodl-tmp/models/Qwen3.5-9B')
VERL = Path('/tmp/glasseek-verl-v080')
SOURCE = ROOT/'data/Glasseek-SFT-5.7K/train.jsonl.part'
OUT = Path(__file__).resolve().parent


def load_functions(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, str(path), 'exec'), namespace)


def describe(values):
    return {"count": len(values), "min": int(min(values)), "mean": float(np.mean(values)),
            **{f'p{p}': float(np.percentile(values, p)) for p in (50, 75, 90, 95, 99)},
            "max": int(max(values)), "sum": int(sum(values))}


def model_stats():
    index = json.loads((MODEL/'model.safetensors.index.json').read_text())
    groups = collections.Counter()
    dtypes = collections.Counter()
    for filename in sorted(set(index['weight_map'].values())):
        with (MODEL/filename).open('rb') as file:
            size = struct.unpack('<Q', file.read(8))[0]
            header = json.loads(file.read(size))
        for name, tensor in header.items():
            if name == '__metadata__':
                continue
            n = math.prod(tensor['shape'])
            kind = 'mtp' if 'mtp' in name else 'vision' if 'visual' in name else 'text'
            groups[kind] += n
            dtypes[tensor['dtype']] += n
    return {'parameters_by_component': dict(groups), 'parameters_by_dtype': dict(dtypes),
            'checkpoint_total_bytes': index['metadata'].get('total_size')}


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    namespace = {}
    load_functions(VERL/'verl/utils/tokenizer.py', {'normalize_token_ids'}, namespace)
    load_functions(VERL/'verl/utils/chat_template.py', {'extract_system_prompt_and_generation', 'apply_chat_template'}, namespace)
    system_ids, generation_ids = namespace['extract_system_prompt_and_generation'](tokenizer)
    records, truncated, template_errors = [], None, []
    with SOURCE.open() as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                truncated = {'line': line_no, 'error': error.msg}
                break
            by_role = collections.Counter()
            total = supervised = 0
            for i, message in enumerate(row['messages']):
                # Identical per-turn call and prefix removal to MultiTurnSFTDataset.
                options = {}
                if row.get('enable_thinking') is not None:
                    options['enable_thinking'] = bool(row['enable_thinking'])
                try:
                    result = namespace['apply_chat_template'](
                        tokenizer, messages=[message], tools=row.get('tools') if i == 0 else None,
                        add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors='pt', **options)
                    ids = result['input_ids'][0]
                except Exception as error:
                    if message['role'] != 'system' or i != 0:
                        raise
                    template_errors.append({'line': line_no, 'message_index': i, 'error': str(error)})
                    # Audit-only workaround: keep system first and APPEND an
                    # empty user, then strip that suffix. No training code change.
                    dummy = {'role': 'user', 'content': ''}
                    rendered = tokenizer.apply_chat_template([message, dummy], tools=row.get('tools'),
                        add_generation_prompt=False, tokenize=False, **options)
                    tail = tokenizer.apply_chat_template([dummy], add_generation_prompt=False, tokenize=False, **options)
                    if not rendered.endswith(tail):
                        raise ValueError('Cannot isolate the system prefix for length audit')
                    ids = tokenizer.encode(rendered[:-len(tail)], add_special_tokens=False)
                if i != 0 and message['role'] != 'system':
                    ids = ids[len(system_ids):]
                n = len(ids)
                total += n
                by_role[message['role']] += n
                if message['role'] == 'assistant':
                    supervised += max(n-len(generation_ids), 0)
            whole_ids = namespace['normalize_token_ids'](tokenizer.apply_chat_template(
                row['messages'], tools=row.get('tools'), add_generation_prompt=False, tokenize=True))
            records.append({'line': line_no, 'sft_tokens_per_turn_rendering': total,
                            'supervised_assistant_tokens': supervised, 'whole_chat_template_tokens': len(whole_ids),
                            'message_count': len(row['messages']), 'tokens_by_role': dict(by_role)})
    lengths = [r['sft_tokens_per_turn_rendering'] for r in records]
    stats = model_stats()
    nparams = sum(stats['parameters_by_component'].values())
    stats['fp32_adamw_states_per_rank_gib_4gpu_16_bytes_per_param'] = nparams*16/4/2**30
    stats['fp32_adamw_text_only_per_rank_gib_4gpu'] = stats['parameters_by_component']['text']*16/4/2**30
    cfg = json.loads((MODEL/'config.json').read_text())['text_config']
    h, layers, vocab = cfg['hidden_size'], cfg['num_hidden_layers'], cfg['vocab_size']
    stats['sequence_memory_indicators_gib'] = {
        str(length): {'one_bf16_hidden_state': length*h*2/2**30,
                      'bf16_checkpoint_inputs_one_per_layer': length*h*2*layers/2**30,
                      'bf16_dense_vocab_logits': length*vocab*2/2**30}
        for length in [int(np.median(lengths)), max(lengths), 65536, 131072]}
    report = {'measured_at_utc': datetime.now(timezone.utc).isoformat(), 'source': str(SOURCE),
              'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'tokenizer': str(MODEL), 'tokenizer_sha256': hashlib.sha256((MODEL/'tokenizer.json').read_bytes()).hexdigest(),
              'coverage': 'Complete prefix records only, not a random sample of the full dataset',
              'whole_chat_matches_per_turn_lengths': all(r['whole_chat_template_tokens'] == r['sft_tokens_per_turn_rendering'] for r in records),
              'gpu_topology': subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True),
              'truncated_record': truncated, 'current_loader_template_errors': template_errors,
              'length_method_note': 'per-turn rendering; system-only turns need an audit-only workaround because the current pinned loader fails',
              'sft_sequence_tokens': describe(lengths),
              'assistant_supervised_tokens': describe([r['supervised_assistant_tokens'] for r in records]),
              'exceeds_tokens': {str(t): sum(n>t for n in lengths) for t in (4096,8192,16384,32768,65536,98304,131072)},
              'role_token_totals': dict(sum((collections.Counter(r['tokens_by_role']) for r in records), collections.Counter())),
              'model': stats,
              'gpu_snapshot': subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,memory.free', '--format=csv'], text=True),
              'records': records}
    (OUT/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records', 'current_loader_template_errors')}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
