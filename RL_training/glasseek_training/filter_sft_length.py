"""Select whole SFT trajectories by the actual verl per-turn tokenization."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

_tokenizer = None
_dataset = None


def initialize(model):
    global _tokenizer, _dataset
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    from transformers import AutoTokenizer
    from verl.utils.tokenizer.chat_template import extract_system_prompt_and_generation
    from .sft_dataset import GlasseekSFTDataset
    torch.set_num_threads(1)
    _tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    _dataset = object.__new__(GlasseekSFTDataset)
    _dataset.tokenizer = _tokenizer
    _dataset.processor = None
    _dataset.apply_chat_template_kwargs = {}
    _dataset.system_prompt, _dataset.generation_prompt = extract_system_prompt_and_generation(_tokenizer)


def process(item):
    from .prepare_data import convert_record
    line_no, raw, cap, seed, val_ratio = item
    try:
        record = json.loads(raw)
        row, question = convert_record(record, 'sft')
        # Training config's default enable_thinking_default='none' is truthy in
        # verl's loader; explicit values are converted to bool in __getitem__.
        enable = bool(record.get('enable_thinking', 'none'))
        total = 0
        for index, message in enumerate(record['messages']):
            ids, _, _, _ = _dataset._process_single_message(
                index, message, record['messages'], record.get('tools') if index == 0 else None, enable)
            total += len(ids)
            if total > cap:
                return (line_no, total, False, None, None, None)
        digest = hashlib.sha256(f'{seed}:{question}'.encode()).digest()
        split = 'val' if int.from_bytes(digest[:8], 'big') / 2**64 < val_ratio else 'train'
        return (line_no, total, True, split, row, hashlib.sha256(question.encode()).hexdigest())
    except Exception as e:
        raise RuntimeError(f'Line {line_no}: {type(e).__name__}: {e}') from e


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--max-length', type=int, default=43008)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-ratio', type=float, default=.05)
    args = parser.parse_args()
    if args.max_length <= 0 or args.workers <= 0 or not 0 < args.val_ratio < 1:
        raise ValueError('Invalid max-length, workers or validation ratio')
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = [args.output / name for name in ('train.parquet', 'val.parquet', 'manifest.json', 'selection.jsonl')]
    if any(p.exists() for p in outputs):
        raise FileExistsError('Output already exists; choose a new directory')
    schema = pa.schema([('messages', pa.string()), ('tools', pa.string()), ('enable_thinking', pa.bool_())])
    counts = {'train': 0, 'val': 0, 'excluded': 0}
    questions = {'train': set(), 'val': set()}
    buffers = {'train': [], 'val': []}
    max_kept = 0
    temp = {key: args.output / f'{key}.parquet.tmp' for key in ('train', 'val')}
    writers = {key: pq.ParquetWriter(path, schema=schema) for key, path in temp.items()}
    try:
        with args.input.open(encoding='utf-8') as source, (args.output/'selection.jsonl.tmp').open('w') as audit:
            work = ((i, line, args.max_length, args.seed, args.val_ratio) for i, line in enumerate(source, 1) if line.strip())
            with mp.Pool(args.workers, initializer=initialize, initargs=(str(args.model),)) as pool:
                for n, (line_no, length, keep, split, row, question_hash) in enumerate(pool.imap(process, work, chunksize=2), 1):
                    audit.write(json.dumps({'source_line': line_no, 'tokens': length,
                                            'exact': keep, 'split': split}, ensure_ascii=False)+'\n')
                    if keep:
                        counts[split] += 1
                        max_kept = max(max_kept, length)
                        questions[split].add(question_hash)
                        buffers[split].append(row)
                        if len(buffers[split]) >= 64:
                            writers[split].write_table(pa.Table.from_pylist(buffers[split], schema=schema))
                            buffers[split].clear()
                    else:
                        counts['excluded'] += 1
                    if n % 250 == 0:
                        print(f'Processed {n}: {counts}', flush=True)
        for split in buffers:
            if buffers[split]:
                writers[split].write_table(pa.Table.from_pylist(buffers[split], schema=schema))
    finally:
        for writer in writers.values():
            writer.close()
    if min(counts['train'], counts['val']) == 0 or questions['train'] & questions['val']:
        raise ValueError('Invalid empty or overlapping question split')
    for split in temp:
        temp[split].rename(args.output/f'{split}.parquet')
    (args.output/'selection.jsonl.tmp').rename(args.output/'selection.jsonl')
    result = {'source': str(args.input.resolve()), 'source_sha256': file_hash(args.input),
              'tokenizer': str(args.model.resolve()), 'tokenizer_sha256': file_hash(args.model/'tokenizer.json'),
              'max_length': args.max_length, 'tokenization': 'verl 0.9.1 MultiTurnSFTDataset per-message; Qwen system fallback; no truncation',
              'seed': args.seed, 'val_ratio': args.val_ratio, 'split_unit': 'normalized first user question',
              'counts': counts, 'unique_questions': {k:len(v) for k,v in questions.items()},
              'maximum_retained_tokens': max_kept,
              'output_sha256': {k:file_hash(args.output/f'{k}.parquet') for k in ('train','val')}}
    (args.output/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
