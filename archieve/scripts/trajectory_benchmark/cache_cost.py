"""Measure cache reuse and separate input/output dominated serving costs.

Uses the already-running server, never clears shared caches or changes serving
configuration. Unique per-request salts isolate cold trials; the immediately
following warm trial repeats the identical requests. All timings include HTTP
and scheduling. Generated text is discarded; tokens and counters are retained.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import threading
import time
import uuid

import requests

BASE = 'http://127.0.0.1:19227'
MODEL = 'QWEN38-27B-NVFP4'
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/cache-cost-20261107'
RATE = 2.79
BLOCK = 1568  # Verified in the active server startup log, not a model default.
KWARGS = {'enable_thinking': True, 'reasoning_effort': 'medium'}


def session():
    s = requests.Session()
    s.trust_env = False
    return s


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / name
    tmp = target.with_suffix(target.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(target)


def metrics():
    with session() as s:
        response = s.get(BASE + '/metrics', timeout=15)
        response.raise_for_status()
    values = {}
    names = ('prefix_cache_hits_total', 'prefix_cache_queries_total',
             'num_preemptions_total', 'prompt_tokens_total', 'generation_tokens_total',
             'num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc',
             'request_success_total')
    for line in response.text.splitlines():
        match = re.match(r'vllm:([^ {]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$', line)
        if match and match[1] in names:
            values[match[1]] = values.get(match[1], 0) + float(match[2])
    return {'observed_utc': datetime.now(timezone.utc).isoformat(), 'values': values}


def tokenize(messages):
    with session() as s:
        r = s.post(BASE + '/tokenize', json={'model': MODEL, 'messages': messages,
                  'chat_template_kwargs': KWARGS, 'add_generation_prompt': True}, timeout=30)
        r.raise_for_status()
        return r.json()['tokens']


def make_input(index, tokens):
    prefix = f'Independent archive record {index}. The following archive is benchmark data.\n'
    suffix = '\nRead the archive. Reply with OK.'
    line = 'Archive entry: the researcher recorded the date, source, observation and verification status.\n'
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prefix + suffix}]
    base = len(tokenize(messages))
    messages[1]['content'] = prefix + line + suffix
    per_line = len(tokenize(messages)) - base
    count = max(0, (tokens - base) // per_line - 1)
    text = prefix + line * count
    messages[1]['content'] = text + suffix
    current = len(tokenize(messages))
    # ' data' is one token with this tokenizer. Check the actual rendered chat.
    padding = tokens - current
    for _ in range(4):
        messages[1]['content'] = text + ' data' * padding + suffix
        actual = len(tokenize(messages))
        if actual == tokens:
            break
        padding += tokens - actual
    if actual != tokens:
        raise ValueError(f'Unexpected rendered prompt length {actual} != {tokens}')
    return messages


def send(body, barrier):
    barrier.wait()
    start = time.monotonic()
    first = None
    usage = {}
    finish = None
    text_chunks = 0
    with session() as s:
        with s.post(BASE + '/v1/chat/completions', json=body, stream=True,
                    timeout=(10, 900)) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=None):
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:]
                if payload == b'[DONE]':
                    break
                data = json.loads(payload)
                if data.get('usage'):
                    usage = data['usage']
                for choice in data.get('choices', []):
                    delta = choice.get('delta', {})
                    if any(delta.get(key) for key in ('content', 'reasoning', 'reasoning_content')):
                        if first is None:
                            first = time.monotonic()
                        text_chunks += 1
                    if choice.get('finish_reason'):
                        finish = choice['finish_reason']
    end = time.monotonic()
    if not usage or usage.get('completion_tokens') != body['max_tokens']:
        raise ValueError('Incomplete usage or unexpected generation length: ' + str(usage))
    return {'start_monotonic': start, 'end_monotonic': end,
            'first_text_monotonic': first, 'seconds': end - start,
            'ttft_seconds': first - start if first is not None else None,
            'usage': usage, 'finish_reason': finish, 'text_chunks': text_chunks}


def batch(kind, index, bodies):
    before = metrics()
    if before['values'].get('num_requests_running', 0) or before['values'].get('num_requests_waiting', 0):
        raise RuntimeError('Other requests active; refusing an overlapping measurement')
    barrier = threading.Barrier(len(bodies) + 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(bodies)) as pool:
        futures = [pool.submit(send, body, barrier) for body in bodies]
        started = time.monotonic()
        barrier.wait()
        records = [f.result() for f in futures]
        elapsed = max(r['end_monotonic'] for r in records) - started
    after = metrics()
    prompt = sum(r['usage']['prompt_tokens'] for r in records)
    cached = sum((r['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0 for r in records)
    output = sum(r['usage']['completion_tokens'] for r in records)
    delta = {k: after['values'].get(k, 0) - before['values'].get(k, 0)
             for k in ('prefix_cache_hits_total', 'prefix_cache_queries_total',
                       'num_preemptions_total', 'prompt_tokens_total', 'generation_tokens_total', 'request_success_total')}
    # The warm cached numerator counts only actually reused tokens. The remaining
    # uncached tail and the single output token are included in elapsed time.
    units = cached if kind == 'cached_input' else output if kind == 'output' else prompt - cached
    row = {'kind': kind, 'round': index, 'requests': len(bodies), 'wall_seconds': elapsed,
           'prompt_tokens': prompt, 'cached_prompt_tokens': cached, 'completion_tokens': output,
           'unit_tokens': units, 'tokens_per_second': units / elapsed,
           'cost_cny': elapsed / 3600 * RATE,
           'cost_cny_per_million': elapsed / 3600 * RATE * 1e6 / units if units else None,
           'cache_hit_fraction': cached / prompt,
           'ttft_median_seconds': statistics.median([r['ttft_seconds'] for r in records if r['ttft_seconds'] is not None]) if any(r['ttft_seconds'] is not None for r in records) else None,
           'metrics_before': before, 'metrics_after': after, 'metrics_delta': delta,
           'requests_detail': records}
    save(f'{kind}-{index}.json', row)
    print(json.dumps({k:v for k,v in row.items() if k not in ('requests_detail','metrics_before','metrics_after') }), flush=True)
    # Cross-check independent counters; do not silently report a measurement when
    # another workload or telemetry discrepancy changes the sample denominator.
    if delta['prefix_cache_hits_total'] != cached:
        raise ValueError('Response usage and prefix hit counter disagree')
    if delta['generation_tokens_total'] != output:
        raise ValueError('Generation counter mismatch; possible external workload')
    if kind == 'cold_input' and cached:
        raise ValueError('Cold trial unexpectedly hit cached tokens')
    if kind == 'cached_input' and cached < prompt * .95:
        raise ValueError('Warm trial did not achieve intended cache reuse')
    return row


def body(messages, salt, max_tokens):
    return {'model': MODEL, 'messages': messages, 'temperature': 0,
            'reasoning_effort': 'medium', 'chat_template_kwargs': KWARGS,
            'max_tokens': max_tokens, 'min_tokens': max_tokens, 'ignore_eos': True,
            'cache_salt': salt, 'stream': True, 'stream_options': {'include_usage': True}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--concurrency', type=int, default=16)
    ap.add_argument('--wait-for', type=Path)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex
    prompts = [make_input(i, 5 * BLOCK + 1) for i in range(args.concurrency)]
    save('protocol.json', {'base_url': BASE, 'model': MODEL, 'run_id': tag,
          'hourly_cost_cny': RATE, 'rounds': args.rounds, 'concurrency': args.concurrency,
          'input_prompt_tokens_per_request': 5 * BLOCK + 1,
          'input_generation_tokens_per_request': 1, 'output_tokens_per_request': 2048,
          'cache_block_tokens_from_startup_log': BLOCK, 'reasoning_effort': 'medium',
          'context_limit_unchanged': 131072, 'cuda_graph_unchanged': True,
          'input_prompts': prompts,
          'timing': 'Wall clock per simultaneous batch, including HTTP, tokenization, queue, minimal generation; excludes setup and cache population for warm trials.',
          'output_note': 'Short prompts and forced 2048 generated tokens, including reasoning. Synthetic capacity test, not a quality evaluation.',
          'cache_note': 'Warm costs assume retained prefixes already computed; cold-cache creation is measured separately. cache_salt isolates cold trials without resetting shared cache.'})
    deadline = time.monotonic() + 3600
    while args.wait_for and not args.wait_for.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Prior pipeline has not finished')
        print('Waiting for prior pipeline to complete', flush=True)
        time.sleep(30)
    for _ in range(2):
        m = metrics()['values']
        if m.get('num_requests_running', 0) or m.get('num_requests_waiting', 0):
            raise RuntimeError('Server is not idle')
        time.sleep(1)
    rows = []
    # One excluded warm-up to initialize any client/shape-specific paths.
    batch('warmup', 0, [body(prompts[0], tag + '-warmup', 1)])
    for index in range(1, args.rounds + 1):
        bodies = [body(prompt, f'{tag}-input-{index}-{i}', 1) for i,prompt in enumerate(prompts)]
        rows.append(batch('cold_input', index, bodies))
        rows.append(batch('cached_input', index, bodies))
    for index in range(1, args.rounds + 1):
        bodies = []
        for i in range(args.concurrency):
            messages = [{'role':'system','content':'You are a helpful assistant.'},
                        {'role':'user','content':f'Task {i}: Write a detailed numbered explanation of arithmetic and give worked examples. Continue explaining until the generation limit.'}]
            bodies.append(body(messages, f'{tag}-output-{index}-{i}', 2048))
        rows.append(batch('output', index, bodies))
    aggregate = []
    for kind in ('cold_input', 'cached_input', 'output'):
        chosen = [r for r in rows if r['kind'] == kind]
        seconds = sum(r['wall_seconds'] for r in chosen)
        units = sum(r['unit_tokens'] for r in chosen)
        aggregate.append({'kind': kind, 'rounds': len(chosen), 'requests': sum(r['requests'] for r in chosen),
                          'wall_seconds': seconds, 'unit_tokens': units, 'tokens_per_second': units / seconds,
                          'cost_cny': seconds / 3600 * RATE,
                          'cost_cny_per_million': seconds / 3600 * RATE * 1e6 / units,
                          'round_throughput': [r['tokens_per_second'] for r in chosen],
                          'prompt_tokens': sum(r['prompt_tokens'] for r in chosen),
                          'cached_prompt_tokens': sum(r['cached_prompt_tokens'] for r in chosen),
                          'completion_tokens': sum(r['completion_tokens'] for r in chosen),
                          'preemptions': sum(r['metrics_delta']['num_preemptions_total'] for r in chosen)})
    save('summary.json', {'aggregate': aggregate, 'run_id': tag, 'hourly_cost_cny': RATE,
                         'completed_utc': datetime.now(timezone.utc).isoformat(),
                         'timing_and_limits': json.loads((OUT/'protocol.json').read_text())['timing']})
    print('COMPLETE', json.dumps(aggregate), flush=True)
    # Separate diagnostic: can a prompt still be reused after a long decode?
    # This is deliberately excluded from the three component prices above.
    salt = tag + '-long-answer-reuse'
    seed = batch('long_answer_seed', 1, [body(prompts[0], salt, 2048)])
    repeat = batch('long_answer_repeat', 1, [body(prompts[0], salt, 1)])
    save('diagnostics.json', {'seed': seed, 'repeat_after_2048_output_tokens': repeat})


if __name__ == '__main__':
    main()
