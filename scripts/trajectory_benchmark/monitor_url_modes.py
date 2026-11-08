"""Record vLLM counters during an existing URL-mode experiment (no requests generated)."""
import argparse
import json
from pathlib import Path
import re
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:19109')
    parser.add_argument('--interval', type=float, default=10)
    args = parser.parse_args()
    names = {'num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc',
             'num_preemptions_total', 'prompt_tokens_total', 'generation_tokens_total',
             'prefix_cache_queries_total', 'prefix_cache_hits_total'}
    previous = None
    with (args.output / 'vllm_metrics.jsonl').open('a', buffering=1) as stream:
        while True:
            row = {'timestamp': time.time()}
            try:
                with urllib.request.urlopen(args.endpoint + '/metrics', timeout=5) as response:
                    raw = response.read().decode()
                metrics = {}
                for line in raw.splitlines():
                    match = re.match(r'vllm:(\w+)(?:\{[^}]*\})?\s+(\S+)', line)
                    if match and match[1] in names:
                        metrics[match[1]] = metrics.get(match[1], 0) + float(match[2])
                row.update(metrics)
                if previous and row['timestamp'] > previous['timestamp']:
                    dt = row['timestamp'] - previous['timestamp']
                    for counter, label in [('prompt_tokens_total', 'prompt_tokens_per_second'),
                                           ('generation_tokens_total', 'output_tokens_per_second')]:
                        if counter in row and counter in previous and row[counter] >= previous[counter]:
                            row[label] = (row[counter] - previous[counter]) / dt
                previous = row
            except Exception as exc:
                row['error'] = type(exc).__name__ + ': ' + str(exc)
            stream.write(json.dumps(row) + '\n')
            try:
                status = json.loads((args.output / 'status.json').read_text())
                if all(run['status'] != 'running' for run in status['runs'].values()):
                    break
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(max(1, args.interval))


if __name__ == '__main__':
    main()
