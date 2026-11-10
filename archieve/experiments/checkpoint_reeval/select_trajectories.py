"""Extract judged, completed RL9B sessions with explicit provenance."""
import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from browse_comp_eval.session_reader import load_session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run_dir / 'judge_official_summary.jsonl').open() if line.strip()]
    out, excluded, seen = [], Counter(), set()
    for judge in rows:
        if judge.get('status') != 'judged' or judge.get('pred_status') != 'success' or judge.get('score') not in (0, 1):
            excluded['not_judged_success'] += 1
            continue
        run_id = judge['run_id']
        if run_id in seen:
            raise ValueError(f'duplicate judged run: {run_id}')
        pred_path = Path(judge['pred_path'])
        pred = json.loads(pred_path.read_text())
        if pred.get('status') != 'success' or pred.get('timed_out') or not pred.get('trace_complete'):
            excluded['incomplete_or_timed_out'] += 1
            continue
        if pred.get('run_id') != run_id or pred.get('attempt') != judge.get('pred_attempt'):
            raise ValueError(f'judge/prediction mismatch: {run_id}')
        session = load_session(pred.get('session_id'), pred.get('profile'))
        if not session or not session.get('messages'):
            excluded['missing_session'] += 1
            continue
        seen.add(run_id)
        out.append({'run_id': run_id, 'question_id': judge['question_id'], 'score': judge['score'],
                    'label': 'success' if judge['score'] == 1 else 'failure',
                    'prediction_sha256': hashlib.sha256(pred_path.read_bytes()).hexdigest(),
                    'prediction_path': str(pred_path), 'judge_path': judge.get('result_path'),
                    'session_id': pred.get('session_id'), 'profile': pred.get('profile'),
                    'question': pred.get('question'), 'answer': pred.get('answer'), 'session': session})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as f:
        for row in sorted(out, key=lambda r: r['run_id']):
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'selected': len(out), 'labels': dict(Counter(r['label'] for r in out)),
                      'excluded': dict(excluded), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
