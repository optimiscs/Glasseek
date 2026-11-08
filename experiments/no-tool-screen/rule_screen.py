#!/usr/bin/env python3
"""No-tool base inference and local exact-answer screening; never calls a judge API."""
import argparse, asyncio, json, re, sqlite3, time, unicodedata
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parent
SYSTEM = ('Answer the question directly using only your existing knowledge. You have no tools, web access, or filesystem. '
          'Do not call tools, pretend to search, invent sources, or claim to create a file. '
          'Give only the answer, at most 100 words; do not show reasoning, speculate, or repeat yourself. '
          'If you cannot answer reliably from memory, reply exactly UNCERTAIN and stop.')
MARKER = re.compile(r'(?i)\bfinal\s+answer\s*:\s*([^\n\r]+)')

def extract_answer(text):
    # Only explicit final-answer markers outside a teacher's reasoning are eligible.
    visible = re.sub(r'<think\b[^>]*>.*?</think\s*>', '', text, flags=re.I | re.S)
    matches = list(MARKER.finditer(visible))
    if not matches:
        return None, 'no_final_answer_marker'
    value = matches[-1].group(1).strip().strip('*_`').strip()
    value = re.sub(r'^[*_`]+|[*_`]+$', '', value).strip()
    if not value or len(value) > 200 or value.lower() in {'unknown', 'uncertain', 'n/a', 'none'}:
        return None, 'ambiguous_final_answer'
    return value, None

def normalized(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()

def contains_answer(output, answer):
    needle = normalized(answer)
    haystack = normalized(output)
    # Avoid matching a short answer inside another alphanumeric word.
    left = r'(?<!\w)' if needle[0].isalnum() else ''
    right = r'(?!\w)' if needle[-1].isalnum() else ''
    return bool(re.search(left + re.escape(needle) + right, haystack))

def decide(teacher, response, status, finish):
    answer, error = extract_answer(teacher)
    if error:
        return {'decision': 'review', 'reason': error, 'teacher_final_answer': None}
    if status != 'ok':
        return {'decision': 'review', 'reason': 'inference_error', 'teacher_final_answer': answer}
    if contains_answer(response, answer):
        return {'decision': 'remove_match', 'reason': 'literal_answer_in_output', 'teacher_final_answer': answer}
    if finish != 'stop':
        return {'decision': 'review', 'reason': 'incomplete_response', 'teacher_final_answer': answer}
    return {'decision': 'keep_no_match', 'reason': 'literal_answer_not_in_output', 'teacher_final_answer': answer}

async def infer(args):
    questions = [json.loads(s) for s in (ROOT / 'questions.jsonl').open()]
    db = sqlite3.connect(ROOT / 'rule_results_2k.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS answers(question_id TEXT PRIMARY KEY, status TEXT, response TEXT, finish TEXT, error TEXT, elapsed REAL)')
    existing = {r[0] for r in db.execute('SELECT question_id FROM answers WHERE status = "ok"')}
    queue = asyncio.Queue()
    for q in questions:
        if q['question_id'] not in existing: queue.put_nowait(q)
    client = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10), trust_env=False)
    async def worker(gpu):
        while True:
            try: q = queue.get_nowait()
            except asyncio.QueueEmpty: return
            start = time.monotonic(); status = 'error'; response = ''; finish = ''; error = ''
            try:
                payload = {'model': 'glasseek-base-9b', 'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': q['question']}], 'temperature': 0, 'top_p': 1, 'max_tokens': 2048, 'repetition_penalty': 1.1, 'stream': False, 'chat_template_kwargs': {'enable_thinking': False}}
                res = await client.post(f'http://127.0.0.1:{18300+gpu}/v1/chat/completions', json=payload)
                res.raise_for_status(); choice = res.json()['choices'][0]; msg = choice['message']
                response = '\n'.join(str(msg.get(k) or '') for k in ('reasoning_content', 'reasoning', 'content'))
                finish = choice.get('finish_reason') or ''
                if msg.get('tool_calls') or re.search(r'<tool_call>|<function=|\b(?:web_search|web_extract)\s*\(', response): error = 'attempted_tool_call'
                elif not response.strip(): error = 'empty_response'
                else: status = 'ok'
            except Exception as exc: error = type(exc).__name__ + ': ' + str(exc)[:300]
            db.execute('INSERT OR REPLACE INTO answers VALUES(?,?,?,?,?,?)', (q['question_id'], status, response, finish, error, time.monotonic()-start)); db.commit()
            print(json.dumps({'question_id': q['question_id'], 'status': status, 'finish': finish, 'error': error}, ensure_ascii=False), flush=True)
            queue.task_done()
    try: await asyncio.gather(*(worker(g) for g in range(args.gpus) for _ in range(args.per_gpu)))
    finally: await client.aclose(); db.close()

def audit(args):
    db = sqlite3.connect(ROOT / 'rule_results_2k.sqlite3')
    answers = {r[0]: r[1:] for r in db.execute('SELECT question_id,status,response,finish,error FROM answers')}
    counts = {}; out = ROOT / 'rule_decisions_2k.jsonl'
    with out.open('w') as f:
        for line in (ROOT / 'rows.jsonl').open():
            row = json.loads(line); a = answers.get(row['question_id'])
            result = decide(row['teacher_answer'], a[1], a[0], a[2]) if a else {'decision': 'review', 'reason': 'inference_missing', 'teacher_final_answer': None}
            obj = {'source_line': row['source_line'], 'split': row['split'], 'question_id': row['question_id'], **result, 'finish': a[2] if a else None, 'error': a[3] if a else None}
            counts[result['decision']] = counts.get(result['decision'], 0) + 1
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    db.close(); print(json.dumps({'counts': counts, 'output': str(out)}, ensure_ascii=False))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['infer', 'audit']); ap.add_argument('--gpus', type=int, default=4); ap.add_argument('--per-gpu', type=int, default=4); args = ap.parse_args()
    if args.action == 'infer': asyncio.run(infer(args))
    else: audit(args)
