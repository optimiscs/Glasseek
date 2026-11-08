"""Common screening primitives with explicit, behavior-preserving audit policies."""
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


def parse_json(text):
    return json.loads(re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', text.strip(), flags=re.I))


@dataclass(frozen=True)
class ScreeningPolicy:
    name: str
    inference: str
    judge: str
    reaudit_first: bool
    normalize_sentinels: bool
    max_candidates: int
    judge_tokens: int
    judge_thinking: bool
    retry_final_sleep: bool

    def teacher_answer(self, text):
        visible = re.sub(r'<think\b[^>]*>.*?</think\s*>', '', text, flags=re.I | re.S)
        matches = list(re.finditer(r'(?i)\bfinal\s+answer[ \t]*:[ \t]*([^\r\n]*)', visible))
        if not matches:
            return None
        answers = [m.group(1).strip().strip('*_`').strip() for m in matches]
        for answer in answers:
            sentinel = normalize(answer) if self.normalize_sentinels else answer.casefold()
            if not answer or len(answer) > 200 or sentinel in ('unknown', 'uncertain', 'none', 'n/a'):
                return None
        return answers[-1] if len({normalize(a) for a in answers}) == 1 else None

    def parse_candidates(self, text):
        candidates = parse_json(text)['candidates']
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= self.max_candidates:
            raise ValueError('candidate_count')
        for rank, item in enumerate(candidates, 1):
            if (not isinstance(item, dict) or type(item.get('rank')) is not int or item['rank'] != rank
                    or not isinstance(item.get('answer'), str) or not item['answer'].strip()
                    or not isinstance(item.get('aliases'), list)
                    or any(not isinstance(a, str) or not a.strip() for a in item['aliases'])):
                raise ValueError('invalid_candidate')
        if len({normalize(item['answer']) for item in candidates}) != len(candidates):
            raise ValueError('duplicate_candidate')
        return candidates

    def initially_pending(self, qid, rows, answers, completed):
        incomplete = any(row['id'] not in completed for row in rows)
        return qid in answers and incomplete if self.reaudit_first else qid not in answers or incomplete

    async def request(self, client, gpu, payload):
        for attempt in range(3):
            try:
                response = await client.post(f'http://127.0.0.1:{18300 + gpu}/v1/chat/completions', json=payload)
                response.raise_for_status()
                return response.json()['choices'][0]
            except Exception:
                if attempt < 2 or self.retry_final_sleep:
                    await asyncio.sleep(2 ** attempt)
                if attempt == 2:
                    raise


def policy(name):
    values = json.loads(Path(__file__).with_name('screening_profiles.json').read_text())[name]
    return ScreeningPolicy(name=name, **values)
