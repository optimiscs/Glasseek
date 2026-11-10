"""Strict native finish contract, deterministic benchmark rendering and audit records."""
import copy
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit
from .prompts import dataset_name

FINISH_TOOL = {'type': 'function', 'function': {
    'name': 'finish', 'description': 'Submit the complete verified deliverable and terminate research. Must be called alone. This does not judge correctness.',
    'strict': True, 'parameters': {'type': 'object', 'additionalProperties': False,
        'required': ['answer', 'evidences', 'confidence'], 'properties': {
            'answer': {'type': 'string', 'minLength': 1, 'description': 'Exact short answer, or FULL report/table when requested.'},
            'evidences': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                'required': ['evidence', 'url'], 'properties': {'evidence': {'type': 'string', 'minLength': 1},
                'url': {'type': 'string', 'pattern': '^https?://[^\\s]+$'}}}},
            'confidence': {'type': 'number', 'minimum': 0, 'maximum': 100}}}}}
REVIEW_TOOL = {'type': 'function', 'function': {'name': 'review_candidate', 'strict': True,
    'description': 'Assess support and specify remaining work, without access to gold answers.',
    'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['accept', 'issues', 'next_steps'],
        'properties': {'accept': {'type': 'boolean'}, 'issues': {'type': 'array', 'items': {'type': 'string'}},
                       'next_steps': {'type': 'array', 'items': {'type': 'string'}}}}}}


def enabled():
    return os.environ.get('GLASSEEK_RESEARCH_PROTOCOL') == 'arex-native'


def native_kwargs(kwargs):
    """Applied AFTER transport overrides, so generic config cannot disable constraints."""
    if not enabled():
        return kwargs
    result = dict(kwargs)
    allowed = {'web_search', 'web_extract', 'execute_code', 'vision_analyze'}
    result['tools'] = [copy.deepcopy(t) for t in kwargs.get('tools', []) if t.get('function', {}).get('name') in allowed]
    for tool in result['tools']:
        function = tool['function']
        if function['name'] == 'web_extract':
            function['description'] = ('Read 1 to 5 HTTP(S) webpages/PDFs. Returns results with url, content, error and retrieval metadata. '
                'Content consists of original passages selected using BM25 plus embeddings, never an LLM summary; '
                'selection may omit other passages. Use a specific goal to retrieve relevant original evidence. '
                'On failure change source or query; no browser tool is provided.')
            goal = function.get('parameters', {}).get('properties', {}).get('goal')
            if goal is not None:
                goal['description'] = 'Specific facts or original evidence to locate. If omitted, selection uses the task question.'
    result['tools'].append(copy.deepcopy(FINISH_TOOL))
    result['tool_choice'] = 'required'
    result['parallel_tool_calls'] = False
    return result


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON field: {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'invalid JSON constant: {value}')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def observed_urls(messages):
    found = set()

    def scan(value):
        if isinstance(value, dict):
            for item in value.values():
                scan(item)
        elif isinstance(value, list):
            for item in value:
                scan(item)
        elif isinstance(value, str):
            # Structured URL fields are authoritative: parentheses, commas,
            # query parameters and signatures are part of the exact address.
            if re.fullmatch(r'https?://[^\s<>"\']+', value):
                found.add(value)
                return
            for url in re.findall(r'https?://[^\s<>"\']+', value):
                url = url.rstrip('.,;:')
                # Remove Markdown/prose closing delimiters only when unmatched.
                while url.endswith(')') and url.count(')') > url.count('('):
                    url = url[:-1]
                while url.endswith(']') and url.count(']') > url.count('['):
                    url = url[:-1]
                found.add(url)

    for message in messages:
        if message.get('role') != 'tool':
            continue
        content = message.get('content') or ''
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (ValueError, TypeError):
                pass
        scan(content)
    return found


def validate_finish(raw, dataset, messages):
    name = dataset_name(dataset)
    value = strict_json(raw)
    if not isinstance(value, dict) or set(value) != {'answer', 'evidences', 'confidence'}:
        raise ValueError('finish requires exactly answer, evidences, confidence')
    if not isinstance(value['answer'], str) or not value['answer'].strip():
        raise ValueError('answer must be nonempty text')
    confidence = value['confidence']
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 100:
        raise ValueError('confidence must be a finite number from 0 to 100')
    if not isinstance(value['evidences'], list):
        raise ValueError('evidences must be an array')
    seen = observed_urls(messages)
    for item in value['evidences']:
        if not isinstance(item, dict) or set(item) != {'evidence', 'url'}:
            raise ValueError('each evidence must contain exactly evidence and url')
        if not all(isinstance(v, str) and v.strip() for v in item.values()):
            raise ValueError('evidence and url must be nonempty strings')
        u = urlsplit(item['url'])
        if u.scheme not in ('http', 'https') or not u.hostname or u.username or re.search(r'\s', item['url']):
            raise ValueError('evidence URL must be HTTP(S) without credentials or whitespace')
        if item['url'] not in seen:
            raise ValueError('evidence URL not present in observed tool results')
    if name not in {'gaia', 'hle'} and not value['evidences']:
        raise ValueError('external research requires source-backed evidences')
    return value


def render_finish(value, dataset):
    """Preserve report/table bytes; render only established short-answer envelopes."""
    name = dataset_name(dataset)
    answer = value['answer']
    explanation = '\n'.join(f"{e['evidence']} ({e['url']})" for e in value['evidences'])
    if name in {'hle', 'browsecomp'}:
        label = 'Answer' if name == 'hle' else 'Exact Answer'
        return f"Explanation: {explanation or 'Self-contained calculation or reasoning; see trajectory.'}\n{label}: {answer}\nConfidence: {value['confidence']:g}%"
    return answer


def write_record(record):
    target = os.environ.get('GLASSEEK_RESEARCH_RESULT')
    if not target:
        return
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
