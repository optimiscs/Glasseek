"""Stage-specific OpenAI-compatible transport. No credentials in logs or manifests."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import random
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlparse


class ProviderError(RuntimeError):
    def __init__(self, status):
        self.status = status
        self.fatal = status in (401, 402, 403)
        super().__init__('API HTTP ' + str(status))

import requests


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def resolve_key(value):
    match = re.fullmatch(r'\$\{([A-Z0-9_]+)\}', value)
    if match:
        key = os.getenv(match[1], '')
        if not key:
            raise ValueError('Missing credential environment variable: ' + match[1])
        return key
    return value


def public_config(config):
    result = copy.deepcopy(config)
    for provider in result['providers'].values():
        provider['key'] = '[redacted]'
    return result


class Client:
    def __init__(self, config, output, policy_hash, *, privacy_mode=False, scrubber=None):
        if privacy_mode and scrubber is None:
            raise ValueError('Privacy mode requires a scrubber')
        self.privacy_mode, self.scrubber = privacy_mode, scrubber
        self.config, self.output, self.policy_hash = config, Path(output), policy_hash
        self.output.mkdir(parents=True, exist_ok=True)
        self.models, self.lock = {}, threading.Lock()
        self.provider_lock = threading.Lock()
        self.provider_blocked = {}
        self.provider_limits = {p['url']: threading.BoundedSemaphore(p.get('max_concurrent_requests',config['execution']['workers'])) for p in config['providers'].values()}

    def request(self, provider, method, suffix, body=None):
        url = provider['url'].rstrip('/')
        if urlparse(url).scheme not in ('http', 'https') or urlparse(url).username or urlparse(url).query:
            raise ValueError('Use a plain HTTP(S) API base URL; put credentials in key')
        key = resolve_key(provider.get('key', ''))
        credential_scope = (url, hashlib.sha256(key.encode()).hexdigest())
        limit = self.provider_limits[provider['url']]
        with limit, requests.Session() as session:
            session.trust_env = provider.get('trust_env', False)
            for transport_attempt in range(1, provider.get('transport_attempts', 4) + 1):
                with self.provider_lock:
                    if credential_scope in self.provider_blocked:
                        raise ProviderError(self.provider_blocked[credential_scope])
                response = session.request(method, url + suffix,
                    headers={'Authorization': 'Bearer ' + key} if key else {}, json=body,
                    timeout=(15, provider.get('timeout_seconds', 180)))
                status = response.status_code
                if status == 200:
                    result = response.json()
                    result['_transport_attempts'] = transport_attempt
                    return result
                if status in (401, 402, 403):
                    with self.provider_lock:
                        self.provider_blocked[credential_scope] = status
                    raise ProviderError(status)
                if status not in (429, 500, 502, 503, 504) or transport_attempt == provider.get('transport_attempts', 4):
                    raise ProviderError(status)
                try:
                    delay = float(response.headers.get('Retry-After', 2 ** transport_attempt))
                except ValueError:
                    delay = 2 ** transport_attempt
                time.sleep(min(60, max(1, delay) + random.random()))

    def endpoint(self, stage):
        spec = self.config['stages'][stage]
        name = spec['provider']
        provider = self.config['providers'][name]
        model = spec.get('model', provider.get('model', 'auto'))
        if model == 'auto':
            with self.lock:
                if name not in self.models:
                    body = self.request(provider, 'GET', '/models')
                    available = sorted({m['id'] for m in body.get('data', []) if isinstance(m.get('id'), str)})
                    preferred = [m for m in provider.get('preferred_models', []) if m in available]
                    if preferred:
                        selected = preferred[0]
                    elif len(available) == 1:
                        selected = available[0]
                    else:
                        raise ValueError('Model discovery ambiguous/unavailable; set provider.model explicitly')
                    self.models[name] = selected
                model = self.models[name]
        return provider, model

    def call(self, stage, ident, prompt, system, validator):
        if self.privacy_mode:
            prompt, system = self.scrubber(prompt), self.scrubber(system)
        provider, model = self.endpoint(stage)
        extra = provider.get('extra_body_by_host', {}).get(urlparse(provider['url']).hostname, {})
        if set(extra) & {'model', 'messages'}:
            raise ValueError('extra_body cannot override model or messages')
        generation = {**self.config['generation'], **self.config['stages'][stage].get('generation', {})}
        fingerprint = digest([stage, model, provider['url'], extra, generation, system, prompt] +
                             ([self.policy_hash, 'privacy-v1'] if self.privacy_mode else []))
        cache = self.output / 'cache' / digest(ident)[:24] / (stage + '.json')
        if cache.exists():
            cached_text = cache.read_text()
            saved = json.loads(cached_text) if not self.privacy_mode or self.scrubber(cached_text) == cached_text else {}
            if saved.get('fingerprint') == fingerprint:
                # Revalidate raw JSON, not previously-mutated validator output.
                try:
                    return validator(copy.deepcopy(saved['raw']))
                except (ValueError,KeyError,TypeError):
                    pass  # Changed validators may reject old output; retry this stage only.
        feedback = ''
        for attempt in range(1, generation.get('attempts', 2) + 1):
            started = time.monotonic()
            record = {'id': ident, 'stage': stage, 'model': model, 'attempt': attempt, 'fingerprint': fingerprint,
                      'started_at': datetime.now(timezone.utc).isoformat(), 'max_tokens': generation.get('max_tokens', 10000)}
            try:
                body = self.request(provider, 'POST', '/chat/completions', {
                    'model': model, 'temperature': generation.get('temperature', 0),
                    'max_tokens': generation.get('max_tokens', 10000),
                    'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt + feedback}], **extra})
                record['usage'] = body.get('usage', {})
                record['transport_attempts'] = body.get('_transport_attempts', 1)
                record['returned_model'] = body.get('model')
                choice = body['choices'][0]
                record['finish_reason'] = choice.get('finish_reason')
                record['reasoning_chars'] = len(choice['message'].get('reasoning') or choice['message'].get('reasoning_content') or '')
                record['raw_response'] = choice['message'].get('content') or ''
                if self.privacy_mode:
                    original = record.pop('raw_response')
                    sanitized = self.scrubber(original)
                    record['response_fingerprint'] = digest(sanitized)
                    if sanitized != original:
                        raise ValueError('Sensitive reviewer output withheld')
                if choice.get('finish_reason') != 'stop':
                    raise ValueError('Incomplete completion')
                content = choice['message'].get('content') or ''
                if not self.privacy_mode:
                    record['raw_response'] = content
                # Some local servers wrap JSON in a single Markdown code block.
                match = re.fullmatch(r'\s*```(?:json)?\s*([\s\S]*?)\s*```\s*', content)
                raw = json.loads(match.group(1) if match else content)
                value = validator(copy.deepcopy(raw))
                cached = {'fingerprint': fingerprint, 'raw': raw, 'value': value, 'model': model, 'validated_policy_hash': self.policy_hash}
                if self.privacy_mode:
                    encoded = json.dumps(cached, ensure_ascii=False)
                    if self.scrubber(encoded) != encoded:
                        raise ValueError('Sensitive cache output withheld')
                save(cache, cached)
                record['status'] = 'success'
                return value
            except Exception as exc:
                record['status'] = 'failed'
                # Transport exception strings can contain URLs/headers; never serialize them.
                message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
                if self.privacy_mode:
                    message = type(exc).__name__  # validator/transport strings may contain source data
                key = resolve_key(provider.get('key', ''))
                if key:
                    message = message.replace(key, '[redacted]')
                record['error'] = message
                if isinstance(exc, ProviderError):
                    record['failure_category'] = 'provider_unavailable' if exc.fatal else 'transport'
                    # Provider errors cannot be repaired by changing model output.
                    # In particular, never spend generation retries on 402.
                    raise RuntimeError(stage + ': ' + message) from None
                feedback = '\nValidation failed: ' + message + '\nReturn the complete corrected JSON object.'
                if attempt == generation.get('attempts', 2):
                    raise RuntimeError(stage + ': ' + message) from None
                time.sleep(min(attempt * 2, 10))
            finally:
                record['seconds'] = round(time.monotonic() - started, 3)
                record['finished_at'] = datetime.now(timezone.utc).isoformat()
                with self.lock:
                    with (self.output / 'calls.jsonl').open('a') as file:
                        encoded = json.dumps(record, ensure_ascii=False)
                        file.write((self.scrubber(encoded) if self.privacy_mode else encoded) + '\n')
                print(digest(ident)[:16] if self.privacy_mode else ident, stage, record.get('status'), flush=True)
