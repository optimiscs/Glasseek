"""Deterministic multi-page envelope. No fetching, summarization or URL guessing."""
import collections
import copy
import json

MAX_URLS = 5
VERSION = 'web-extract-results-v1'


def validate_urls(urls):
    if not isinstance(urls, list) or not 1 <= len(urls) <= MAX_URLS:
        raise ValueError('urls must be an array containing 1 to 5 URLs; split larger batches explicitly')
    if any(not isinstance(url, str) or not url.strip() for url in urls):
        raise ValueError('Every URL must be a nonempty string')
    return urls


def runtime_results(urls, results):
    """Match by exact requested URL, retaining duplicates in provider order.

    Missing provider output is a runtime error, never a fabricated successful page.
    Extra/unidentifiable output fails the whole operation instead of being dropped.
    """
    by_url = collections.defaultdict(collections.deque)
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get('url'), str):
            raise ValueError('Provider returned a result without a URL')
        by_url[result['url']].append(copy.deepcopy(result))
    output = []
    for url in urls:
        if by_url[url]:
            result = by_url[url].popleft()
            result.setdefault('error', None)
        else:
            result = {'url': url, 'content': '', 'error': 'Provider returned no result for this URL',
                      'error_code': 'missing_provider_result'}
        output.append(result)
    if any(by_url.values()):
        raise ValueError('Provider returned unexpected or surplus URL results')
    return output


def historical_results(text, urls):
    """Only accept an explicit JSON per-page envelope with complete URL coverage.

    Do not infer page boundaries or success from prose, headings, order, or presence
    of content. A missing error field remains missing (unknown historical status).
    """
    body = text.strip()
    if body.startswith('<tool_response>') and body.endswith('</tool_response>'):
        body = body[len('<tool_response>'):-len('</tool_response>')].strip()
    def unique_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate JSON key')
            value[key] = item
        return value
    try:
        payload = json.loads(body, object_pairs_hook=unique_keys)
    except (ValueError, TypeError):
        return None, 'extract_page_boundaries_unverified'
    if not isinstance(payload, dict) or set(payload) != {'results'} or not isinstance(payload['results'], list):
        return None, 'extract_page_boundaries_unverified'
    pages = payload['results']
    if any(not isinstance(p, dict) or not isinstance(p.get('url'), str) or
           not isinstance(p.get('content'), str) or
           ('error' in p and p['error'] is not None and not isinstance(p['error'], str)) for p in pages):
        return None, 'extract_result_schema_unverified'
    if collections.Counter(p['url'] for p in pages) != collections.Counter(urls):
        return None, 'extract_result_url_coverage_mismatch'
    # Preserve original order and all fields. URL-based verification needs no sort.
    return copy.deepcopy(payload), None
