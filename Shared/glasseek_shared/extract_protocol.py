"""Deterministic multi-page envelope. No fetching, summarization or URL guessing."""
import collections
import copy
import json
import ast
import re
from glasseek_shared.historical_tools import unwrap, load, containers

MAX_URLS = 5
VERSION = 'web-extract-results-v1'
HISTORICAL_POLICY_VERSION = 'historical-web-call-batch-v3'


def response_body(text):
    """Remove only the transport envelope; preserve the actual return verbatim."""
    stripped = text.strip()
    opening, closing = '<tool_response>', '</tool_response>'
    if stripped.startswith(opening) and stripped.endswith(closing):
        if stripped.count(opening) != 1 or stripped.count(closing) != 1:
            raise ValueError('ambiguous_tool_response_envelope')
        return stripped[len(opening):-len(closing)]
    return text


def merged_response_slices(text, parts):
    """Verify call-level concatenation spans, without inventing per-page spans."""
    if not isinstance(parts, list) or len(parts) < 2:
        raise ValueError('invalid_merged_response_parts')
    result, cursor, seen = [], 0, set()
    opening, closing = '<tool_response>', '</tool_response>'
    for index, part in enumerate(parts):
        if not isinstance(part, dict):raise ValueError('invalid_merged_response_parts')
        start, end, cid = part.get('start'), part.get('end'), part.get('source_call_id')
        if (type(start) is not int or type(end) is not int or not cursor <= start < end <= len(text)
            or not isinstance(cid, str) or not cid or cid in seen):
            raise ValueError('invalid_merged_response_parts')
        gap = text[cursor:start].strip()
        if gap and not (index == 0 and gap == opening):
            raise ValueError('invalid_merged_response_parts')
        seen.add(cid);result.append((start, end));cursor=end
    if text[cursor:].strip() not in ('', closing):
        raise ValueError('invalid_merged_response_parts')
    return result


def batch_response_slices(text, parts):
    """Original return components of a uniquely associated whole-call batch.

    Components need not map to individual queries/URLs or individual source
    calls. Their boundaries come from source tool messages, never inferred pages.
    """
    import hashlib
    if not isinstance(parts,list) or not parts:
        raise ValueError('invalid_batch_response_parts')
    cursor=0;result=[]
    for index,part in enumerate(parts):
        if not isinstance(part,dict):raise ValueError('invalid_batch_response_parts')
        start,end=part.get('start'),part.get('end')
        if type(start) is not int or type(end) is not int or not cursor<=start<end<=len(text):
            raise ValueError('invalid_batch_response_parts')
        gap=text[cursor:start].strip()
        if gap and not (index==0 and gap=='<tool_response>'):
            raise ValueError('invalid_batch_response_parts')
        if part.get('content_sha256')!=hashlib.sha256(text[start:end].encode()).hexdigest():
            raise ValueError('invalid_batch_response_parts')
        if not isinstance(part.get('source_response'),dict):raise ValueError('invalid_batch_response_parts')
        result.append((start,end));cursor=end
    if text[cursor:].strip() not in ('','</tool_response>'):
        raise ValueError('invalid_batch_response_parts')
    return result


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
    """Recover boundaries from protocol fields/headers or a uniquely linked URL.

    Unknown execution status stays unknown. Never assign an unlabeled multi-URL
    prose block by order, infer a successful fetch, or silently drop surplus pages.
    """
    try:
        body = unwrap(text)
        value = containers(load(text))
    except (ValueError, TypeError):
        return None, 'extract_page_boundaries_unverified'
    payload = None
    if isinstance(value, dict) and isinstance(value.get('results'), list):
        payload = copy.deepcopy(value)
    elif isinstance(value, list) and all(isinstance(p, dict) for p in value):
        payload = {'results': copy.deepcopy(value)}
    elif isinstance(value, dict) and isinstance(value.get('url'), str):
        payload = {'results': [copy.deepcopy(value)]}
    if payload is not None:
        pages = payload['results']
        for page in pages:
            if 'content' not in page:
                field = next((k for k in ('raw_content', 'report', 'summary') if isinstance(page.get(k), str)), None)
                if field: page['content'] = page[field]
                elif isinstance(page.get('error'), str): page['content'] = ''
            if 'error' not in page and type(page.get('success')) is bool:
                if page['success']: page['error'] = None
                elif isinstance(page.get('message'), str): page['error'] = page['message']
    else:
        pages = []
        # The format emitted by the recorded visit tool explicitly separates URL
        # and Report. Only full line-anchored records count as page boundaries.
        headers = list(re.finditer(r'(?m)^URL:\s*(https?://\S+)\s*\nReport:\s*', body))
        useful = list(re.finditer(r'(?m)^The useful information in (https?://\S+) for user goal ([\s\S]*?) as follows:\s*\n', body))
        if headers and not body[:headers[0].start()].strip():
            for i, header in enumerate(headers):
                end = headers[i+1].start() if i+1 < len(headers) else len(body)
                pages.append({'url': header[1], 'content': body[header.end():end],
                              'historical_status': 'unreported', 'boundary_origin': 'url_report_header'})
        elif useful and not body[:useful[0].start()].strip():
            for i, header in enumerate(useful):
                end = useful[i+1].start() if i+1 < len(useful) else len(body)
                pages.append({'url': header[1], 'content': body[header.end():end], 'source_goal': header[2],
                              'historical_status': 'unreported', 'boundary_origin': 'url_goal_header'})
        elif body.startswith('[web_extract]'):
            lines = body.splitlines(); i = 1
            try:
                while i < len(lines):
                    match = re.fullmatch(r'  - (https?://\S+)(?: ERROR=(.*))?', lines[i])
                    if not match: raise ValueError('unrecognized_excerpt_record')
                    page = {'url': match[1], 'content_kind': 'excerpt'}
                    if match[2] is not None:
                        page.update(content='', error=ast.literal_eval(match[2])); i += 1
                    else:
                        if i+2 >= len(lines): raise ValueError('truncated_excerpt_record')
                        title = re.fullmatch(r'    title=(.*)', lines[i+1]); excerpt = re.fullmatch(r'    excerpt=(.*)', lines[i+2])
                        if not title or not excerpt: raise ValueError('unrecognized_excerpt_fields')
                        page.update(title=ast.literal_eval(title[1]), content=ast.literal_eval(excerpt[1]), error=None); i += 3
                    pages.append(page)
            except (ValueError, SyntaxError):
                return None, 'extract_page_boundaries_unverified'
        elif len(urls) == 1 and isinstance(value, str) and value.strip():
            pages = [{'url': urls[0], 'content': body, 'historical_status': 'unreported',
                      'boundary_origin': 'single_url_call_response'}]
        else:
            return None, 'extract_page_boundaries_unverified'
        payload = {'results': pages}
    if any(not isinstance(p, dict) or not isinstance(p.get('url'), str) or
           not isinstance(p.get('content'), str) or
           ('error' in p and p['error'] is not None and not isinstance(p['error'], str)) for p in pages):
        return None, 'extract_result_schema_unverified'
    if collections.Counter(p['url'] for p in pages) != collections.Counter(urls):
        return None, 'extract_result_url_coverage_mismatch'
    return payload, None
