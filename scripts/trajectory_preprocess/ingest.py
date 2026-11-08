"""Streaming intake, conservative normalization, exact dedup and near-duplicate review."""
import copy
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
from urllib.parse import urlparse

from client import digest
from tool_xml import normalize_text
from message_adapter import conversation
from runtime_metadata import clean_item
from call_protocol import ensure_links
from glasseek_shared.dataset_policy import excluded_source


def records(path, max_record_chars=16000000):
    """JSON arrays, JSONL, and Parquet; never load a multi-GB JSON file wholesale."""
    path = Path(path)
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8, use_threads=False):
            yield from batch.to_pylist()
        return
    decoder = json.JSONDecoder()
    with path.open(encoding='utf-8-sig') as file:
        buffer = ''; array = None; eof = False; ended = False; needs_separator = False; allow_end = True
        while True:
            if not eof and len(buffer) < 65536:
                chunk = file.read(65536); buffer += chunk; eof = not chunk
            buffer = buffer.lstrip()
            if array is None and buffer:
                array = buffer.startswith('[')
                if array: buffer = buffer[1:].lstrip()
            if array and needs_separator and buffer:
                if buffer.startswith(','):
                    buffer=buffer[1:].lstrip();needs_separator=False;allow_end=False
                elif not buffer.startswith(']'):
                    raise ValueError('Missing JSON array separator')
            if array and buffer.startswith(']'):
                if not allow_end:raise ValueError('Trailing JSON array comma')
                ended = True; buffer = buffer[1:].strip()
            if ended:
                if buffer: raise ValueError('Trailing content after JSON array')
                for chunk in iter(lambda:file.read(65536),''):
                    if chunk.strip():raise ValueError('Trailing content after JSON array')
                return
            if not buffer and eof:
                if array: raise ValueError('Truncated JSON array')
                return
            try:
                row, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof: raise ValueError('Malformed/truncated JSON input') from None
                if len(buffer) > max_record_chars: raise ValueError('Record exceeds intake size limit')
                chunk = file.read(65536); buffer += chunk; eof = not chunk
                continue
            if not isinstance(row, dict): raise ValueError('Expected a row object; bundle adapter required')
            if end > max_record_chars: raise ValueError('Record exceeds intake size limit')
            yield row
            buffer = buffer[end:]
            needs_separator=True;allow_end=True


def normalize(row, source, path, index, adapter='auto'):
    if excluded_source(source):
        raise ValueError('excluded_dataset_source')
    raw_hash = digest(row)
    if row.get('subagents'):
        raise ValueError('Hierarchical bundle requires explicit main/subagent expansion and provenance adapter')
    if 'steps' in row and 'question' in row:
        item = copy.deepcopy(row)
        item.setdefault('id', raw_hash[:16]); item.setdefault('source', source)
    elif adapter == 'canonical':
        raise ValueError('Canonical record missing steps/question')
    else:
        item = {'id': raw_hash[:16], 'source': source, **conversation(row)}
    item['normalized_schema_version'] = 3
    item['tool_call_format'] = 'hermes_qwen_xml_v1'
    item.setdefault('program_errors', []); item.setdefault('adapter_flags', [])
    item.setdefault('reference_answer', '')
    item['provenance'] = {'source': source, 'path': str(path), 'row_index': index, 'raw_sha256': raw_hash}
    if not isinstance(item.get('question'), str) or not item['question'].strip(): raise ValueError('Empty question')
    if not item.get('steps'): raise ValueError('Empty trajectory')
    # Preserve evidence; only versioned delegate telemetry is removed below.
    for number, step in enumerate(item['steps'], 1):
        if step.get('step_id') != number: raise ValueError('Non-contiguous step IDs')
        if not all(isinstance(step.get(k), str) for k in ('assistant', 'observation')): raise ValueError('Invalid step text')
        step.setdefault('calls', [])
        step['assistant'], step['calls'] = normalize_text(step['assistant'], step['calls'])
    ensure_links(item)
    clean_item(item)
    item['question_hash'] = digest(unicodedata.normalize('NFC', re.sub(r'\s+', ' ', item['question']).strip()).casefold())
    item['content_hash'] = semantic_hash(item)
    return item


def semantic_hash(item):
    semantic_steps = [{k:v for k,v in s.items() if k not in ('source_message_index','source_tool_payload',
                      'call_links','responses','source_call_links','source_responses','association_status')}
                      for s in item['steps']]
    semantic = {'question': unicodedata.normalize('NFC', item['question']).strip(), 'steps': semantic_steps}
    for key in ('context_messages','instruction_messages'):
        if item.get(key):
            semantic[key] = [{k:v for k,v in m.items() if k not in
                             ('source_message_index','source_content_start','source_content_end')} for m in item[key]]
    return digest(semantic)


def harness_plan(item, config):
    if config.get('contract'):
        from glasseek_contract import plan
        return plan(item,config,harness_plan)
    plan, errors, flags = [], [], []
    for step in item['steps']:
        calls = step.get('calls', [])
        if not isinstance(calls, list): errors.append('invalid_calls'); continue
        if calls and not step['observation'].strip(): errors.append('missing_observation')
        for call in calls:
            if not isinstance(call, dict): errors.append('invalid_call'); continue
            original_name, args = call.get('name'), call.get('arguments')
            name = original_name.casefold() if isinstance(original_name, str) else original_name
            if not isinstance(args, dict): errors.append('invalid_arguments'); continue
            if name in ('search', 'websearch', config['search_name'].casefold(), 'browser.search'):
                queries = args.get('query'); queries = [queries] if isinstance(queries, str) else queries
                if not isinstance(queries, list) or not queries or not all(isinstance(q,str) and q.strip() for q in queries):
                    errors.append('invalid_search_query'); continue
                if set(args) - {'query', 'limit', 'topn'}: flags.append('unmapped_search_arguments')
                if 'limit' in args and 'topn' in args and args['limit'] != args['topn']:
                    errors.append('conflicting_search_limits'); continue
                limit = args.get('limit', args.get('topn', 5))
                if type(limit) != int or not 1 <= limit <= 100: errors.append('invalid_search_limit'); continue
                for query in queries:
                    plan.append({'step_id': step['step_id'], 'name': config['search_name'], 'arguments': {'query': query, 'limit': limit}})
                if len(queries)>1: flags.append('search_batch_split_requires_observation_alignment')
                if name == 'browser.search': flags.append('search_backend_semantics_require_replay')
            elif name in ('visit', 'visit_urls', 'webextract', config['extract_name'].casefold(), 'browser.open'):
                if name == 'browser.open':
                    target = args.get('id')
                    if not isinstance(target, str) or not target.startswith(('https://', 'http://')):
                        errors.append('unsupported_tool:browser.open.cursor_or_local_resource'); continue
                    args = {**args, 'url': target}
                    flags.append('browser_viewport_semantics_require_replay')
                if 'url' in args and 'urls' in args:
                    one = args['url'] if isinstance(args['url'],list) else [args['url']]
                    many = args['urls'] if isinstance(args['urls'],list) else [args['urls']]
                    if one != many: errors.append('conflicting_extract_urls'); continue
                urls = args.get('urls', args.get('url')); urls = [urls] if isinstance(urls, str) else urls
                if isinstance(urls,list) and urls and any(isinstance(u,str) and urlparse(u).scheme and urlparse(u).scheme not in ('http','https') for u in urls):
                    errors.append('unsupported_tool:web_extract.url_scheme'); continue
                if not isinstance(urls, list) or not urls or not all(isinstance(u,str) and urlparse(u).scheme in ('http','https') and urlparse(u).netloc for u in urls):
                    errors.append('invalid_extract_urls'); continue
                # Extraction query is an alias for the reading objective, not a search call.
                goals=[args[k] for k in ('goal','query') if k in args]
                if any(not isinstance(g,str) for g in goals):
                    errors.append('invalid_extract_goal'); continue
                if len(goals)==2 and goals[0]!=goals[1]:
                    errors.append('conflicting_extract_goals'); continue
                goal_args={'goal':goals[0]} if goals else {}
                if 'query' in args: flags.append('extraction_query_mapped_to_goal')
                if set(args) - {'url', 'urls','goal','query'}: flags.append('extraction_semantics_require_replay')
                if any(g.strip() for g in goals): flags.append('goal_conditioned_extraction_requires_replay')
                size = config['extract_max_urls']
                if len(urls) > size:
                    errors.append('extract_url_batch_limit_requires_explicit_split')
                    continue
                for start in range(0, len(urls), size):
                    plan.append({'step_id': step['step_id'], 'name': config['extract_name'], 'arguments': {'urls': urls[start:start+size],**goal_args}})
                if len(urls)>size: flags.append('extract_batch_split_requires_observation_alignment')
            else: errors.append('unsupported_tool:' + str(original_name))
    if item['steps'][-1].get('calls'): errors.append('missing_final_answer')
    if item['steps'][-1].get('channel') == 'analysis': errors.append('missing_final_answer')
    return {'calls': plan, 'errors': sorted(set(errors)), 'flags': sorted(set(flags)),
            'status': 'requires_runtime_replay', 'replayed': False,
            'note': 'Mapping proposal only; original calls/observations preserved. Schema match is not runtime equivalence.'}


def simhash(question):
    text = re.sub(r'\s+', ' ', unicodedata.normalize('NFC', question).casefold()).strip()
    features = {text[i:i+3] for i in range(max(1,len(text)-2))}
    weights = [0]*64
    for feature in features:
        value = int(digest(feature)[:16],16)
        for bit in range(64): weights[bit] += 1 if value & (1<<bit) else -1
    return sum(1<<bit for bit,n in enumerate(weights) if n>0)


class Dedup:
    def __init__(self, path, config):
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS exact (hash TEXT PRIMARY KEY, id TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS buckets (band INTEGER, value INTEGER, hash TEXT, id TEXT)')
        self.db.execute('CREATE INDEX IF NOT EXISTS lookup ON buckets(band,value)')
        self.config = config

    def add(self, item, question_simhash=None):
        prior = self.db.execute('SELECT id FROM exact WHERE hash=?',(item['content_hash'],)).fetchone()
        if prior: return {'exact_duplicate_of': prior[0], 'near_duplicates': []}
        value = simhash(item['question']) if question_simhash is None else question_simhash
        neighbors = set()
        for band in range(4):
            bucket = (value >> (band*16)) & 65535
            rows = self.db.execute('SELECT hash,id FROM buckets WHERE band=? AND value=? LIMIT ?',
                                  (band,bucket,self.config['max_bucket_candidates']))
            for prior_hash, ident in rows:
                if (value ^ int(prior_hash,16)).bit_count() <= self.config['near_question_hamming_distance']:
                    neighbors.add(ident)
            self.db.execute('INSERT INTO buckets VALUES (?,?,?,?)',(band,bucket,format(value,'016x'),item['id']))
        self.db.execute('INSERT INTO exact VALUES (?,?)',(item['content_hash'],item['id']))
        return {'exact_duplicate_of': None, 'near_duplicates': sorted(neighbors),
                'near_action': 'review_only; keep different trajectories for the same question'}

    def close(self):
        self.db.commit(); self.db.close()
