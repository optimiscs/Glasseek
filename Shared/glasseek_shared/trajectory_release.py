"""Fail-closed publication and training-loader contract. No service/model imports."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import re

import jsonschema
from glasseek_shared.dataset_policy import excluded_source, VERSION as SELECTION_VERSION

PROTOCOL = 'glasseek-call-response-v1'
SCHEMA_VERSION = 1
RELEASE_VERSION = 'glasseek-sft-release-v1'
RESPONSE_FORMAT = 'qwen-user-tool-response-v1'
REQUIRED_CHECKS = ('safety', 'quality', 'replay', 'review')
_VALIDATORS = {}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def without_descriptions(value):
    if isinstance(value, dict):
        return {k: without_descriptions(v) for k, v in value.items() if k != 'description'}
    if isinstance(value, list):
        return [without_descriptions(v) for v in value]
    return value


def schemas(tools):
    if isinstance(tools, str):
        tools = json.loads(tools)
    result = {}
    for raw in tools:
        schema = without_descriptions(raw.get('function', raw))
        name = schema['name']
        if name in result:
            raise ValueError('duplicate_tool_schema')
        result[name] = schema
    return result


def runtime_schemas():
    root = Path(__file__).resolve().parents[2]
    result = {}
    # AST literals only: do not import tool servers or execute configuration code.
    for filename, symbol, function in (
        ('web_tools.py', 'WEB_SEARCH_SCHEMA', False), ('web_tools.py', 'WEB_EXTRACT_SCHEMA', False),
        ('code_execution_tool.py', 'build_execute_code_schema', True),
        ('delegate_tool.py', 'DELEGATE_TASK_SCHEMA', False), ('vision_tools.py', 'VISION_ANALYZE_SCHEMA', False)):
        tree = ast.parse((root / 'Evaluation_backend/tools' / filename).read_text())
        node = next(n for n in tree.body if
                    (function and isinstance(n, ast.FunctionDef) and n.name == symbol) or
                    (not function and isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == symbol for t in n.targets)))
        value = next(n.value for n in reversed(node.body) if isinstance(n, ast.Return)) if function else node.value
        def literal(n):
            if isinstance(n, ast.Dict):
                return {ast.literal_eval(k): literal(v) for k, v in zip(n.keys, n.values)
                        if ast.literal_eval(k) != 'description'}
            if isinstance(n, (ast.List, ast.Tuple)):
                return [literal(v) for v in n.elts]
            return ast.literal_eval(n)
        schema = literal(value)
        result[schema['name']] = schema
    return result


def decoded(value):
    return json.loads(value) if isinstance(value, str) else value


def payload(row):
    value = {'id': row['id'], 'messages': decoded(row['messages']), 'tools': decoded(row['tools']),
            'source': row.get('source', ''),
            'provenance': decoded(row.get('provenance', {})),
            'split_group_id': row.get('split_group_id') or '',
            'bundle_id': row.get('bundle_id') or '', 'parent_id': row.get('parent_id') or '',
            'enable_thinking': bool(row.get('enable_thinking', True)),
            'agent_role': row.get('agent_role', 'parent'),
            'url_mode': row.get('url_mode', 'raw')}
    if row.get('supervision') is not None:
        value['supervision'] = decoded(row['supervision'])
    return value


def validate_messages(messages, tool_schemas, *, require_page_status=False):
    """Verify executable calls and batch responses. Page parsing is diagnostic.

    require_page_status is an explicit legacy diagnostic, never the default SFT
    acceptance gate. Historical multi-URL tool text need not be per-page JSON.
    """
    from glasseek_shared.extract_protocol import historical_results, merged_response_slices, batch_response_slices
    from glasseek_shared.hermes.qwen_xml_protocol import has_batch_wrapper
    calls, responses = {}, set()
    for mi, message in enumerate(messages):
        role, text = message.get('role'), message.get('content')
        if role not in ('system', 'developer', 'user', 'assistant', 'tool') or not isinstance(text, str):
            raise ValueError('invalid_message_schema')
        if message.get('association_status') == 'review_required':
            raise ValueError('unverified_response_association')
        if role == 'assistant':
            if has_batch_wrapper(text):raise ValueError('legacy_tool_batch_wrapper')
            visible = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
            blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', visible, re.S)
            links = message.get('call_links', [])
            if len(blocks) != len(links) or visible.count('<tool_call>') != len(blocks) or visible.count('</tool_call>') != len(blocks):
                raise ValueError('xml_call_metadata_mismatch')
            if message.get('tool_calls'):
                raise ValueError('duplicate_native_and_xml_calls')
            for block, link in zip(blocks, links):
                cid, name, args = link.get('call_id'), link.get('name'), link.get('arguments')
                if not isinstance(cid, str) or not cid or cid in calls:
                    raise ValueError('missing_or_duplicate_call_id')
                if name not in tool_schemas or not isinstance(args, dict):
                    raise ValueError('unknown_tool_or_invalid_arguments')
                schema = tool_schemas[name]['parameters']
                if set(args) - set(schema['properties']):
                    raise ValueError('unknown_tool_argument')
                try:
                    key = digest(schema)
                    if key not in _VALIDATORS:
                        _VALIDATORS[key] = jsonschema.Draft7Validator(schema)
                    _VALIDATORS[key].validate(args)
                except jsonschema.ValidationError:
                    raise ValueError('invalid_tool_arguments') from None
                expected = [f'<function={name}>']
                for key, value in args.items():
                    body = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)
                    if re.search(r'</?(?:tool_call|function|parameter)(?:[=>\s])', body):
                        raise ValueError('xml_delimiter_collision')
                    expected.extend([f'<parameter={key}>', body, '</parameter>'])
                expected.append('</function>')
                if block.strip() != '\n'.join(expected):
                    raise ValueError('xml_arguments_metadata_mismatch')
                calls[cid] = (mi, name, args)
        elif role == 'tool' or (role == 'user' and text.lstrip().startswith('<tool_response>')):
            if role == 'user' and (not text.rstrip().endswith('</tool_response>') or
                text.count('<tool_response>') != 1 or text.count('</tool_response>') != 1):
                raise ValueError('ambiguous_tool_response_envelope')
            cid = message.get('response_to')
            if not cid or cid != message.get('tool_call_id') or cid not in calls or cid in responses:
                raise ValueError('missing_duplicate_or_orphan_response_to')
            responses.add(cid)
            call_mi, name, args = calls[cid]
            if 'search_return_audit' in message:
                from glasseek_shared.web_return_audit import fingerprint
                audit=message['search_return_audit']
                if (name!='web_search' or not isinstance(audit,dict)
                    or audit.get('request_sha256')!=fingerprint(args['query'])
                    or audit.get('rendered_response_sha256')!=hashlib.sha256(text.encode()).hexdigest()
                    or audit.get('association',{}).get('call_id')!=cid):
                    raise ValueError('invalid_search_return_audit')
            if 'merged_response_parts' in message:
                if name not in ('web_search','web_extract'):raise ValueError('invalid_merged_response_parts')
                parts=message['merged_response_parts'];merged_response_slices(text,parts)
                link=next(x for x in messages[call_mi]['call_links'] if x['call_id']==cid)
                source_ids=link.get('source_call_ids',[])
                if (not isinstance(source_ids,list) or any(not isinstance(s,str) or not s for s in source_ids)
                    or len(source_ids)!=len(set(source_ids)) or set(source_ids)!={p['source_call_id'] for p in parts}):
                    raise ValueError('invalid_merged_response_parts')
            if 'batch_response_parts' in message:
                if name not in ('web_search','web_extract') or 'merged_response_parts' in message:
                    raise ValueError('invalid_batch_response_parts')
                batch_response_slices(text,message['batch_response_parts'])
                link=next(x for x in messages[call_mi]['call_links'] if x['call_id']==cid)
                source_ids=link.get('source_call_ids')
                if (not isinstance(source_ids,list) or not source_ids
                    or any(not isinstance(s,str) or not s for s in source_ids)
                    or len(set(source_ids))!=len(source_ids)
                    or message.get('source_call_ids')!=source_ids
                    or link.get('association_scope')!='closed_round_call_batch'):
                    raise ValueError('invalid_batch_response_parts')
                native_ids=[p['source_response'].get('source_response_id') for p in message['batch_response_parts']]
                native_ids=[value for value in native_ids if value is not None]
                if (any(not isinstance(value,str) for value in native_ids)
                    or len(set(native_ids))!=len(native_ids)
                    or (native_ids and set(native_ids)!=set(source_ids))):
                    raise ValueError('invalid_batch_response_parts')
            if name == 'web_extract' and require_page_status:
                results, reason = historical_results(text, args['urls'])
                if reason:
                    raise ValueError(reason)
                if any('error' not in p for p in results['results']):
                    raise ValueError('extract_page_status_unknown')
    if set(calls) != responses:
        raise ValueError('missing_tool_response')
    if not messages or messages[-1]['role'] != 'assistant' or messages[-1].get('call_links'):
        raise ValueError('missing_final_answer')


def validate_row(row, expected_schemas):
    if excluded_source(row.get('source')):
        raise ValueError('excluded_dataset_source')
    if excluded_source(decoded(row.get('provenance', {})).get('source')):
        raise ValueError('excluded_dataset_source')
    if row.get('protocol_version') != PROTOCOL or row.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('trajectory_protocol_or_schema_version_mismatch')
    if row.get('training_ready') is not True or row.get('harness_replayed') is not True or row.get('acceptance_status') != 'passed':
        raise ValueError('trajectory_not_accepted_for_training')
    acceptance = decoded(row.get('acceptance', {}))
    if any(acceptance.get(k) != 'passed' for k in REQUIRED_CHECKS):
        raise ValueError('missing_acceptance_stage')
    value = payload(row)
    if 'supervision' in value:
        from .curation.supervision import validate_plan
        validate_plan(value['supervision'],value['messages'],require_learn=True)
    if row.get('content_sha256') != digest(value) or acceptance.get('content_sha256') != digest(value):
        raise ValueError('trajectory_content_hash_mismatch')
    role = row.get('agent_role', 'parent')
    allowed = {k:v for k,v in expected_schemas.items() if role != 'child' or k not in ('execute_code', 'delegate_task')}
    if role not in ('parent', 'child') or schemas(value['tools']) != allowed:
        raise ValueError('runtime_tool_schema_mismatch')
    validate_messages(value['messages'], allowed)


def verify_release_file(path, expected_schemas=None):
    """Called before loader sampling/tokenization; every stored row is checked."""
    import pyarrow.parquet as pq
    path = Path(path).resolve()
    manifest_path = path.parent / 'release.manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Missing required release.manifest.json: ' + str(path))
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('version') != RELEASE_VERSION or manifest.get('protocol_version') != PROTOCOL or
        manifest.get('response_format') != RESPONSE_FORMAT or manifest.get('dataset_selection_version') != SELECTION_VERSION or
        manifest.get('schema_version') != SCHEMA_VERSION or manifest.get('acceptance_status') != 'passed'):
        raise ValueError('release_version_or_acceptance_mismatch')
    expected_schemas = runtime_schemas() if expected_schemas is None else expected_schemas
    if manifest.get('tool_schema_sha256') != digest(expected_schemas):
        raise ValueError('release_runtime_schema_mismatch')
    entries = [e for e in manifest.get('files', []) if e.get('path') == path.name]
    if len(entries) != 1 or entries[0].get('sha256') != file_hash(path):
        raise ValueError('release_file_hash_mismatch')
    count = 0
    actual = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            validate_row(row, expected_schemas)
            if row['id'] in actual:
                raise ValueError('duplicate_release_id')
            actual[row['id']] = row['content_sha256']
            count += 1
    if count != entries[0].get('rows') or count <= 0:
        raise ValueError('release_row_count_mismatch_or_empty')
    inventory = manifest.get('content_inventory', {})
    inv_path = path.parent / 'content_hashes.jsonl'
    if inventory.get('path') != inv_path.name or inventory.get('sha256') != file_hash(inv_path):
        raise ValueError('release_inventory_hash_mismatch')
    with inv_path.open() as stream:
        inventory_rows = [json.loads(line) for line in stream]
    expected_inventory = {}
    for item in inventory_rows:
        if item['id'] in expected_inventory:
            raise ValueError('duplicate_inventory_id')
        expected_inventory[item['id']] = item['content_sha256']
    if expected_inventory != actual or manifest.get('rows') != count:
        raise ValueError('release_content_inventory_mismatch')
    return manifest
