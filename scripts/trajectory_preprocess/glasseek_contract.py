"""Current Glasseek evaluation tool contract, read from repository source.

This is a compatibility screen, not a sandbox/security proof or runtime replay.
Dataset code is parsed as data and never executed by this module.
"""
import ast
import builtins
import copy
import hashlib
import json
import re
import posixpath
from pathlib import Path
import symtable
from urllib.parse import urlparse

from jsonschema import Draft7Validator
from client import digest
from tool_xml import CALL, THINK, normalize_text, serialize

ROOT = Path(__file__).resolve().parents[2]
_VALIDATORS = {}


def schema_value(node):
    """Extract JSON Schema data without importing service modules or evaluating code."""
    if isinstance(node, ast.Dict):
        return {ast.literal_eval(k): schema_value(v) for k,v in zip(node.keys,node.values)
                if ast.literal_eval(k) != 'description'}
    if isinstance(node, (ast.List,ast.Tuple)):
        return [schema_value(v) for v in node.elts]
    return ast.literal_eval(node)


def schema_from_file(path, name, function=False):
    tree=ast.parse(path.read_text())
    for node in tree.body:
        if function and isinstance(node,ast.FunctionDef) and node.name==name:
            returns=[n for n in node.body if isinstance(n,ast.Return)]
            return schema_value(returns[-1].value)
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return schema_value(node.value)
    raise ValueError('Schema not found: '+name)


def build_contract():
    launch=ROOT/'Evaluation/services/eval_run.py'
    whitelists=[]
    if launch.is_file():
        tree=ast.parse(launch.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.List):
                for i,element in enumerate(node.elts[:-1]):
                    if isinstance(element,ast.Constant) and element.value=='--tool-whitelist':
                        whitelists.append(ast.literal_eval(node.elts[i+1]))
        profile_source='evaluation_launcher'
    else:
        # A clean checkout also supports the tracked standard shell launcher.
        # This is a source-level profile, never evidence of runtime replay.
        launch=ROOT/'Evaluation/scripts/run_eval.sh'
        whitelists=re.findall(r'^\s*DEFAULT_TOOL_WHITELIST="(web,code_execution,delegation,vision)"\s*$',launch.read_text(),re.M)
        profile_source='standard_shell_default'
    if len(set(whitelists))!=1:raise ValueError('Ambiguous active evaluation tool whitelist')
    toolsets=whitelists[0].split(',')
    if set(toolsets)!={'web','code_execution','delegation','vision'}:
        raise ValueError('Evaluation tool profile changed; update the reviewed compatibility adapter')
    declared=set()
    for node in ast.parse((ROOT/'Evaluation_backend/toolsets.py').read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TOOLSETS' for t in node.targets):
            for key,value in zip(node.value.keys,node.value.values):
                if ast.literal_eval(key) in toolsets:
                    definition=schema_value(value)
                    if definition.get('includes'):raise ValueError('Composite toolsets require explicit resolution')
                    declared.update(definition['tools'])
    schemas={};paths=[launch,ROOT/'Evaluation_backend/toolsets.py',ROOT/'Evaluation/browse_comp_eval/hermes_client.py',
                      ROOT/'Evaluation_backend/tools/remote_pure_compute.py',
                      ROOT/'Evaluation_backend/tools/scientific_container_worker.py',
                      ROOT/'Evaluation_backend/tools/scientific_runtime.json',
                      ROOT/'Evaluation_backend/tools/scientific_runtime.Dockerfile',
                      ROOT/'Evaluation_backend/tools/execute_code_fs_sandbox.py',
                      ROOT/'Evaluation_backend/tools/quest_page_retrieval.py',
                      ROOT/'Shared/glasseek_shared/hermes/url_registry.py',
                      ROOT/'Shared/glasseek_shared/extract_protocol.py',ROOT/'Shared/glasseek_shared/search_protocol.py',
                      ROOT/'Shared/glasseek_shared/hermes/qwen_xml_protocol.py']
    for filename,symbol,fn in [('web_tools.py','WEB_SEARCH_SCHEMA',False),('web_tools.py','WEB_EXTRACT_SCHEMA',False),
                               ('code_execution_tool.py','build_execute_code_schema',True),
                               ('delegate_tool.py','DELEGATE_TASK_SCHEMA',False),('vision_tools.py','VISION_ANALYZE_SCHEMA',False)]:
        path=ROOT/'Evaluation_backend/tools'/filename;paths.append(path)
        schema=schema_from_file(path,symbol,fn);schemas[schema['name']]=schema
    if declared!=set(schemas):raise ValueError('Configured tool names do not match extracted schemas')
    client=(ROOT/'Evaluation/browse_comp_eval/hermes_client.py').read_text()
    for field,value in [('mode','strict'),('filesystem_access','none'),('network_access','none'),('tool_access','none')]:
        if f'("code_execution.{field}", "{value}")' not in client:
            raise ValueError('Evaluation code execution policy changed')
    import yaml
    configpath=ROOT/'Evaluation/data/runtime/hermes/config.yaml'
    local=yaml.safe_load(configpath.read_text()) if configpath.exists() else {}
    delegation=(local or {}).get('delegation',{})
    scientific=json.loads((ROOT/'Evaluation_backend/tools/scientific_runtime.json').read_text())
    contract={'profile':'glasseek-eval','profile_source':profile_source,'toolsets':toolsets,'schemas':schemas,
              'code_policy':{'mode':'strict','filesystem_access':'none','network_access':'none','tool_access':'none',
                             'dependencies':scientific['packages'],'image_id':scientific['image_id'],
                             'temporary_files':'private /tmp; discarded per call',
                             'limits':scientific['limits'],'state':'fresh_container_per_call',
                             'python_syntax':[3,8]},
              'delegation':{'max_concurrent_children':delegation.get('max_concurrent_children',3),
                            'max_spawn_depth':delegation.get('max_spawn_depth',1),
                            'blocked_child_tools':sorted(ast.literal_eval(next(n.value.args[0] for n in ast.parse((ROOT/'Evaluation_backend/tools/delegate_tool.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DELEGATE_BLOCKED_TOOLS' for t in n.targets))))},
              'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))},
              'runtime_replay':False}
    contract['sha256']=digest(contract)
    return contract


# A conservative in-memory subset. Other stdlib modules need compatibility
# review; this list is not a claim that static analysis proves safe execution.
COMPUTE_MODULES={'math','cmath','decimal','fractions','statistics','itertools','functools','collections',
                 'heapq','bisect','re','string','json','csv','datetime','calendar','random','operator',
                 'enum','dataclasses','typing','array','io'}
COMPUTE_MODULES.update({'textwrap', 'difflib', 'unicodedata', 'colorsys', 'ipaddress'})
# Only distributions in the pinned target image are eligible for migration.
SCIENTIFIC_MODULES = {'numpy', 'scipy', 'pandas', 'matplotlib', 'sympy', 'dateutil'}
SCIENTIFIC_IO = {'load', 'save', 'savez', 'savez_compressed', 'loadtxt', 'savetxt',
                 'genfromtxt', 'fromfile', 'tofile', 'memmap', 'savefig', 'imread',
                 'imsave', 'read_csv', 'read_json', 'read_excel', 'read_pickle',
                 'read_parquet', 'read_sql', 'read_html', 'to_csv', 'to_json',
                 'to_excel', 'to_pickle', 'to_parquet', 'to_sql', 'load_library',
                 'lambdify', 'sympify', 'parse_expr'}
COMPUTE_MODULES.update(SCIENTIFIC_MODULES)
DYNAMIC_NAMES={'open','exec','eval','compile','__import__','getattr','setattr','delattr','globals','locals','vars',
               'input','breakpoint','help'}


def private_temp_target(node):
    """Literal paths inside the fresh container's /tmp, never host paths."""
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str): return False
    path = node.value
    if not path or '\x00' in path or '://' in path: return False
    resolved = posixpath.normpath(posixpath.join('/tmp', path))
    return resolved.startswith('/tmp/')


def code_checks(code):
    errors=set()
    if not isinstance(code,str) or not code.strip():return ['tool_schema:execute_code.empty_code']
    try:tree=ast.parse(code,feature_version=(3,8));table=symtable.symtable(code,'<trajectory>','exec')
    except (SyntaxError,ValueError):return ['tool_runtime:execute_code.python_syntax']
    aliases = {}
    memory_buffers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names: aliases[a.asname or a.name.split('.')[0]] = a.name if a.asname else a.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            for a in node.names: aliases[a.asname or a.name] = (node.module or '')+'.'+a.name
    def qualified(node):
        if isinstance(node, ast.Name): return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute): return qualified(node.value)+'.'+node.attr
        return ''
    def buffer(node):
        return (isinstance(node, ast.Call) and qualified(node.func) in {'io.BytesIO','io.StringIO'}) or (isinstance(node, ast.Name) and node.id in memory_buffers)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and buffer(node.value):
            memory_buffers.update(t.id for t in node.targets if isinstance(t, ast.Name))
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            for a in node.names:
                root = a.name.split('.')[0]
                if root not in COMPUTE_MODULES:errors.add('tool_runtime:execute_code.import:'+root)
        if isinstance(node,ast.ImportFrom):
            if node.level or (node.module or '').split('.')[0] not in COMPUTE_MODULES:
                errors.add('tool_runtime:execute_code.import:'+(node.module or 'relative'))
            if any(a.name=='*' or a.name in DYNAMIC_NAMES for a in node.names):errors.add('tool_runtime:execute_code.dynamic_or_io')
        if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load) and node.id in DYNAMIC_NAMES - {'open'}:
            errors.add('tool_runtime:execute_code.dynamic_or_io')
        if isinstance(node,ast.Attribute) and (node.attr.startswith('__') or node.attr in ('open','read_text','write_text','read_bytes','write_bytes')):
            errors.add('tool_runtime:execute_code.dynamic_or_io')
        if isinstance(node, ast.Call):
            name = qualified(node.func)
            if name in {'open','io.open'}:
                target = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg=='file'), None)
                if not private_temp_target(target): errors.add('tool_runtime:execute_code.dynamic_or_io')
            if name.split('.')[0] in SCIENTIFIC_MODULES and name.rsplit('.',1)[-1] in SCIENTIFIC_IO:
                operation = name.rsplit('.',1)[-1]
                target = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg in {'file','fname','filepath_or_buffer','path_or_buf'}), None)
                # A buffer is not host I/O. Unknown targets remain held; native
                # isolation, not this heuristic, is the execution security boundary.
                if operation in {'load_library','lambdify','sympify','parse_expr'} or not (target is not None and (buffer(target) or private_temp_target(target))):
                    errors.add('tool_runtime:execute_code.dynamic_or_io')
    defined={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    allowed=defined | set(dir(builtins)) | {'__name__'}
    def unresolved(scope):
        for symbol in scope.get_symbols():
            if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in allowed:
                errors.add('tool_runtime:execute_code.external_state')
        for child in scope.get_children():unresolved(child)
    unresolved(table)
    return sorted(errors)


def decode_typed(value,schema):
    """Decode stringified containers only when the target schema requires them."""
    if isinstance(value,str) and schema.get('type') in ('array','object','boolean','integer','number'):
        try:decoded=json.loads(value)
        except (ValueError,TypeError):pass
        else:
            if Draft7Validator({'type':schema['type']}).is_valid(decoded):value=decoded
    if isinstance(value,dict):
        return {k:decode_typed(v,schema.get('properties',{}).get(k,{})) for k,v in value.items()}
    if isinstance(value,list):return [decode_typed(v,schema.get('items',{})) for v in value]
    return value


def validate_args(name,args,contract):
    schema=contract['schemas'][name]['parameters']
    if not isinstance(args,dict) or set(args)-set(schema['properties']):
        return ['tool_schema:'+name+'.unknown_arguments']
    key=(contract['sha256'],name)
    if key not in _VALIDATORS:_VALIDATORS[key]=Draft7Validator(schema)
    if not _VALIDATORS[key].is_valid(args):return ['tool_schema:'+name+'.invalid_arguments']
    errors=[]
    if name=='execute_code':errors.extend(code_checks(args['code']))
    elif name=='delegate_task':
        tasks=args.get('tasks') or [args]
        if not tasks or any(not isinstance(t.get('goal'),str) or not t['goal'].strip() for t in tasks):
            errors.append('tool_schema:delegate_task.missing_goal')
        if len(tasks)>contract['delegation']['max_concurrent_children']:
            errors.append('tool_runtime:delegate_task.batch_limit')
        for t in [args]+(args.get('tasks') or []):
            if set(t.get('toolsets') or [])-set(contract['toolsets']):
                errors.append('tool_runtime:delegate_task.disabled_child_toolset')
            if set(t.get('toolsets') or []) & {'code_execution','delegation'}:
                errors.append('tool_runtime:delegate_task.blocked_child_tools')
            if t.get('acp_command') or t.get('acp_args'):
                errors.append('tool_runtime:delegate_task.external_acp_dependency')
            if t.get('role')=='orchestrator' and contract['delegation']['max_spawn_depth']<=1:
                errors.append('tool_runtime:delegate_task.depth_limit')
        if len(args.get('tasks') or [])>1 and args.get('background'):
            errors.append('tool_schema:delegate_task.background_batch')
    elif name=='vision_analyze':
        url=args['image_url']
        if not url.strip():errors.append('tool_schema:vision_analyze.empty_image')
        elif not (urlparse(url).scheme in ('http','https') and urlparse(url).netloc) and not url.startswith('data:image/'):
            errors.append('tool_runtime:vision_analyze.unstaged_local_asset')
    return sorted(set(errors))


def plan(item, config, legacy):
    contract=config['contract'];basic={k:v for k,v in config.items() if k!='contract'}
    calls,errors,flags=[],[],list(item.get('adapter_flags',[]))
    extras={'execute_code':'execute_code','pythoninterpreter':'execute_code','functions.pythoninterpreter':'execute_code',
            'python':'execute_code','delegate_task':'delegate_task','vision_analyze':'vision_analyze'}
    for step in item['steps']:
        for index,call in enumerate(step['calls']):
            mapped=extras.get(call['name'].casefold())
            if mapped:
                args=decode_typed(copy.deepcopy(call['arguments']),contract['schemas'][mapped]['parameters']);issues=validate_args(mapped,args,contract)
                errors.extend(issues)
                if not step['observation'].strip():errors.append('missing_observation')
                if not issues:
                    calls.append({'step_id':step['step_id'],'source_call_index':index,'name':mapped,'arguments':args})
                    flags.append(mapped+'_runtime_replay_required')
            else:
                probe={'steps':[{**step,'calls':[call]},{'step_id':-1,'calls':[],'assistant':'end','observation':''}]}
                p=legacy(probe,basic);errors.extend(p['errors']);flags.extend(p['flags'])
                for c in p['calls']:
                    errors.extend(validate_args(c['name'],c['arguments'],contract))
                    calls.append({**c,'source_call_index':index})
    if item['steps'][-1]['calls'] or item['steps'][-1].get('channel')=='analysis':errors.append('missing_final_answer')
    return {'calls':calls,'errors':sorted(set(errors)),'flags':sorted(set(flags)),
            'status':'schema_compatible_pending_replay' if not errors else 'held',
            'profile':contract['profile'],'contract_sha256':contract['sha256'],'replayed':False,
            'note':'Active Glasseek tool contract; static code/dependency checks are not runtime replay or factual-quality approval.'}


def materialize(item):
    """Write actual target tool names/arguments into executable XML, preserving observations."""
    if item.get('program_errors') or item['harness'].get('errors'):
        raise ValueError('Cannot materialize a held trajectory')
    reserved_ids={link['call_id'] for step in item['steps'] for link in step.get('call_links',[]) if link.get('call_id')}
    obsolete={'extract_page_boundaries_unverified','extract_result_schema_unverified',
              'extract_result_url_coverage_mismatch','extract_page_status_unknown'}
    item['harness']['flags']=[f for f in item['harness']['flags'] if f not in obsolete]
    for step in item['steps']:
        original_calls=copy.deepcopy(step['calls'])
        text=step['assistant'];protected=[(m.start(),m.end()) for m in THINK.finditer(text)]
        matches=[m for m in CALL.finditer(text) if not any(a<=m.start()<b for a,b in protected)]
        if len(matches)!=len(step['calls']):raise ValueError('Materialization source call count mismatch')
        parts=[];cursor=0;target=[];mapped_by_call=[]
        for index,match in enumerate(matches):
            mapped=[{'name':c['name'],'arguments':c['arguments']} for c in item['harness']['calls']
                    if c['step_id']==step['step_id'] and c['source_call_index']==index]
            if not mapped:raise ValueError('Materialization would drop a call')
            mapped_by_call.append(mapped)
            parts.extend([text[cursor:match.start()],'\n'.join(serialize(c) for c in mapped)])
            cursor=match.end();target.extend(mapped)
        parts.append(text[cursor:]);converted=''.join(parts)
        if normalize_text(converted,target)[1]!=target:raise ValueError('Materialized XML roundtrip mismatch')
        from call_protocol import materialize_links
        item['harness']['flags'] = sorted(set(item['harness']['flags'] + materialize_links(step, mapped_by_call, reserved_ids)))
        step['assistant']=converted;step['calls']=target
        from web_batching import coalesce_step
        item['harness']['flags']=sorted(set(item['harness']['flags']+coalesce_step(
            item,step,original_calls,reserved_ids)))
        from glasseek_shared.web_return_audit import analyze_search_return
        from glasseek_shared.extract_protocol import merged_response_slices,batch_response_slices
        by_id={link['call_id']:link for link in step['call_links']}
        for response in step['responses']:
            link=by_id.get(response['response_to'])
            if link and link['name']=='web_search':
                spans=[]
                if 'merged_response_parts' in response:spans=merged_response_slices(response['content'],response['merged_response_parts'])
                elif 'batch_response_parts' in response:spans=batch_response_slices(response['content'],response['batch_response_parts'])
                components=[response['content'][start:end] for start,end in spans]
                response['search_return_audit']=analyze_search_return(link['arguments']['query'],response['content'],link,components)
    if 'target_call_response_alignment_required' not in item['harness']['flags']:
        item['harness']['flags']=[f for f in item['harness']['flags'] if f!='search_batch_split_requires_observation_alignment']
    item['tools_materialized']=True
    from tool_prompt import reconcile
    item['harness']['flags']=sorted(set(item['harness']['flags']+reconcile(item)))
    if any(f in item['harness']['flags'] for f in ('same_turn_extract_calls_merged','same_turn_search_calls_merged')):
        item['harness'].setdefault('source_mapping_calls',copy.deepcopy(item['harness']['calls']))
        item['harness']['calls']=[{'step_id':s['step_id'],'source_call_index':link.get('source_call_index',i),
            'source_call_indices':link.get('source_call_indices',[link.get('source_call_index',i)]),
            'target_call_index':i,**copy.deepcopy(call)}
            for s in item['steps'] for i,(call,link) in enumerate(zip(s['calls'],s['call_links']))]
