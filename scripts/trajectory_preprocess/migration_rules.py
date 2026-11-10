"""Overlapping trajectory-level rule membership, captured before tool rewriting.

These are audit labels, not acceptance decisions. Final acceptance determines
whether a member is migrated, held, quarantined, or an exact duplicate.
"""

SEARCH = {'search', 'websearch', 'web_search', 'browser.search'}
EXTRACT = {'visit', 'visit_urls', 'webextract', 'web_extract'}
CODE = {'execute_code', 'pythoninterpreter', 'functions.pythoninterpreter', 'python'}


def memberships(item):
    rules = set()
    for step in item['steps']:
        for call in step.get('calls', []):
            if not isinstance(call, dict):
                rules.add('unparsed'); continue
            name = str(call.get('name', '')).casefold()
            args = call.get('arguments')
            allowed = None
            if name in SEARCH:
                rules.add('search'); allowed = {'query', 'topn', 'limit'}
            elif name in EXTRACT:
                rules.add('extract'); allowed = {'url', 'urls', 'goal', 'query'}
            elif name == 'browser.open':
                rules.add('browser.open'); allowed = {'id', 'url', 'urls', 'goal', 'query'}
            elif name in CODE:
                rules.add('code')
            elif name in {'delegate_task', 'call_sub_agent'}:
                rules.add('subagent')
            elif name == 'vision_analyze':
                rules.add('vision')
            else:
                rules.add('unsupported_tool')
            if allowed is not None and isinstance(args, dict):
                extra = set(args) - allowed
                rules.update('argument.' + k for k in extra & {'source', 'parse_type'})
                if extra - {'source', 'parse_type'}:
                    rules.add('argument.other')
    return sorted(rules or {'no_tool'})
