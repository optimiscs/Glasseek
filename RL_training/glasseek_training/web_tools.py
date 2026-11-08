"""Small web tool environment; no reference answers are exposed to it."""
import asyncio
import json
import os
import re
import urllib.request
import urllib.error
from html import unescape
from glasseek_shared.extract_protocol import validate_urls
from glasseek_shared.hermes.tool_errors import ToolFailure,exception_failure,http_json

TOOLS = [
    {"type": "function", "function": {"name": "web_search", "description": "Search the web.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "web_extract", "description": "Read 1 to 5 webpages. Returns one results array with URL, content and error per page.",
     "parameters": {"type": "object", "properties": {"urls": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
         "goal": {"type": "string", "description": "Optional objective for selecting evidence from the pages."}}, "required": ["urls"]}}},
]


def search_pool_worker_env():
    """Forward routing to Ray actors without distributing upstream credentials."""
    endpoint = os.environ.get('GLASSEEK_SEARCH_POOL_ENDPOINT') or os.environ.get('SEARCH_SERVER_ENDPOINT')
    if not endpoint:
        raise ValueError('Shared search pool endpoint is not configured')
    env = {
        'GLASSEEK_SEARCH_POOL_ENDPOINT': endpoint,
        'SEARCH_SERVER_ENDPOINT': endpoint,
        'GLASSEEK_SEARCH_POOL_TIMEOUT': os.environ.get('GLASSEEK_SEARCH_POOL_TIMEOUT', '60'),
    }
    if os.environ.get('GLASSEEK_URL_MODE', 'raw') != 'raw':
        env['GLASSEEK_URL_MODE'] = os.environ['GLASSEEK_URL_MODE']
        env['GLASSEEK_URL_PROMPT'] = os.environ.get('GLASSEEK_URL_PROMPT', 'auto')
    for key in ('GLASSEEK_QUEST_RETRIEVAL_URL', 'GLASSEEK_QUEST_RETRIEVAL_TIMEOUT'):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def request_json(url, body, headers=None, timeout=60):
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                    headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def parse_tool_calls(text: str) -> list[dict]:
    # Ignore tool-like text in reasoning. Only calls after the last think block execute.
    text = text.rsplit("</think>", 1)[-1]
    calls = []
    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S):
        if block.lstrip().startswith("{"):
            call = json.loads(block)
            args = call.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            name = call["name"]
        else:
            match = re.fullmatch(r"\s*<function=([^>]+)>\s*(.*?)\s*</function>\s*", block, re.S)
            if not match:
                raise ValueError("Malformed Qwen tool call")
            name, params = match.groups()
            args = {key: unescape(value.strip()) for key, value in
                    re.findall(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", params, re.S)}
        if not isinstance(args, dict):
            raise ValueError("Tool arguments must be an object")
        if name == 'web_extract' and isinstance(args.get('urls'), str):
            args['urls'] = json.loads(args['urls'])
        calls.append({"name": name, "arguments": args})
    if "<tool_call>" in text and not calls:
        raise ValueError("Unclosed tool call")
    return calls


class WebTools:
    schemas = TOOLS

    def __init__(self, timeout=60, max_chars=24000, url_mode=None):
        self.timeout, self.max_chars = timeout, max_chars
        from glasseek_shared.hermes.url_registry import URLRegistry, mode_value, annotate_schemas
        self.url_mode = mode_value(url_mode)
        self.url_registry = URLRegistry(self.url_mode)
        self.schemas = annotate_schemas(TOOLS, self.url_mode)
        self.search_endpoint = os.environ.get('GLASSEEK_SEARCH_POOL_ENDPOINT') or os.environ.get('SEARCH_SERVER_ENDPOINT')
        if not self.search_endpoint:
            raise ValueError('Source Evaluation/services/search_pool_env.sh before web rollouts')
        from urllib.parse import urlsplit
        endpoint = urlsplit(self.search_endpoint)
        if endpoint.scheme not in ('http', 'https') or not endpoint.hostname or endpoint.hostname == 'google.serper.dev':
            raise ValueError('SEARCH_SERVER_ENDPOINT must point to the shared search pool')
        self.search_timeout = float(os.environ.get('GLASSEEK_SEARCH_POOL_TIMEOUT', timeout))

    async def execute(self, call):
        if self.url_mode == 'raw' and call['name'] != 'web_extract':
            return await asyncio.to_thread(self._execute, call)
        try:
            name, args = call['name'], call['arguments']
            if name == 'web_extract':
                if 'url' in args and 'urls' in args:
                    raise ValueError('Use urls only; ambiguous url/urls arguments')
                validate_urls(args.get('urls', [args['url']] if 'url' in args else None))
            cached = self.url_registry.cached_evidence(name, args)
            resolved = self.url_registry.resolve_args(name, args)
            if cached is not None:
                result = cached
            elif name == 'web_extract':
                # Legacy singular calls remain readable; the advertised contract uses urls.
                if 'url' in resolved and 'urls' in resolved:
                    raise ValueError('Use urls only; ambiguous url/urls arguments')
                urls = validate_urls(resolved.get('urls', [resolved['url']] if 'url' in resolved else None))
                async def extract_one(url):
                    single = {**resolved, 'url': url}
                    single.pop('urls', None)
                    try:
                        text = await asyncio.to_thread(self._execute, {'name':name,'arguments':single})
                    except Exception as exc:
                        error=exception_failure(exc).payload
                        return {'url':url, 'content':'', 'error':error['error'],'error_code':error['error_code'],'retryable':error['retryable']}
                    if isinstance(text,ToolFailure):
                        return {'url':url,'content':'',**{k:v for k,v in text.payload.items() if k!='success'}}
                    failed = text.startswith('Tool error:')
                    return {'url':url, 'content':'' if failed else text, 'error':text if failed else None}
                results = await asyncio.gather(*(extract_one(url) for url in urls))
                result = json.dumps({'results':results},ensure_ascii=False)
            else:
                result = await asyncio.to_thread(self._execute, {'name':name,'arguments':resolved})
            return self.url_registry.observe(name, result)
        except ValueError as exc:
            return json.dumps({'error':str(exc),'status':'url_reference_error'})

    def begin_turn(self):
        self.url_registry.begin_turn()

    def _execute(self, call):
        name, args = call["name"], call["arguments"]
        try:
            if name == "web_search":
                request = urllib.request.Request(self.search_endpoint,
                    data=json.dumps({'q': args['query'], 'num': 10}).encode(),
                    headers={'Content-Type': 'application/json'})
                # Internal service requests must not inherit an external proxy.
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                result=http_json(opener,request,self.search_timeout)
                if isinstance(result,ToolFailure):return result
                return json.dumps(result, ensure_ascii=False)
            if name == "web_extract":
                goal = args.get('goal', '')
                if not isinstance(goal, str):
                    return ToolFailure('goal must be a string','invalid_tool_arguments')
                selector = os.environ.get('GLASSEEK_QUEST_RETRIEVAL_URL', '').rstrip('/')
                if goal.strip() and not selector:
                    return ToolFailure('goal-aware extraction requires GLASSEEK_QUEST_RETRIEVAL_URL','tool_not_configured')
                url = args["url"]
                if not url.startswith(("https://", "http://")):
                    return ToolFailure('expected an HTTP(S) URL','invalid_tool_arguments')
                headers = {"Accept": "text/plain"}
                if os.environ.get("JINA_API_KEY"):
                    headers["Authorization"] = "Bearer " + os.environ["JINA_API_KEY"]
                request = urllib.request.Request("https://r.jina.ai/" + url, headers=headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    if not goal.strip():
                        return response.read(self.max_chars * 4).decode("utf-8", errors="replace")[:self.max_chars]
                    # Select from the full bounded source, before output truncation.
                    raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    return ToolFailure('page exceeds goal-selection input limit','tool_input_too_large')
                selection = urllib.request.Request(selector + '/select',
                    data=json.dumps({'text':raw.decode('utf-8', errors='replace'), 'goal':goal}).encode(),
                    headers={'Content-Type':'application/json'})
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(selection, timeout=float(os.environ.get('GLASSEEK_QUEST_RETRIEVAL_TIMEOUT', '180'))) as response:
                    selected = json.load(response)['content']
                if not isinstance(selected, str) or not selected.strip():
                    return ToolFailure('goal-aware extraction returned no content','tool_empty_response')
                return selected[:self.max_chars]
            return ToolFailure(f'unknown tool {name}','unknown_tool')
        except (KeyError, ValueError, OSError) as error:
            # Error type is enough for the policy; do not include API credentials/headers.
            return exception_failure(error)
