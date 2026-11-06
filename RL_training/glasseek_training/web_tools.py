"""Small web tool environment; no reference answers are exposed to it."""
import asyncio
import json
import os
import re
import urllib.request
from html import unescape

TOOLS = [
    {"type": "function", "function": {"name": "web_search", "description": "Search the web.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "web_extract", "description": "Read a webpage by URL.",
     "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
]


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
        calls.append({"name": name, "arguments": args})
    if "<tool_call>" in text and not calls:
        raise ValueError("Unclosed tool call")
    return calls


class WebTools:
    schemas = TOOLS

    def __init__(self, timeout=60, max_chars=24000):
        self.timeout, self.max_chars = timeout, max_chars
        if not os.environ.get("SERPER_API_KEY"):
            raise ValueError("Set SERPER_API_KEY for web rollouts")

    async def execute(self, call):
        return await asyncio.to_thread(self._execute, call)

    def _execute(self, call):
        name, args = call["name"], call["arguments"]
        try:
            if name == "web_search":
                result = request_json("https://google.serper.dev/search", {"q": args["query"], "num": 10},
                                      {"X-API-KEY": os.environ["SERPER_API_KEY"]}, self.timeout)
                return json.dumps(result, ensure_ascii=False)[:self.max_chars]
            if name == "web_extract":
                url = args["url"]
                if not url.startswith(("https://", "http://")):
                    return "Tool error: expected an HTTP(S) URL"
                headers = {"Accept": "text/plain"}
                if os.environ.get("JINA_API_KEY"):
                    headers["Authorization"] = "Bearer " + os.environ["JINA_API_KEY"]
                request = urllib.request.Request("https://r.jina.ai/" + url, headers=headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read(self.max_chars * 4).decode("utf-8", errors="replace")[:self.max_chars]
            return f"Tool error: unknown tool {name}"
        except (KeyError, ValueError, OSError) as error:
            # Error type is enough for the policy; do not include API credentials/headers.
            return f"Tool error: {type(error).__name__}"
