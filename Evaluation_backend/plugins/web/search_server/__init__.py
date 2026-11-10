"""Search Server backend compatible with both Hermes provider APIs."""

from __future__ import annotations

import json
import logging
import os
import copy

from tools.web_tools import WEB_SEARCH_SCHEMA

from .provider import SearchServerProvider, check_search_server_requirements

logger = logging.getLogger(__name__)
_SEARCH = SearchServerProvider()


def _search_check() -> bool:
    try:
        return bool(check_search_server_requirements())
    except Exception:
        return False


def handle_search_server(args: dict, **_) -> str:
    from tools.interrupt import is_interrupted

    args = args if isinstance(args, dict) else {}
    query = args.get("query", "")
    try:
        limit = 10 if os.getenv("GLASSEEK_SEARCH_TOP10") == "1" else max(1, int(args.get("limit", 5)))
    except (TypeError, ValueError):
        limit = 5
    if is_interrupted():
        return json.dumps({"success": False, "error": "Interrupted"})
    try:
        # Share the canonical batch dispatcher; providers accept one query only.
        from tools.web_tools import web_search_tool
        return web_search_tool(query, limit=limit)
    except Exception:
        logger.error("Search Server web_search handler failed")
        return json.dumps({"success": False, "error": "web_search failed"})


_handle_search = handle_search_server


def register(ctx) -> None:
    ctx.register_web_search_provider(_SEARCH)
    schema = copy.deepcopy(WEB_SEARCH_SCHEMA)
    if os.getenv("GLASSEEK_SEARCH_TOP10") == "1":
        schema["description"] = "Search the web for information. Returns up to 10 results with titles, URLs, and descriptions."
        schema["parameters"]["properties"]["limit"]["default"] = 10
        schema["parameters"]["properties"]["limit"]["description"] = "The evaluator returns up to 10 results per query."
    ctx.register_tool(
        name="web_search",
        toolset="web",
        schema=schema,
        handler=handle_search_server,
        check_fn=_search_check,
        requires_env=[],
        emoji="🔍",
        override=True,
    )
