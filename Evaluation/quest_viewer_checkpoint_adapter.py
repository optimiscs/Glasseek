#!/usr/bin/env python3
"""Publish Hermes evaluation snapshots in Quest's existing viewer format."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from browse_comp_eval.hermes_client import profile_name

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "data/runtime"
RUN = os.environ.get("GLASSEEK_VIEWER_RUN", "glasseek_sft9b_quest128")
DATASET = os.environ.get("GLASSEEK_VIEWER_DATASET", "quest_qwen27b_128")
WORKSPACE = os.environ.get("GLASSEEK_VIEWER_WORKSPACE", "glasseek-sft9b-browsecomp-128")
MODEL = os.environ.get("GLASSEEK_VIEWER_MODEL", "Glasseek SFT-9B")
LABEL = os.environ.get("GLASSEEK_VIEWER_LABEL", f"{MODEL} · BrowseComp · Quest 128")


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".viewer-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def object_from_text(value):
    if isinstance(value, dict):
        return value
    text = str(value or "")
    for i, char in enumerate(text):
        if char == "{":
            try:
                obj, _ = json.JSONDecoder().raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except ValueError:
                continue
    return {}


def tool_view(name: str, args: dict, output: str):
    obj = object_from_text(output)
    data = obj.get("data", {})
    if name in ("web_search", "search_server"):
        error = str(obj.get('error') or '') if obj.get('error') or obj.get('success') is False else ''
        status = 'error' if error else 'pending' if not output else 'success'
        found = data.get("web", []) if isinstance(data, dict) else []
        results = []
        for item in found if isinstance(found, list) else []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or item.get("link") or "")
            results.append({"url": url, "title": str(item.get("title") or ""),
                            "domain": urlsplit(url).hostname or "", "snippet": str(item.get("description") or item.get("snippet") or "")})
        return "search", {"name": name, "kind": "search", "goal": "", "status": status,
            "error": error, "errorCode": obj.get('error_code'), "queries": [{"query": str(args.get("query") or ""),
            "count": len(results) if status == 'success' else None, "results": results,
            "status": status, "error": error}], "queryCount": 1, "issuedQueryCount": 1, "resultCount": len(results)}
    if name in ("web_extract", "fetch_server"):
        docs = obj.get("results") if "results" in obj else (data.get("results", data.get("web", [])) if isinstance(data, dict) else data)
        if isinstance(docs, dict):
            docs = [docs]
        if not isinstance(docs, list):
            docs = []
        pages = []
        for item in docs:
            if isinstance(item, dict):
                url = str(item.get("url") or "")
                error = item.get("error")
                pages.append({"url": url, "domain": urlsplit(url).hostname or "", "title": str(item.get("title") or url), "summary": f"读取失败：{error}" if error else str(item.get("content") or item.get("text") or "")})
        if not pages and (not output or obj.get("error")):
            urls = args.get("urls") or ([args['url']] if args.get('url') else [])
            if isinstance(urls, str):
                urls = [urls]
            if isinstance(urls, list):
                for url in urls:
                    if isinstance(url, str):
                        pages.append({"url": url, "domain": urlsplit(url).hostname or "", "title": url, "summary": f"读取失败：{obj['error']}" if obj.get('error') else "正在读取，尚未返回结果。"})
        requested = args.get('urls') or ([args['url']] if args.get('url') else [])
        if isinstance(requested, str):
            requested = [requested]
        return "read", {"name": name, "kind": "read", "goal": "", "pages": pages, "pageCount": len(pages) if output else 0,
                        "singlePageRequest": isinstance(requested, list) and len(requested) == 1}
    return "compute", {"name": name or "tool", "kind": "compute", "goal": "", "code": str(args.get("code") or ""), "stdout": output, "language": "python" if name == "execute_code" else "text"}


def build_steps(messages):
    outputs = {str(m.get("tool_call_id")): str(m.get("content") or "") for m in messages if m.get("role") == "tool" and m.get("tool_call_id")}
    steps = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        thought = str(message.get("reasoning") or message.get("reasoning_content") or message.get("content") or "")
        calls = message.get("tool_calls") or []
        if not calls:
            if not thought:
                continue
            calls = [None]
        for call_index, call in enumerate(calls):
            tool = None
            kind = "answer"
            raw = ""
            call_text = ""
            if call:
                fn = call.get("function") or {}
                name = str(fn.get("name") or "")
                args = object_from_text(fn.get("arguments"))
                arguments = fn.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                call_text = f'{name}({arguments})'
                raw = outputs.get(str(call.get("id") or call.get("call_id")), "")
                kind, tool = tool_view(name, args, raw)
                if '"status":"tool_budget_denied"' in raw:
                    kind, tool = "answer", None
                    call_text = "[未执行：120 次工具预算已耗尽] " + call_text
            display_thought = thought if call_index == 0 else ""
            if call_text:
                display_thought = f'{display_thought}\n\n{call_text}' if display_thought else call_text
            steps.append({"n": len(steps) + 1, "kind": kind, "label": {"search": "搜索", "read": "读取", "compute": "工具", "answer": "综合"}[kind], "thought": display_thought, "model_seconds": None, "tool_seconds": None, "modelStart": None, "modelEnd": None, "toolStart": None, "toolEnd": None, "tool": tool, "rawToolOutput": raw, "toolCallId": str(call.get('id') or call.get('call_id') or '') if call else '', "sourceSessionId": message.get('_viewer_session_id', ''), "sourceContentHash": hashlib.sha256(str(message.get("content") or "").encode()).hexdigest()})
    return steps


def duration(session):
    try:
        return max(0.0, (datetime.fromisoformat(session["last_updated"]) - datetime.fromisoformat(session["session_start"])).total_seconds())
    except (KeyError, ValueError, TypeError):
        return 0.0


def merge_session_chain(snapshots):
    """Keep pre-compaction history and skip inherited child messages."""
    if not snapshots:
        return None
    by_id = {s.get('session_id'): s for s in snapshots if s.get('session_id')}
    current = snapshots[0]
    chain = []
    seen = set()
    while current and current.get('session_id') not in seen:
        seen.add(current.get('session_id'))
        chain.append(current)
        current = by_id.get(current.get('parent_session_id'))
    chain.reverse()
    merged = dict(chain[-1])
    merged['session_start'] = chain[0].get('session_start') or merged.get('session_start')
    merged['messages'] = [{**m, '_viewer_session_id': session.get('session_id', '')} for i, session in enumerate(chain)
                          for m in session.get('messages', [])
                          if i == 0 or not m.get('_inherited_from_previous_session')]
    return merged


def unwrap_tool_output(text):
    if text.startswith('<untrusted_tool_result'):
        start = text.find('\n\n')
        end = text.rfind('\n</untrusted_tool_result>')
        if start >= 0 and end >= 0:
            return text[start + 2:end]
    return text


def enrich_timing(session, steps, db_path, log_path):
    """Use only uniquely matched recorded spans; return False on ambiguity."""
    if not steps or not db_path.is_file() or not log_path.is_file():
        return False
    try:
        origin = datetime.fromisoformat(session['session_start']).timestamp()
        with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as db:
            rows = db.execute('SELECT session_id, role, timestamp, tool_call_id, tool_name, content, tool_calls FROM messages WHERE COALESCE(inherited_from_previous_session, 0)=0 ORDER BY id').fetchall()
        api, tools, warnings, identified_tools = [], [], [], []
        finalizations = []
        for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines():
            stamp = re.match(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)', line)
            if not stamp:
                continue
            ts = datetime.strptime(stamp.group(1), '%Y-%m-%d %H:%M:%S,%f').timestamp()
            m = re.search(r'\[([^\]]+)\] agent\.conversation_loop: API call #\d+: .*?latency=([\d.]+)s', line)
            if m:
                api.append((ts, m.group(1), float(m.group(2))))
            m = re.search(r'agent\.tool_executor: tool (\S+) (completed|failed) \(([\d.]+)s(?:, (\d+) chars)?\)', line)
            if m:
                tools.append((ts, m.group(1), m.group(2), float(m.group(3)), int(m.group(4)) if m.group(4) else None))
            m = re.search(r'agent\.tool_executor: Tool (\S+) returned error \(([\d.]+)s\): (.+)$', line)
            if m:
                warnings.append((ts, m.group(1), 'failed', float(m.group(2)), m.group(3)))
            m = re.search(r'agent\.chat_completion_helpers: glasseek_finalization_timing (\{.*\})', line)
            if m:
                try:
                    record = json.loads(m.group(1))
                    finalizations.append((str(record['session_id']), str(record['content_sha256']),
                                          float(record['started_at']), float(record['duration_seconds'])))
                except (KeyError, TypeError, ValueError):
                    continue
            m = re.search(r'agent\.tool_executor: glasseek_tool_timing (\{.*\})', line)
            if m:
                try:
                    record = json.loads(m.group(1))
                    identified_tools.append((ts, str(record['session_id']), str(record['tool_call_id']),
                                             str(record['tool_name']), float(record['duration_seconds'])))
                except (KeyError, TypeError, ValueError):
                    continue
        assistant = []
        results = {}
        for sid, role, ts, call_id, name, content, calls in rows:
            if role == 'assistant':
                try:
                    ids = tuple(str(x.get('id') or x.get('call_id') or '') for x in json.loads(calls or '[]'))
                except (ValueError, TypeError):
                    return False
                assistant.append((sid, ts, ids, content or ''))
            elif role == 'tool' and call_id:
                results[(sid, str(call_id))] = (ts, name, content or '')
        complete = True
        for step in steps:
            sid, cid = step['sourceSessionId'], step['toolCallId']
            final = [e for e in finalizations if not cid and e[0] == sid
                     and e[1] == step.get('sourceContentHash')]
            if final:
                if len(final) != 1:
                    complete = False
                    continue
                _, _, start, seconds = final[0]
                step.update(modelStart=max(0.0, start - origin), modelEnd=max(0.0, start + seconds - origin),
                            model_seconds=seconds, modelTimingKind='finalization')
                continue
            candidates = [r for r in assistant if r[0] == sid and (cid in r[2] if cid else not r[2] and (hashlib.sha256(r[3].encode()).hexdigest() == step['sourceContentHash'] if step.get('sourceContentHash') else (r[3] or '') == step['thought']))]
            if len(candidates) != 1:
                complete = False
                continue
            _, at, _, _ = candidates[0]
            model = [e for e in api if e[1] == sid and 0 <= at - e[0] < 5]
            if len(model) != 1:
                complete = False
            else:
                end, _, seconds = model[0]
                step.update(modelStart=max(0.0, end - seconds - origin), modelEnd=max(0.0, end - origin), model_seconds=seconds)
            if not cid:
                continue
            row = results.get((sid, cid))
            if row is None:
                complete = False
                continue
            done, name, content = row
            identified = [e for e in identified_tools if e[1] == sid and e[2] == cid and e[3] == name
                          and at - 0.5 <= e[0] <= done + 0.5]
            if len(identified) == 1:
                end, _, _, _, seconds = identified[0]
                step.update(toolStart=max(0.0, end - seconds - origin), toolEnd=max(0.0, end - origin), tool_seconds=seconds)
                tool = step.get('tool') or {}
                if step.get('kind') == 'read' and tool.get('singlePageRequest') and len(tool.get('pages', [])) == 1:
                    tool['pages'][0]['readerSeconds'] = seconds
                continue
            if len(identified) > 1:
                complete = False
                continue
            matching = [e for e in tools if e[1] == name and at - 0.5 <= e[0] <= done + 0.5 and (e[4] == len(unwrap_tool_output(content)) if e[2] == 'completed' else True)]
            if not matching:
                matching = [e for e in warnings if e[1] == name and at - 0.5 <= e[0] <= done + 0.5
                            and unwrap_tool_output(content).startswith(e[4])
                            and sum(1 for peer in candidates[0][2]
                                    if (other := results.get((sid, peer))) is not None
                                    and other[1] == name and unwrap_tool_output(other[2]).startswith(e[4])) == 1]
            if len(matching) > 1:
                # Concurrent calls to the same tool can finish at different times.
                # Only disambiguate when exactly one completion is close to this
                # tool-result timestamp; otherwise keep the timeline unavailable.
                matching = [e for e in matching if abs(e[0] - done) <= 0.5]
            if len(matching) != 1:
                complete = False
                continue
            end, _, _, seconds, _ = matching[0]
            step.update(toolStart=max(0.0, end - seconds - origin), toolEnd=max(0.0, end - origin), tool_seconds=seconds)
            tool = step.get('tool') or {}
            if step.get('kind') == 'read' and tool.get('singlePageRequest') and len(tool.get('pages', [])) == 1:
                tool['pages'][0]['readerSeconds'] = seconds
        return complete
    except (OSError, sqlite3.Error, KeyError, TypeError, ValueError):
        return False


def session_statistics(session, log_path, config_path, steps=None):
    """Use per-call Hermes usage logs; never infer cache hits from absent fields."""
    session_ids = {str(m.get('_viewer_session_id')) for m in session.get('messages', []) if m.get('_viewer_session_id')}
    if session.get('session_id'):
        session_ids.add(str(session['session_id']))
    calls = []
    try:
        for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines():
            match = re.search(r'\[([^\]]+)\] agent\.conversation_loop: API call #(\d+): .*?\bin=(\d+) out=(\d+) total=(\d+) latency=([\d.]+)s(?: cache=(\d+)/(\d+) \(\d+%\))?(?: ttft=([\d.]+)s)?', line)
            if match and match.group(1) in session_ids:
                calls.append(match)
    except OSError:
        pass
    if not calls:
        return {}
    inputs = sum(int(m.group(3)) for m in calls)
    outputs = sum(int(m.group(4)) for m in calls)
    usage = {'inputTokens': inputs, 'outputTokens': outputs, 'totalTokens': inputs + outputs,
             'usageSamples': len(calls)}
    cache_calls = [m for m in calls if m.group(7) is not None]
    if cache_calls:
        cached = sum(int(m.group(7)) for m in cache_calls)
        measured = sum(int(m.group(8)) for m in cache_calls)
        usage.update(cachedInputTokens=cached, uncachedInputTokens=max(0, measured - cached),
                     cacheMeasuredInputTokens=measured,
                     cacheSamples=len(cache_calls))
    context = {'usedTokens': int(calls[-1].group(3))}
    try:
        match = re.search(r'(?m)^  context_length:\s*(\d+)\s*$', config_path.read_text(encoding='utf-8'))
        if match:
            context['capacityTokens'] = int(match.group(1))
    except OSError:
        pass
    result = {'usage': usage, 'context': context,
              'modelSeconds': sum(float(m.group(6)) for m in calls), 'completedCalls': len(calls)}
    ttfts = [float(m.group(9)) for m in calls if m.group(9) is not None]
    if ttfts:
        result.update(ttftSeconds=sum(ttfts) / len(ttfts), ttftSamples=len(ttfts))
    tool_steps = [s for s in (steps or []) if s.get('tool')]
    measured = [s['tool_seconds'] for s in tool_steps
                if isinstance(s.get('tool_seconds'), (int, float)) and not isinstance(s.get('tool_seconds'), bool)
                and s['tool_seconds'] >= 0]
    if measured:
        result.update(toolSeconds=sum(measured), toolSamples=len(measured), toolCalls=len(tool_steps))
    return result


def find_profile(run, case, session, result, profiles_dir):
    """Resolve Hermes' truncated, hashed profile names without guessing by prefix."""
    previous_attempt = result.get('attempt', 1) if isinstance(result, dict) else 1
    if not isinstance(previous_attempt, int) or isinstance(previous_attempt, bool):
        previous_attempt = 1
    attempts = range(1, max(3, previous_attempt + 1) + 1)
    candidates = [profiles_dir / profile_name(run, f'{case}__r1', n) for n in attempts]
    candidates = [path for path in candidates if path.is_dir()]
    if len(candidates) <= 1:
        return candidates[0] if candidates else profiles_dir / '__missing_profile__'

    session_ids = {str(m.get('_viewer_session_id')) for m in session.get('messages', [])
                   if m.get('_viewer_session_id')}
    if session.get('session_id'):
        session_ids.add(str(session['session_id']))
    matched = []
    for path in candidates:
        try:
            log = (path / 'logs/agent.log').read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        if any(f'[{sid}]' in log for sid in session_ids):
            matched.append(path)
    if len(matched) == 1:
        return matched[0]
    # If session matching is ambiguous, do not attribute another attempt's
    # timing or token usage to this trajectory.
    return profiles_dir / '__missing_profile__'


def build_package():
    rows = [json.loads(line) for line in (RUNTIME / f"{DATASET}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    sessions = RUNTIME / "hermes/sessions" / RUN / DATASET
    output = ROOT / "output/preds" / RUN / DATASET
    results = {}
    for path in (output / "conv").glob("*/*.json"):
        item = read_json(path)
        if isinstance(item, dict) and item.get("question_id"):
            results[str(item["question_id"])] = item
    judges = {}
    for path in (output / "judge_official").rglob("*.json"):
        item = read_json(path)
        if isinstance(item, dict) and item.get("question_id"):
            judges[str(item["question_id"])] = item
    active = False
    active_profiles = set()
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = proc.read_bytes().split(b'\0')
            cmd = b' '.join(args).decode(errors='replace')
            if 'browse_comp_eval.run' in cmd and RUN in cmd:
                active = True
            if b'--profile' in args:
                index = args.index(b'--profile')
                if index + 1 < len(args):
                    active_profiles.add(args[index + 1].decode(errors='replace'))
        except OSError:
            pass
    cases = []
    for index, row in enumerate(rows, 1):
        case = str(row["id"])
        snapshots = list((sessions / case / "main").glob("session_*.json"))
        snapshots.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
        session = merge_session_chain([x for x in (read_json(p) for p in snapshots) if isinstance(x, dict)])
        result = results.get(case)
        if not session and not result:
            continue
        session = session or ((result.get("history") or [{}])[0] if result else {})
        steps = build_steps(session.get("messages", []))
        profile = find_profile(RUN, case, session, result, RUNTIME / 'hermes/profiles')
        timing_available = enrich_timing(session, steps, profile / 'state.db', profile / 'logs/agent.log')
        session_stats = session_statistics(session, profile / 'logs/agent.log', profile / 'config.yaml', steps)
        rounds = sum(1 for m in session.get('messages', []) if m.get('role') == 'assistant' and (m.get('tool_calls') or m.get('reasoning') or m.get('reasoning_content') or m.get('content')))
        counts = Counter(s["kind"] for s in steps if s["tool"])
        judge = judges.get(case) or {}
        score = judge.get("score") if judge.get("status") == "judged" else None
        correct = bool(score) if isinstance(score, (int, float)) else None
        running = bool(active and profile.name in active_profiles and
                       (not result or result.get('status') == 'interrupted' or result.get('attempt_consumed') is False))
        error_text = str((result or {}).get("error") or "")
        tails = " ".join(str((result or {}).get(k) or "") for k in ("stdout_tail", "stderr_tail", "agent_log_tail"))
        if (result or {}).get('tool_call_budget_exceeded'):
            correct = False
            eval_status = "tool_budget_exceeded"
        elif running:
            eval_status = "running"
        elif not result:
            eval_status = "interrupted"
        elif result.get("user_terminated"):
            eval_status = "user_terminated"
        elif result.get("status") != "success":
            eval_status = "tool_error" if any(x in error_text.lower() for x in ("tool", "sandbox", "code_execution")) else "inference_error"
        elif "Reached maximum iterations (200)" in tails:
            eval_status = "round_limit"
        elif judge.get("status") == "judge_failed":
            eval_status = "judge_error"
        elif judge.get("status") == "judged":
            eval_status = "correct" if bool(judge.get("score")) else "incorrect"
        else:
            eval_status = "judging"
        elapsed = duration(session)
        answer = str((result or {}).get("model_response") or "")
        stats = {"steps": len(steps), "searchCalls": counts['search'], "readCalls": counts['read'], "computeCalls": counts['compute'], "searchQueries": counts['search'], "resultsScanned": sum(s['tool'].get('resultCount', 0) for s in steps if s['tool']), "pagesRead": sum(s['tool'].get('pageCount', 0) for s in steps if s['tool']), "uniqueSources": 0, "uniqueDomains": 0, "averageStepSeconds": None, "longestStepSeconds": None}
        cases.append({"id": case, "index": index, "field": "BrowseComp", "fieldZh": "BrowseComp", "topic": case, "topicZh": case, "question": row['question'], "answer": answer, "answerNote": "运行中" if running else "时长为会话记录窗口", "judge": "DeepSeek Flash · BrowseComp official", "correct": correct, "evaluationStatus": eval_status, "evaluationError": error_text, "judgeStatus": judge.get("status"), "inProgress": running, "interrupted": bool(not active and not result), "rounds": rounds, "durationSeconds": elapsed, "durationText": f"{int(elapsed // 60)}m {int(elapsed % 60)}s", "model": MODEL, "risk": {"level": "unknown", "label": "未审计"}, "riskKey": "not_audited", "stats": stats, "sessionStats": session_stats, "timing": {}, "timingAvailable": timing_available, "domains": [], "steps": steps, "finalAnswer": answer, "keyIndices": [], "sources": []})
    run = {"name": LABEL, "model": MODEL, "benchmark": "BrowseComp · Quest 128", "date": datetime.now().date().isoformat(), "dateShort": datetime.now().date().isoformat(), "totalCases": len(cases), "requestedCases": len(rows), "judgePassed": sum(c['correct'] is True for c in cases), "judgeScored": sum(c['correct'] is not None for c in cases), "rounds": sum(c['rounds'] for c in cases), "toolCalls": sum(sum(c['stats'][k] for k in ('searchCalls','readCalls','computeCalls')) for c in cases), "pagesRead": sum(c['stats']['pagesRead'] for c in cases), "searchQueries": sum(c['stats']['searchQueries'] for c in cases), "searchCalls": sum(c['stats']['searchCalls'] for c in cases), "inProgress": any(c['inProgress'] for c in cases), "wiki": "Quest 27B 选择性难题子集；DeepSeek Flash 判分；未审计的轨迹不标记为安全。"}
    counts = Counter(c['evaluationStatus'] for c in cases)
    run['statusCounts'] = dict(counts)
    run['errorCount'] = sum(counts[x] for x in ('inference_error', 'tool_error', 'judge_error', 'interrupted', 'user_terminated'))
    run['roundLimitCount'] = counts['round_limit']
    run['userTerminatedCount'] = counts['user_terminated']
    run['unsuccessfulCount'] = counts['incorrect'] + counts['user_terminated']
    run['wiki'] = f"BrowseComp {len(rows)}；已评分 {run['judgeScored']}，通过 {run['judgePassed']}；异常 {run['errorCount']}；达到轮数上限 {run['roundLimitCount']}。"
    return {"generatedAt": datetime.now().astimezone().isoformat(), "run": run, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval must be at least one second')
    while True:
        package = build_package()
        if package['cases']:
            atomic_json(args.output, package)
            print(f"published {len(package['cases'])} cases; running={package['run']['inProgress']}", flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
