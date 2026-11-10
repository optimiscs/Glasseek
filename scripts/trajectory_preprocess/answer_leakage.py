"""Offline evidence checks derived from Glasseek's BrowseComp guard.

Use its pure overlap/marker helpers, never its runtime hooks or blanket hosting
blocklist. A HuggingFace URL, task prompt, or answer string alone proves no leak.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from glasseek_shared.historical_tools import load, search_sections

PLUGIN = Path(__file__).resolve().parents[2] / 'Evaluation_backend/plugins/browse_comp_guard/__init__.py'
SPEC = importlib.util.spec_from_file_location('_curation_browsecomp_guard', PLUGIN)
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)
PLUGIN_SHA256 = hashlib.sha256(PLUGIN.read_bytes()).hexdigest()
GOLD = re.compile(r'''(?im)(?:^|[\n{,])\s*["']?(?:ground[ _-]?truth|reference[ _-]?answers?|gold(?:en)?[ _-]?answer|answer[ _-]?key|expected[ _-]?answer|标准答案|参考答案|标注答案|答案库)["']?\s*[:=：]\s*\S''')
ANSWER_RESOURCE = re.compile(r'''(?ix)(?:\[\s*(?:answer[ _-]?keys?|reference[ _-]?answers?|标准答案|参考答案|标注答案)\s*\]\s*\(|(?:[/=?&][^\s<>"'`]{0,300}[_/.-])?(?:answer[_-]?keys?|reference[_-]?answers?|ground[_-]?truth|gold[_-]?answers?)(?:\.(?:jsonl?|csv|parquet|txt|html?|pdf))?(?=$|[\s"'<>)\]\x60&#?]))''')
VISIBLE = guard._VISIBLE_RESULT_FIELDS | {'summary', 'answer', 'reference_answer', 'ground_truth', 'question', 'evidence'}
IGNORED = {'query','queries','goal','source_goal','request','request_args','arguments','metadata','diagnostics','url','urls'}


def visible_text(text):
    try: value = load(text)
    except ValueError: return text
    def walk(value, visible=False, depth=0):
        if depth>32: raise ValueError('answer_structure_depth_limit')
        if isinstance(value, str):
            if value.lstrip().startswith(('{','[','"')):
                try: nested=load(value)
                except ValueError: nested=value
                if not isinstance(nested,str):return walk(nested,visible,depth+1)
            sections=search_sections(value)
            if sections is not None:
                return '\n'.join(body for _,body,_ in sections) if visible else ''
            header=re.match(r'^\s*The useful information in https?://\S+ for user goal [\s\S]*? as follows:\s*\n',value)
            return (value[header.end():] if header else value) if visible else ''
        if isinstance(value, list): return '\n'.join(walk(v, visible,depth+1) for v in value)
        if isinstance(value, dict):
            return '\n'.join((str(k)+': ' if str(k).lower() in {'answer','reference_answer','ground_truth','question'} else '')+
                walk(v, visible or str(k).lower() in VISIBLE or str(k).lower()=='result',depth+1) for k,v in value.items() if str(k).lower() not in IGNORED)
        return ''
    return walk(value,isinstance(value,str))


def question_text(item):
    if isinstance(item.get('question'), str) and item['question'].strip(): return item['question']
    for message in item.get('messages', []):
        text = message.get('content', '')
        if message.get('role') == 'user' and isinstance(text, str) and not text.lstrip().startswith('<tool_response>'):
            return text
    return ''


def external_signals(text, question, max_compare_chars=120000):
    try: body = visible_text(text)
    except ValueError:
        return [('coverage.answer_structure_depth',{'limit':32})]
    signals = []
    marker = bool(GOLD.search(body) or ANSWER_RESOURCE.search(body))
    if guard._BENCHMARK_RE.search(body) or marker:
        signals.append(('benchmark.reference', {'scope':'external_evidence', 'gold_marker':marker}))
    if not question.strip(): return signals
    # This limit is surfaced, never reported as a complete clean result.
    if len(body) > max_compare_chars:
        signals.append(('coverage.answer_overlap_limit', {'limit':max_compare_chars}))
        body = body[:max_compare_chars]
    normalized_question = re.sub(r'\s+', ' ', question).strip().casefold()
    normalized_body = re.sub(r'\s+', ' ', body).strip().casefold()
    qtokens = [t for t in guard._word_tokens(question) if t not in guard._QUESTION_STOP_WORDS]
    rtokens = set(guard._word_tokens(body))
    overlap = False
    if len(qtokens) >= 8 and len(set(qtokens) & rtokens) / max(1,len(set(qtokens))) >= .2:
        overlap = guard._contains_question_leak({'content':body}, question)
    # The guard's word tokenizer is English-only; a long exact Chinese prompt
    # can be checked without guessing semantic similarity or using a model.
    if len(normalized_question) >= 40 and normalized_question in normalized_body:
        overlap = True
    if overlap:
        answer_marker = marker or bool(re.search(r'(?im)(?:^|\n)\s*(?:answer|答案)\s*[:：]', body))
        signals.append(('benchmark.current_question_answer' if answer_marker else 'benchmark.question_overlap',
                        {'question_overlap':True,'answer_marker':answer_marker,
                         'scope':'external_visible_content','causality':'exposure signal; use by assistant requires review'}))
    return signals
