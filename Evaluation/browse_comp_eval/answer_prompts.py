"""Dataset-selectable final-answer instructions, independent of research prompts."""
from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path

DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / 'config' / 'answer_prompts.json'


def load_answer_prompt(dataset: str, registry_path: str | Path | None = None) -> tuple[str, str]:
    registry = Path(registry_path or os.environ.get('GLASSEEK_ANSWER_PROMPT_REGISTRY') or DEFAULT_REGISTRY).expanduser().resolve()
    data = json.loads(registry.read_text(encoding='utf-8'))
    mappings = data.get('datasets')
    if not isinstance(mappings, dict) or not isinstance(data.get('default'), str):
        raise ValueError('answer prompt registry requires datasets and default')
    if dataset in mappings:
        key = dataset
    else:
        matching = [pattern for pattern in mappings if fnmatch.fnmatchcase(dataset, pattern)]
        if len(matching) > 1:
            raise ValueError(f'ambiguous answer prompt mappings for {dataset!r}')
        key = matching[0] if matching else 'default'
    relative = data['default'] if key == 'default' else mappings[key]
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f'invalid answer prompt path for {key!r}')
    path = (registry.parent / relative).resolve()
    prompt = path.read_text(encoding='utf-8').strip()
    if not prompt:
        raise ValueError(f'empty answer prompt: {path}')
    return key, prompt
