"""Versioned, static URL-protocol prompt plugin; never reads a live registry."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from .url_registry import add_url_argument, mode_value

DEFAULT_CONFIG = Path(__file__).with_name('url_prompt_config.json')
MARKER = '<glasseek_url_prompt '


def configure_url_arguments(args):
    """Resolve user paths before launchers change cwd or start workers."""
    spec = args.url_prompt
    if spec not in ('auto', 'off'):
        spec = str(Path(spec).expanduser().resolve())
    os.environ['GLASSEEK_URL_MODE'] = args.url_mode
    os.environ['GLASSEEK_URL_PROMPT'] = spec
    return prompt_module(args.url_mode, spec)


def prompt_module(mode=None, spec=None):
    mode = mode_value(mode)
    spec = spec if spec is not None else os.environ.get('GLASSEEK_URL_PROMPT', 'auto')
    if mode == 'raw' or spec == 'off':
        return {'enabled': False, 'mode': mode, 'text': ''}
    path = DEFAULT_CONFIG if spec == 'auto' else Path(spec).expanduser().resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    version = config.get('version')
    if not isinstance(version, str) or not version or not all(c.isalnum() or c in '.-_' for c in version):
        raise ValueError('URL prompt config needs an alphanumeric version')
    blocks = [config['common'], config['modes'][mode], *config['few_shot']['common']]
    blocks.extend(config['few_shot'].get(mode, []))
    descriptions = dict(config.get('tool_descriptions', {}))
    for tool, extra in config.get('tool_mode_descriptions', {}).get(mode, {}).items():
        descriptions[tool] = (descriptions.get(tool, '') + '\n' + extra).strip()
    if not all(isinstance(tool, str) and isinstance(text, str) and text.strip()
               for tool, text in descriptions.items()):
        raise ValueError('Tool prompt descriptions must be nonempty strings')
    blocks.extend('Tool protocol: ' + tool + '\n' + text for tool, text in sorted(descriptions.items()))
    if not all(isinstance(block, str) and block.strip() for block in blocks):
        raise ValueError('URL prompt blocks must be nonempty strings')
    body = '\n\n'.join(blocks)
    digest = hashlib.sha256((mode + '\n' + body).encode()).hexdigest()
    text = (f'{MARKER}version="{version}" mode="{mode}" sha256="{digest}">\n'
            + body + '\n</glasseek_url_prompt>')
    return {'enabled': True, 'mode': mode, 'version': version, 'sha256': digest,
            'config': str(path), 'text': text, 'tool_descriptions': descriptions}


def append_system_prompt(base, mode=None, spec=None):
    module = prompt_module(mode, spec)
    if not module['enabled']:
        if mode_value(mode) != 'raw' and MARKER in base:
            raise ValueError('Prepared system prompt already contains a URL plugin; cannot disable it silently')
        return base
    text = module['text']
    if MARKER in base:
        if base.count(MARKER) == 1 and text in base:
            return base
        raise ValueError('Existing URL prompt differs from selected plugin; regenerate the data or start a new session')
    return base + ('\n\n' if base else '') + text


def apply_to_messages(messages, mode=None, spec=None):
    if not prompt_module(mode, spec)['enabled'] and not any(
            m.get('role') == 'system' and MARKER in (m.get('content') or '') for m in messages):
        return messages
    result = copy.deepcopy(messages)
    if result and result[0].get('role') == 'system':
        result[0]['content'] = append_system_prompt(result[0].get('content') or '', mode, spec)
    else:
        text = append_system_prompt('', mode, spec)
        if text:
            result.insert(0, {'role': 'system', 'content': text})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_url_argument(parser)
    args = parser.parse_args()
    print(prompt_module(args.url_mode, args.url_prompt)['text'])
