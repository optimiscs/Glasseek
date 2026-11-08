"""Bind upstream harness globals to a private task module, never shared globals."""
import hashlib
import json
import importlib.util
from pathlib import Path
import re
import sys
import uuid
from . import activate, UPSTREAM


def create_agent(*, workspace, **kwargs):
    activate()
    workspace = Path(workspace).resolve()
    # Upstream interpolates paths into bash and Python snippets without quoting.
    # Refuse ambiguous paths rather than changing upstream command semantics.
    if not re.fullmatch(r'[A-Za-z0-9_./-]+', str(workspace)):
        raise ValueError('Official CLM workspace path must contain only letters, digits, _, ., / and -')
    ctx = workspace / '.clm/context'
    state = workspace / '.clm/bash'
    for directory in (ctx, state):
        directory.mkdir(parents=True, exist_ok=True)
    name = 'clm_harness.clm_agent._glasseek_' + uuid.uuid4().hex
    source = UPSTREAM / 'clm/clm_harness/clm_agent/harness.py'
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        from .completion import patch_loop, prompt_template, install, PROTOCOL
        source_text = source.read_text()
        patched_source = patch_loop(source_text)
        exec(compile(patched_source, str(source), 'exec'), module.__dict__)
        original = module._SYSTEM_TEMPLATE
        replacements = {module._CTX_FILE: str(ctx / 'LIVE_CTX_MAIN.txt'),
                        module._CTX_DIR: str(ctx), module._STATE_DIR: str(state)}
        # Longest-first prevents replacing the directory inside the file path twice.
        prompt = prompt_template(original)
        for old, new in sorted(replacements.items(), key=lambda item: -len(item[0])):
            prompt = prompt.replace(old, new)
        module._CTX_DIR = str(ctx)
        module._CTX_FILE = str(ctx / 'LIVE_CTX_MAIN.txt')
        module._STATE_DIR = str(state)
        module._SYSTEM_TEMPLATE = prompt
        kwargs['finish_policy'] = 'open_ended'
        kwargs['task_template'] = 'open_problems'
        agent = module.ClmAgent(**kwargs)
        install(agent, module)
        agent.glasseek_paths = dict(workspace=str(workspace), context_dir=str(ctx),
            context_file=module._CTX_FILE, bash_state_dir=str(state))
        (workspace/'.clm/paths.json').write_text(json.dumps(agent.glasseek_paths,indent=2))
        agent.glasseek_prompt = dict(source_sha256=hashlib.sha256(original.encode()).hexdigest(),
            rendered_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            path_substitutions=replacements, completion_protocol=PROTOCOL,
            loop_source_sha256=hashlib.sha256(patched_source.encode()).hexdigest(),
            completion_implementation_sha256=hashlib.sha256((Path(__file__).parent/'completion.py').read_bytes()).hexdigest(),
            completion_patch_sha256=agent.glasseek_completion_patch_sha256)
        agent.glasseek_module_name = name
        return agent
    except BaseException:
        sys.modules.pop(name, None)
        raise


def release_agent(agent):
    sys.modules.pop(agent.glasseek_module_name, None)
