"""Opt-in CLM working-note engine; builtin compression remains the budget fallback."""
import json
from glasseek_shared.context.manager import WorkingContext
from glasseek_shared.context.config import SCHEMA
from agent.model_metadata import estimate_messages_tokens_rough

class CLMContextEngine:
    name = 'clm'
    def __init__(self, fallback):
        self.fallback = fallback
        self.working = WorkingContext()
        self.pending = None

    def __getattr__(self, name):
        return getattr(self.fallback, name)

    def get_tool_schemas(self):
        return [SCHEMA["function"]]

    def has_content_to_compress(self, messages):
        return self.pending is not None or self.fallback.has_content_to_compress(messages)

    def should_compress(self, *args, **kwargs):
        return self.pending is not None or self.fallback.should_compress(*args, **kwargs)

    def should_compress_preflight(self, *args, **kwargs):
        return self.pending is not None or self.fallback.should_compress_preflight(*args, **kwargs)

    def handle_tool_call(self, name, args, **kwargs):
        messages = kwargs.get('messages') or []
        # Freeze visibility at this action. Assistant claims never publish IDs.
        latest = next((m for m in reversed(messages) if m.get('role') == 'assistant'), {})
        if len(latest.get('tool_calls') or []) > 1:
            return json.dumps({'accepted':False,'reason':'edit_requires_exclusive_turn','version':self.working.state.version})
        observations = [str(m.get('content') or '') for m in messages if m.get('role') == 'tool']
        receipt = self.working.edit(args, observations,
            lambda text: estimate_messages_tokens_rough([{'role':'user','content':text}]),
            validate=lambda text: estimate_messages_tokens_rough(
                [m for m in messages if m.get('role') == 'system'] +
                [next((m for m in messages if m.get('role') == 'user'), {'role':'user','content':''})] +
                [{'role':'user','content':text}, latest]) + 256 < self.fallback.threshold_tokens)
        if json.loads(receipt)['accepted']:self.pending = self.working.note()
        return receipt

    def compress(self, messages, *args, **kwargs):
        if self.pending is None:return self.fallback.compress(messages, *args, **kwargs)
        # Keep the latest complete action and all its tool results together.
        last = next((i for i in range(len(messages)-1,-1,-1) if messages[i].get('role')=='assistant'),len(messages))
        pinned = [m for m in messages[:last] if m.get('role')=='system']
        task = next((m for m in messages[:last] if m.get('role')=='user'),None)
        if task is not None:pinned.append(task)
        candidate = pinned + [self.pending] + messages[last:]
        self.pending = None
        if estimate_messages_tokens_rough(candidate) >= self.fallback.threshold_tokens:
            self.working.events.append({'accepted':False,'reason':'fallback_request_over_budget'})
            return self.fallback.compress(messages, *args, **kwargs)
        self.fallback.compression_count += 1
        self.fallback.last_prompt_tokens = estimate_messages_tokens_rough(candidate)
        return candidate
