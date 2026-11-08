"""Transactional validation for model-proposed, versioned working contexts.

This module does not modify an agent's messages, URL registry, or sampled tokens.
Callers own persistence and may commit the returned immutable state only after
constructing and validating the complete next model request.
"""
from dataclasses import dataclass
import re
from typing import Callable, FrozenSet

_REFERENCE = re.compile(r'(?<![A-Za-z0-9_])(?:U[1-9]\d*|D[1-9]\d*@v[1-9]\d*(?:#E[1-9]\d*)?)(?![A-Za-z0-9_])')


@dataclass(frozen=True)
class ContextState:
    version: int
    content: str


@dataclass(frozen=True)
class EditResult:
    state: ContextState
    accepted: bool
    reason: str
    input_tokens: int
    proposed_tokens: int
    references: tuple[str, ...]


def propose_edit(state: ContextState, *, base_version: int, content: str,
                 visible_refs: FrozenSet[str], count_tokens: Callable[[str], int],
                 max_tokens: int) -> EditResult:
    """Validate an editable progress note and preserve old state on rejection.

    visible_refs must be the caller's pre-action visibility snapshot, including
    individual evidence spans only when the model has actually observed them.
    The caller must include pinned messages and tool framing in its budget.
    """
    before = count_tokens(state.content)
    after = count_tokens(content) if isinstance(content, str) else 0
    refs = tuple(dict.fromkeys(_REFERENCE.findall(content))) if isinstance(content, str) else ()
    reason = ''
    if base_version != state.version:
        reason = 'stale_version'
    elif not isinstance(content, str) or not content.strip():
        reason = 'empty_context'
    elif max_tokens < 1 or after > max_tokens:
        reason = 'over_budget'
    elif any(ref not in visible_refs for ref in refs):
        reason = 'unknown_or_future_reference'
    if reason:
        return EditResult(state, False, reason, before, after, refs)
    return EditResult(ContextState(state.version + 1, content), True, 'accepted', before, after, refs)
