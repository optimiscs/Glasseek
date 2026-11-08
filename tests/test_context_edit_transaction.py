from glasseek_shared.context.edit import ContextState, propose_edit


def call(state, content, **kwargs):
    return propose_edit(state, base_version=kwargs.get('version', state.version), content=content,
        visible_refs=frozenset(kwargs.get('visible', {'U1', 'D1@v1', 'D1@v1#E1'})),
        count_tokens=len, max_tokens=kwargs.get('limit', 1000))


def test_edit_commits_new_version_without_mutating_old_state():
    old = ContextState(3, 'evidence <url>U1</url>')
    result = call(old, 'claim <url>D1@v1#E1</url>')
    assert result.accepted and result.state.version == 4
    assert old.content == 'evidence <url>U1</url>' and old.version == 3


def test_unknown_and_future_refs_are_rejected_transactionally():
    old = ContextState(1, 'prior')
    for content in ('<url>U2</url>', 'D1@v1#E2', '<url>D2@v1</url>'):
        result = call(old, content)
        assert not result.accepted and result.state is old
        assert result.reason == 'unknown_or_future_reference'


def test_stale_over_budget_and_empty_edits_keep_old_context():
    old = ContextState(3, 'original')
    for result in (call(old, 'valid', version=2), call(old, 'too long', limit=2), call(old, '  ')):
        assert not result.accepted and result.state is old
    assert [call(old, 'valid', version=2).reason, call(old, 'too long', limit=2).reason] == ['stale_version', 'over_budget']
