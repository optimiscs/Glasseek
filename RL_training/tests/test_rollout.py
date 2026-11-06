import asyncio
import pytest

from glasseek_training.rollout import RolloutSettings, SearchRollout
from glasseek_training.web_tools import parse_tool_calls
from glasseek_training.reward import exact_match


class Tokenizer:
    """Character IDs keep tests independent of model downloads."""
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text) if tokenize else text


class Tools:
    schemas = []
    async def execute(self, call):
        return "evidence:" + "x" * 200


def settings():
    return RolloutSettings(max_model_len=2200, max_prompt_length=200, max_response_length=1600,
                           compact_at=250, max_turn_tokens=200, max_observation_tokens=400,
                           summary_tokens=20, summary_reserve=600, max_turns=4, max_compactions=2)


def test_compaction_preserves_context_and_masks_summary_and_observations():
    tokenizer = Tokenizer()
    call = '<tool_call>{"name":"web_search","arguments":{"query":"test"}}</tool_call>'
    outputs = [call, "short summary", "<answer>Paris</answer>"]
    requests = []
    async def generate(prompt, budget):
        requests.append(list(prompt))
        ids = tokenizer.encode(outputs[len(requests)-1])
        return ids, [-.5] * len(ids)
    loop = SearchRollout(tokenizer, generate, Tools(), settings())
    result = asyncio.run(loop.run([{"role": "user", "content": "question"}]))
    assert result["compactions"] == 1
    assert result["stop_reason"] == "answer"
    first, second = result["segments"]
    assert first["prompt_ids"] == requests[0]
    assert second["prompt_ids"] == requests[2]
    assert "short summary" in tokenizer.decode(second["prompt_ids"])
    assert "evidence" in tokenizer.decode(first["response_ids"])
    optimized = [token for s in result["segments"] for token, mask in zip(s["response_ids"], s["response_mask"]) if mask]
    assert tokenizer.decode(optimized) == call + "<answer>Paris</answer>"
    assert "evidence" in tokenizer.decode(requests[1])


def test_generation_budget_is_failure_and_not_final_answer():
    async def generate(prompt, budget):
        return [ord('x')] * budget, [-1.] * budget
    result = asyncio.run(SearchRollout(Tokenizer(), generate, Tools(), settings()).run(
        [{"role": "user", "content": "question"}]))
    assert result["stop_reason"] == "generation_limit"
    assert result["final_answer"] == ""
    assert len(result["segments"]) == 1


def test_tools_and_answer_extraction():
    xml = '<tool_call><function=web_search><parameter=query>Paris &amp; London</parameter></function></tool_call>'
    assert parse_tool_calls(xml) == [{"name": "web_search", "arguments": {"query": "Paris & London"}}]
    assert parse_tool_calls('<think>' + xml + '</think>Done') == []
    with pytest.raises(ValueError):
        parse_tool_calls('<tool_call>unfinished')
    assert exact_match('<think>London</think><answer>Paris.</answer>', ['Paris', '巴黎']) == 1
    assert exact_match('not Paris', 'Paris') == 0
    assert exact_match('', '') == 0


def test_template_trailing_newline_is_preserved_after_eos():
    class NewlineTokenizer(Tokenizer):
        def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
            text = "".join(f"<{m['role']}>{m['content']}<eos>\n" for m in messages)
            if add_generation_prompt:
                text += '<assistant>'
            return self.encode(text) if tokenize else text
    tokenizer = NewlineTokenizer()
    loop = SearchRollout(tokenizer, None, Tools(), settings())
    suffix = tokenizer.decode(loop.continuation([{'role': 'tool', 'content': 'evidence'}]))
    assert suffix.startswith('\n<tool>')


def test_multiple_compactions_keep_summary_out_of_all_training_likelihoods():
    from glasseek_training.cm_gspo import Rollout, Segment, pack_rollouts
    tokenizer = Tokenizer()
    call = '<tool_call>{"name":"web_search","arguments":{"query":"test"}}</tool_call>'
    outputs = [call, 'SUMMARY_A', call, 'SUMMARY_B', '<answer>Paris</answer>']
    requests = []
    async def generate(prompt, budget):
        index = len(requests)
        requests.append(list(prompt))
        ids = tokenizer.encode(outputs[index])
        return ids, [-7777. if index in (1, 3) else -.5] * len(ids)
    result = asyncio.run(SearchRollout(tokenizer, generate, Tools(), settings()).run(
        [{'role': 'user', 'content': 'question'}]))
    assert result['compactions'] == 2
    segments = [Segment(**item) for item in result['segments']]
    assert len(segments) == 3
    for segment, execution_call in zip(segments, (0, 2, 4)):
        assert segment.prompt_ids == requests[execution_call]
        assert -7777. not in segment.log_probs
    assert 'SUMMARY_A' in tokenizer.decode(segments[1].prompt_ids)
    assert 'SUMMARY_B' in tokenizer.decode(segments[2].prompt_ids)
    expected = call + call + outputs[-1]
    actual = tokenizer.decode([t for s in segments for t, m in zip(s.response_ids, s.response_mask) if m])
    assert actual == expected
    rollouts = [Rollout('q', 1., segments), Rollout('q', 0., segments)]
    data, _ = pack_rollouts(rollouts, group_size=2, pad_token_id=0)
    assert data['response_mask'].sum() == 2*len(expected)
    assert data['old_log_probs'][data['response_mask'].bool()].eq(-.5).all()
    assert data['attention_mask'].sum() > data['response_mask'].sum()
