# CLM context with Hermes completion

Smoke run: `clm_hermes_completion_smoke_20261108_122642`

The original 34-task evaluation remains paused. The smoke uses two original tasks with the existing evaluation entrypoint and --context-manager clm.

| Task | Submission | Official judge | Wall seconds | Search / extract | Input / output tokens |
|---|---|---:|---:|---|---|
| row0084 | Native assistant | 1.0 | 202.27 | 23 / 5 | 604478 / 7636 |
| row0130 | Native assistant | 1.0 | 213.86 | 16 / 2 | 351306 / 12096 |

Both tasks have a final assistant message without tool calls, no answer.txt, real viewer timing and saved request-context breakdowns.

37 deterministic checks passed, covering real shell execution and mirror edits, checkpoint save/restore, unchanged upstream budget/rollback decisions, incomplete-output rejection, and default CLI behavior.

A real vLLM RL sampling check completed a tool action and a final answer (72 + 25 = 97 tokens). Every prompt/token/logprob/mask array matched the captured sampler return exactly. A separate deterministic test verifies the same equality across an actual mirror edit.

Implementation: pinned upstream source remains pristine. A source-anchored change to its no-action branch runs in each private task module. Only action/completion instructions and completion wording in budget/resume notices change; upstream edit, budget and checkpoint mechanisms are retained. Source and patch hashes are recorded.

Validation limits:

- Only two original research tasks; not a 34-task quality benchmark.
- RL check exercises real sampling and the shared loop, not a full optimizer step.
- Answer formatting remains prompt-controlled as in Hermes; row0130 did not start its explanation with the literal Explanation: label.
- File/echo checkpoints cannot be silently resumed under the new completion protocol.
