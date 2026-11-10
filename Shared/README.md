# Shared runtime contracts

`glasseek_shared` owns the first eight role-neutral Hermes implementations:
search provider and provider ABC, message text normalization, Gemini schema
conversion, iteration budgets, retry bookkeeping/backoff, and tool-result
classification. Both Hermes frontends retain their public module paths as API
adapters and import the same implementations. There is one package link in each
frontend; new logic belongs here, not in those adapters.

These modules were selected because their contracts could be tested independently
of model inference and because they do not resolve assets relative to `__file__`.
Role-specific plugin registration, URL safety, context compression, prompts and
execution safeguards remain with each frontend. The 29 differing modules are
inventoried in `archieve/reports/code-consolidation-20261106/hermes-role-differences.json`.
Unmigrated upstream modules are not automatically classified as obsolete.

`screening.py` and `screening_profiles.json` share answer extraction, candidate
validation, request retries and initial queue-selection policies for the top10
and strong audits. Their prompts, database schemas, result-evidence validation,
and the strong audit's re-audit-first workflow preserve existing semantics.
For example, the historical handling of full-width “unknown” differs between
these policies and is covered by a regression test rather than silently changed.

The monorepo's package links work without installation. For a separately deployed
frontend, install this package alongside that frontend with `pip install ./Shared`
(and its existing HTTP dependency for search), rather than copying its source.
Both roles are tested in separate processes, matching their normal deployment.

## CLM context with Hermes completion

Training and evaluation use the same `--context-manager clm` option and shared
`context/official/run.py` runtime. The default `--context-manager hermes` is unchanged.
CLM retains the pinned upstream context mirror, parser, edit gate, budget decisions,
rollback and checkpoints. `completion.py` changes only the action/finish instructions,
the no-action completion branch, and finish wording in budget/resume notices: a nonempty assistant
answer without tool calls ends the task. There is no answer file or echo submission.
Empty, truncated, reasoning-only and malformed tool responses do not complete it.

The upstream checkout is not edited. Its no-action branch is replaced in each
private task module against one exact source anchor, with a hard failure if that
anchor changes. Provenance records both the pinned commit and modified source hash.
This is an explicit Hermes protocol adaptation, not an unmodified official prompt.
Checkpoints carry `completion-protocol.json` (`hermes-assistant-v1`); old file/echo
checkpoints are retained for audit and cannot be silently resumed under this protocol.

RL stores every sampled token ID, behavior logprob and mask before interpreting the
response. Parsed reasoning/tool/answer fields are execution metadata and do not
rewrite these training arrays. CLM SFT requires canonical bash trajectories ending
with an actual assistant answer, rather than a synthetic answer recovered from a file.

## CLM 上游依赖

官方 CLM 通过 `third_party/context-language-models` 子模块固定到 `483198229ab7b90c0eed4651306ab3b3505aaf50`。检出后运行：

```bash
git submodule update --init third_party/context-language-models
```

运行适配器会验证上游提交与工作区未被修改；模型、凭证和本机运行目录不包含在子模块中。
