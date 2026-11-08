# Trajectory curation and human review

Install the shared package and preprocessing dependencies from the repository root:

```bash
python -m pip install -e Shared -r scripts/trajectory_preprocess/requirements.txt
```

The pipeline normalizes XML tool calls, preserves Qwen `user` / `<tool_response>` messages, records call/response identities, removes runtime telemetry, performs exact deduplication and flags similar questions. `glasseek_contract.py` checks the evaluation launcher when present; a clean checkout uses the tracked standard `Evaluation/scripts/run_eval.sh` profile. Both are static checks, not execution receipts. OpenResearcher is excluded.

Run `full_prepare.py --help`, `migrate_searchswarm.py --help`, `safety_pipeline.py --help`, and `publish_release.py --help` for independent batch entrypoints. Use a new output directory whenever rules or inputs change. `config.example.json` maps stages to OpenAI-compatible API or local providers. Supply credentials through environment variables or local key files; do not commit credentials or datasets. The default safety policy ignores email/phone and checks credentials, answer leakage, URLs, code and tool contracts.

The launcher `start_full_prepare.sh OUTPUT` accepts `PREPROCESS_PYTHON` and `PREPROCESS_WORKERS`. Its AutoDL defaults are conveniences; override them on other hosts. Input patterns are configured in `inputs.datasets.json`.

## Human review

The UI lives in the companion [QUEST viewer](https://github.com/optimiscs/QUEST). Check out both repositories as sibling folders, or set `QUEST_ROOT` and `GLASSEEK_ROOT`. The viewer backend mounts the review API only when the Glasseek review package exists. Set `GLASSEEK_REVIEW_DB` to a prepared batch database, then run:

```bash
bash scripts/trajectory_review/start.sh --host 127.0.0.1 --port 6080
```

Open `/curation.html`. Prepare a bounded batch with `python scripts/trajectory_review/prepare.py --help`, and attach redacted source snapshots with `python scripts/trajectory_review/attach_sources.py --help`. Source browsing offers role-based messages, original JSON, downloads and full-record search. Source coordinates and hash mismatches remain visible. Review identities are self-reported; this is a local review tool, not an authenticated multi-user service.

The 12-question rubric is scored server-side. Critical failures override totals; unknown answers cannot pass. Program labels are revealed after submission, with per-check agreement. Single reviews are not adjudicated gold labels. No human approvals, tool replay receipts or training readiness are fabricated. Publishing requires hash-bound safety, quality, replay and review acceptance; the SFT loader verifies the release manifest before loading.

## Offline tests

```bash
export PYTHONPATH=Shared:scripts/trajectory_preprocess:scripts/trajectory_preprocess_pilot:scripts
python -m unittest discover -s scripts/trajectory_preprocess -p 'test_*.py' -q
python -m unittest discover -s scripts/trajectory_preprocess_pilot -p 'test_*.py' -q
python -m unittest discover -s scripts/trajectory_review -p 'test_*.py' -q
python -m unittest discover -s scripts/trajectory_benchmark -p 'test_unify_trajectory_candidates.py' -q
```

These tests use synthetic records and mocked transports; no paid model calls or live tool execution are required. Generated datasets, review databases, credentials and local reports stay outside this change.
