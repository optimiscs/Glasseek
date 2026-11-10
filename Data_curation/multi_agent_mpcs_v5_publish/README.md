# Multi-Agent MPCS v5

A multi-agent workflow for generating, expanding, and validating multi-hop web research questions:

`Seed → Fact verification → Root constraints → Local expansion → Question structure checks → Solver / Ambiguity review / Repair`

This is the public distribution of v5p1. The Python package name `browsecomp_v2` and some script names retain their historical naming. The `hermes/` directory bundles the Hermes runtime source used by the workflow, including local modifications, so execution no longer depends on an external Hermes checkout on the author's machine. Historical reports, legacy evaluations, test suites, and development maintenance materials are excluded. Skill instructions, prompts, plugin configurations, dependency manifests, assets, and licenses needed at runtime are retained.

**The templates contain no working credentials or private service addresses. Configure your own model and web services before a live run. Original run data, evaluation datasets, login credentials, and Python virtual environments are not included.**

## Directory layout

| Path | Purpose |
| --- | --- |
| `browsecomp_v2/` | Workflow, prompts, validators, and runners |
| `hermes/` | Modified Hermes 0.17.0 source, original license, dependency manifests, and lockfiles |
| `scripts/` | Full workflow, individual stage entrypoints, and result export tools |
| `start_v5p1.sh` | Launcher using the bundled Hermes runtime |
| `.env_v5.example` | v5 workflow configuration template |
| `config/` | Hermes configuration and service credential templates |
| `examples/sample_seed.json` | Fictional seed for offline dry-runs only |
| `workflow_viewer.html` | Local viewer for workflow results |
| `data/`, `.hermes/` | Generated at runtime and ignored by Git |

## 1. Installation

Linux, Bash, and Python 3.11 are recommended. The bundled Hermes declares support for Python `>=3.11,<3.14`; the launcher and process management have been validated on Linux. Run all commands below from this pipeline directory (`Data_curation/multi_agent_mpcs_v5_publish/`):

```bash
python3.11 -m venv hermes/.venv
hermes/.venv/bin/python -m pip install -e ./hermes
```

Dependencies are defined in `hermes/pyproject.toml`, and `hermes/uv.lock` is retained. If uv is installed, use the following command instead of the installation steps above to create `hermes/.venv` from the lockfile:

```bash
uv sync --project hermes --locked
```

Do not copy a `.venv` from another machine. The pip installation does not lock every transitive dependency; use the retained uv lockfile when stricter dependency reproducibility is needed.

The default example uses web services and does not enable browser tools. Enabling browser tools requires Node.js 20+, `agent-browser`, and the corresponding browser dependencies; check these through the Hermes tool configuration interface. Other optional tools may also require additional installation.

## 2. Configure models and web services

```bash
cp .env_v5.example .env_v5
mkdir -p .hermes
cp config/hermes-config.example.yaml .hermes/config.yaml
cp config/.env.hermes.example .hermes/.env
chmod 700 .hermes
chmod 600 .env_v5 .hermes/config.yaml .hermes/.env
```

Edit `.env_v5` and fill in at least the following:

```dotenv
DEFAULT_API_KEY=YOUR_MODEL_SERVICE_API_KEY
DEFAULT_BASE_URL=YOUR_MODEL_SERVICE_API_URL
DEFAULT_MODEL=YOUR_SUPPORTED_MODEL_NAME
DEFAULT_PROVIDER=custom
DEFAULT_BACKEND=hermes
DEFAULT_API_MODE=chat_completions
```

The `YOUR_...` values are placeholders, not working configuration. Use a model that supports tool calling and a matching API mode; the default template targets a Chat Completions-compatible service. Empty role-specific values such as `SEED_MODEL`, `CONSTRAINT_MODEL`, and `SOLVER_MODEL` fall back to `DEFAULT_MODEL`. For experiments using multiple models, set `*_MODEL`, `*_API_KEY`, and `*_BASE_URL` for each role.

The default Solver toolset includes `vision`. If the default model does not support images, configure a working vision model under `auxiliary.vision` in `.hermes/config.yaml`, or explicitly remove tools you will not use.

Values in the workflow's `.env_v5` override process environment variables with the same names. Its simple parser **does not support inline comments, `export KEY=...`, variable interpolation, or general `~` expansion**. Put comments on separate lines. Hermes YAML supports `${VARIABLE}`; the examples use this syntax to reference model settings loaded by the workflow without duplicating credentials.

Search routing is an exception: the shared endpoint selected before launch is retained across workflow and Hermes dotenv reloads.

### Shared search service

Search uses the same pool as training and evaluation. Start it using the [shared search pool instructions](../../Evaluation/services/README.search-pool.md). `start_v5p1.sh` and the workflow configuration loader select the central endpoint; a stale Hermes profile cannot redirect searches to a different provider. Upstream Serper keys stay in the pool service. The example still uses Tavily for page extraction, so configure `TAVILY_API_KEY` for extraction or use a fetch service below.

### Fetch service and remote search endpoint

To use a fetch service, replace the `web` section with `config/hermes-web-services.example.yaml` and configure its URL and credentials. To reach a shared search pool on another host, set `SEARCH_SERVER_ENDPOINT` before launch. A standalone copy outside the Glasseek repository must set `GLASSEEK_SEARCH_POOL_ENDPOINT` explicitly and run its Python workflow entry point. See the client implementations for the expected protocols:

- `hermes/plugins/web/search_server/provider.py`
- `hermes/plugins/web/fetch_server/provider.py`

Changing retrieval services changes search results, extracted page text, tool call counts, and question difficulty. The public templates do not guarantee reproduction of historical experiment results.

## 3. Check and run

### Runtime check

```bash
./start_v5p1.sh --check
```

This checks Python, Hermes imports, and YAML structure. The reported `aiagent_source` should point to the bundled `hermes/run_agent.py`. It sends no model requests and **does not validate credentials, network connectivity, or retrieval service availability**.

### Offline dry-run

```bash
hermes/.venv/bin/python scripts/run_workflow.py \
  --env .env_v5.example \
  --seed examples/sample_seed.json \
  --out-dir data/runs_dry \
  --dry-run
```

The dry-run uses a fictional seed to exercise the program flow. It does not call a model or validate facts, retrieval, or question quality. Do not use this seed as production input.

### Live run with one seed

After configuring your services:

```bash
./start_v5p1.sh 1 v5_demo
```

This command makes real service calls and incurs costs. The launcher defaults to one concurrent Seed pipeline, three Solver rollouts, and one concurrent Solver rollout. Local concurrency in the template is two. Increase the workload after a small run succeeds:

```bash
SEED_CONCURRENCY=2 SOLVER_CONCURRENCY=2 ./start_v5p1.sh 5 v5_batch
```

The launcher writes `SEED_CONCURRENCY`, `SEED_PREFETCH`, `SOLVER_ROLLOUTS`, and `SOLVER_CONCURRENCY` into a temporary configuration, overriding their corresponding values from `.env_v5` without modifying the original file. Edit `.env_v5` directly for other workflow settings.

Use `ENV_FILE`, `HERMES_PROJECT_ROOT`, `HERMES_VENV`, `HERMES_HOME`, and `PYTHON_BIN` to override configuration or runtime locations; absolute paths are recommended. Default paths are resolved relative to the launcher's directory.

### Existing seeds and individual stages

```bash
hermes/.venv/bin/python scripts/run_workflow.py \
  --env .env_v5 \
  --seed path/to/verified_seed.json \
  --run-id v5_existing \
  --resume
```

Real seeds use the same structure as the example but must contain verifiable targets, answers, and sources. Individual stage entrypoints are `run_seed_stage.py`, `run_expand_stage.py`, and `run_question_stage.py`; use each script's `--help` for its arguments. Running the Python scripts directly also selects the bundled Hermes by default and uses the concurrency settings in `.env_v5`.

## 4. Pipeline logic

The following describes the defaults in `.env_v5.example`:

1. **Seed**: Generate or read a target entity and reference answer. After deduplication, perform two independent fact checks to verify source support and answer cardinality. Failed seeds may be repaired and checked again.
2. **Root**: Build core and distractor constraints. Each core constraint defines a candidate set: no single core constraint uniquely identifies the target, while the intersection of all core candidate sets contains only the target. Check facts, sources, and ambiguity.
3. **Local**: Recursively expand intermediate entities in the constraints into a multi-hop constraint tree. The default maximum depth is four. Check factual support and distinct parent/child sources.
4. **Question**: Select tree paths to construct a question, checking its structure, paths, and coverage of deeper clues. The default limit is 220 words. With `V2_QUESTION_STRUCTURAL_ONLY=1`, separate semantic and uniqueness checks after question generation are skipped; ambiguity in the public wording is reviewed after Solver attempts. Seed and Root checks still run.
5. **Solver / Judge**: Run three independent solution attempts by default. The Solver receives neither the reference answer nor the internal constraint tree; an independent Judge scores its response. Compute separate medians for API calls and tool calls across successful trajectories. The minimum acceptable effort is at least 10 API calls **and** at least 20 tool calls.
6. **Ambiguity review and Repair**: Independently verify incorrect answers. A question is considered ambiguous only when evidence shows that an alternative answer satisfies every condition in the question. Revise ambiguous or overly easy questions using the existing tree, then retest. The default budget allows up to five question revisions, including up to three further shortcut-pruning revisions, while retaining the better accepted version.

For the default three Solver attempts:

| Result | Status and meaning |
| --- | --- |
| No correct attempts | `review:all_wrong`: retained for human review; this does not prove the question is unambiguous |
| One correct attempt with sufficient effort | `accepted:hard` |
| Two or three correct attempts with sufficient effort | `accepted:high_effort` |
| A correct answer with insufficient effort | Revise the question; reject it if the revision budget is exhausted |
| Execution or judging failure | `verification_failed`; not counted as an incorrect answer |

The launcher's seed count specifies how many seeds to process, not a guaranteed number of valid questions. The driver's "effective question" count includes only `accepted:hard` and `review:all_wrong` results with at most one correct attempt, so it includes questions awaiting human review. Models, search services, and configuration changes all affect results.

## 5. Outputs and resuming runs

For `--run-id v5_demo`, the main outputs are:

- `data/seeds/generated_v5_demo/`: generated seeds.
- `data/runs_v5_demo/`: artifacts, constraint trees, question revisions, Solver results, and trajectories.
- `data/tool_workspaces/`, `data/solver_workspaces/`: tool workspaces managed by the launcher.
- `.hermes/`: Hermes configuration, sessions, logs, and runtime state.

The launcher enables `--resume`. Reusing a run ID may reuse completed results; use a new ID for an independent experiment. Delete run directories only when their contents are no longer needed.

## 6. Resources, permissions, and publication notes

- **Cost and duration**: Low concurrency does not imply a low total request count. Verification, retries, Repair, and subagents add requests. The template allows up to 200 Solver iterations and 7,200 seconds per attempt; configure service-side spending limits yourself.
- **Memory**: The template requires at least 6,144 MB of available memory and reserves 1,024 MB per call for admission control. Small containers may wait for resources. Lower concurrency first, then tune protection settings to the available resources; do not disable protection simply to eliminate waiting.
- **Execution permissions**: `terminal`, `file`, and `code_execution` can execute code or access files visible to the runtime account. A tool working directory is not an operating-system sandbox. Use a separate container or dedicated account and avoid mounting personal credentials or unrelated data.
- **Logs and data**: Conversations, page content, and tool outputs are saved by default and may contain sensitive information or third-party material. Log redaction cannot cover every returned value. Review outputs before sharing them, and do not force-add ignored directories.
- **Configuration isolation**: The workflow defaults to its own `.hermes/` directory. There is no need to copy the author's or another Hermes installation's login state. Keep API keys in private local configuration files.
- **Historical result processing**: v49-specific exporters are retained under `archieve/Data_curation/multi_agent_mpcs_v5_publish/scripts/` at the repository root. Current trajectory processing lives in `scripts/trajectory_preprocess/`.
- **Licensing**: The Hermes MIT license and component attribution are retained; see below. No license has been assigned to the first-party workflow. Its rights holder must confirm ownership and licensing before public release.
- **Redaction scope**: Original deployment addresses, personal paths, user configuration, and authentication state have been removed. Example paths containing specific usernames have been replaced with generic examples. Loopback addresses, container service accounts, generic system paths, public service domains, and required attribution are retained. Newly generated runtime data must be reviewed separately.

## 7. Validation scope

This distribution provides `--check` and the offline dry-run as runtime checks. Historical validation reports and development test suites are not included.

End-to-end validation with real models and search services requires your own configuration. Passing offline checks does not establish model quality or reproduction of historical experiment results.

## 8. Third-party provenance and licensing

The bundled `hermes/` is based on [NousResearch/Hermes-Agent](https://github.com/NousResearch/Hermes-Agent) 0.17.0 at base commit `1456f09e46bd842e7958e29c315681cddfe276f0`, with local modifications. It is not an unmodified upstream release. Local changes cover auxiliary models, response transport, web search and extraction, BrowseComp guards, and request trace capture.

This distribution additionally removes development documentation, historical evaluations, tests, website presentation content, and release maintenance materials, and replaces personal example paths with generic ones. The original Hermes MIT license and copyright notice are in [hermes/LICENSE](hermes/LICENSE); other component licenses remain in their respective directories. The Hermes license does not automatically apply to the separate first-party workflow code.

Dependency manifests and lockfiles, skills, and plugin runtime resources are retained. `hermes/website/static/api/model-catalog.json` is kept because runtime code reads it. Only client adapters for the separate search and fetch services are included, not their server implementations or credentials.
