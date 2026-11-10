# CLM + Hermes 34-task evaluation

Status: **completed**. Updated: 2026-11-08T08:07:37.920105+00:00.

Run: `clm_hermes_34_c12_20261108_123828`. Workers: 5. Previous natural terminal cases: 14; current: 34/34.

| Metric | Previous natural terminal | Current natural terminal |
|---|---:|---:|
| Cases | 14 | 34 |
| Successful submissions | 6 | 34 |
| Scored | 6 | 34 |
| Correct | 3 | 14 |
| Input tokens (includes cache) | 12,452,121 | 108,491,190 |
| Cached input tokens | 2,561,856 | 42,278,016 |
| Output tokens | 109,892 | 590,640 |
| Model calls | 415 | 2,495 |
| bash calls | 412 | 2,460 |
| web_search | 338 | 2,212 |
| web_extract | 36 | 178 |
| Real context edits | 0 | 28 |
| Rollback retries | 0 | 0 |
| Accuracy among scored answers | 50.0% | 41.2% |

Common naturally ended tasks: 14. Common scored tasks: 6.

| Task | Previous status / score | Current status / score | Input token delta | Wall seconds delta |
|---|---|---|---:|---:|
| row0026 | failed / None | success / 0.0 | 1,356,017 | 1,302 |
| row0048 | success / 0.0 | success / 0.0 | 736,282 | 453 |
| row0055 | failed / None | success / 0.0 | 1,167,828 | 1,074 |
| row0057 | success / 1.0 | success / 1.0 | 5,905 | 22 |
| row0084 | success / 1.0 | success / 1.0 | 9,268 | 60 |
| row0098 | success / 0.0 | success / 1.0 | 211,760 | 278 |
| row0114 | failed / None | success / 0.0 | 1,581,742 | 1,341 |
| row0117 | success / 1.0 | success / 1.0 | -378,632 | -364 |
| row0130 | failed / None | success / 1.0 | -143,247 | 636 |
| row0149 | success / 0.0 | success / 0.0 | 984,046 | 1,310 |
| row0150 | failed / None | success / 0.0 | 4,275,584 | 2,670 |
| row0170 | failed / None | success / 1.0 | -304,517 | -451 |
| row0171 | failed / None | success / 0.0 | 5,045,975 | 2,520 |
| row0175 | failed / None | success / 0.0 | 6,834,377 | 2,394 |

## Accounting limits

- Input tokens include cached tokens; do not add cache reads a second time.
- Current run contains the original c12/3600-second phase and a c5/no-total-timeout continuation with a cumulative 120 tool-round budget.
- Resumed cases use cumulative native token counts and cumulative execution wall time; paused wall time is excluded.
- bash counts model-facing calls; web counts underlying bridge calls. They overlap.
- Exact context-edit counts come from official n_ctx_syncs_real for finished cases. Live observations are separate lower bounds.
- Previous operator-interrupted attempts are excluded from natural-terminal and paired comparisons.
- Previous missing/invalid submissions have no answer score; they are not silently assigned a judge score of zero.
- Previous run was paused and changed recording code mid-run. Search results and server load also vary; comparison is descriptive.
