# CLM + Hermes 34-task evaluation

Status: **completed_with_errors**. Updated: 2026-11-08T10:10:29.088058+00:00.

Run: `clm_hermes_34_c5_20261108_170243`. Workers: 5. Previous natural terminal cases: 34; current: 34/34.

| Metric | Previous natural terminal | Current natural terminal |
|---|---:|---:|
| Cases | 34 | 34 |
| Successful submissions | 34 | 33 |
| Scored | 34 | 33 |
| Correct | 14 | 15 |
| Input tokens (includes cache) | 108,491,190 | 87,734,486 |
| Cached input tokens | 42,278,016 | 77,909,568 |
| Output tokens | 590,640 | 492,533 |
| Model calls | 2,495 | 2,154 |
| bash calls | 2,460 | 2,120 |
| web_search | 2,212 | 1,907 |
| web_extract | 178 | 155 |
| Real context edits | 28 | 25 |
| Rollback retries | 0 | 0 |
| Accuracy among scored answers | 41.2% | 45.5% |

Common naturally ended tasks: 34. Common scored tasks: 33.

| Task | Previous status / score | Current status / score | Input token delta | Wall seconds delta |
|---|---|---|---:|---:|
| row0001 | success / 0.0 | success / 0.0 | 3,242,901 | -950 |
| row0026 | success / 0.0 | success / 0.0 | -2,292,158 | -1,774 |
| row0031 | success / 1.0 | success / 1.0 | -991,767 | -2,678 |
| row0048 | success / 0.0 | success / 1.0 | -516,412 | -685 |
| row0050 | success / 0.0 | success / 0.0 | 2,926,177 | -2,516 |
| row0055 | success / 0.0 | success / 0.0 | 1,581,730 | -2,190 |
| row0057 | success / 1.0 | success / 1.0 | 71,719 | 15 |
| row0084 | success / 1.0 | success / 1.0 | 19,424 | -7 |
| row0093 | success / 0.0 | success / 0.0 | -1,428,551 | -2,860 |
| row0098 | success / 1.0 | success / 1.0 | 1,036,105 | -56 |
| row0104 | success / 0.0 | failed / None | -470,218 | -2,893 |
| row0114 | success / 0.0 | success / 0.0 | 185,527 | -2,868 |
| row0117 | success / 1.0 | success / 1.0 | 36,476 | -493 |
| row0130 | success / 1.0 | success / 1.0 | -311,006 | -1,799 |
| row0149 | success / 0.0 | success / 1.0 | -1,470,461 | -2,488 |
| row0150 | success / 0.0 | success / 0.0 | -1,665,821 | -3,379 |
| row0157 | success / 1.0 | success / 0.0 | 2,345,813 | -2,053 |
| row0170 | success / 1.0 | success / 1.0 | -103,858 | -1,096 |
| row0171 | success / 0.0 | success / 1.0 | -2,328,933 | -3,317 |
| row0175 | success / 0.0 | success / 1.0 | -6,961,553 | -3,749 |
| row0187 | success / 0.0 | success / 0.0 | -2,153,951 | -3,406 |
| row0194 | success / 0.0 | success / 0.0 | -3,621,585 | -3,907 |
| row0200 | success / 1.0 | success / 1.0 | -4,546,812 | -3,939 |
| row0204 | success / 0.0 | success / 0.0 | 634,808 | -2,882 |
| row0207 | success / 1.0 | success / 0.0 | 1,225,759 | -3,183 |
| row0211 | success / 1.0 | success / 1.0 | 519,480 | -2,877 |
| row0230 | success / 0.0 | success / 0.0 | -117,582 | -3,483 |
| row0233 | success / 1.0 | success / 0.0 | 1,069,063 | -2,674 |
| row0239 | success / 1.0 | success / 1.0 | -6,409 | -850 |
| row0243 | success / 0.0 | success / 0.0 | 1,228,625 | -1,900 |
| row0265 | success / 0.0 | success / 0.0 | -5,639,883 | -2,686 |
| row0279 | success / 1.0 | success / 1.0 | -255,000 | -1,370 |
| row0296 | success / 0.0 | success / 0.0 | 2,222,079 | -964 |
| row0306 | success / 0.0 | success / 0.0 | -4,220,430 | -915 |

## Accounting limits

- Input tokens include cached tokens; do not add cache reads a second time.
- bash counts model-facing calls; web counts underlying bridge calls. They overlap.
- Exact context-edit counts come from official n_ctx_syncs_real for finished cases. Live observations are separate lower bounds.
- Previous operator-interrupted attempts are excluded from natural-terminal and paired comparisons.
- Previous missing/invalid submissions have no answer score; they are not silently assigned a judge score of zero.
- Previous run was paused and changed recording code mid-run. Search results and server load also vary; comparison is descriptive.
