# CLM versus pre-CLM Hermes

Status: completed. Common scored tasks: 23. Updated: 2026-11-08T08:07:37.920105+00:00.

| Metric | Hermes first run | Hermes repeat | CLM + Hermes |
|---|---:|---:|---:|
| Correct | 10 | 12 | 9 |
| Model calls | 882 | 651 | 1,602 |
| web_search | 1,664 | 1,147 | 1,436 |
| web_extract | 66 | 62 | 106 |
| Input including cache | 35,907,233 | 23,352,648 | 69,802,543 |
| Uncached input | 3,370,817 | 2,399,496 | 46,629,679 |
| Output | 268,993 | 205,799 | 376,413 |
| Mean seconds | 719.66 | 422.40 | 2,578.87 |
| Context operations | 28 | 13 | 14 |

| Question | Hermes first | Hermes repeat | CLM |
|---|---:|---:|---:|
| row0001 | 0 | 0 | 0 |
| row0026 | 0 | 0 | 0 |
| row0031 | 1 | 1 | 1 |
| row0048 | 0 | 0 | 0 |
| row0050 | 0 | 0 | 0 |
| row0055 | 0 | 0 | 0 |
| row0057 | 1 | 1 | 1 |
| row0084 | 1 | 1 | 1 |
| row0093 | 0 | 1 | 0 |
| row0098 | 0 | 0 | 1 |
| row0104 | 1 | 1 | 0 |
| row0114 | 1 | 0 | 0 |
| row0117 | 1 | 0 | 1 |
| row0130 | 1 | 1 | 1 |
| row0149 | 1 | 1 | 0 |
| row0150 | 0 | 0 | 0 |
| row0157 | 0 | 1 | 1 |
| row0170 | 1 | 1 | 1 |
| row0171 | 0 | 0 | 0 |
| row0175 | 1 | 1 | 0 |
| row0187 | 0 | 1 | 0 |
| row0200 | 0 | 1 | 1 |
| row0204 | 0 | 0 | 0 |

## Limits

- Baselines use the same model, raw URLs, identical questions, 128K context and 16384 output budget.
- Historical URL modes each used concurrency 2 and shared a max-seqs-8 server; current CLM began at c12 and resumed unfinished/error tasks at c5 without a total timeout, with 120 cumulative tool rounds.
- The first raw run had a retrieval outage. Counts include only identical, jointly scored tasks.
- Hermes allowed parallel native web tools per model response; official CLM uses a single bash action.
- Repeat-baseline tool counts use complete native DB history deduplicated by call ID; prediction tool_calls only described the visible post-compression tail.
- Prompt, tool error propagation, web results and load also differ. This is not an isolated causal estimate of context editing.
- Traditional summary compressions and CLM mirror edits are different operations.
