# CLM versus pre-CLM Hermes

Status: completed_with_errors. Common scored tasks: 22. Updated: 2026-11-08T10:10:29.088058+00:00.

| Metric | Hermes first run | Hermes repeat | CLM + Hermes |
|---|---:|---:|---:|
| Correct | 9 | 11 | 12 |
| Model calls | 806 | 634 | 1,289 |
| web_search | 1,539 | 1,123 | 1,141 |
| web_extract | 58 | 60 | 96 |
| Input including cache | 32,504,782 | 22,992,925 | 51,933,855 |
| Uncached input | 3,094,126 | 2,353,405 | 5,687,391 |
| Output | 245,283 | 201,413 | 291,671 |
| Mean seconds | 676.99 | 435.67 | 468.63 |
| Context operations | 25 | 13 | 13 |

| Question | Hermes first | Hermes repeat | CLM |
|---|---:|---:|---:|
| row0001 | 0 | 0 | 0 |
| row0026 | 0 | 0 | 0 |
| row0031 | 1 | 1 | 1 |
| row0048 | 0 | 0 | 1 |
| row0050 | 0 | 0 | 0 |
| row0055 | 0 | 0 | 0 |
| row0057 | 1 | 1 | 1 |
| row0084 | 1 | 1 | 1 |
| row0093 | 0 | 1 | 0 |
| row0098 | 0 | 0 | 1 |
| row0114 | 1 | 0 | 0 |
| row0117 | 1 | 0 | 1 |
| row0130 | 1 | 1 | 1 |
| row0149 | 1 | 1 | 1 |
| row0150 | 0 | 0 | 0 |
| row0157 | 0 | 1 | 0 |
| row0170 | 1 | 1 | 1 |
| row0171 | 0 | 0 | 1 |
| row0175 | 1 | 1 | 1 |
| row0187 | 0 | 1 | 0 |
| row0200 | 0 | 1 | 1 |
| row0204 | 0 | 0 | 0 |

## Limits

- Baselines use the same model, raw URLs, identical questions, 128K context and 16384 output budget.
- Historical URL modes each used concurrency 2 and shared a max-seqs-8 server; current CLM uses concurrency 5, tool-round limit 120, total timeout None. Opinions about isolated compression effects require matched settings.
- The first raw run had a retrieval outage. Counts include only identical, jointly scored tasks.
- Hermes allowed parallel native web tools per model response; official CLM uses a single bash action.
- Repeat-baseline tool counts use complete native DB history deduplicated by call ID; prediction tool_calls only described the visible post-compression tail.
- Prompt, tool error propagation, web results and load also differ. This is not an isolated causal estimate of context editing.
- Traditional summary compressions and CLM mirror edits are different operations.
