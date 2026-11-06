# prompts.py 中英对照（未完成）

本文件是部分翻译，不是完整译文。英文原文以同目录 `prompts.py` 为准；字段名、变量名和枚举值保留英文。

## CONSTRAINT_SCHEMA — 约束路径结构

| 字段 | 中文含义 |
| --- | --- |
| `path_id` | 稳定的路径标识 |
| `role` | 路径角色：`core`（核心）或 `distractor`（干扰） |
| `branch` | 独立的证据分支 |
| `hop_count` | 跳数 |
| `clue` | 一条简洁、原子化且有证据支持的事实或关系；不要透露目标名称或答案 |
| `candidates` | 目标实体的 ID 或名称，以及有代表性的近似候选 |
| `estimated_candidate_count` | 估计的候选数量 |
| `terminal_type` | 与目标相同的类型 |
| `local_target_id` | 可选：需要递归识别的中间实体的稳定 ID |
| `local_target_name` | 从线索中原样复制的中间实体名称 |
| `local_target_canonical_name` | 可选：中间实体的规范名称 |
| `local_target_type` | 可选：中间实体的类型 |
| `evidence` | 证据列表，包含 URL、简短引用或转述、支持的论断及来源站点 |
| `notes` | 此路径有用的原因 |
| `local_constraints` | 可选：递归展开的 `ConstraintPath` 记录，用于识别线索中的重要具名实体 |

## ROOT_CONSTRAINT_SCHEMA — 根约束结构

| 字段 | 中文含义 |
| --- | --- |
| `path_id` | 稳定且唯一的路径 ID |
| `role` | `core`（核心）或 `distractor`（干扰） |
| `branch` | `property:<kind>`（属性类型）或 `relation:<entity type>`（关系实体类型） |
| `hop_count` | 跳数，示例值为 1 |
| `clue` | 一条适用于 Seed 目标的简洁原子事实或关系 |
| `candidates` | Seed 目标的实体 ID 和有代表性的近似候选 |
| `estimated_candidate_count` | 估计的候选数量 |
| `terminal_type` | 与 Seed 目标相同的类型 |
| `evidence` | 证据 URL、包含 Seed 目标名称的简短佐证、所支持的线索和来源站点 |
| `notes` | 候选集合及线索用途的简要说明 |
