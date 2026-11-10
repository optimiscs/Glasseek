# 历史归档

本目录统一保存历史实验、聊天中的对比展示、旧版本和迁移记录。原根目录的
`reports/`、`experiments/` 已整体迁入这里，不保留原路径软链接。

| 目录 | 内容 |
| --- | --- |
| `reports/` | 历史报告、展示、日志及源码快照 |
| `experiments/` | 历史实验代码、配置及资产链接 |
| `scripts/trajectory_benchmark/` | 一次性对比、采样、性能测量和展示脚本 |
| `scripts/trajectory_preprocess_pilot/` | 旧 pilot 的入口和测试 |
| `scripts/trajectory_preprocess/` | 被正式策略替代的旧实现及报告兼容入口 |
| `scripts/maintenance/` | 已完成使命的旧 analysis 迁移工具 |
| `Data_curation/` | 历史 v49 数据导出脚本 |
| `tests/` | 仅服务于历史工具和对比的测试 |
| `migration/` | 本次移动清单、软链接修复清单及本机修改前状态 |

正式代码不能导入归档；默认测试发现不进入这里。旧脚本和报告内保留当时的
路径、版本及源码哈希，供追溯使用，不承诺直接运行。重新开展工作时，应调用
正式入口并选择新输出目录，不在旧报告目录中继续写入。

当前入口：

- 轨迹处理、导出与校验：`scripts/trajectory_preprocess/`，策略统一为 `policy.py`。
- Web 审核与原始记录浏览：`scripts/trajectory_review/`。
- 评测、工具服务与 viewer 适配：`Evaluation/`。
- 数据构建：`Data_curation/multi_agent_mpcs_v5_publish/scripts/run_workflow.py`。
- 基准评测套件：`scripts/evaluation_suite/`，配置为 `config.json`。
- 训练：`RL_training/`；新的训练日志输出到 `outputs/sft/`。

大型数据、模型、运行环境保持原有专用目录，归档中的链接已按新路径修复。
评测套件使用的 34 题排除集迁入 `data/evaluation/exclusions/clm-hermes-34.jsonl`。
目录通过同盘重命名移动，已有服务打开的日志句柄继续指向同一文件。
本次移动不删除历史内容，也不释放其占用空间。

`migration/moves.json` 记录每项迁移的源路径、目标路径及原因；恢复单项文件时
应先核对当前正式实现。`migration/preexisting.patch` 和 `preexisting-status.txt`
仅在本机保存，用于核对本次整理之前已有的未提交修改，不进入 Git。
