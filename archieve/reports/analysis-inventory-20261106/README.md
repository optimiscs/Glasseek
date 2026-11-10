# analysis 目录清单与归档记录（2026-11-06）

**后续状态：旧 `analysis/` 目录和 35 个兼容入口已移除。当前路径及校验方式见 [入口清理记录](analysis-retirement.md)。以下保留第一阶段归档记录，原兼容路径验证和布局回滚说明已不再适用。**

已先完成只读盘点，再执行资产归档、实验目录迁移、正式服务归位三步。
基线共 **35 个目录、31,972 个文件、202.15 GiB**（按文件逻辑大小统计）。
迁移采用同文件系统 rename，保留相对兼容链接；没有删除数据，也没有复制大文件。

| 分类 | 文件数 | GiB | 正式目录 |
| --- | ---: | ---: | --- |
| 结果与日志 | 692 | 2.326666 | `reports/experiments/` |
| 输入数据 | 86 | 0.017731 | `data/experiments/` |
| 实验代码与配置 | 160 | 0.000411 | `experiments/` |
| 可再生成缓存和下载 | 30,106 | 5.812233 | `.cache/experiments/` |
| 检查点和导出模型 | 74 | 193.663819 | `checkpoints/archive/` |
| 历史运行环境快照 | 854 | 0.329729 | `Evaluation/data/runtime/archive/` |

检查点约 193.66 GiB，不能当作可丢弃缓存。模型目录整体迁移，权重、配置、
tokenizer 和优化器状态保留在一起。Hy-MT2 下载目录归入缓存，其中包含未完成下载。
梯度诊断产物归入结果。运行快照单独归档并忽略版本管理，避免混入历史凭据。

## 服务归位

| 原入口 / 内容 | 正式位置 | 处理 |
| --- | --- | --- |
| `clean2899-full64k/service.py` | `Evaluation/services/tool_services.py` | 旧入口只转发；搜索始终调用既有 `search_pool.py` |
| runtime `services/jina_fetch.py` | `Evaluation/services/jina_fetch.py` | 移动原实现，旧路径换为链接 |
| `checkpoint_reeval/start_search_services.sh` | `Evaluation/services/tool_services.py` | 保留参数和凭据环境兼容，转发正式入口 |
| `checkpoint_reeval/async_judge.py` | `Evaluation/services/async_judge.py` | 使用原有评判库，旧入口只转发；4 个调用点改为正式入口 |
| `clean2899-full64k/eval_env.sh` 的通用工具路由 | `Evaluation/config/tool_services.json` + `Evaluation/services/tool_services_env.sh` | 实验名字、数据集、模型和实验开关留在实验配置 |
| 增量评判默认设置 | `Evaluation/config/async_judge.json` | 密钥仍来自私有文件或环境变量 |
| BM25、embedding、bridge 的启动命令 | `Evaluation/services/tool_services.py` | 正式 `start_quest_retrieval.sh` 也复用该入口；复用 Quest 现有实现 |

固定 128 题 preflight、一次性审计、历史 viewer、磁盘监控和 benchmark 都保留为实验。
没有把这些脚本包装成新的通用服务。对原版本化代码按哈希扫描未发现完全相同的文件；
这不代表不存在逻辑相似的实验。本次消除的是已识别的通用服务启动和配置重复来源。

## 清单文件与验证

- `directories.csv`：35 个目录的文件数和大小。
- `migration-plan.csv`：653 个实际迁移动作及目标位置。
- `files.csv`：逐文件类别、目标路径、大小和是否原已版本化（完整本地清单）。
- `inventory.json`：原始文件 inode、大小、mtime，以及小型版本化文件 SHA-256（本地清单）。
- `migration-journal.jsonl`：每次 rename 的 intent/complete 记录，可用于布局回滚。
- `verification.json`：服务归位前 31,972 个文件全部通过原始身份校验。
- `source-edits.json`：服务归位涉及的 8 个源码修改，逐项记录原/新哈希与原因。
- `final-verification.json`：8 项声明修改按新哈希检查，其余保持原身份校验，全部通过。
- `validation.json`：23 项测试通过；含迁移/回滚、正式及旧入口启动、搜索池和增量评判回归；另记录语法检查与线上搜索池健康状态。

迁移前发现 TensorBoard 持有 3 个只读句柄；没有发现使用这些目录的写入进程、
工作目录或直接脚本入口。工具保留读者 inode，遇到写入者或活动实验入口会拒绝迁移。
兼容链接无断链，原先版本化的 251 个文件均有可继续版本管理的目标位置。
本次未重跑 GPU 训练、完整评测或上游计费抓取；运行中的共享搜索池健康检查正常。

重新校验（从仓库根目录执行）：

```bash
python scripts/maintenance/archive_analysis.py verify \
  --output reports/analysis-inventory-20261106 \
  --changes reports/analysis-inventory-20261106/source-edits.json
```

不加 `--changes` 会将上述 8 项有意源码改动报告为与盘点不一致。原始盘点不会被覆盖。
布局回滚入口为 `archive_analysis.py rollback --output reports/analysis-inventory-20261106`；
它仅逆转有日志记录的目录/资产迁移，保留后续源码编辑，不撤回正式服务的代码改动。
回滚前应停止归档路径的写入并核对新增文件；不要在运行实验期间调整布局。

`analysis/<实验名>` 是历史兼容入口，新实验代码写入 `experiments/`，新服务只修改正式实现。
旧实验采用原子替换写文件时可能替换兼容链接，重跑请先配置独立输出目录。
缓存、模型和输入数据不进入 Git；在新机器克隆仓库后需要另行恢复这些本地资产。

## 原目录清单

| 实验目录 | 文件数 | 归档前 GiB | 原版本化文件数 |
| --- | ---: | ---: | ---: |
| `32k-parallel-benchmark` | 71 | 1.636119 | 13 |
| `checkpoint_reeval` | 806 | 1.069245 | 28 |
| `clean2899-full64k` | 2,285 | 17.938693 | 26 |
| `ddp-lora-probe` | 32 | 0.002377 | 2 |
| `extract-128-audit` | 6 | 0.000383 | 4 |
| `faircompare120-20261106` | 1,838 | 0.300389 | 24 |
| `fsdp-turbo-cp` | 14,082 | 1.597462 | 1 |
| `full64k-fsdp2-smoke-20261105` | 24 | 52.602735 | 1 |
| `full64k-reeval` | 1,751 | 17.713268 | 8 |
| `full64k-turns-formal-20261105` | 6 | 0.000692 | 3 |
| `lora-sp4-probe` | 143 | 0.944684 | 4 |
| `megatron-dcp` | 9,032 | 0.752374 | 0 |
| `no-tool-screen` | 24 | 0.025475 | 11 |
| `no-tool-strong-audit` | 7 | 0.003216 | 4 |
| `no-tool-top10` | 19 | 0.538615 | 7 |
| `paper-audit` | 127 | 0.003728 | 0 |
| `pro6000-lora` | 230 | 0.349925 | 12 |
| `prompt-audit` | 18 | 0.000336 | 6 |
| `public-trace-token-comparison` | 8 | 0.000845 | 5 |
| `released-model-inference-url` | 8 | 0.000348 | 5 |
| `remaining2899-token-distribution` | 2 | 0.000003 | 2 |
| `sft-4gpu-feasibility` | 5 | 0.000042 | 4 |
| `sft-eval-aligned` | 9 | 0.182924 | 6 |
| `sft-eval-fixed-64k` | 3 | 0.000007 | 3 |
| `sft-quality-20261105` | 16 | 0.011938 | 10 |
| `sft-translation` | 242 | 1.231117 | 4 |
| `sft128k` | 1,022 | 0.028796 | 11 |
| `sft64k-round-curriculum` | 1 | 0.000003 | 0 |
| `sp4-30g` | 64 | 0.006076 | 18 |
| `tool-health-dashboard` | 2 | 0.000051 | 2 |
| `tool-repair-20261106` | 30 | 0.000812 | 15 |
| `top10-filtered-tools` | 3 | 0.000025 | 3 |
| `turn-resume-smoke-20261105` | 39 | 105.205005 | 1 |
| `url-provenance` | 13 | 0.002870 | 4 |
| `url-token-audit` | 4 | 0.000009 | 4 |
