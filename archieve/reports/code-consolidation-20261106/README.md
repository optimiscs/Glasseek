# 入口与共享实现清理记录（2026-11-06）

本批先完成 runtime 收口，再合并参数变体、删除失去消费者的入口，最后独立推进
有行为回归覆盖的共享核心和策略抽取。未重启线上搜索池，未启动 GPU 训练或整套评测。

## 已落地

| 范围 | 结果 | 唯一实现 / 配置 |
| --- | --- | --- |
| runtime 旧入口 | 20 个 Python/Shell 脚本、1 个旧 Docker SSH 代理和 3 个字节码文件核验后归档移除；2 个源码链接移除；旧 services 目录已不存在 | 正式实现位于 `Evaluation/services/`；私有可恢复 ZIP 位于 `Evaluation/data/runtime/archive/` |
| 评测启动 | RL-9B、SFT-9B、RL-4B 共用入口；supervisor 不再读取 runtime 脚本 | `eval_run.py`、`eval_profiles.json`，读取 `tool_services.json` 和 `search_pool.json` |
| LoRA probe | 4 份 Python 实现合成 1 份，原启动脚本选择 profile | `RL_training/glasseek_training/lora_probe.py`、`lora_probe_profiles.json` |
| 权重校验 | 2 份实现合为参数化命令，更新实际调用方 | `verify_weights.py --source --target --world-size` |
| vLLM 启动 | RL-9B、step39、step76 共用启动逻辑，保留原默认并发及 thinking 差异 | `model_server.py`、`model_profiles.json` |
| SFT 启动 | 128k、64k、turns、clean2899、length 共用 launcher | `train_sft.sh` → `sft_launch.py`、`sft_launch_profiles.json` |
| 退役转发入口 | 删除实验 `service.py`、`async_judge.py`、`start_search_services.sh` 和 `retired_remote_wait.py.disabled`；先改 portable 启动调用方及测试 | 直接调用正式服务，旧文件归入私有 ZIP |
| Hermes 首批共享 | 8 个实现只有一份源码；两端公共 import 路径转发到共享包 | `Shared/glasseek_shared/hermes/` |
| 数据筛选策略 | 共享提示词配置、答案提取、候选校验、HTTP 重试和初始队列策略；保留 top10/strong 的不同 schema 和证据判定 | `Shared/glasseek_shared/screening.py`、`screening_profiles.json` |
| 判题调度 | 显式 record/summary 策略；supervisor 不再修改另一模块的全局 SCRIPT/SUMMARY | `judge_policy.py`、`incremental_judging.py` |
| 课程训练 | 共用一个 trainer，通过 length/turns 策略选择采样和验证；移除旧 turns trainer 文件 | `curriculum_sft.py`、`curriculum_policy.py`、`turn_sampler.py` |

保留的小型 `run_*` 和 `start_model.sh` 是仍有调用方的命名 profile 入口，内部不再复制
执行逻辑。泛用评测入口读取公共工具路由，同时保留调用者对工具启用状态和 fetch URL
显式参数的控制。

## 行为边界

- LoRA 长度、截断、offload、rank/alpha/lr 默认值与原变体一致。
- 课程 length 策略继续要求单卡 SP=1；turns 策略继续要求双卡 SP=1 和 64K manifest。
  turn sampler 使用迁移前冻结类做差分测试，覆盖不同大小、rank、epoch 和补齐行为。
- top10 与 strong 的判断语义没有强行归一，包括 Unicode sentinel 的历史差异；
  strong 继续先复核已有答案，再生成缺失答案。两者的数据库结构与最终证据校验保留。
- 增量判题继续排除未完成/未消耗尝试，复用相同 fingerprint 的终态结果，并保留失败冷却。
- 两套 Hermes 的插件注册、防泄漏、URL 安全、上下文压缩和工具执行策略没有整目录覆盖。

## 第三阶段尚未整体合并的部分

本次完成共享核心的首批 8 个模块及上述策略接入，不代表整个 Hermes 双副本已消除。
29 个角色差异文件仍保留各自实现，清单见 `hermes-role-differences.json`；其余上游同源
公共模块尚未批量迁移。后续迁移应先增加对应的角色行为测试，特别是资源根目录、
上下文压缩、工具权限和防泄漏行为，不能仅凭文件相同或相似就替换。
筛选流程的不同 schema/证据验证和两种判题执行方式也继续作为独立策略存在。

## 验证和记录

- `baseline.json`：本批开始时各脚本 SHA-256。
- `retired-runtime.json`：旧 runtime 文件及可恢复归档位置。
- `hermes-shared-modules.json`：8 个共享模块迁移前的来源和哈希。
- `hermes-role-differences.json`：29 个仍保留角色差异的模块。
- `turn-sampler-reference.txt`：用于差分回归的冻结采样类，非运行入口。
- `validation.json`：最终回归、配置组合、资产完整性和线上搜索池检查。

相关测试在 Quest 环境运行；涉及 Arrow/Hydra/verl 的 11 项训练回归在原训练环境运行，
只临时从已有环境加载 pytest，无安装或升级依赖。另实际组合了全部 5 个 SFT profile 的
Hydra 配置。权重校验使用合成 CPU tensors，未重新读取数百 GiB 真实模型做全量比较。

此前 `analysis-inventory-20261106` 的源码校验记录是历史快照，经过本批源码合并和删除后
不再作为当前源码的无修改断言；其中模型、数据、结果和缓存资产继续单独校验原始身份。
