# 实验代码归档

这里保存原 `analysis/<实验名>` 的实验脚本和实验专用配置。长期使用的工具服务
统一放在 [Evaluation/services](../Evaluation/services/README.tool-services.md)，
服务默认配置放在 `Evaluation/config/`；不要在实验目录复制搜索池、抓取服务或检索启动逻辑。

| 内容 | 正式位置 |
| --- | --- |
| 实验脚本、实验配置、提示词 | `experiments/<实验名>/` |
| 结果、日志、图表、梯度诊断产物 | `reports/experiments/<实验名>/` |
| 依赖包、wheel、编译缓存、未完成的预训练模型下载 | `.cache/experiments/<实验名>/` |
| 训练检查点、优化器状态、适配器、导出的模型权重 | `checkpoints/archive/<实验名>/` |
| 实验输入数据 | `data/experiments/<实验名>/` |
| 历史运行环境和远端快照 | `Evaluation/data/runtime/archive/<实验名>/` |

本机旧 `analysis/` 目录及其 35 个兼容链接已移除，实验直接使用本目录。
脚本中的旧路径已更新；已明确归档位置的完整资产路径直接指向对应归档目录。
实验目录内的资产链接仍保留，用于相对路径读取历史数据。
检查点与模型目录整体移动，配置和 tokenizer 文件与权重一起保留。没有删除或
重新复制大文件；这次整理不会释放磁盘空间。

这些是历史实验档案。复现实验前应配置新的输出目录；某些脚本采用原子替换写入，
重跑可能替换兼容链接或覆盖旧结果。不要把历史路径当作新的生产写入接口。
本次未重跑 GPU 训练或整套评测。大型本地资产不进入 Git，克隆源码不会附带它们。

新增实验先查阅 [归档清单](../reports/analysis-inventory-20261106/README.md)。
可复用的逻辑修改正式实现，实验侧只保留调用和参数；固定数据集 preflight、
一次性审计、旧 viewer、磁盘监控和 benchmark 脚本继续留在对应实验中。

移除旧入口的范围、验证与远端路径说明见
[清理记录](../reports/analysis-inventory-20261106/analysis-retirement.md)。
