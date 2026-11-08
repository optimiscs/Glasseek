# 移除旧 analysis 入口

第一阶段归档完成后，进一步移除本机 `analysis/` 目录、35 个目录兼容链接和
该目录的过渡 README。资产仍位于第一阶段指定的正式目录；本阶段没有删除实验数据、
模型或缓存，也没有移动这些大文件。

85 个实验脚本及 1 份实验说明的路径已更新。已知完整资产路径改为直接指向
`reports/experiments/`、`checkpoints/archive/`、`.cache/experiments/`、
`data/experiments/` 或历史 runtime 归档；脚本入口和实验工作目录指向 `experiments/`。
正式服务入口继续位于 `Evaluation/services/`。实验内部用于相对读取资产的链接仍保留。

另外更新了 `/root/tf-logs/glasseek-full64k-2gpu-turns` 和归档中一个 TensorBoard
事件链接，两者均直接指向正式结果目录，保留相同目标 inode。未重启 TensorBoard。
删除前未发现进程以本机旧 analysis 路径作为工作目录或直接脚本参数。

可运行代码的本机旧路径依赖已清除。历史报告、迁移清单、已禁用的历史脚本和迁移工具
中的旧路径用于追溯，不作为本机运行入口。部分远端部署脚本使用
`checkpoint-reeval/analysis/...`；这是不同目录、不同机器的部署约定，本次未迁移远端。

`tests/test_repository_layout.py` 检查旧目录不存在，且可运行源码/配置不再依赖
本机旧路径。原抓取入口测试改为从 `experiments/` 执行转发脚本，继续验证正式服务复用。

验证命令：

```bash
python scripts/maintenance/archive_analysis.py verify --canonical \
  --output reports/analysis-inventory-20261106 \
  --changes reports/analysis-inventory-20261106/retired-source-edits.json
```

`--canonical` 直接检查正式目标位置。`retired-source-edits.json` 声明源码修改后的哈希；
其他文件继续校验原 inode、大小、mtime，以及原清单中已有的哈希。
旧清单与第一阶段验证报告保持原样。移除旧入口后，原布局 rollback 命令会明确拒绝执行，
避免无声跳过或恢复已淘汰的目录；不应再按第一阶段文档恢复兼容入口。

相关记录：`analysis-retirement.json`、`analysis-retirement-edits.json`、
`external-analysis-links.json`、`retirement-verification.json`、`retirement-validation.json`。
验证不包含重新训练、完整评测或 GPU 检索服务启动。

最终验证：27 项测试通过；47 个 Python 脚本和 38 个 Shell 脚本通过语法检查；
31,972 个文件在正式位置校验通过，相关链接无断链，线上搜索池健康检查正常。
