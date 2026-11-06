# Glasseek：verl SFT 与 CM-GSPO

这里提供 SFT 配置、在线搜索 rollout、同策略摘要、CM-GSPO 数据流、终局奖励、检查点恢复和 CPU 回归测试。实现固定到 [verl v0.9.1 的 commit `1876b06`](https://github.com/verl-project/verl/tree/1876b06d0a3e4e71e06230be10af14492ca8a75b)，不修改安装目录中的 verl 源码。该 tag 内的版本字符串为 `0.9.1`。

训练预算和学习率参考论文附录 C；本机 SFT 已按显存实测调整为 42K、LoRA 和四卡运行，因此不是论文全参 128K SFT 的复现。CM-GSPO 算法对应 §3.2 公式 (3)–(6)。

| 设置 | SFT | RL（CM-GSPO） |
| --- | --- | --- |
| 训练预算 | 3 epochs | 100 optimizer steps |
| 学习率 | `2e-5` | `1e-6` |
| 最小学习率 | `2e-6`（`min_lr_ratio=0.1`） | —，constant |
| Warmup 比例 | `0.03` | `0` |
| Global batch size | 64 条训练序列 | 16 prompts |
| Rollouts / prompt | — | 8，共 128 条原始 rollout / step |
| 最大序列长度 | 43008（42×1024） | 论文未指定，当前工程默认 131072 / 上下文窗口 |

SFT 使用继承 verl `MultiTurnSFTDataset` 的 `GlasseekSFTDataset`，仅训练 assistant 目标，`no_padding + use_dynamic_bsz` 在引擎内部做变长序列打包。全局 batch 的计数单位是数据集序列；没有把多条数据离线拼接成固定长度 43008 的新样本。超长输入报错，不静默截断。Qwen thinking 模板默认保留逐轮渲染得到的训练 token，避免整段渲染删除旧轮次的 reasoning。

## 安装

本机已提供独立环境 `/root/autodl-tmp/glasseek-verl091-env`（先 `source /root/autodl-tmp/glasseek-verl091-env/bin/activate`）。该环境已安装 verl 及 SFT 核心依赖；vLLM 尚未安装，RL 启动前须安装与 CUDA/PyTorch 匹配的 vLLM 并验证。也可在新的 Python 3.11/3.12 CUDA 环境安装：

```bash
cd /path/to/Glasseek
python -m pip install -r RL_training/requirements.txt
# 根据集群 CUDA / PyTorch 安装匹配的 FlashAttention；不要覆盖已有集群驱动。
python -m pip install flash-attn --no-build-isolation
export PYTHONPATH="$PWD/RL_training${PYTHONPATH:+:$PYTHONPATH}"
```

`requirements.txt` 固定 verl commit，vLLM 使用 `0.20.x`、Transformers 使用 `>=5.5.3,<5.11`（排除 5.6.0），并固定 `tensordict==0.10.0`。这是与该源码接口及 Qwen3.5 支持相匹配的安装范围，并非经过 GPU 集成验证的完整依赖锁文件。

本机 SFT 默认 4 GPU、FSDP2、SP=1、LoRA rank=16、alpha=32、BF16 模型权重及融合输出层；全局 batch 为 64，每 GPU 动态 micro batch 上限 43008 tokens。实测单条 47,014 tokens 可完成一步，但显存余量小，52,080 tokens OOM，因此本配置选择 42K；该实测不足以保证所有批次或其他负载下均不 OOM。论文使用不同硬件和全参 128K 设置。

## 数据

SFT 输入为完整 JSONL，每行 `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`，支持 system/tool 消息及原有 tools 字段。RL 每行为 `{"question": "...", "answer": "..."}`；也接受 `query`、`reference_answer`、`ground_truth`，answer 可为答案别名列表。

```bash
python -m glasseek_training.filter_sft_length \
  --input data/Glasseek-SFT-5.7K/train.jsonl --output data/glasseek/sft \
  --model /root/autodl-tmp/models/Qwen3.5-9B --max-length 43008
python -m glasseek_training.prepare_data rl \
  --input /data/Glasseek-RL/train.jsonl --output data/glasseek/rl
```

先以逐轮渲染的实际 token 数筛选完整轨迹，超过 43008 的整条排除，不截断；再按归一化的问题文本确定性划分约 95%/5%，同问题的多个 SFT 轨迹不会跨集合。输出 `train.parquet`、`val.parquet` 和带源文件 SHA256 的 `manifest.json`。SFT 的 messages/tools 以 JSON 字符串保存在 Parquet 中，由自定义 loader 无损还原，避免工具参数字典被 Arrow 合并成含多余 null 键的结构。空验证集、非法记录、已有输出和 `.part` 未完成下载会报错。本机已校验的源文件位于 `data/Glasseek-SFT-5.7K/train.jsonl`。转换器针对文本轨迹，多模态内容需要单独扩展。

参考答案只进入 `reward_model.ground_truth`，不会传入搜索模型 prompt 或工具。

## SFT

```bash
MODEL_PATH=/root/autodl-tmp/models/Qwen3.5-9B \
NPROC_PER_NODE=4 \
bash RL_training/scripts/train_sft.sh
```

更换模型前须使用对应 tokenizer 重新筛选并重新验证显存；数据路径可通过 `SFT_TRAIN_FILE`、`SFT_VAL_FILE` 设置；Hydra 参数置于命令末尾，例如 `optim.lr=2e-5 trainer.experiment_name=sft-4b`。

两节点 16 GPU：在每个节点分别设置 `NNODES=2 NODE_RANK=0/1 MASTER_ADDR=<主节点地址> NPROC_PER_NODE=8` 后运行同一脚本。代码、数据和输出目录需在节点间可访问。

默认每 10 steps 保存一次；verl 也在训练结束时保存，`resume_mode=auto` 恢复模型、优化器和训练进度。SFT → RL 先导出 Hugging Face checkpoint：

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir checkpoints/glasseek/sft/global_step_<实际步数> \
  --target_dir /models/glasseek-sft-hf
```

具体目录以训练输出为准。RL checkpoint 的模型分片位于 `global_step_N/actor`，导出时把该子目录传给 `--local_dir`。

## CM-GSPO

```bash
export MODEL_PATH=/models/glasseek-sft-hf
export SERPER_API_KEY=...
export JINA_API_KEY=...  # 可选
export GLASSEEK_JUDGE_BASE_URL=https://your-judge-service/v1
export GLASSEEK_JUDGE_MODEL=your-judge-model
export GLASSEEK_JUDGE_API_KEY=...
bash RL_training/scripts/train_cm_gspo.sh
```

默认使用 Serper `web_search` 和 Jina `web_extract`，支持 Qwen JSON/XML tool-call 格式；没有复用整个 Hermes 运行时，浏览器、代码执行等工具需要在 `web_tools.py` 扩展。模型通过 tokenizer 的 tools 参数得到工具 schema，生成 token 原样保留，工具返回值以 mask=0 追加。上下文压缩使用当前 rollout 的同一策略服务器；摘要调用不参与训练。

默认奖励为外部 OpenAI-compatible judge 的正确性判断（0/1）；题目、参考答案和最终回答只在轨迹结束后送给 judge。judge 配置和判断提示是工程默认，论文没有给出可直接复用的完整训练奖励实现。网络/解析失败会使该训练 step 报错，避免把服务故障当作错误答案。工具调用错误作为工具观察返回。回合、生成或压缩次数耗尽的轨迹奖励为 0，但保留实际生成过的执行 token。

调试时可设置 `cm_gspo.reward_mode=exact_match`，此时不需要 judge 服务。精确匹配只适合严格短答案，不能替代正式搜索训练中的语义判分。

多节点 RL：先在同一个 Ray 集群上启动所有节点，确保相同代码路径、依赖、数据、模型、API 环境变量及共享 checkpoint 目录；只在 head 节点执行一次：

```bash
NNODES=4 NPROC_PER_NODE=8 bash RL_training/scripts/train_cm_gspo.sh \
  +ray_kwargs.ray_init.address=auto
```

`RL_TRAIN_FILE`、`RL_VAL_FILE` 可覆盖数据路径；RL 脚本导出绝对 `GLASSEEK_AGENT_CONFIG` 路径。默认 100 steps、每 10 steps 保存、自动恢复；验证默认关闭，可设置 `trainer.test_freq=10`。验证仍使用每题 8 次采样，其指标为平均 rollout reward。

## CM-GSPO 如何落到 verl

- `rollout.py` 记录每个压缩边界后的真实 prompt token、执行/工具 token、执行 mask、behavior logprobs。摘要只出现在下一段已固定的 prompt 中，不占有效长度；训练时不重新摘要。
- `cm_gspo.py` 先对每题的 8 个原始终局 reward 计算 GRPO advantage，使用 verl 一致的样本标准差。再为段赋值 `A_cm = A_rollout * L_segment / L_rollout`。相同 reward 的组 advantage 为 0。
- `trainer.py` 将段展开为训练行，通过 verl 的 loss registry 调用 `loss.py` 中的专用 `cm_gspo_loss`。`model.external_lib=glasseek_training.verl_loss` 确保所有 worker 注册此损失。执行 token 的段级平均 log-ratio 对应公式 (3)，`detach` 后乘 token 局部似然因子对应公式 (4)。不能直接使用上游 `gspo`：它的 `log_ratio <= 10` 截断会让负 advantage、大 ratio 区域的惩罚梯度消失。专用实现不作该截断；数值溢出明确报错。
- 每 step 所有段累积完成一次 optimizer update。`mini_batch_size` 是补齐后的段数；传入 loss 的 `global_batch_size` 始终是 **128 个原始 rollout**。因此实现的是 `sum(segment_loss) / 128`，不会多除一个段数。按实际 actor DP size 补齐行，补齐副本的执行 mask 和 advantage 都为 0；SP/TP 的进程数不作为独立样本数。
- 直接使用采样时记录的 behavior logprobs；没有用摘要后的拼接长历史重算旧策略概率。采样固定 `temperature=1, top_p=1, top_k=-1`，使采样概率与训练似然一致。无 critic、KL 或 entropy 附加项。

论文未给出的默认值：对称 clip `ε=0.0003`（通过 `cm_gspo.clip_epsilon` 同时控制上下界）、98304 token 压缩阈值、8192 单轮生成、8192 工具观察、4096 摘要、128 轮和 16 次压缩上限。压缩输入是当前可见的完整历史（包含上一次摘要及后续工具观察）。所有这些值集中在 `configs/cm_gspo.yaml`；超预算会明确报错或标记失败，不静默截断训练上下文。

### 公式与分布式归一化核对

对于 P 个 prompts、每题 G 个 rollouts，令 N=P×G（默认 128），D 为 actor 的数据并行规模。每个 rank、每个 microbatch 使用：

```text
loss_local = -D/N * sum_segments [ (1/L_ik) * sum_execution_tokens
    min(ratio_token * A_cm, clip(ratio_token, 1-epsilon, 1+epsilon) * A_cm) ]
A_cm = L_ik/L_i * A_i
```

每个 rank 对 microbatch 求和，再由 DDP/FSDP 对 rank 梯度取平均，D 抵消，只剩 N 的原始 rollout 归一化。没有再次除以段数、有效 token 总数或 microbatch 数。固定相同 G 时，`1/N` 等价于对 P 个问题的公式 (6) 取平均。

未被 clipping 截断的执行 token 的 loss 梯度为 `-A_i * s_ik / (N * L_i)`；不是再乘一个 `1/L_ik`。摘要生成动作不进入 ratio、advantage 权重或损失；复用摘要的 token 仍保持 attention=1，作为固定条件影响后续执行的概率。屏蔽摘要的 likelihood **不等于**切断执行损失经过摘要上下文表示的参数梯度。

## 验证

```bash
python -m pip install -r RL_training/requirements-test.txt
python -m pytest -q RL_training/tests
```

仅依赖 CPU 的测试覆盖权重、上下文保留、工具与摘要屏蔽、奖励分组、截断处理和数据划分。完整兼容性测试额外读取固定版本的 verl 源码，执行真实上游 GSPO 损失和 Hydra 配置组合：

```bash
git clone --branch v0.9.1 --depth 1 https://github.com/verl-project/verl.git /tmp/verl-v091
VERL_SOURCE=/tmp/verl-v091 python -m pytest -q RL_training/tests
```

测试验证公式 (3)–(6) 的 ratio 数值、token 局部 Jacobian、离策略解析梯度、正负 advantage clipping、多次摘要屏蔽和分段补齐；另有真正的两进程 CPU Gloo/DDP 测试，将分布式 microbatch 累积结果与单进程全 batch 的损失及梯度逐项比较。未配置 `VERL_SOURCE` 时，上游兼容性测试会明确跳过。GPU 内核、Ray 集群、在线搜索与 judge 的端到端运行需在训练环境另行验证。
