# Glasseek：verl SFT 与 CM-GSPO

启动入口已统一为 `scripts/train_sft.sh`。默认模式保留 `MODEL_PATH` 等环境变量；
历史实验使用 `--profile 128k|64k|turns|clean2899|length`，配置集中在
`configs/sft_launch_profiles.json`。`--print-command` 可在不加载训练模型的情况下
检查实际命令；额外 Hydra 参数放在命令末尾并覆盖 profile 默认值。

```bash
bash RL_training/scripts/train_sft.sh --profile clean2899 --print-command
bash RL_training/scripts/train_sft.sh --profile turns optim.lr=1e-5
PYTHONPATH=RL_training python -m glasseek_training.lora_probe 4856 --profile offload --print-config
python RL_training/glasseek_training/verify_weights.py --source /path/to/checkpoint --target /path/to/model.safetensors
```

`lora_probe.py` 合并单卡、DDP、128K 和 activation-offload 变体；其配置位于
`configs/lora_probe_profiles.json`。`verify_weights.py` 共用精确 dtype/数值比较，
支持通过 `--world-size` 指定分片数。这些参数预检不替代 GPU 训练或大权重实测。

课程训练只保留 `curriculum_sft.py` 一个 trainer 实现，以
`SFT_CURRICULUM_POLICY=length|turns` 选择策略。profile 自动设置该变量。
长度策略仍为单卡、SP=1，按 epoch 随机选样后按长度排序；turns 策略仍要求双卡、
SP=1、64K manifest，保留原补齐和分片次序。历史验证模块继续继承该 trainer。

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

### URL 表示模式

数据准备、SFT、RL 和评测使用同一个 `--url-mode raw|hybrid|ref|evidence` 参数，
也支持 `GLASSEEK_URL_MODE` 环境变量；命令行优先。未指定时为 `raw`，保持原有 URL、
工具 schema 和工具返回值。公共实现位于 `Shared/glasseek_shared/hermes/url_registry.py`。

非 raw 模式的正文来源引用统一写成 `<url>U1</url>`，证据片段写成
`<url>D1@v1#E3</url>`。工具参数只传内部 ID，例如 XML 工具调用：

```xml
<tool_call>
<function=web_extract>
<parameter=urls>["U1"]</parameter>
</function>
</tool_call>
```

注册表内部以 `U1` 为键，程序恢复完整 URL。正文/工具观察显示 XML 引用，
离线转换后的工具参数输出裸 ID；旧的完整 XML 参数仍兼容解析。
协议版本为 `glasseek-url-v3-text-xml-args-id`；
旧版非 raw SFT 数据应重新转换，旧评测结果保留，不与新协议混算。

system prompt 插件由 `--url-prompt` 控制，训练和评估使用相同入口：

```bash
# 自动注入对应模式的映射说明与 XML few-shot（默认）
python Evaluation/services/eval_run.py base9b full --url-mode ref --url-prompt auto
# 关闭 system prompt 和工具描述的提示插件，仍保留 ref 程序解析，用于消融
python Evaluation/services/eval_run.py base9b full --url-mode ref --url-prompt off
# 读取自定义 JSON 配置；SFT、RL 和数据准备命令也接受这两个参数
python Evaluation/services/eval_run.py base9b full --url-mode ref --url-prompt /path/to/url-prompts.json
# 只预览插件，不启动模型或评估
PYTHONPATH=Shared python -m glasseek_shared.hermes.url_prompt --url-mode evidence
```

默认配置为 `Shared/glasseek_shared/hermes/url_prompt_config.json`，包含 `version`、
`common`、`modes`、`few_shot`、`tool_descriptions` 和 `tool_mode_descriptions`，
可复制后修改，无需改 trainer。工具描述由同一插件追加到 system prompt 和 API tool schema；
`--url-prompt off` 同时关闭两处附加说明，基础工具描述保留。
环境变量为 `GLASSEEK_URL_PROMPT`，命令行优先；多机 RL 的自定义配置路径需所有 worker 可读。
插件追加到默认或自定义 system prompt，已有基础提示保留；相同插件不会重复注入。
示例不写入真实来源注册表，不包含未来轨迹映射；正文为稳定文本，便于前缀缓存。
版本和内容 SHA256 随数据/轨迹元数据记录。更换配置后用新会话、新输出目录运行，
不要在活动评测中编辑配置。`raw` 无论是否选择插件都保持原提示。

| 模式 | 模型看到的来源表示 |
| --- | --- |
| `raw` | 原始 URL，原有默认行为 |
| `hybrid` | 来源首次出现保留 URL 并附短 ID，后续使用短 ID；额外元数据可能增加 token |
| `ref` | 短 ID 和域名，程序还原实际请求 URL |
| `evidence` | 来源短 ID、带版本的页面快照及证据片段；可用 `D1@v1#E3` 读取已存片段 |

直接在现有命令末尾添加参数。例如以 `ref` 为例：

```bash
python -m glasseek_training.prepare_data sft \
  --input /data/trajectories.jsonl --output /data/sft-ref --url-mode ref
SFT_TRAIN_FILE=/data/sft-ref/train.parquet SFT_VAL_FILE=/data/sft-ref/val.parquet \
  bash RL_training/scripts/train_sft.sh --url-mode ref
bash RL_training/scripts/train_cm_gspo.sh --url-mode ref
python Evaluation/services/eval_run.py base9b full --url-mode ref
# 直接调用原评测脚本时，在原有参数之后增加 --url-mode ref 也可。
```

SFT 必须先按所选模式准备数据，loader 会拒绝训练模式与数据标签不一致的样本。
RL 的题目/参考答案数据无需 URL 改写，模式作用在在线工具边界。
模型采样的 token 和 logprob 原样保留，程序只解析工具参数副本并转换新产生的工具观察；
SFT 的离线消息转换与 RL 的在线采样是两个不同入口。

短 ID 按每条轨迹独立分配，只在工具观察公布来源后可用于下一轮；同轮并行工具不能引用
彼此尚未公布的来源。上下文压缩继承来源表，子任务使用独立来源表。
评测映射和页面快照默认保存于 `$HERMES_HOME/url_registry`，可用
`GLASSEEK_URL_STATE_DIR` 指定独立目录，包含原始 URL，按轨迹数据管理访问权限。

34 条历史轨迹的确定性回放入口为 `scripts/trajectory_benchmark/benchmark_url_modes.py`；
`run_url_modes_live.py --output <独立目录>` 复用现有评测入口执行四种模式端到端重跑。
回放的 token 节省仅统计去重后的历史消息，不等于包含重复上下文的实际推理输入 token；
工具参数还原率也不等于网络请求成功率或最终答案正确率。

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
source Evaluation/services/search_pool_env.sh
export JINA_API_KEY=...  # 可选
export GLASSEEK_JUDGE_BASE_URL=https://your-judge-service/v1
export GLASSEEK_JUDGE_MODEL=your-judge-model
export GLASSEEK_JUDGE_API_KEY=...
bash RL_training/scripts/train_cm_gspo.sh
```

`web_search` 使用与评测、数据生产相同的共享搜索池，启动及配置见 [搜索池说明](../Evaluation/services/README.search-pool.md)。Serper 密钥只保存在服务端，不需要分发给训练 worker；启动器会把池地址传入 Ray runtime environment。多机训练需设置所有 worker 都能访问的 `SEARCH_SERVER_ENDPOINT`，不能使用各机器自身的 localhost。`web_extract` 仍使用 Jina，支持 Qwen JSON/XML tool-call 格式；没有复用整个 Hermes 运行时，浏览器、代码执行等工具需要在 `web_tools.py` 扩展。模型通过 tokenizer 的 tools 参数得到工具 schema，生成 token 原样保留，工具返回值以 mask=0 追加。上下文压缩使用当前 rollout 的同一策略服务器；摘要调用不参与训练。

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
