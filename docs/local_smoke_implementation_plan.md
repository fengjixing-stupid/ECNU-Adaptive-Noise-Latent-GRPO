---
id: DOC-PLAN-001
type: design
status: active
title: 本地 Smoke 与 Kaggle Notebook 实现计划
created: 2026-09-30
updated: 2026-10-02
related_docs:
  - DOC-SPEC-001
  - DOC-MINFIX-001
  - DOC-TRAIN-001
  - DOC-POLICY-001
  - DOC-AGENT-001
planned_code:
  - adaptive_noise/noise_head.py
  - adaptive_noise/features.py
  - adaptive_noise/controller.py
  - adaptive_noise/checkpoint.py
  - scripts/smoke_local.py
  - notebooks/adanoise_kaggle.ipynb
---
# 本地 Smoke 与 Kaggle Notebook 实现计划

> 执行者：采用 superpowers:executing-plans 逐任务实施。已读 [子 Agent 协议](AGENT_USE.md)，本轮零子 Agent。后续只有清晰独立任务或有价值的独立审查才派发，明确文件所有权；根 Agent 集成、复核并维护 progress。

本地 CPU smoke 范围已获用户确认（2026-09-30）；此 active 状态只表示本地工程执行获准。Kaggle 数据规模、特征和训练/评估决策仍是待讨论的实验方案，执行前与用户交流。

**Goal:** 本地 CPU 验证原始 sampler 数学契约，Kaggle baseline 通过后再实现并验证 Noise Head，Kaggle Notebook 验证真实 1B latent 推理并执行公平对比。

**Architecture:** 一个纯 PyTorch adaptive_noise 包用于 CPU 与 CUDA；作者固定版本提供实际 latent inference。主项目维护可核验的源码补丁，不直接把作者整个 CUDA/verl 依赖栈安装到本地。Notebook 仅调用同一组 CLI，负责资源探测、输入挂载和阶段门禁。

**Tech Stack:** 本地现有 Python/PyTorch/pytest；Kaggle 单 GPU 自定义 SGLang、离线训练用 PyTorch、结果用 JSONL/Parquet。新增依赖在实施时明确版本与用途，不自动安装。

当前训练路线按 [两阶段训练与共享题集](training_route.md)：基础筛选即可，两阶段共享train题目，RL独立on-policy采样，validation/test隔离；不要求同prefix分支搜索。

**Spec:** [任务规范](AdaNoise_LGRPO_Codex_Kaggle_Spec.md)；[当前 minfix 审计](minfix_submodule_update.md)。返回 [文档索引](README.md)。

## 1. 当前边界与前置条件

Task 1 本地 sampler smoke 已实现，见 [命令与验收](local_cpu_smoke.md)。其余任务仍是待实现接口；本轮不执行模型加载、训练、数据下载或Kaggle任务。

| 阶段 | 做什么 | 可证明什么 |
| --- | --- | --- |
| L0 本地 CPU smoke | 小张量、toy frozen backbone、合成轨迹 | 数学、梯度隔离、保存加载、恢复和 sampler kernel 等价 |
| K0 Kaggle inference smoke | 真 checkpoint、1题/1 rollout、单 GPU | 真实 latent/Gumbel/hidden 路径及资源可用 |
| K1 Kaggle engineering | 32 train/32 validation/32 test | Stage A → adaptive → paired evaluation 全流程 |
| K2 pilot/final | 先 200 train/100 validation/200 test，按实测预算扩展 | 方法有效性、运行成本、置信区间 |

项目和子 Agent 规范已核验；真实模型阶段必须先获得用户提供的 checkpoint/data 路径。任何阶段发现该阶段必需文件或输入缺失，立即停止并报告，不继续工作或猜测模型。
本地 CPU 运行不代表作者 CUDA Engine 在 macOS 可用，也不代表真实模型 smoke 已通过。

## 2. 全局约束与设计决定

冻结 backbone：eval 模式、所有 requires_grad=False、forward 使用 no_grad；head 输入 detach。optimizer 只含 hidden projection 和 MLP 参数。
Noise Head：d→64 trainable projection，拼接六维统计→70→128→32→1，GELU，sigmoid；scale=0+(1-0)*ratio。固定 K=10，Gumbel temperature=1.0，one-sided=True。Stage B 已于2026-10-02确认纳入首版；本段单输出sigmoid接口只描述Stage A，RL随机动作参数化需讨论后扩展。
固定 baseline 和 adaptive 使用同 checkpoint、prompt、top-p、temperature、response cap、verifier、split、seeds、rollout 数；只改变 scale policy。
不得自动量化、启用无 hidden fallback、改成每题单 scale 或改变实验预算。
环境/网络/设备/OOM 错误立即停止实验并写失败记录；不得根据 Spec 的降级序列自动重试。项目代码错误可修复，三次失败停止并指出可疑假设。

### 2.1 特征的确定定义

在当前 logits 加噪前取原始 top-K logits，在 K 内 softmax 成 p；统计在 float32 计算。这是 controller 特征，不替换作者的 full-vocabulary noisy top-K selection。
H=-sum(p*log(p+1e-8))；H_norm=H/log(K)；margin=p1-p2；ESS=1/sum(p²)；ESS_norm=(ESS-1)/(K-1)；pmax=max(p)。
delta_entropy 使用 raw H_t-H_prev；首步为0。t 采用1起始，step_ratio=t/T_max；首版 T_max=max_new_tokens（K0为64），不额外改变作者 latent 终止机制。六维 scalar 顺序：[H_norm,margin,ESS_norm,pmax,delta_entropy,step_ratio]。
features 只返回 detached hidden 与 scalar，不持有 trainable projection；projection 属于 NoiseHead，因而 Stage A 必须存 raw detached h_t，不能缓存随训练变化的投影。
采集时 raw hidden 单独保存 CPU fp16 分片，记录 hidden_size/dtype/row key；统计与标签使用 float32。不得缓存 GPU 全序列 hidden。
K<2、非有限概率、无效步号或 shape 不一致都抛项目配置错误；不通过隐式 clamp 掩盖错误。

### 2.2 identity 的适用范围

原版固定路径使用第一行 scale；所以旧版 identity 仅在所有请求 scale、temperature 相同的原作者语义下逐张量核对。
新版 fixed 按请求 [B,1] scale 应用；异构 batch 对每行独立参考路径核对，明确这是原作者首行广播问题的修复。
仅 adaptive 生效请求进入 head；请求 entropy/step 状态以 request id 存储并随 filter/merge/reset 更新，完成和取消请求清理。
保留 minfix 的8192 chunk及随机调用顺序、原有 clamp、one-sided、top-p 保底 K、Gumbel temperature、配置化 latent_end_token_id mask、latent/explicit 切换及所有随机抽样顺序。

### 2.3 可复现源码组织

主项目 submodule 固定用户修订 commit `0b7e85f15e9859033653964282348e518f9f6291`，路径 `latent_grpo_minfix/Latent-GRPO-final/`；源码在其 `Latent-GRPO/` 子目录。
实施时创建 patches/sglang_adanoise.patch 和 scripts/prepare_upstream.py；在 artifacts/upstream/ 复制需要的上游工作树、校验版本并应用补丁，原参考 checkout 保持干净。
Kaggle 输入的 code-package 必须含固定上游源码或预构建包、补丁和主项目包；manifest 记录主项目 commit、上游 commit、patch SHA256。不能假定离线 Notebook 能访问 submodule URL。

## 3. Review Focus

| 失败输入/条件 | 验收证据 | 归属 |
| --- | --- | --- |
| raw/noisy p 混用造成循环 | 相同 raw logits、改变噪声时特征不变 | Task 3 |
| 请求 filter/merge 后串行号 | 重排、完成、取消后逐请求 H_prev/t 不串线 | Task 3 |
| backbone/projection 梯度边界错 | backbone 无 grad，projection/MLP 有有限非零 grad | Task 3 |
| 停止/取消/断点造成重复结果 | resume key 去重，manifest 不匹配拒绝恢复 | Task 4 |
| 训练/validation/test 或增强题泄漏 | parent id 分组切分、test 标签禁止被 selection 消费 | Task 4/5 |

## 4. 实现任务（顺序执行）

### Task 1 — 固定 sampler 参考与本地 smoke 入口（约 2–3 小时）

**Files:** tests/reference_fixed_sampler.py、tests/test_fixed_mode_identity.py、adaptive_noise/sampling_kernel.py、scripts/smoke_local.py、configs/local_smoke.yaml。
**Interfaces:** sample_latent_kernel(logits:[B,V], fixed_scales:[B,1], top_p:[B,1], K:int, gumbel_temperature:float, one_sided:[B,1], active_mask:[B], gumbels=None, latent_end_token_id:int) → indices:[B,K], probs:[B,K], next_latent_id:[B]。
参考函数从锁定 minfix `_streaming_latent_noisy_topk` 抽取数学路径，独立保存并注明来源；不导入 SGLang engine、Ray、verl 或 CUDA kernels。

- [x] 先写测试：B=1/3、V=32、K=10、seed=0/1/2、one-sided 两种、top-p=.2/.95，另覆盖V=8192/8205边界；同一分块 RNG 初态下 minfix reference 与 kernel 的 indices、probs、latent id 相等。
- [x] 验证 active/nonlatent/配置化 latent_end_token_id mask，scale=0 消去噪声且保持原 top-K 概率；full vocabulary 噪声可改变候选集合。
- [x] 实现8192分块 kernel 与 smoke CLI；测试确认noisy workspace不恢复为[B,V]；生产采样共用该 kernel，Kaggle patched sampler 接入此函数，避免本地与真实路径各写一份算法。
- [x] 执行 `python -m pytest tests/test_fixed_mode_identity.py -q`；通过后生成 artifacts/local_smoke.json，schema_version=1、scope=tensor_only、device=cpu、seed、版本、case 名称和 pass/fail、runtime_sec。
- [x] 提交此任务明确文件；运行测试失败即不进入下一任务。

**Gate:** 单 CPU 小张量通过；不要求本地下载或加载 1B。预期 smoke 运行 10–60 秒，超时记录并排查，不能伪称通过。

### Task 2 — Kaggle 固定 baseline 与 Hidden Probe（约 2–3 小时）

**Files:** scripts/{prepare_upstream,smoke_kaggle}.py、configs/adanoise_kaggle.yaml、notebooks/adanoise_kaggle.ipynb。
**Interfaces:** fixed baseline 使用原始作者 Engine；仅导出当前 LAST hidden probe、latent step 计数及资源记录。现有 LAST capture 先 probe，不先修改 model runner。

- [ ] Notebook 首 cell 检查 GPU、显存、torch/CUDA/依赖版本与输入路径，写 artifacts/environment.json；失败立即终止后续 cells，禁止自动安装补救。
- [ ] 先运行未启用 head 的 frozen baseline：1题、1 rollout、batch=1、max_running_requests=1、tp=1、max_new_tokens=64、top_k=10、response temperature=.6、top_p=.95、Gumbel temp=1、scale=1、one-sided=True；bf16 支持由硬件检测，明确记录实际 dtype，否则选择兼容 fp16，不能执行失败后再降级。
- [ ] 验证 prompt 以 `<think>` 进入 latent；核验 tokenizer 的 `</think>` id 与配置化 latent_end_token_id mask一致；输出至少1个 latent step、hidden 非空 [B,d]、有限统计。未进入 latent 的普通生成视为 smoke 失败。
- [ ] 两次相同输入/seed 的 baseline 输出一致；显式固定 sampling seed，验证实际 worker 使用的 seed，不仅设置 Notebook 父进程 RNG。
- [ ] 保存 smoke_test.json：output_text、num_latent_steps、peak_gpu_memory_mb、runtime_sec、method、hidden probe、baseline 复现证据和失败类别；shutdown engine 后提交。

**Gate:** Kaggle 的 K0 是所有真实数据采集和训练的先决条件。记录 PyTorch peak allocated/reserved 与进程 GPU 占用，不能把 SGLang 全部显存仅用 parent process 的 torch 峰值代替。CPU artifact 的 peak_gpu_memory_mb=null。

### Task 3 — Features、Noise Head 与 Sampler 集成（约 5–8 小时）

**Files:** adaptive_noise/{features,noise_head,controller,checkpoint}.py、tests/test_noise_features.py、tests/test_noise_head.py、tests/test_adaptive_scale_bounds.py、tests/test_sampler_adaptive_scale_shape.py、tests/test_request_state.py、patches/sglang_adanoise.patch。
**前置:** Task 2 的真实 fixed baseline smoke 已通过；未通过时只写测试，不实现 Noise Head。
**Interfaces:** compute_noise_features(hidden_state:[B,d],topk_probs:[B,K],step_idx:[B],max_latent_steps:int,prev_entropy:[B]|None) → dict(hidden_input,scalar_input,entropy,entropy_norm,margin,ess,ess_norm,p_max,delta_entropy,step_ratio)。
NoiseHead(hidden_size,hidden_proj_size=64,mlp_hidden_1=128,mlp_hidden_2=32,min_scale=0,max_scale=1).forward(hidden_input,scalar_input) → (scale:[B,1],ratio:[B,1])。
controller.resolve_scale(enabled,head,features,fixed_scales) → [B,1]；disabled 不调用 head；enabled 缺 head/hidden 立即报错。
checkpoint.save/load 保存 head state、architecture、feature schema、upstream/model 身份、scale bounds、训练状态；参数不匹配拒绝加载。

- [ ] 测试 uniform K=10：H_norm≈1、ESS_norm≈1、margin=0、pmax=.1；one-hot：H_norm≈0、ESS_norm≈0、margin=1；首步 delta=0，第二步 raw entropy 差值正确。
- [ ] 测试 detach、K/shape/nonfinite 拒绝；统计不受当前 Gumbel 随机变量影响。
- [ ] 实现 head/controller，测试 B=1/3、极大正负输入、有限输出及上下界；两个不同 scale 行只影响各自采样。
- [ ] 在 tiny frozen nn.Linear backbone 上构造32条固定合成记录，SmoothL1/Adam(lr=.001) 训练20步；断言 backbone 参数逐位不变且 grad=None，projection/MLP 获得梯度，有限 loss 比初始更低。该 toy loss 不作为方法有效性证据。
- [ ] 保存加载得到相同 scale；测试 architecture mismatch。接入 worker 启动加载一次 head，sampler 用 Task 1 kernel，SamplingBatchInfo/request 同步 step/entropy/enable 状态；测试 filter/merge/reset/完成/取消，执行五个 Spec 指定测试文件和完整 local smoke。
- [ ] Kaggle 用 toy head 做工程 adaptive smoke，标记非正式方法结果；验证每 latent step scale 与 hidden 对齐、head no_grad、不存 GPU history。之后替换真实 Stage A head，提交本任务文件。

**Gate:** 本地 L0 smoke 验证全部数学/冻结/保存加载案例，记录每项证据。合成样本训练可以在本地运行，真实 head 训练必须在 K0 通过后。

### Task 4 — Stage A 采集、训练与恢复（约 4–6 小时，不含推理耗时）

**Files:** scripts/{collect_fixed_sweep,train_noise_head}.py、adaptive_noise/{dataset,resume}.py、tests/test_oracle_labels.py、tests/test_resume.py；更新 Notebook 对应 cells。
**Interfaces:** collection row key=(dataset,split,parent_problem_id,problem_id,seed,scale,rollout_id,latent_step)，结果与 hidden shard 由此关联；build_oracle(rows) → problem_id 到最优 scale。

- [ ] 测试 sample-level 平均 correctness argmax、平手选择小 scale（validation baseline 同样按平均 Pass@1 最大、小 scale 平手规则）；同题各轨迹 step 共享标签；重复/缺失 rollout 不能默认为完整，all-wrong 题明确标记为 noisy label。
- [ ] Engineering：32 train/32 validation/32 test，seeds=[0]，候选[0,.25,.5,.75,1]，每scale 2 rollout；训练采集共320 rollout。按 parent_problem_id 分组，固定 split manifest，禁止同一增强题跨 split。
- [ ] 实现 no_grad 采集 raw hidden/统计/正确性/时长/invalid；train 拟合 oracle，validation 用于选择 fixed c* 与 head checkpoint，test 不参与选择。head Adam lr=.001、batch=64、epochs=20、SmoothL1；按题等权避免长轨迹主导。
- [ ] 每25样本写 progress.json/results_partial.parquet/hidden shards；训练另写 noise_head_latest.pt，包含 optimizer/epoch/RNG。恢复验证 model/split/feature/config manifest，原子写文件，row key 去重；测试中断恢复与不中断结果一致。
- [ ] K0 通过后执行真实 Stage A；每 epoch 在独立 validation oracle 记录上计算按题等权 SmoothL1，best 按最小 validation loss 保存，平手保留更早 checkpoint；不在每 epoch 额外运行模型；最后用锁定 best head 做 validation 推理核验。论文说明 sample-level oracle 不代表真正 step-optimal policy。

**Gate:** 采集文件完整、无 split 泄漏、resume 不重复，backbone 未改变；保存 noise_head_best.pt。Pilot 200/100/200，2 rollout/scale；Final seeds=[0,1,2]，规模由实测耗时决定。
预算在 K0 后按 `(题数×候选scale数×rollout数×秒/rollout)` 估算并包含训练/验证时间；超 session 预算先停止计划扩展，不启动注定超时任务。

### Task 4B — Head-Only RL（首版必需，详细实施计划待实验设置确认）

**前置:** Kaggle baseline/hidden probe与Stage A已通过，用户确认随机动作分布、warm-start衔接、更新方式及预算；必需checkpoint/data输入齐备。未确认前不实现或运行本任务。
**数据契约:** 共享Task 4的train题目与split manifest；使用当前head的新rollout，不把固定sweep或旧策略轨迹直接用于on-policy更新。最终正确性奖励，只更新head/projection，冻结backbone。

- [ ] 明确随机动作与log_prob接口、行为版本、head梯度获取、Stage A权重迁移和确定性评估规则；另写具体可执行测试步骤。
- [ ] 测试逐token动作、最终奖励到head的梯度、backbone冻结和同题独立轨迹，以及validation/test不进入更新。
- [ ] 确认预算后进行head-only RL并支持checkpoint/optimizer/RNG恢复，记录独立Stage A/Stage B checkpoint身份。
- [ ] Kaggle或资源失败按用户规则停止；不得把省略Stage B视为首版完成。

### Task 5 — 公平评估、Notebook 阶段门禁与分析（约 3–5 小时，不含推理耗时）

**Files:** scripts/{evaluate_paired,analyze_results}.py、tests/test_split_selection.py、tests/test_metrics.py；完成 Notebook 及文档使用命令。
**Interfaces:** evaluate 使用选定 c* 和锁定 Stage B head，两种 method 的完整 manifest 相同，仅 scale policy 不同；每题 seed/rollout 顺序配对。

- [ ] 测试 verifier 正确/错误/无法解析/截断；invalid 保留在 Pass@1 分母；latent、explicit、total token 数分别记录。测试选择脚本拒绝 test split；可选 Pass@4 默认关闭，需要预算允许后启用，两方法均为4 rollout。
- [ ] 测试成对合并时缺失或重复 method 记录报错；按题聚类 paired bootstrap 95% CI，final 三 seed 单独汇报，不能把同题不同 seed 当独立题增加样本量。
- [ ] Engineering 按两方法各32 test×1 rollout 运行；固定 baseline scale 由 validation 锁定，test 不再调参，记录所有 per-example/per-step Spec 字段和额外 scale mean/min/max/std。
- [ ] 输出 summary.csv、paired_results.parquet、trajectory/entropy_scatter/method_comparison/scale_distribution 的 PDF/PNG；没有正确或错误样本时明确写 unavailable，禁止制造示例。
- [ ] Notebook cells 依次执行资源探测→baseline smoke→hidden probe→collection→Stage A head train→Stage B head-only RL→adaptive smoke→paired eval→plots→导出；前一步 artifact 不通过则下一步不执行。运行相应测试、同步文档并提交。

**Gate:** scale 是乘法系数，方差比例=(s/c*)²；c*=0 时比例未定义，只报告绝对 scale 与扰动方差。最终报告区分 L0/K0/K1/K2，只有真实结果支持方法结论。测试 head 输出依输入变化，但研究结果学成常数或更差必须如实报告。

## 5. 待实现命令与验收输出

本地指定虚拟环境中的 python 执行下列命令；命令不写入机器专用路径。

```sh
python -m pytest tests -q
python scripts/smoke_local.py --config configs/local_smoke.yaml --device cpu
```

Kaggle code-package 放入 `/kaggle/input/<code-package>/`；工作目录 `/kaggle/working/adanoise_lgrpo/`，输出统一 `artifacts/`。模型与数据由 CLI 显式映射，不修改挂载输入。

```sh
python scripts/smoke_kaggle.py --platform kaggle --config configs/adanoise_kaggle.yaml --method fixed
python scripts/collect_fixed_sweep.py --platform kaggle --config configs/adanoise_kaggle.yaml --resume
python scripts/train_noise_head.py --platform kaggle --config configs/adanoise_kaggle.yaml --resume
python scripts/smoke_kaggle.py --platform kaggle --config configs/adanoise_kaggle.yaml --method adanoise
python scripts/evaluate_paired.py --platform kaggle --config configs/adanoise_kaggle.yaml --resume
```

Stage B CLI尚未定义，须在批准详细RL接口后加入；上述train_noise_head.py命令仅代表Stage A。

所有 CLI 支持 `--model-path`、`--data-path`、`--output-dir` 与对应标量 override；合并优先级 CLI>YAML>默认值，写 resolved_config.json。分析脚本单独读取 artifacts，不重复启动模型。

## 6. 设计阶段验收

- [x] 固定上游版本并审计 sampler/hidden/weighted embedding 调用链。
- [x] 定义本地张量 smoke 和 Kaggle 真实 inference 的证据边界。
- [x] 明确特征、冻结、scale、原始 identity、Stage A 和公平比较契约。
- [x] 已补读 docs/AGENT_USE.md；按委派门禁确定顺序执行和零子 Agent。
- [x] 用户确认本地 CPU smoke 范围；实验设计在执行前与用户交流。
- [ ] 真实模型步骤前提供 checkpoint/data 路径并确认实验方案。

本地 CPU smoke 已获准；真实模型执行仍需 checkpoint/data 输入，后续实验设计先与用户交流。Task 1已通过；未实现或运行的其余测试不标记为通过。
