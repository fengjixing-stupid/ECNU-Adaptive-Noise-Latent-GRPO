---
id: DOC-DATA-001
type: design
status: proposed
title: 数据集构建脚本设计
created: 2026-10-02
updated: 2026-10-02
related_docs:
  - DOC-TRAIN-001
  - DOC-PLAN-001
---
# 数据集构建脚本设计

返回 [文档索引](README.md)。已确定题源、不再去重边界与首轮子池方案：先分别抽取，再固定 240 train / 80 validation，之后 probing。probing scale、递增重复次数和 train 筛选规则已确定；抽样/切分采用 master_seed=42 的三路派生 seed；生成配置等模型实施参数尚待确认。本文设计脚本接口，尚未实现代码或运行模型。

## 输入与目标

先输出候选子池和固定切分清单；随后由冻结 1B checkpoint 的 probing 估计经验正确率，按批准的基本筛选规则形成实验训练题集。Stage A/B 共享最终 train 题目，Stage B 使用同题的新 on-policy rollout。不能把候选子池直接称为最终训练集。

| 文件（相对于 data/） | 用户指定角色 | 元数据记录数 |
| --- | --- | ---: |
| GSM8k-Aug-oss-dup-all.parquet | 训练候选 gsm8k_aug | 301852 |
| DAPO-Math-17k-en-train.parquet | 训练候选 dapo_math | 14116 |
| GSM8k-Aug-test.parquet | 保留测试 gsm8k_aug_test | 1319 |
| Math-500-test.parquet | 保留测试 math500_test | 500 |

2026-10-02：指定虚拟环境的 PyArrow 25.0.1 已成功读取四个文件的 schema 和首条记录，尚未扫描全部记录。此前读取 DAPO 出现 `Repetition level histogram size mismatch`；用户更新 PyArrow 后本次重试成功。没有代用户安装依赖或修改环境。

四个文件均包含 `prompt`、`reward_model.ground_truth` 和 `extra_info.question/index/split`。`data_source` 值不是可靠的实验题源身份，使用上表显式别名。Math-500 内部 `extra_info.split` 为 train，必须按文件角色输出为 test，并保留内部原始字段供追溯。

用户决定信任作者已有处理：不再次去重，不比较训练/测试的文本相似度，不要求 parent_problem_id，不做语义聚类；保留来源中的重复记录。记录级 ID 隔离不代表语义去重已验证。

## 拟实现接口

新增 `scripts/build_dataset.py`、`adaptive_noise/dataset_builder.py`、`configs/dataset_build.yaml`、`tests/test_dataset_builder.py`；独立于作者 submodule，不修改其代码。

拟定命令为 `python scripts/build_dataset.py --config configs/dataset_build.yaml`。配置显式指定四个文件、输出目录、master_seed、每来源候选题数和 train/validation 配额。首轮每来源 160 题，分为 120 train / 40 validation；master_seed 固定为用户确认的42，三个派生 seed 按下节算法生成。缺失文件或必填配置立即失败，不自动下载或寻找替代文件。

实现分为三个顺序步骤：

1. 流式读取两个训练候选文件，验证 question 非空、prompt 为有效 role/content 消息、ground_truth 为非空字符串；保留消息边界、原始答案及 LaTeX，不改写 prompt、不凭长度或猜测难度删除题目。无效训练记录写入 rejected 清单并注明原因；结构不兼容或文件读取错误立即停止。测试记录不自动删除，遇到无效字段停止并报告。
2. 按来源生成稳定行 ID。先从 GSM8K-Aug 和 DAPO 的有效记录中分别无放回抽取 160 题，再分别以独立切分随机流划分 120 train / 40 validation；总计 240 train 候选、80 validation 候选。同文件、配置和 seed 得到相同抽样及切分结果。必须在模型 probing 之前写出并锁定 split manifest。指定测试文件直接复制为独立 test 分区，不从候选子池再划 test，不用内部 split 字段路由。
3. 写出题目池、split manifest 与统计报告；先完成文件写入，再发布完成标记。已有输出目录拒绝覆盖，要求新的输出路径。只报告实际读取和筛选数量，不把文件元数据数量当作最终可用数量。

基础字段检查并不证明所有答案可自动评分。保留原始 ground_truth 和 reward_model.style，后续使用统一的最终答案 verifier；不因 Math-500 包含符号答案而提前删题。verifier 未实现或不支持必需格式时，按缺失前置条件停止采集。

## 已确认 master seed 与派生规则

用户确认 `master_seed=42`，只派生三个用于数据构建的 seed；不把三个不同 seed 当作三轮独立实验。算法为 `sha256-v1`：对 UTF-8 字符串 `adanoise.dataset-seeds.v1:{master_seed}:{role}` 计算 SHA-256，取前4字节按无符号大端整数解释。避免依赖进程随机化的 Python `hash()`。

| role | 派生 seed（master=42） | 职责 |
| --- | ---: | --- |
| gsm8k_aug_sample | 2028899853 | GSM8K-Aug无放回抽样160题 |
| dapo_sample | 3785434924 | DAPO无放回抽样160题 |
| train_validation_split | 262147858 | 抽样后各来源120/40切分 |

抽样两个 RNG 独立；切分只使用第三个 seed 初始化的一个独立 RNG，按固定来源顺序 GSM8K-Aug → DAPO 连续消费，分别切分各来源160题。随机操作前将记录按 source_row 排序，避免流式读取批次影响排序。保存 master seed、全部 role/派生值、派生算法版本、采样 RNG 算法及 Python 版本到 build manifest。改 master seed 属于新构建，已锁定 split 不被覆盖。

三个派生 seed 的职责只覆盖数据抽样和切分；模型 rollout 的随机数安排不默认为这三个 seed，需在模型采集配置中另行明确。

## 数据契约与产物

题目记录包含 `problem_id`、`source`、`source_file_sha256`、`source_row`、`original_index`、`original_split`、`split`、`question`、`prompt`、`ground_truth`、`reward_style`。problem_id 由来源别名、文件内容身份和行号构成；不把题目哈希用作去重 ID，不猜测原题关系。

输出目录包含：

| 产物 | 用途 |
| --- | --- |
| train.parquet / validation.parquet | 首轮抽取的候选子池按固定切分分配：240 / 80，尚未经过模型筛选 |
| test_gsm8k_aug.parquet / test_math500.parquet | 保留全部指定测试记录，分来源报告结果 |
| split_manifest.parquet | problem_id、来源、split，供两个训练阶段共享 |
| build_manifest.json | 输入哈希、构建配置、版本、PyArrow版本、实际统计与检查状态 |
| rejected.parquet | 无效训练候选记录的位置和原因；不存模型 rollout |

顺序固定为：基础检查 → 每来源抽样 → 来源内切分 → 锁定候选 split manifest → probing → 基本筛选 → 锁定最终 train selection。240:80 指 train/validation，不是两个现有测试集的规模。不能根据 probing 结果移动记录的 split 或重新切分。Stage A/B 复用同一份最终 train selection；validation 只用于选择，test 只用于最终评估。建议 validation 不按经验正确率筛掉难题，保留完整 80 题；该筛选细节尚待讨论。

## 与模型阶段的连接

本地构建仅操作数据，不加载 checkpoint、不生成 hidden、不估计成功率。模型路径已由用户提供：`/Users/fengjixing/Python_Project/models/LLaMA3.2-1B-Instruct-Latent-GRPO-Top10`；真实采集仍须先通过 Kaggle GPU baseline smoke。

后续 `collect_fixed_sweep.py` 接受固定 train/validation selection，在批准的 scales 和 M 下记录最终正确性，计算每题每个 scale 的经验成功率。保存每题每个 scale 的成功数、完成 rollout 数及经验正确率 q_hat_i(s)，用于基本难度筛选和 warm-start 初始化；不要求细致难度分桶或同 prefix 搜索。probing 与 Stage A sweep 是否共用轨迹及采集配置尚未确定，不能自动重复采集，也不能未校验契约就宣称可复用。不回头修改 split；train 筛选按下节最新用户决定执行，替代此前不剔除全失败题的建议。筛选移出本轮训练题集，不删除原始数据。

两阶段优化只读取 train。Stage B 重新 rollout，不能复用 Stage A 轨迹作当前策略样本。测试文件不提供训练 scale 标签，不参与候选 scale、数据比例或 checkpoint 选择。

## 已确认 probing 与 train 筛选规则

首轮 scale 集合 S 为 `[0, 0.25, 0.5, 1.0, 2.0]`，保守备用集合为 `[0, 0.25, 0.5, 0.75, 1.0]`。s 是 Gumbel 乘法 scale，方差随 s² 变化。probing 每条轨迹固定同一个 s，不使用逐 token head 动作。

首轮按 scale 报告 invalid generation 比例与最终答案正确率。用户根据 s=2.0 下是否大量 invalid 或明显正确率下降，决定后续沿用首轮集合还是改用保守集合；“大量”未设数值阈值，脚本不自动切换。已完成 probing 的结果始终标注其实际 scale 集合，不能用缺失的 s=0.75 结果冒充保守集合完成结果。是否修改 Noise Head 输出上界独立于 probing 候选集合，尚未批准。

s=0 始终 M=1，在所有阶段、追加和恢复时均不得再请求本题的 s=0 rollout。四个非零 scale 首轮分别 M=3；如需追加，各累计到 M=5，再累计到 M=7，每次每个非零 scale 只新增2条，保留既有结果。每题正常完成的累计 probing 次数为13、21或29；这是随机扰动响应的粗略筛选，不把 s=0 一次成功当作统计上确定的成功概率。

在每个完整采样阶段计算 q_hat_i(s)=success_count/completed_count，delta=max(q_hat)-min(q_hat)，包含 s=0 的单次结果。禁止从不完整记录默认为完整或把缺失 rollout 计作错误答案。train 判定按以下优先级执行：

| 条件 | 本轮训练题集处理 | reason |
| --- | --- | --- |
| 所有 scale 的已完成结果全错 | 剔除 | all_wrong |
| 所有 scale 的已完成结果全对 | 剔除 | all_correct |
| 有对有错且 delta > 0.4 | 保留，停止追加该题 | noise_sensitive |
| 有对有错且 delta ≤ 0.4，非零 M=3或5 | 非零 scale 累计追加到5或7 | continue_probing |
| 有对有错且 delta ≤ 0.4，非零 M=7 | 剔除并保存 problem_id | not_sensitive_at_cap |

阈值严格大于0.4，等于0.4不保留。不额外增加重复次数，不为达到目标训练题数自动补抽或放宽筛选。筛选后 train 数量及两来源比例由结果决定。全错/全对剔除是有限次观测下的本轮选择，不宣称证明题目永久不可解或必然正确。

输出 `probe_rollouts.parquet`（problem_id、split、scale、rollout_id、seed、最终答案、reward、终止/invalid状态）、`probe_summary.parquet`（每 scale 的样本数/成功数/q_hat、delta、采样阶段）、`selection_decisions.parquet`（每题 problem_id、来源身份、原始行号、保留/剔除与原因），以及 `train_selected.parquet`。所有剔除题都保留可回查 ID，尤其 not_sensitive_at_cap；原始数据与320题候选清单不删除。可用 problem_id 关联候选清单恢复题目，或按输入文件哈希与 source_row 找回原始记录。

环境、网络、设备故障或OOM立即停止，不生成奖励或筛选决定。validation 不进入训练筛选；完整80题保留，具体 validation probing 采样预算另行确定。两阶段仅使用 train_selected，RL重新生成当前策略轨迹。

只对240个train候选按此规则 probing，首轮为3120条，全部追加到上限为6960条；实际通常介于两者之间。该计算不包含validation、Kaggle smoke及正式训练。GPU wall time须由实测吞吐估算。

## 待确认参数与验收

首轮候选子池 320 题，两来源各 160；每来源先抽样再切分 120 train / 40 validation，合计 240:80。此前 90/10 的完整池切分建议，以及 256:64 子池分配建议均被本方案替代。1:1 是 probing 前的来源配比，不要求筛选后的训练题集仍为 1:1，也不以题源名称代替经验难度。

待确认：模型rollout随机数安排、生成长度/explicit采样设置、invalid判定接口、validation probing预算、probing/sweep轨迹复用规则。train筛选规则、scale与递增M已确定，最终train规模由结果决定。必须先完成 Kaggle baseline smoke、测量 rollout 吞吐并核算预算，才执行真实 probing。

实施验收覆盖：同文件/seed 切分可复现；train/validation/test 的记录 ID 隔离；两来源各抽160且各划120/40，probing前manifest已锁定，筛选不能移动split；原始 prompt 和答案逐项保留；重复题保留；Math-500 内部 train 标记不能进入训练；缺失输入、读文件异常及无效测试记录停止；Stage A/B 的 train selection 完全一致。使用合成小文件测试接口，再对真实文件仅运行 CPU 数据构建，产物不提交 Git。

probing 控制器与筛选实现另新增 `adaptive_noise/probing.py`、`scripts/select_probe_dataset.py` 和 `tests/test_probing.py`。验收覆盖首轮13条、递增只加非零scale、s=0绝不追加、delta=0.4边界、全错/全对优先剔除、7次仍不敏感的ID回查、恢复不重复采样、未完成记录禁止筛选和错误不变成R=0。只用合成reward记录验证控制器，真实模型probing仍在Kaggle验收后进行。
