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

返回 [文档索引](README.md)。已确定题源、不再去重边界与首轮子池方案：先分别抽取，再固定 240 train / 80 validation，之后 probing。seed、scale、重复次数及筛选规则尚待确认。本文设计脚本接口，尚未实现代码或运行模型。

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

拟定命令为 `python scripts/build_dataset.py --config configs/dataset_build.yaml`。配置显式指定四个文件、输出目录、抽样与切分 seed、每来源候选题数和 train/validation 配额。首轮每来源 160 题，分为 120 train / 40 validation；尚未确认的 seed 不设置可执行默认值。缺失文件或必填配置立即失败，不自动下载或寻找替代文件。

实现分为三个顺序步骤：

1. 流式读取两个训练候选文件，验证 question 非空、prompt 为有效 role/content 消息、ground_truth 为非空字符串；保留消息边界、原始答案及 LaTeX，不改写 prompt、不凭长度或猜测难度删除题目。无效训练记录写入 rejected 清单并注明原因；结构不兼容或文件读取错误立即停止。测试记录不自动删除，遇到无效字段停止并报告。
2. 按来源生成稳定行 ID。先从 GSM8K-Aug 和 DAPO 的有效记录中分别无放回抽取 160 题，再分别以独立切分随机流划分 120 train / 40 validation；总计 240 train 候选、80 validation 候选。同文件、配置和 seed 得到相同抽样及切分结果。必须在模型 probing 之前写出并锁定 split manifest。指定测试文件直接复制为独立 test 分区，不从候选子池再划 test，不用内部 split 字段路由。
3. 写出题目池、split manifest 与统计报告；先完成文件写入，再发布完成标记。已有输出目录拒绝覆盖，要求新的输出路径。只报告实际读取和筛选数量，不把文件元数据数量当作最终可用数量。

基础字段检查并不证明所有答案可自动评分。保留原始 ground_truth 和 reward_model.style，后续使用统一的最终答案 verifier；不因 Math-500 包含符号答案而提前删题。verifier 未实现或不支持必需格式时，按缺失前置条件停止采集。

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

后续 `collect_fixed_sweep.py` 接受固定 train/validation selection，在批准的 scales 和 M 下记录最终正确性，计算每题每个 scale 的经验成功率。保存每题每个 scale 的成功数、完成 rollout 数及经验正确率 q_hat_i(s)，用于基本难度筛选和 warm-start 初始化；不要求细致难度分桶或同 prefix 搜索。probing 与 Stage A sweep 是否共用轨迹及采集配置尚未确定，不能自动重复采集，也不能未校验契约就宣称可复用。不回头修改 split，不按少量全失败结果永久剔除题目。筛选阈值、预算与全失败标签处理实施前另行确定。

两阶段优化只读取 train。Stage B 重新 rollout，不能复用 Stage A 轨迹作当前策略样本。测试文件不提供训练 scale 标签，不参与候选 scale、数据比例或 checkpoint 选择。

## 待确认参数与验收

首轮候选子池 320 题，两来源各 160；每来源先抽样再切分 120 train / 40 validation，合计 240:80。此前 90/10 的完整池切分建议，以及 256:64 子池分配建议均被本方案替代。1:1 是 probing 前的来源配比，不要求筛选后的训练题集仍为 1:1，也不以题源名称代替经验难度。

待确认：抽样/切分 seed、probing scale 集合、每题每 scale 的重复次数 M、基本筛选规则与最终 train 规模、validation 是否保持完整、probing/sweep 轨迹复用规则。必须先完成 Kaggle baseline smoke、测量 rollout 吞吐并核算预算，才执行真实 probing。

实施验收覆盖：同文件/seed 切分可复现；train/validation/test 的记录 ID 隔离；两来源各抽160且各划120/40，probing前manifest已锁定，筛选不能移动split；原始 prompt 和答案逐项保留；重复题保留；Math-500 内部 train 标记不能进入训练；缺失输入、读文件异常及无效测试记录停止；Stage A/B 的 train selection 完全一致。使用合成小文件测试接口，再对真实文件仅运行 CPU 数据构建，产物不提交 Git。
