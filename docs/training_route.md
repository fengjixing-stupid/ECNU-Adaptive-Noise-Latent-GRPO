---
id: DOC-TRAIN-001
type: design
status: active
title: 两阶段 Noise Head 训练与共享题集
created: 2026-10-02
updated: 2026-10-15
related_docs:
  - DOC-SPEC-001
  - DOC-PLAN-001
---
# 两阶段 Noise Head 训练与共享题集

返回 [文档索引](README.md)。2026-10-02 用户已确认本路线；这是训练与数据契约，尚未实现训练代码或执行实验。
本文件覆盖原 Spec 中 Stage B 可选及只做 Stage A 的首版范围；后续实验设置仍需与用户交流。

2026-10-14：正式 probing 已完成，本地选定数据已验收；Noise Head 与 Stage A/B 训练尚未实现。用户暂停后续讨论的无监督/Test-Time 分支，相关想法保存在 [归档设计](test_time_exploration_archived.md)，当前实验不使用。

## 1. 已确认路线

Stage A：监督 warm start，类似SFT但监督目标是noise scale。冻结checkpoint生成固定scale sweep，按题的最终答案正确率产生目标scale，监督训练Noise Head。
Stage B：首版必须包含head-only policy gradient。初始化自Stage A；在每个latent step随机选择scale，最终答案正确奖励1，否则0，只更新Noise Head及其输入投影。
backbone两阶段始终冻结；不进行backbone反向传播，不训练LoRA，不启动完整Latent-GRPO/GRPO/PPO。
各步骤可以使用不同scale；“先大后小”只是待观察的假设，不作为强制目标或标签。
题目级scale标签用于初始化，不声称它监督了逐step最优动作。
输出 scale 范围已确认采用 `[0,2]`；Stage A 使用 Adam，学习率已确认 `3e-4`，覆盖旧计划的 `.001` 建议。该学习率不自动用于 Stage B。
2026-10-14 用户确认head输入为当前clean hidden加6个统计特征，不采用hidden-only。输入来源已定；原方案中hidden投影64维与后续MLP层宽尚未单独最终确认，不把“直接输入”推定为已批准某种内部投影或拼接位置。
用户讨论异质特征的尺度、冗余和容量后，决定当前继续采用拼接作为融合方式。拼接不要求两组特征原本属于同一语义空间；是否有效仍需验证。归一化、内部投影和层宽尚未最终确定，不自动增加双分支编码器或复杂融合模块。
用户确认Stage A其余提议：SmoothL1、batch size 64、20 epochs、按题等权、按最小validation loss保存best checkpoint。此处每题等权要求先平均同题的训练损失，再在题之间平均，避免长轨迹或多轨迹题主导。best选择使用离线validation标签，不在validation上更新参数。
正式评估锁定训练完成的 head；validation/test 不进行参数更新，不使用轨迹距离奖励、聚类奖励或逐题 Test-Time 适应。

### 2026-10-15 已确认的四项Stage B设置

| 项目 | 已确认内容 |
| --- | --- |
| 双卡布局 | Kaggle T4×2，每张卡一个完整冻结1B模型，独立Engine、各自tp=1，并行采样；两个worker使用同一head版本，汇总后统一更新并同步 |
| warm-start衔接 | 用warm-start对当前状态的预测初始化RL策略在该状态的scale均值；保留逐状态差异，不取全局标签平均值作为初始化中心 |
| 采样量 | 每题每轮8条新轨迹，双卡各4条；同一批采样期间使用相同策略版本，批次间按下述更新安排同步 |
| RL优化器与学习率 | Adam（随后单独确认），学习率`3e-5`；Stage A使用Adam、学习率`3e-4` |

用户随后另外确认Beta动作分布及固定浓度`kappa=20`。令归一化均值为`mu_t`，参数为`alpha_t=20*mu_t`、`beta_t=20*(1-mu_t)`；浓度不参与训练，RL更新head所表示的逐状态均值。初始化均值来自warm-start；不能把warm-start head永久冻结为RL均值。`mu_t`必须位于`(0,1)`，端点附近的具体数值处理仍需确定。

用户随后确认Stage B更新安排：每轮将13道train题打乱，分成4＋4＋5三批；每题采样8条新轨迹，因此三批分别为32、32、40条，每轮共104条。每批完成采样后仅执行一次head优化器更新，再将新head版本同步给两个worker后开始下一批；同一批内不更新参数，更新后不复用该批轨迹继续优化。每轮覆盖全部13题一次，共3次更新，不丢弃最后5题。此处批次是RL题目批次，与Stage A的batch size 64不同；训练总轮数尚未确认。

基础强度sigma_0=1、端点处理、10轮/30次更新/1040轨迹预算、梯度裁剪1.0、leave-one-out baseline均仍为待讨论建议。输入融合与Stage A此前批准项继续有效。

双卡布局是后续训练采集目标，尚未实现或GPU验收；已有K0、请求seed及probing证据为单Engine/GPU0。Stage A离线训练无需因此启用双卡，已完成probing不重跑。单次Kaggle实验最多8小时的约束继续有效。

## 2. 共享题目、独立轨迹

两阶段使用同一份train题目及答案，复用相同problem_id和split manifest。
Stage A复用probing中每题最佳经验正确率scale的全部离线轨迹与输入特征（包含正确/错误轨迹，平手取较小scale）；其他scale保留探测摘要。Stage B使用当前head策略重新采样的on-policy轨迹。
旧固定scale轨迹和旧head策略轨迹不能直接当作当前head的on-policy样本；策略版本改变后，按批准的采样/更新契约处理，不静默复用历史缓存。
RL每步记录输入特征、实际动作/scale及行为策略身份和动作概率信息；每条轨迹记录最终答案、正确性、seed、rollout_id和终止状态。
采样时backbone和hidden inputs不保留反向图；head的log probability必须具有针对head参数的可用梯度，不能把no_grad下导出的浮点log_prob直接用于反向传播。采用保留head图或按行为版本重算的方式，具体接口实施前确定。

## 3. 基本筛选与切分

2026-10-02 用户确定训练来源为 GSM8K-Aug 和 DAPO-Math-17k；GSM8K-Aug-test 与 Math-500-test 是独立测试集，不增加其他题源。
用户决定信任作者已有数据处理，本项目不再去重、不要求增强题与原题映射；此前 parent_problem_id 分组切分要求由本决定替代。基本检查只验证题目、prompt 与最终答案字段有效，保留原始题目和答案。首轮先从每来源抽取160题，再各分120 train / 40 validation，合计240:80；固定split后才进行probing。候选题经冻结1B checkpoint的经验正确率基本筛选后，锁定两阶段共享的最终train题集。
不重复审计语义重复及原题关系，因此只保证文件角色和记录 ID 的隔离，不宣称已验证语义去重或 checkpoint 历史训练污染。构建脚本设计见 [数据集构建设计](dataset_construction_design.md)。
固定sweep的成功率可作粗略模型相对难度参考，不要求复杂难度分层、同prefix分支搜索或逐step oracle标签。
最新用户决定：首轮S=[0,.25,.5,1,2]；s=0仅1次且不追加，非零scale累计3→5→7。train全错/全对剔除，有对有错且经验正确率极差>0.4保留；到7次仍≤0.4剔除并记录problem_id。剔除仅针对本轮训练选择，保留原始题目和候选清单。详见数据构建设计；替代此前全失败题暂不剔除的建议。
validation保留全部80题，使用相同递增采样控制，标记all_correct/all_wrong/noise_sensitive/not_sensitive_at_cap，不按类别剔除。
先锁定train/validation/test，再生成训练轨迹；train供两个训练阶段更新参数，validation只用于选择固定baseline scale、head checkpoint和超参数，test仅作最终评估。
validation与test都不进入监督/策略梯度更新；两个指定测试文件不得参与训练池构建或参数选择；不据内部 extra_info.split 改变用户指定的文件角色。

## 4. 奖励与终止

R=1仅代表最终答案经统一规则判定正确；无法判定正确记0，并单独记录invalid/截断等状态，不能用格式奖励或中间过程奖励替代正确性。
环境、网络、设备故障及OOM不是错误答案样本；按用户规则立即停止实验和报告，不转成R=0，也不自动更改预算。
资源不足时Stage B保持“未完成”；不得静默跳过RL，把Stage A-only结果宣称为批准的首版方法。

## 5. 尚需讨论的实现与实验设置

| 待确定内容 | 当前边界 |
| --- | --- |
| head 内部结构 | 输入已确认hidden＋6统计量；原候选为2048维hidden投影64维后拼接，MLP 70→128→32→1。内部投影和层宽仍待最终确定 |
| 动作参数化与数值处理 | Beta及固定浓度20已确认；归一化均值的内部参数化和端点处理仍需确定，保证alpha/beta严格为正 |
| warm-start到RL的衔接 | 已确认按状态初始化scale均值；分布参数化、端点处理、基础强度与倍率的映射、确定性评估输出规则待定 |
| RL更新与稳定性 | 每题每轮8条、Adam学习率3e-5、每轮打乱后按4＋4＋5分批且每批仅更新一次已确认；reward baseline、正则项和总预算待定 |
| 正式评估 | validation选择固定scale与head，锁定后评估两种方法；每题rollout数、重复seed数、最终汇总指标和配对预算仍需确定 |
| 数据与模型 | 题源与路径已提供；子池320题、两来源各160、train/validation=240:80；probing scales/M和筛选规则已确定；数据master_seed=42，三路派生负责两个来源抽样与切分；生成设置沿用作者：GSM8K-Aug max_new_tokens=128，DAPO=4096；validation分类保留与最佳scale轨迹复用已确认；模型rollout根seed=12345逐轨迹派生、作者答案判定与invalid/截断独立标记已确认 |

本轮只记录批准的路线；不因确认路线就自动选择上述参数、下载安装依赖或启动实验。

### Gumbel噪声与head动作分布的区别

作者原始Gumbel噪声机制继续沿用，无需重新选择噪声类型。当前代码为标准Gumbel采样、裁剪到`[-1.5,3]`、乘scale，再加到logits上。这里随机的是噪声，scale可以由head确定性输出；轨迹仍然随机。

原Spec在Stage B另提议一个scale策略，先采样scale，再使用作者Gumbel噪声。2026-10-15用户已选择Beta且固定浓度20；`a~Beta(alpha,beta)`，按已确认的scale范围映射（归一化动作对应`scale=2*a`），与基础强度倍率记号的最终衔接仍待明确。该策略下score-function更新使用动作的`log pi_theta(a|features)`（或正确变换后的scale密度），与作者sampler返回的Gumbel噪声相关数值不是同一个概率。

这里的“原Spec”指本项目设计文档，不是Latent-GRPO论文。论文第2.2节式(2)–(5)及第3.4节式(22)–(23)使用latent perturbation/显式token的策略目标更新模型；没有提出Beta scale策略。本项目额外将每步scale作为head动作，使最终正确性奖励可通过`(R-b)*sum_t grad log pi_head(a_t|x_t)`更新head，而不需要对奖励或冻结的SGLang生成过程反向传播。每步可共用终局奖励，这只是有噪声的轨迹级归因，不证明某个scale的单次选择就是正确原因。Beta提供有界连续动作参数化，现已获用户采用。论文依据：[Latent-GRPO](https://arxiv.org/html/2604.27998v1)。

随机scale不是所有RL方法的必然要求，但当前已选择Beta动作策略。若未来改回确定性scale，需要另行建立对head有效的梯度估计；不能因生成轨迹有随机性就将标准Gumbel的log probability直接用于更新head。作者实际路径还有裁剪、top-K选择及scale=0，直接使用普通连续Gumbel密度不能视为已验证的正确head梯度。

## 6. 当前数据与实验准备状态（2026-10-14）

本地数据目录为 `artifacts/probing_v1/selected/`。manifest状态complete：4608次probing，13道训练题（11 GSM8K-Aug、2 DAPO），80道validation题。训练标签为10题0、1题0.5、2题2；13题对应21条轨迹、526个latent steps。validation目标为78题0、2题2。数据少和标签偏斜是已知事实，不据此自动扩池、重采样或改变筛选规则。

本次重新核验6个Parquet文件哈希与manifest一致，362条保留轨迹引用的724个JSON文件均存在，训练轨迹/步数与标签统计一致。此前完整本地验收已逐对验证schema、维度、有限值、对齐与选择决定；本次未重复全部深度检查，不声称JSON另有上游哈希证明。

`retained_trajectories.parquet`包含全部320题各自最佳scale的轨迹，不能直接全部作为训练输入；必须按`train_selected.parquet`中的13个problem_id过滤。validation以`validation_marked.parquet`中的80题独立加载。

| 部分 | 当前状态 | 开展实验前的工作 |
| --- | --- | --- |
| backbone与推理环境 | 真实GRPO 1B、T4单GPU的K0及请求seed检查已通过，正式probing完成；后续双卡独立模型布局已确认 | 增加双worker并行采样、版本同步和seed隔离；训练模块集成后验证adaptive GPU smoke |
| 离线输入 | selected本地完整；原始题库和指定测试集已有 | 提供selected在Kaggle上的实际挂载/恢复路径；无需重新probing或额外下载模型 |
| head及Stage A | 输入与Stage A超参数已确认；训练模块和CLI尚未实现 | 确定内部结构及Stage B衔接，完成数据加载、训练、保存恢复和CPU验收 |
| Stage B | 正确性奖励、冻结范围、双卡、均值初始化、采样量、Adam学习率3e-5、Beta/固定浓度20及4＋4＋5单次更新安排已定，随机策略/训练代码未实现 | 确认端点与scale映射、baseline及总预算；实现当前策略rollout和head梯度更新 |
| 最终对比 | 两种方法与validation/test隔离已定 | 实现validation选择、固定预算配对评估、结果导出，确认seed和指标 |

所有单次Kaggle实验不超过8小时；Stage A已确认20 epochs，RL和评估的具体预算尚未确定。当前Notebook仍是probing入口，不是训练Notebook。下一工作单元应先明确head动作及Stage A/B衔接，再锁定内部结构和最小训练实施计划；当前不运行实验。
