---
id: DOC-EXPERIMENT-SUMMARY-001
type: design
status: active
title: AdaNoise-LGRPO：实验设计、当前进度与统计方法
created: 2026-10-17
updated: 2026-10-17
related_docs:
  - DOC-TRAIN-001
  - DOC-DATA-001
  - DOC-PLAN-001
---

# AdaNoise-LGRPO：实验设计、当前进度与统计方法

**当前阶段：数据准备与 probing 已完成，Noise Head 的训练实现尚未开始。** 当前训练材料包含13道题、21条保留轨迹、526个 latent steps；validation 保留80道题。首版采用监督 warm-start 和 head-only policy gradient 两阶段训练。

本文汇总截至2026-10-17的讨论与本地数据核验结果。参数状态分为“已确认”“待确认”和“统计改进建议”；后两者不自动进入实验。正式训练契约见[训练路线](training_route.md)，返回[文档索引](README.md)。

## 1. 研究目标与实验问题

### 1.1 要学习什么

使用作者已经训练完成的1B Latent-GRPO checkpoint，冻结语言模型全部参数，只训练一个逐 latent step 决定噪声强度的 Noise Head，以及为它服务的可训练输入投影。

希望回答的问题是：

> 在相同模型、题目和评估采样预算下，根据当前推理状态动态选择 Gumbel 噪声 scale，是否比 validation 选出的固定 scale 获得更高的最终答案正确率？

控制对象是每一步的噪声幅度。它可以随题目、推理状态和阶段变化。“前期大、后期小”是可以观察的现象，不是训练标签或必须满足的规律。

当前方案不更新 backbone，不训练 LoRA，不执行完整模型的 GRPO/PPO。这里的“Latent-GRPO”指基础 checkpoint 与 latent 生成机制；当前优化算法是对 Noise Head 的策略梯度。

### 1.2 主实验比较

| 方法 | 噪声控制方式 | 参数选择 |
| --- | --- | --- |
| Fixed-Gumbel | 整条轨迹使用同一个固定 scale | 由 validation 选择，进入 test 前锁定 |
| AdaNoise | 每个 latent step 根据状态产生 scale 策略 | train 上 warm-start＋RL；由 validation 选择最终 head |

两种方法使用相同 checkpoint、prompt、答案判定、生成长度上限、temperature、top-p、top-K 和评估采样预算。正式评估时 head 参数固定。

Fixed-Gumbel 是对比方法；后文 leave-one-out baseline 是策略梯度里的奖励参照值。两者用途不同。

### 1.3 整体流程

~~~mermaid
flowchart TD
    A[GSM8K-Aug与DAPO训练源] --> B[各抽160题]
    B --> C[先固定240 train与80 validation]
    C --> D[冻结1B进行固定scale probing]
    D --> E[筛出13道train及最佳scale轨迹]
    D --> F[保留全部80道validation并标记]
    E --> G[Stage A 监督warm-start]
    G --> H[Stage B 同题重新采样并训练head]
    F --> I[选择固定scale和head checkpoint]
    H --> I
    J[独立GSM8K-Aug-test与Math-500-test] --> K[锁定配置后的配对评估]
    I --> K
~~~

图中数据构建、probing 和离线选择已完成；Stage A、Stage B 和最终对比仍待实现与执行。

## 2. 噪声机制与 Noise Head

### 2.1 当前沿用的 Gumbel 扰动

对当前步的词表 logits \(z_{t,v}\)，作者采样路径使用 Gumbel 随机量，并进行裁剪和缩放。用简化记号表示为：

\[
g_{t,v}\sim\mathrm{Gumbel}(0,1),\qquad
\widetilde g_{t,v}=\mathrm{clip}(g_{t,v},-1.5,3),
\]

\[
\widetilde z_{t,v}=z_{t,v}+s_t\widetilde g_{t,v}.
\]

当前沿用双侧设置。随后的 top-p、noisy top-K、mixture 权重及 latent 终止规则遵守作者实现。Noise Head 改变的是 \(s_t\)，并不重新定义这些生成规则。

由于使用裁剪后的噪声，在给定 \(s_t\) 的条件下：

\[
\mathrm{Var}(s_t\widetilde g)=s_t^2\mathrm{Var}(\widetilde g).
\]

因此本文统一称 \(s_t\) 为 **scale／幅度系数**，不直接称其为“方差”。裁剪后的 Gumbel 也不能不加区分地代入未裁剪标准 Gumbel 的密度公式。

作者论文提供 latent 推理和相应优化背景；本项目的 Beta scale 策略是额外设计。[Latent-GRPO 原论文](https://arxiv.org/html/2604.27998v1)。

### 2.2 已确认的控制范围与基础强度记号

实际噪声 scale 的设计范围为 \([0,2]\)。若采用倍率写法：

\[
s_t=\sigma_0 r_t,
\]

则 \(\sigma_0\) 是基础强度，\(r_t\) 是动态倍率。上一轮提出的 \(\sigma_0=1\) 会使倍率与已有 scale 标签数值一致；它在现有正式记录中仍是待确认项，本文沿用该状态。

如果未来改用其他 \(\sigma_0\)，必须同时明确倍率范围和标签换算，保证实际 scale 范围仍与已确认设计一致。例如保持 \(r_t\in[0,2]\) 却把 \(\sigma_0\) 改为0.35，会把实际范围改成 \([0,0.7]\)，这就改变了实验。

### 2.3 输入：clean hidden＋6个统计特征

输入由当前步的2048维 hidden state 与6个标量特征组成，融合方式已确认采用拼接。

“clean”表示在**当前步添加 Gumbel 噪声之前**读取；该状态仍可能受到此前各步噪声选择的影响。

在当前 clean logits 的 top-10 内做 softmax，记条件概率为 \(p_{t,1},\ldots,p_{t,10}\)，按概率降序排列。统计特征在 float32 下计算：

| 顺序 | 特征 | 定义 | 含义 |
| --- | --- | --- | --- |
| 1 | 归一化熵 | \(H_t/\log K\)，\(H_t=-\sum_jp_{t,j}\log(p_{t,j}+\epsilon)\) | top-K 内概率分散程度 |
| 2 | 概率间隔 | \(p_{t,1}-p_{t,2}\) | 前两候选的竞争程度 |
| 3 | 归一化 ESS | \((1/\sum_jp_{t,j}^2-1)/(K-1)\) | 有效候选数量的归一化表示 |
| 4 | 最大概率 | \(\max_jp_{t,j}\) | top-K 内最强候选的相对占比 |
| 5 | 熵变化 | \(H_t-H_{t-1}\)，首步为0 | 当前与上一步的不确定性变化 |
| 6 | 步数比例 | \(t/T_{\max}\)，\(t\) 从1开始 | 已使用的生成预算比例 |

这里 \(K=10\)，熵计算的现有设计使用 \(\epsilon=10^{-8}\)。\(T_{\max}\) 使用 max_new_tokens，而不是生成结束后才知道的实际轨迹长度。因此步数比例不会读取未来信息，但它也不是“思考已完成百分之多少”的直接测量。

输入统计使用 clean top-K，生成时则使用作者的 noisy top-K 路径。二者的 token 集合不必相同。

### 2.4 拼接的数学解释与内部结构状态

拼接不同类型的特征，并不要求它们原本处于同一个语义坐标系。第一层线性变换满足：

\[
W[h;c]+b=W_hh+W_cc+b.
\]

模型可以分别学习两部分输入的权重。需要处理的是尺度、冗余和训练样本数量，而不是“两个向量能否拼接”。

当前结构状态如下：

| 内容 | 状态 |
| --- | --- |
| 使用2048维 clean hidden＋6统计量 | 已确认 |
| 两部分输入采用拼接融合 | 已确认 |
| hidden 先投影为64维 | 候选，待最终确认 |
| 拼接后 MLP 为70→128→32→1，使用 GELU | 候选，待最终确认 |
| hidden／统计量归一化方法 | 待确认 |
| 输出映射和 Beta 均值的端点数值处理 | 待确认 |

如果采用上述带 bias 的线性层结构，其参数量为144,385，其中2048→64投影占131,136个。该数值是对候选结构的计算，不代表已经实现了这个模型。减少六个标量并不能解决大部分容量问题；最大的参数来源是 hidden 投影。

## 3. 数据集构建思路

### 3.1 数据来源与角色

| 本地文件 | 角色 | 实际文件记录数 |
| --- | --- | ---: |
| GSM8k-Aug-oss-dup-all.parquet | GSM8K-Aug训练候选源 | 301,852 |
| DAPO-Math-17k-en-train.parquet | 较难训练候选源 | 14,116 |
| GSM8k-Aug-test.parquet | 独立测试集 | 1,319 |
| Math-500-test.parquet | 独立测试集 | 500 |

这些记录数于2026-10-17从本地 Parquet 元数据重新读取。DAPO 文件名称中的“17k”不代表当前文件恰有17,000条记录。

Math-500 仅用于测试。训练使用 GSM8K-Aug 与 DAPO，不再额外加入题源。按用户决定信任作者已有数据处理，不再次去重，也不要求增强题与原题的对应关系；因此记录身份隔离已建立，语义重复与基础模型历史训练污染不是本次已验证的结论。

### 3.2 先抽样、先切分，再 probing

从两个训练来源各无放回抽取160题，再分别固定120 train／40 validation：

| 来源 | train候选 | validation | 合计 |
| --- | ---: | ---: | ---: |
| GSM8K-Aug | 120 | 40 | 160 |
| DAPO | 120 | 40 | 160 |
| 合计 | 240 | 80 | 320 |

测试文件始终独立，不从这320题中再划出 test。先划分再 probing，使训练题选择不改变 validation 的成员身份。

数据 master seed 为42，三个派生 seed 分别承担两来源抽样与切分；它们不是三次独立实验：

| 用途 | 派生 seed |
| --- | ---: |
| GSM8K-Aug抽样 | 2028899853 |
| DAPO抽样 | 3785434924 |
| train／validation切分 | 262147858 |

模型 rollout 使用另一根 seed 12345，按题目身份、scale 与 rollout 序号派生可追溯的逐轨迹 seed。追加采样和恢复保留既有轨迹身份。

### 3.3 用 checkpoint 的经验成功率描述难度

对题目 \(i\)、固定 scale \(s\)，定义：

\[
\widehat q_i(s)=\frac{1}{M_i(s)}\sum_{m=1}^{M_i(s)}R_{i,s,m}.
\]

\(R=1\) 仅表示最终答案通过既定 verifier；其他正常完成但答案不正确的结果为0。数据集名称只决定题源，不直接给题目贴“简单／困难”标签。

当前首轮 scales 为：

\[
S=\{0,0.25,0.5,1.0,2.0\}.
\]

每条 probing 轨迹从头到尾使用同一个 scale，尚未使用逐步 Noise Head。scale=0只采1次，始终不追加；其余四种 scale 从各3次开始，必要时累计到5次、最多7次。每题的总 probing 数因此为13、21或29。

保守集合 \(\{0,0.25,0.5,0.75,1.0\}\) 只作为后续 probing 的候选调整。如果 scale=2 出现大量 invalid 或正确率下降，由用户决定是否采用；当前数据实际来自首轮集合。

### 3.4 train筛选与validation标记

每完成一个采样阶段，计算：

\[
\widehat\Delta_i=\max_{s\in S}\widehat q_i(s)-\min_{s\in S}\widehat q_i(s).
\]

| 当前观测结果 | train处理 | validation处理 |
| --- | --- | --- |
| 全部scale的结果全错 | 剔除，记 all_wrong | 保留并标记，停止该题追加 |
| 全部scale的结果全对 | 剔除，记 all_correct | 保留并标记，停止该题追加 |
| 有对有错，且极差严格大于0.4 | 保留，记 noise_sensitive | 保留并标记，停止该题追加 |
| 极差≤0.4，非零scale尚未达到7次 | 非零scale追加至5或7次 | 同样追加 |
| 到7次仍有对有错且极差≤0.4 | 剔除、保留problem_id | 保留，记 not_sensitive_at_cap |

不根据最终训练题数量自动补抽、放宽阈值或改变数据比例。

“noise_sensitive”是当前筛选标签；其统计证据强度见第9节。全对／全错也只描述有限次观测，不证明真实成功率为1／0。

### 3.5 最佳scale与轨迹保留

每题终态定义：

\[
s_i^*=\arg\max_s\widehat q_i(s),
\]

平手取较小 scale。这里最大化的是**经验最终答案正确率**，不是 token 概率或轨迹似然。

保留 \(s_i^*\) 下全部完整轨迹及输入特征，包括其中答错的轨迹；其余 scale 保留成功数、样本数、答案判定和 seed 等轻量记录，不再永久保留完整 hidden 序列。

Stage A 为该题所有保留 latent steps 赋相同标签 \(s_i^*\)。这个标签表达“哪一个固定 scale 在此次 probing 中最好”，不是逐步最优 scale 的标注。

### 3.6 生成与评分设置

| 配置 | 当前值 |
| --- | --- |
| GSM8K-Aug生成上限 | max_new_tokens=128 |
| DAPO／Math-500生成上限 | max_new_tokens=4096 |
| temperature／top_p | 0.6／0.95 |
| max_topk／Gumbel softmax temperature | 10／1.0 |
| Gumbel设置 | 启用噪声分支，包括scale=0；one-sided=False |

保留来源 prompt 和 chat 消息结构，沿用作者最终答案提取与比较方法。GSM8K-Aug优先提取 boxed，辅以既有数字格式；DAPO／Math-500沿用作者 high-task 的 boxed 与规范化文本判定。

invalid、截断与最终正确性分别记录。环境、设备或 OOM 故障不记成答错样本，应停止实验并报告。

## 4. 已获取的数据与一条真实示例

### 4.1 实际产物规模

本地目录为 **artifacts/probing_v1/selected/**。

| 项目 | 实际数量 |
| --- | ---: |
| 固定候选题 | 320 |
| 完成的probing轨迹 | 4,608 |
| 最终train题 | 13：GSM8K-Aug 11题，DAPO 2题 |
| 保留validation题 | 80 |
| 供Stage A使用的train轨迹／latent steps | 21／526 |
| 全部候选题的最佳scale保留轨迹 | 362 |
| 保留轨迹对应的JSON文件 | 724 |

全部320题中，291题在非零scale各3次时结束，2题在5次时结束，27题在7次时结束：

\[
291\times13+2\times21+27\times29=4608.
\]

train 的筛选结果为：

| 标签 | 题数 | 是否进入训练 |
| --- | ---: | --- |
| all_wrong | 153 | 否 |
| all_correct | 52 | 否 |
| noise_sensitive | 13 | 是 |
| not_sensitive_at_cap | 22 | 否 |

最终保留率为13/240，约5.42%。validation的四类数量依次为49、27、2、2，全部保留。

训练 scale 标签分布：

| 标签 | 题数 |
| --- | ---: |
| 0 | 10 |
| 0.5 | 1 |
| 2 | 2 |

validation 的标签分布为78题0、2题2。上述标签偏斜是实测结果，不代表已经证明大部分状态最适合零噪声。

**训练加载必须按 train_selected 中的13个 problem_id 过滤。** retained_trajectories 中的362条轨迹还包含未选中的train候选和validation，不能整体作为训练集。

2026-10-17重新核验了六个 Parquet 与 manifest 的哈希、724个引用文件的存在性，以及21条训练轨迹的526步计数。此前已有完整schema／对齐验收；本次未重复所有轨迹的深度检查。

### 4.2 示例题目与身份

以下来自真实 train_selected 的第一条记录，没有改写题目或重新生成答案。

| 字段 | 内容 |
| --- | --- |
| source | gsm8k_aug |
| source_row | 32611 |
| split | train |
| ground_truth | 606 |
| noise_target | 0.0 |

完整 problem_id：

~~~text
gsm8k_aug:3766e3a83cd82ddd686392d8bc6ef6f262821490a09b694b2caed44f1a482501:32611
~~~

原题：

> The town's summer fair has 15 booths. Each booth sells 7 lemonade cups per hour. The fair runs for 5 hours. After the first 2 hours, the organizers add 3 extra booths that each sell 9 cups per hour for the remaining time. How many total lemonade cups are sold during the fair?

原始 user 消息在题目前带有指令：

~~~text
Please reason step by step, and put your final answer within \boxed{}.
~~~

其答案可直接核对为 \(15\times7\times5+3\times9\times3=606\)。

### 4.3 这道题如何进入训练集

| 固定scale | 成功／完成次数 | 经验成功率 |
| --- | ---: | ---: |
| 0 | 1/1 | 1.000000 |
| 0.25 | 7/7 | 1.000000 |
| 0.5 | 7/7 | 1.000000 |
| 1 | 6/7 | 0.857143 |
| 2 | 4/7 | 0.571429 |

最终：

\[
\widehat\Delta_i=1-\frac47=\frac37\approx0.428571>0.4.
\]

因此它被保留并标记 noise_sensitive。前三个 scale 的经验成功率并列最高，按既定规则选较小值0，故 warm-start 标签为0。

这张表不能推出“scale=0比0.25和0.5更可靠”：三者的样本量不同，标签0主要由当前的平手处理规则决定。

### 4.4 该题保留轨迹的一条记录

| 字段 | 实际内容 |
| --- | --- |
| scale／rollout_id | 0.0／0 |
| seed | 50549235 |
| reward | 1 |
| latent mixture steps | 29 |
| all_output_ids长度 | 33 |
| feature_schema／feature_source | raw-latent-v1／pre_gumbel |
| 每步hidden维度 | 2048 |

原始记录中的输出文本：

~~~text
84445-.</-5253344 *201</-8138144300201</-606</think>\boxed{606}<|eot_id|>
~~~

latent部分解码后的字符不等于可读的自然语言推理过程；该轨迹的最终 boxed 答案为606，得到reward=1。

第一步的 hidden 前8维为：

~~~text
[-1.3701171875, 1.9736328125, -0.83154296875, -0.08758544921875,
  0.5712890625, 1.2509765625, -0.986328125, 4.30078125]
~~~

第一步保存的 clean top-10 概率，保留8位小数展示：

~~~text
[0.12957054, 0.11257283, 0.10910933, 0.10575239, 0.10090964,
 0.09934518, 0.09628864, 0.09045481, 0.07982609, 0.07617059]
~~~

按第2节定义，由这些保存值计算出的六个特征约为：

| 特征 | 数值 |
| --- | ---: |
| H_norm | 0.99515748 |
| margin | 0.01699771 |
| ESS_norm | 0.97576758 |
| p_max | 0.12957054 |
| delta_entropy | 0 |
| step_ratio | 0.00781250，即1/128 |

这些六维数值是本次说明时计算的派生展示值；原始特征JSON保存的是 hidden 与 topk_probs。实现时使用float32可能有末位舍入差异。记录step_idx从0开始，特征中的 \(t\) 按第1步换算为1。

这条轨迹会提供29条共享同一题目级标签0的训练记录。它们属于同一个问题和同一条轨迹，不是29道独立训练题。

## 5. Stage A：监督 warm-start 的优化设计

### 5.1 训练目标

Stage A 使用已保留的固定scale轨迹，学习一个依赖当前状态的 scale 预测器：

\[
\widehat s_{i,\ell}=f_\theta(h_{i,\ell},c_{i,\ell}).
\]

监督目标为题目级标签 \(s_i^*\)。按题等权可以表达为：

\[
L_A(\theta)=\frac1N\sum_{i=1}^{N}
\frac1{L_i}\sum_{\ell=1}^{L_i}
\mathrm{SmoothL1}(\widehat s_{i,\ell}-s_i^*),
\]

其中 \(L_i\) 为该题保留的所有latent训练记录数量，\(N=13\)。实现需要保证其采样和加权确实达到按题等权，而非直接让所有step等权导致长题占比过大。

SmoothL1在小误差区域为二次惩罚、大误差区域为线性惩罚。损失类型已定，具体平滑参数需要在实施配置中显式写清；本文不另行设定该参数。

### 5.2 已确认设置

| 参数 | 值 |
| --- | --- |
| 训练参数 | Noise Head及输入投影 |
| backbone | 冻结 |
| 优化器／学习率 | Adam／3e-4 |
| loss | SmoothL1 |
| batch size | 64 |
| epochs | 20 |
| 样本加权 | 按题等权 |
| checkpoint选择 | 最小离线validation loss |
| scale范围 | [0,2] |

Stage A的意义是提供较合理的初始 scale 预测，不把少量固定scale轨迹拟合称为已经学会逐步最优探索。

### 5.3 数据量与标签结构的影响

526个step可以提供不同状态下的输入，但它们只来自13道题。模型可能记住题目状态与标签的关系，未必泛化到新题。

此外，validation有78/80个标签为0。一个恒输出0的预测器，其按题平均绝对标签误差只有：

\[
\frac{78\times0+2\times2}{80}=0.05.
\]

这是标签结构导致的参考计算，不是训练结果。它说明“监督validation loss较低”与“生成答案更准确”必须分开评价。Stage A的既定选模规则继续保留；最终生成正确率需要在后续评估中测量。

### 5.4 与Stage B的衔接

用户已确认：使用 warm-start 对当前状态的预测初始化 RL scale 策略的均值，保留逐状态差异。

不使用13个标签的全局平均值作为所有状态的统一中心。该全局平均为：

\[
(10\times0+1\times0.5+2\times2)/13\approx0.346154,
\]

它会丢失状态信息。是否选用该值做额外诊断不影响当前逐状态初始化方案。

## 6. Stage B：head-only policy gradient 的优化设计

### 6.1 Beta动作策略

逐步归一化动作：

\[
a_t\sim\mathrm{Beta}(\alpha_t,\beta_t),\qquad s_t=2a_t,
\]

\[
\alpha_t=20\mu_t,\qquad\beta_t=20(1-\mu_t),\qquad0<\mu_t<1.
\]

浓度 \(\kappa=20\) 固定，不训练。Noise Head表达状态相关的均值 \(\mu_t\)，在Stage B继续更新。映射 \(s_t=2a_t\) 表达实际scale；与基础倍率记号的最终统一仍需确认。

warm-start预测 \(\widehat s_t\) 对应初始 \(\mu_t\) 的自然关系是 \(\mu_t\approx\widehat s_t/2\)。但标签包含0和2，Beta要求两个形状参数严格为正，因此均值的边界映射与数值处理仍是实施前待定项，不能直接令 \(\alpha=0\) 或 \(\beta=0\)。

存在两层随机性：先抽取scale动作，再使用作者的Gumbel扰动生成latent轨迹。固定head参数并不意味着生成轨迹确定。

### 6.2 目标与梯度归属

概念目标为提高训练问题上的最终正确率：

\[
J(\theta)=\mathbb E_{i,\tau\sim\pi_\theta}[R_i(\tau)].
\]

对第 \(m\) 条轨迹，策略梯度具有以下形式：

\[
g_{i,m}=A_{i,m}\sum_{t=1}^{T_{i,m}}
\nabla_\theta\log\pi_\theta(a_{i,m,t}\mid x_{i,m,t}).
\]

\(x_t\) 是当前hidden和统计特征。只对head的动作log probability求导；冻结backbone的生成过程提供状态转移和最终reward。

动作必须按行为策略真实采样，并与用于计算log probability的分布、head版本和边界变换一致。若在线推理只保存了不带计算图的log probability数值，训练时必须用对应的输入和行为版本重算可求导的log probability。

轨迹内的log probability求和与“除以轨迹长度”不是同一个目标。最终loss的题目／轨迹／token归一化方式尚未单独固化；上式用于说明梯度来源，不自动批准一种新的长度加权规则。

### 6.3 Leave-one-out奖励baseline

对同题、同策略版本的8条新轨迹：

\[
b_{i,m}=\frac17\sum_{n\ne m}R_{i,n},\qquad
A_{i,m}=R_{i,m}-b_{i,m}.
\]

两张GPU各采4条，先汇总成8条，再计算baseline。每条轨迹内各latent动作共用该终局优势值。

如果8条全部答对或全部答错，优势均为0；仍计入采样预算，不追加采样或重新筛题。零优势表示该组没有提供当前reward策略梯度信号；是否调用优化器、以及已有Adam动量如何处理，应在实现中明确，不能直接把“零梯度”解释为“参数一定不变”。

本设计对应REINFORCE的多样本leave-one-out估计思路，不需要把作者Gumbel扰动本身的密度当作Noise Head动作概率。[RLOO原始研究中的算法说明](https://arxiv.org/html/2402.14740v2)。

### 6.4 更新频率、预算与双卡布局

| 参数 | 已确认方案 |
| --- | --- |
| 训练题目 | 与Stage A相同的13题 |
| 每题每轮轨迹 | 8条全新on-policy轨迹 |
| 双卡分工 | T4×2，各完整模型、独立Engine、tp=1；每题每卡4条 |
| 每轮分批 | 打乱13题，分成4＋4＋5题 |
| 各批轨迹数 | 32、32、40 |
| 更新频率 | 每批采样后仅更新head一次 |
| 策略同步 | 批内参数固定；更新后同步双worker，再采下一批 |
| 优化器／学习率 | Adam／3e-5 |
| 梯度裁剪 | head及可训练投影的全局L2范数上限1.0 |
| 首轮预算 | 10轮，30次更新，1040条新训练轨迹 |
| 单次运行限制 | 最多8小时 |

1040条不包含validation／test采样。达到时间限制但预算尚未完成时，记录实际进度并标为未完成。

梯度裁剪在反向传播后、Adam更新前执行；所有可训练参数的梯度联合计算范数。它没有改变Stage A的已确认配置。

双卡方案已确认，但还没有双worker训练采集和同步的GPU验收记录。现有真实GPU证据是单Engine／GPU0的K0、请求seed检查和probing。

### 6.5 新旧轨迹与分布变化

Stage A轨迹来自固定scale，Stage B轨迹来自当前逐步策略；两者的状态分布可能不同，因此共享题目不等于共享可直接用于RL的轨迹。

每批更新后，下一批使用新策略采样，不把历史轨迹直接当作当前on-policy数据重复优化。主模型固定只减少了可训练参数；head改变的噪声仍会改变后续状态分布。

## 7. 当前阶段与执行证据

| 工作内容 | 状态 | 证据与边界 |
| --- | --- | --- |
| 本地数据构建、CPU sampler及probing控制 | 已完成验收 | 既有CPU测试记录；本次未重跑整套代码测试 |
| 真实GRPO checkpoint K0 | 用户提供GPU PASS | hidden维度2048，轨迹／mixture对齐，单GPU |
| 持久Engine请求级seed检查 | 用户提供GPU PASS | 同seed复现，不同seed改变轨迹 |
| 正式probing与离线选择 | 已完成 | manifest complete，4608次rollout，320题终态 |
| 本地selected完整性 | 已验收并于本次轻量复核 | 六个Parquet哈希一致，724引用文件齐全 |
| Noise Head结构与接口 | 设计中 | 输入融合已定，内部宽度和归一化未最终确认 |
| Stage A训练 | 未开始 | 未产生训练后的head checkpoint |
| Stage B双卡采集与训练 | 未开始 | 双worker同步、动作概率与优化流程待实现 |
| Fixed-Gumbel与AdaNoise正式评估 | 未开始 | 尚无两方法准确率提升结论 |

当前Notebook仍承担probing工作，不是已经可以运行Noise Head训练的Notebook。

模型使用用户提供的真实GRPO checkpoint：

~~~text
本地：
/Users/fengjixing/Python_Project/models/LLaMA3.2-1B-Instruct-Latent-GRPO-Top10

Kaggle：
/kaggle/input/models/fengjixing/llama3-2-1b-instruct-latent-grpo-top10/other/default/1/LLaMA3.2-1B-Instruct-Latent-GRPO-Top10
~~~

selected数据的Kaggle挂载／恢复位置仍需提供。当前摘要记录本地已存在的数据，不推定它已挂载到某个Kaggle目录。

## 8. 尚需固定的实验设计

以下内容在实现前需要给出确定配置。本文不代替用户作选择。

| 设计项 | 当前状态／需明确内容 |
| --- | --- |
| 基础强度与倍率 | \(\sigma_0=1\)是上一轮建议；保持实际scale范围与标签语义一致 |
| head结构 | 投影维度、MLP层宽、激活与输入归一化 |
| Beta边界 | 均值如何保持在(0,1)，动作边界如何处理，如何保持采样与log density一致 |
| loss细节 | SmoothL1平滑参数、RL题目／轨迹／长度归一化 |
| 优化器细节 | Adam其余参数、Stage A到B时优化器状态处理、零优势批次行为 |
| 正则化 | 是否使用熵、KL或其他正则项及其系数 |
| RL选模 | validation的生成评估方式、检查频率与best checkpoint标准 |
| 正式评估 | 每题采样数、评估seed数量、汇总指标、来源权重及是否重复训练 |
| 执行输入 | selected在Kaggle上的实际位置 |

这里的未决项主要是补齐已确认设计的数学和实现语义。例如，采样后把动作clip到边界，却继续使用原始Beta密度，会使实际采样过程和训练概率不一致，必须在实现前统一。

每项新增设计均须与已确认设计比较。若新增项与旧设计冲突，先明确冲突和影响，由用户选择；已批准某个新建议本身，不自动授权覆盖冲突旧设置。

## 9. 哪些内容可以用统计理论解释

### 9.1 经验正确率：Bernoulli／Binomial模型

在固定题目、固定scale、固定生成配置下，若不同rollout可视为独立同分布：

\[
R_m\sim\mathrm{Bernoulli}(q),\qquad
K=\sum_mR_m\sim\mathrm{Binomial}(M,q).
\]

固定样本数时：

\[
\mathbb E[\widehat q]=q,\qquad
\mathrm{Var}(\widehat q)=\frac{q(1-q)}{M}.
\]

当 \(q=0.5\) 时，\(M=3,5,7\) 的标准误分别约为0.289、0.224、0.189；相对于0.4的筛选阈值，这种抽样波动不小。

现有M由中间结果决定，因此不能无条件把固定样本数的估计性质和覆盖率搬到最终筛选结果上。先用固定M模型理解波动，再处理选择与停止规则。

### 9.2 小样本区间：1/1与7/7不是同等证据

Wilson区间为比例估计提供比简单正态近似更合适的有界区间。[NIST比例置信区间说明](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)。

对第4节数据作**固定样本数下的说明性95% Wilson区间计算**：

| 成功／次数 | 点估计 | Wilson区间，约 |
| --- | ---: | --- |
| 1/1 | 1.000 | [0.207, 1.000] |
| 7/7 | 1.000 | [0.646, 1.000] |
| 6/7 | 0.857 | [0.487, 0.974] |
| 4/7 | 0.571 | [0.250, 0.842] |

表中区间显示证据精度差异。它们不是经过多scale选择和3→5→7停止规则校正后的联合95%保证，也不能通过“区间是否重叠”直接完成差异检验。

scale=0去除了latent的该项扰动，但沿用的显式生成仍有temperature等采样设置，不能仅由scale=0推出所有生成随机性消失。M=1时，观测到0或1也不能据此估计真实成功率的方差为0。

### 9.3 极差筛选：效应量、选择偏差与一个精确反例

\(\widehat\Delta_i>0.4\)描述观测到的成功率差值。它不是显著性检验，也没有预设的误选概率保证。

考虑一个用于分析规则的理想反例：五个scale的**真实成功率全部为0.5**，各次生成独立，首轮仍使用1、3、3、3、3次。

当scale=0这一次答错时，它的估计值为0。只要另四组中任意一组在3次中答对至少2次，极差就至少为2/3，超过0.4。每组达到这一条件的概率为1/2，因此：

\[
P(\widehat\Delta>0.4\mid R_{s=0}=0)
=1-(1/2)^4=15/16.
\]

当scale=0答对时，推导对称：任意其他组正确不超过1次，就触发阈值。因此：

\[
P(\text{首轮标记noise\_sensitive}\mid q(s)\equiv0.5)
=15/16=93.75\%.
\]

本次用13个二元结果的精确计数复核，触发情况权重为7680/8192。**93.75%是这个明确零效应假设下的规则行为，不是当前13题的实测误选率。** 实际题目成功率可能接近0或1，且未知。

这个反例说明：不等采样量、scale=0单次的极端估计和取最大最小值，共同使“观测敏感”容易与抽样噪声混淆。当前数据继续按已确认规则保留；统计分析没有重新裁定13题的去留。

### 9.4 自适应追加：序贯推断问题

当前在M=3、5、7时重复检查同一个筛选条件，并在满足条件时停止。它属于自适应采样／序贯决策。

若未来希望作正式统计判断，需要为重复查看结果设计有效的边界，例如预先限定查看次数并分配错误概率，或采用time-uniform confidence sequences。后者的目标是在多个观察时点同时维持覆盖保证。[Howard等，Confidence Sequences](https://arxiv.org/abs/1810.08240)。

这不意味着现有追加逻辑不能用于探索性筛选，而是其结论应标注为经验筛选结果。

### 9.5 最佳scale标签：argmax选择与“赢家偏差”

同一批数据既用于比较多个scale，又用于选出最高成功率的标签，最大观测值会倾向于受到有利抽样误差影响。继续用这个最大值评价所选scale，会高估其表现。

此外，全错validation题所有scale均为0，平手规则给出标签0；全对题也给出0。这样产生的0标签，有一部分表达的是“没有观测到区分信息”，而非“零噪声具有强统计优势”。

validation必须与最终test分开。频繁依据validation调整结构、筛选阈值和训练轮数，也会逐渐对validation过拟合。[Cawley与Talbot：模型选择与后续评估偏差](https://www.jmlr.org/papers/v11/cawley10a.html)。

### 9.6 Beta策略：均值与探索幅度的统计关系

由Beta分布的均值和方差，在 \(\alpha=\kappa\mu,\beta=\kappa(1-\mu)\) 下：

\[
\mathbb E[a]=\mu,\qquad
\mathrm{Var}(a)=\frac{\mu(1-\mu)}{\kappa+1}.
\]

实际scale为 \(s=2a\)，且 \(\kappa=20\)，则：

\[
\mathbb E[s]=2\mu,\qquad
\mathrm{Var}(s)=\frac{4\mu(1-\mu)}{21}.
\]

\(\mu=0.5\)时，scale标准差约0.2182；均值靠近0或1时，方差减小。因此“固定浓度20”并不意味着所有状态具有相同探索方差。这是分布本身的性质。[NIST Beta分布](https://www.itl.nist.gov/div898/handbook/eda/section3/eda366h.htm)。

### 9.7 Leave-one-out：控制变量与无偏梯度

设同题8条轨迹在给定策略下相互独立，并令：

\[
S_m=\sum_t\nabla_\theta\log\pi_\theta(a_{m,t}\mid x_{m,t}).
\]

score函数满足 \(\mathbb E[S_m]=0\)。其他轨迹形成的baseline不使用第m条自身的随机结果，因此：

\[
\mathbb E[b_mS_m]=0,\qquad
\mathbb E[(R_m-b_m)S_m]=\mathbb E[R_mS_m].
\]

这解释了在相应条件下，减去leave-one-out baseline不会改变原始策略梯度的期望，并可作为方差缩减手段。八个优势值共享奖励，彼此相关，不能把它们当成八个独立优势观测。

无偏性针对正确采样、正确log probability下的原始估计；后续梯度裁剪是非线性变换，不能继续把裁剪后的更新称为严格无偏。其数值收益要通过实验观察。[REINFORCE与RLOO](https://arxiv.org/html/2402.14740v2)。

### 9.8 按题等权：聚类数据与有效样本量

数据具有“题目→轨迹→latent step”的层级结构。共享题目标签、前缀和相邻状态会引入相关性。

因此，526步不等于526个独立题目。按题等权可以控制不同题目在训练目标里的权重；它不能消除题内相关性，也不能凭空增加泛化信息。

评估标准误或bootstrap时，应优先保留题目这一聚类单位，而不是把全部token或step打散后当成独立观测。当前train只有13个题目簇，即使做题目级重采样，其稳定性也有限。

### 9.9 六个特征：信息论与代理指标

熵和ESS刻画概率集中程度；margin和p_max刻画局部候选竞争。它们有清楚的概率论意义，但不足以直接测量“hidden state的信息密度”。

这里的概率在top-K内部重新归一化，没有包含被截掉词表的总概率质量，因此H_norm接近1只说明这10个候选相对均匀。它不能单独推出整个模型不确定，更不能推出增大噪声一定提高答案正确率。

如果统计特征是hidden经固定输出层得到的函数，部分信息可能与hidden冗余；显式加入这些量仍可提供便于小模型使用的表示。其实际价值需要对照分析或消融，而不是仅靠“信息密度”直觉决定方向。

## 10. 可以进一步采用的统计改造

以下均为**建议，尚未批准或执行**。其中一部分只增加结果分析；另一部分会改变现有筛选、标签或训练目标，必须先确认。

### 10.1 优先级与设计影响

| 优先级 | 建议 | 解决的问题 | 与当前设计关系 |
| --- | --- | --- | --- |
| 高 | 对现有成功数／次数给出不确定性描述 | 1/1和7/7被点估计视为相同 | 可做附加分析；若用于改标签则需确认 |
| 高 | 最终比较报告题目级配对差值与区间 | 单个准确率差缺少精度信息 | 补齐待定评估方案，需确认具体口径 |
| 高 | 把极差筛选升级为有误差控制的判定 | 多scale比较、M不等和重复查看 | 替换现有筛选设计，必须确认 |
| 中 | 对标签不确定性做后验建模或加权 | argmax硬标签过于确定 | 改变warm-start标签／权重，必须确认 |
| 中 | 用容量控制和分组评估检查head过拟合 | 13题对应较多参数 | 改变结构或训练设置时必须确认 |

### 10.2 Beta-Binomial后验：描述正确率的不确定性

对题目i、scale s的未知成功率使用一个先验：

\[
q_{i,s}\sim\mathrm{Beta}(a_0,b_0).
\]

观测到k次成功、n次生成后，后验为：

\[
q_{i,s}\mid D\sim\mathrm{Beta}(a_0+k,b_0+n-k).
\]

例如仅作说明取均匀先验 \(a_0=b_0=1\)，1/1对应后验均值2/3，7/7对应8/9。它们不再因点估计均为1而被视作等量证据。

这里的Beta描述**成功率的不确定性**，与第6节用来**采样scale动作**的Beta是不同随机变量；先验参数也不等于动作浓度20。

在似然模型正确、完整观测和停止规则可忽略的条件下，可以用该后验分析已有记录；它不自动提供频率学意义的序贯错误率保证。先验选择、跨scale依赖和采样机制需要明确。

可以进一步计算：

\[
w_{i,s}=P\{q_{i,s}=\max_{u\in S}q_{i,u}\mid D\},
\]

得到“哪个scale最优”的后验权重，再考虑软标签、期望效用或标签置信权重。直接用多个相距较远scale的平均值作新标签，可能得到一个从未验证过的中间scale，因此也需要实验验证。当前首版仍采用已批准的argmax硬标签。

### 10.3 将“敏感”改为带统计证据的判断

一种候选目标是区分：

\[
\max_s q_i(s)-\min_s q_i(s)>\delta_{\mathrm{effect}},
\]

而不是仅要求观测极差超过阈值。

若为各scale构造同时有效的界 \([L_s,U_s]\)，则：

\[
\max_sL_s-\min_sU_s>\delta_{\mathrm{effect}}
\]

可作为一种较保守的充分条件。这里需要同时处理多个scale和多个采样时点；只给每个scale套一个普通95%区间不够。

本项目M最多7、scale=0只有1次，这种证据要求可能使可确认的题目更少。因此不能保证“换成统计方法后不增加预算还能得到更多训练题”。如需增加scale=0次数、扩大每题预算或保留“不确定”类别，都属于现有设计的明确修改。

### 10.4 自适应分配采样预算

现有追加时对四个非零scale统一各加2次。可选改法是把新增采样分配给“最有可能影响最优scale判断”的候选，而不是平均分配。

这可以用best-arm identification、后验不确定性或预期信息增益来构造。对耗时不同的题目，还可考虑每秒获得的辨别信息。

改造会改变已确认的3→5→7同步追加规则，且需要匹配序贯推断方法。它适合后续数据版本研究；当前版本不自动重跑。

### 10.5 题目级配对评估与bootstrap

对测试题i，两方法各采K条并先得到题目级平均正确率，定义：

\[
d_i=\frac1K\sum_{k=1}^{K}
\left(R^{\mathrm{Ada}}_{i,k}-R^{\mathrm{Fixed}}_{i,k}\right),
\qquad
\widehat D=\frac1N\sum_i d_i.
\]

报告 \(\widehat D\) 及其区间，并分别报告两方法本身的正确率。bootstrap可按题目成组重采样，始终把同题两方法结果放在一起；多来源数据可在来源内分层重采样，再按预先约定权重汇总。配对重采样的实现语义可参考[SciPy bootstrap文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)。

如果每题每方法只产生一个二元结果，可以考虑基于“仅Ada答对／仅Fixed答对”计数的McNemar检验；多次rollout不能直接全部当作互相独立的题目对。[statsmodels McNemar说明](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)。

使用相同seed标签有助于复现，但两方法随机数消费路径可能不同。它不自动保证严格的common-random-number耦合或必然降低方差。

### 10.6 分清推理随机性与训练随机性

固定一个训练完成的head、多次测试seed，只测量该head条件下的推理随机性。要估计训练算法自身的稳定性，需要不同训练seed的独立训练运行。

首版只确定了10轮RL预算，尚未确定独立训练重复次数。若增加训练重复，应同时增加总资源预算，并保持每个运行的选模与测试规则一致。

### 10.7 小样本容量控制与输入统计

可考虑比较较小投影、线性／浅层head、权重衰减或对warm-start head的约束，观察能否减少对13道题的记忆。这些都是待讨论的结构或训练目标修改。

归一化统计应只由train拟合，随后固定用于validation/test。如果在全体数据上拟合缩放参数或降维表示，会破坏训练与评估信息隔离。

hidden范数、熵、ESS与最终正确性的关系，可以先做按题聚合的相关性或回归诊断；连续相邻step不能当作大量独立样本。关联结果也不能直接解释为“增加某项特征就会提高探索收益”。

### 10.8 从诊断走向可解释的控制规律

当前没有强制scale随步数单调下降。完成真实训练后，可以描述scale随step_ratio、entropy和margin的条件变化，并区分题源、成功／失败轨迹。

如果未来要增加单调性或平滑惩罚，应先验证观测是否支持这种结构，再把它作为新的建模假设提交讨论。平滑可以降低控制波动，但也可能抹去必要的阶段切换。

## 11. 统计结论与实施顺序

现有设计可由以下几类统计工具解释：

| 已有设计 | 对应理论 |
| --- | --- |
| 最终答案正确／错误、经验成功率 | Bernoulli／Binomial估计与区间 |
| 多scale比较、argmax标签、追加停止 | 模型选择偏差、多重比较与序贯推断 |
| Beta scale策略 | 有界连续分布的均值—方差参数化 |
| head-only policy gradient与LOO | score-function估计、控制变量与Monte Carlo |
| 按题等权、共享题目多轨迹 | 层级／聚类数据与相关观测 |
| 熵、ESS、margin | 信息论与概率集中度 |
| 最终两方法比较 | 配对设计、分层汇总与聚类重采样 |

最值得优先处理的统计问题是**probing点估计的证据强度**和**最终准确率差的评估精度**。两者直接影响“训练标签代表什么”以及“实验提升是否可信”。

当前顺序应保持为：

1. 明确第8节中影响数学语义的未决项；若选统计改造，先确定是否修订现有设计。
2. 实现Noise Head、数据加载和Stage A，完成CPU验收。
3. 完成Kaggle双worker、Beta动作与log probability、参数同步的smoke验收。
4. 执行已批准的Stage A＋10轮Stage B，并记录实际完成预算。
5. 锁定validation选择与评估预算，在独立test上进行Fixed-Gumbel／AdaNoise对比。

本次文档整理没有修改已选训练题、scale标签、采样次数或优化算法，也没有启动模型训练。当前可以确认的是数据与设计状态；方法是否提高最终正确率，仍待正式实验回答。
