---
id: DOC-TTA-ARCHIVE-001
type: design
status: archived
title: 暂存的无监督轨迹探索与逐题 Test-Time 适应思路
created: 2026-10-14
updated: 2026-10-14
related_docs:
  - DOC-TRAIN-001
---
# 暂存的无监督轨迹探索与逐题 Test-Time 适应思路

返回 [文档索引](README.md)。当前实验以 [两阶段训练路线](training_route.md) 为准。

2026-10-14 用户决定暂停此分支：保留设计供以后研究，当前实验不采用。本文不授权实现、训练或改变测试协议；恢复研究时需要重新讨论。

## 原始动机与讨论范围

在少量离线监督 warm-start 和 head-only RL 后，针对每道新题生成多条轨迹，通过无监督的轨迹结构统计奖励探索，只更新 Noise Head，backbone 冻结。用户希望研究逐 latent step 的状态信息与适宜 noise scale 的关系；“信息密度越高应越多探索”是研究假设，尚未定义可测量的信息密度，也未证明这种单调关系。

用户在该分支中采纳：每道新题均从同一离线 head 初始化，题目结束后丢弃适应参数；与相同采样预算的固定 head 比较。此决定随整个分支暂停，不适用于当前离线训练后冻结 head 的正式评估。

## 用户提出的方向

- 可以研究只使用原始 hidden state 的小型 head，但输入结构尚未最终确认。
- 轨迹表示需要保留顺序和关键细节；不采用整条轨迹简单平均为单个向量的方案。
- 希望最大化不同轨迹映射结果之间的差异，正确性与多样性分开评估；距离不必满足严格数学度量公理。
- 希望利用逐步 Gumbel 扰动的生成机制设计映射与差异量；提到 K-Means、SVM、AIC/BIC 作为可能的工具，未选定算法。

## 讨论过但未批准实施的候选

候选映射为完整有序序列 `Phi(trajectory)=[(clean_hidden_t, noisy_topk_ids_t, mixture_weights_t)]`。每步降维可另行研究，但不通过整轨迹池化丢掉时序。比较 mixture 时需要按 token ID 对齐，不能只比较排序后的概率。scale、seed 可作为生成来源记录，不自动进入差异奖励。

候选无监督目标为同题多条轨迹的平均两两差异的期望：最大化 `E[2/(M*(M-1))*sum_{i<j} D(Phi(tau_i),Phi(tau_j))]`。首轮建议固定映射和距离的计算规则，以免训练映射仅放大数值就提高目标。该目标、距离、长度处理及其策略梯度估计均未实现或验证。

讨论过 hidden 状态演化差异、latent mixture 分布差异、保留顺序的序列比较，以及条件生成过程之间的差异。用户尚未选择主对象、对齐规则或具体算法。聚类或无监督学习不是计算差异的必需前置步骤。

## 已澄清的边界

实际 sampler 从标准 Gumbel 采样，再裁剪到 `[-1.5,3]`，乘 scale，加到当前 logits，经过候选限制、noisy top-K 和 softmax 产生 latent mixture。不同步骤的模型状态依赖历史；完整轨迹不是独立普通 Gumbel 样本的集合。直接奖励噪声幅度差异可能鼓励大 scale，而不代表推理路径更有用。实现依据为 [CPU sampling kernel](../adaptive_noise/sampling_kernel.py)。

AIC/BIC 是基于似然与复杂度的模型选择准则，通常越小越好，不直接等于两条轨迹的距离；可研究它们是否适合选择共同或分组的轨迹动态模型，但需要先定义模型、似然和时序依赖处理。普通 SVM 是监督方法，One-Class SVM 可以用于异常检测，异常不自动等于有用的新路径。

能区分轨迹不保证解题收益。轨迹完成后的统计可指导下一批生成，不能作为尚未生成步骤的已知信息；head-only RL 仍需可用的动作 log probability 和梯度契约。若未来采用测试时更新，需要单独声明适应协议，测试真值不得进入奖励，并明确总生成预算、最终答案选取和独立评估。

## 当前排除项与未决问题

当前实验不使用轨迹距离/聚类奖励、测试时参数更新或逐题适应循环，也不将归档分支的预算和比较方法混入正式实验。距离、无监督算法、代理奖励、Test-Time 更新参数、停止条件和收益验证方法全部保留为未决研究问题。

当前回到监督 warm-start 加真实最终答案正确性奖励的 head-only policy gradient；validation/test 不更新参数。此前单独确认的输出范围 `[0,2]` 和 Stage A Adam 学习率 `3e-4` 继续有效。

参考：[TTRL](https://arxiv.org/abs/2504.16084)、[AIC/BIC 定义](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/AIC.html)。这些工作是讨论背景，不是当前项目方法已验证的证据。
