---
id: DOC-TRAIN-001
type: design
status: active
title: 两阶段 Noise Head 训练与共享题集
created: 2026-10-02
updated: 2026-10-02
related_docs:
  - DOC-SPEC-001
  - DOC-PLAN-001
---
# 两阶段 Noise Head 训练与共享题集

返回 [文档索引](README.md)。2026-10-02 用户已确认本路线；这是训练与数据契约，尚未实现训练代码或执行实验。
本文件覆盖原 Spec 中 Stage B 可选及只做 Stage A 的首版范围；后续实验设置仍需与用户交流。

## 1. 已确认路线

Stage A：监督 warm start，类似SFT但监督目标是noise scale。冻结checkpoint生成固定scale sweep，按题的最终答案正确率产生目标scale，监督训练Noise Head。
Stage B：首版必须包含head-only policy gradient。初始化自Stage A；在每个latent step随机选择scale，最终答案正确奖励1，否则0，只更新Noise Head及其输入投影。
backbone两阶段始终冻结；不进行backbone反向传播，不训练LoRA，不启动完整Latent-GRPO/GRPO/PPO。
各步骤可以使用不同scale；“先大后小”只是待观察的假设，不作为强制目标或标签。
题目级scale标签用于初始化，不声称它监督了逐step最优动作。

## 2. 共享题目、独立轨迹

两阶段使用同一份train题目及答案，复用相同problem_id和split manifest。
Stage A使用固定scale生成的离线轨迹；Stage B使用当前head策略重新采样的on-policy轨迹。
旧固定scale轨迹和旧head策略轨迹不能直接当作当前head的on-policy样本；策略版本改变后，按批准的采样/更新契约处理，不静默复用历史缓存。
RL每步记录输入特征、实际动作/scale及行为策略身份和动作概率信息；每条轨迹记录最终答案、正确性、seed、rollout_id和终止状态。
采样时backbone和hidden inputs不保留反向图；head的log probability必须具有针对head参数的可用梯度，不能把no_grad下导出的浮点log_prob直接用于反向传播。采用保留head图或按行为版本重算的方式，具体接口实施前确定。

## 3. 基本筛选与切分

题池候选为GSM8K-Aug、GSM8K及一个待定较难数据集；当前未选择第三个数据集或确定混合比例。
基本筛选：题目/答案有效、最终答案可自动校验、去重，并将同一原题及增强变体按parent_problem_id分组切分。
固定sweep的成功率可作粗略模型相对难度参考，不要求复杂难度分层、同prefix分支搜索或逐step oracle标签。
少量rollout全部失败不作为永久剔除规则；筛选阈值和抽样预算尚未确认。
先锁定train/validation/test，再生成训练轨迹；train供两个训练阶段更新参数，validation只用于选择固定baseline scale、head checkpoint和超参数，test仅作最终评估。
validation与test都不进入监督/策略梯度更新；官方保留测试题及其增强变体不能混入训练池。

## 4. 奖励与终止

R=1仅代表最终答案经统一规则判定正确；无法判定正确记0，并单独记录invalid/截断等状态，不能用格式奖励或中间过程奖励替代正确性。
环境、网络、设备故障及OOM不是错误答案样本；按用户规则立即停止实验和报告，不转成R=0，也不自动更改预算。
资源不足时Stage B保持“未完成”；不得静默跳过RL，把Stage A-only结果宣称为批准的首版方法。

## 5. 尚需讨论的实现与实验设置

| 待确定内容 | 当前边界 |
| --- | --- |
| 随机动作分布 | 原Spec的Beta是候选，不是已选定方案；当前确定性sigmoid不能直接套用score-function policy gradient |
| warm-start到RL的衔接 | RL动作参数化及其初始化、确定性评估输出规则需要明确 |
| RL更新与稳定性 | 更新周期、reward baseline、学习率、正则项和预算尚未确定 |
| 数据与模型 | 第三个题源、混合比例、rollout数、checkpoint/data路径尚未提供或确认 |

本轮只记录批准的路线；不因确认路线就自动选择上述参数、下载安装依赖或启动实验。
