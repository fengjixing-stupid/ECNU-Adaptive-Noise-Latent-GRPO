---
id: DOC-SPEC-001
type: requirements
status: active
title: AdaNoise-LGRPO：面向 Kaggle 的 Noise Head 自适应 Gumbel 探索实现规范
created: 2026-09-30
updated: 2026-10-02
---
# AdaNoise-LGRPO：面向 Kaggle 的 Noise Head 自适应 Gumbel 探索实现规范

> 2026-10-02 用户确认修订：Stage B 纳入首版；Stage A/B共享train题目，Stage B使用当前head新rollout，validation/test隔离。当前契约见 [训练路线](training_route.md)。本文未确认的参数仍是建议，不作为启动实验授权。

> **用途**：提供给本地 Codex 作为后续代码修改、Kaggle 实验执行与结果整理的统一实现说明。  
> **实验平台硬约束**：**所有主要实验必须能够在 Kaggle Notebook / Kaggle GPU 环境上运行**。  
> **研究范围硬约束**：只比较两种方法：  
> 1. **Original Fixed-Gumbel Baseline**  
> 2. **Noise-Head Gumbel（本文方法，暂命名 AdaNoise-LGRPO）**  
>
> 本文不重新进行完整 Latent-GRPO RL，不更新大模型 backbone，只训练一个很小的 Noise Head。

---

# 0. 一句话目标

在已经训练好的 Latent-SFT / Latent-GRPO 模型上增加一个轻量 **Noise Head**，根据当前 latent reasoning state 自适应输出 Gumbel noise 的 **scale ratio**，从而将原始固定噪声

\[
\text{noise\_scale}_t = c
\]

改为

\[
\text{noise\_scale}_t = s_t
=
s_{\min} + (s_{\max}-s_{\min})r_t,
\qquad
r_t = \sigma(f_\phi(x_t)).
\]

其中主模型全部冻结，只训练参数量很小的 \(f_\phi\)。

---

# 1. 项目背景与原始方法

Latent reasoning 使用连续 latent token 代替较长的显式 Chain-of-Thought。  
Latent-SFT 将 latent token 约束在 vocabulary embedding space 中，典型形式为

\[
z_t = \sum_{i=1}^{K} p_{t,i}e_{t,i}.
\]

Latent-GRPO 在 latent sampling 阶段加入 Gumbel perturbation，通过随机扰动获得多样的 latent reasoning trajectories。

当前仓库中的核心实现已经包含：

```text
add_noise_gumbel_softmax
use_one_sided_gumbel_noise
noise_scale
gumbel_softmax_temperature
```

当前主要噪声强度控制方式是一个**固定 scalar**：

```python
gumbels = sampling_info.noise_scales[0] * gumbels
```

即所有问题、所有 latent step 使用同一 noise scale。

本文要修改的核心只有这一点：

```text
Fixed noise scale
        ↓
State-dependent noise scale predicted by Noise Head
```

---

# 2. 核心研究问题

原始固定 Gumbel noise 隐含假设：

\[
s_t=c,\quad \forall x,t.
\]

即无论：

- 当前问题难度；
- 当前 latent state；
- 当前模型置信程度；
- 当前 reasoning step；

都使用相同的探索强度。

本文认为该假设可能过强。

研究假设：

\[
s_t^* = f(h_t, u_t)
\]

其中：

- \(h_t\)：当前 latent step 的模型 hidden representation；
- \(u_t\)：当前 latent distribution 的不确定性统计量；
- \(s_t^*\)：更合适的 Gumbel exploration scale。

核心问题为：

> **Can a learned Noise Head outperform a fixed Gumbel exploration scale under the same frozen backbone and the same Kaggle inference budget?**

---

# 3. 最终方法名称

暂定：

```text
AdaNoise-LGRPO
```

完整含义：

```text
Adaptive Noise Controlled Latent-GRPO
```

更具体的方法描述：

> **A frozen latent-reasoning backbone with a lightweight state-dependent Noise Head for adaptive Gumbel exploration.**

---

# 4. 最终模型结构

## 4.1 Backbone

使用作者已经训练好的 Latent-SFT / Latent-GRPO checkpoint。

硬约束：

```python
for p in backbone.parameters():
    p.requires_grad = False
```

不允许为了本文实验重新进行：

- 全参数训练；
- LoRA 主模型训练；
- 完整 GRPO；
- PPO / verl actor optimization；
- reference model optimization。

原因：论文实验最终必须以 **Kaggle** 为主要实验平台，完整 RL 成本过高。

---

## 4.2 Noise Head

在模型当前 latent step 的表示后增加一个非常小的 MLP：

\[
x_t
\rightarrow
\operatorname{Linear}
\rightarrow
\operatorname{GELU}
\rightarrow
\operatorname{Linear}
\rightarrow
\operatorname{GELU}
\rightarrow
\operatorname{Linear}
\rightarrow
r_t.
\]

默认推荐：

```text
input_dim -> 128 -> 32 -> 1
```

输出：

\[
r_t=\operatorname{sigmoid}(f_\phi(x_t)),
\qquad
r_t\in(0,1).
\]

将其映射到允许的 Gumbel scale 区间：

\[
s_t
=
s_{\min}
+
(s_{\max}-s_{\min})r_t.
\]

### 初始推荐范围

第一轮实验建议：

\[
s_{\min}=0,\qquad
s_{\max}=1.
\]

如果后续 validation 显示需要比原始 `noise_scale=1.0` 更强的探索，再将：

\[
s_{\max}=1.5.
\]

**第一版代码不要一开始开放无界输出。**

禁止直接：

```python
noise_scale = noise_head_output
```

必须经过 sigmoid + bounded mapping，防止异常 noise scale 导致生成失控。

---

# 5. Noise Head 输入

最终推荐输入：

\[
x_t=
[
\operatorname{Proj}(h_t);
H_t;
M_t;
ESS_t;
p_{\max,t};
\Delta H_t;
t/T_{\max}
].
\]

---

## 5.1 Hidden State

当前 latent position 的最后层表示：

\[
h_t\in\mathbb{R}^{d}.
\]

为了减少 Noise Head 参数，可先投影：

\[
\tilde h_t=W_hh_t,
\]

推荐：

```text
d -> 64
```

然后再与 scalar statistics 拼接。

如果 Kaggle 显存非常紧张，允许第一版直接使用：

```text
hidden-state projection dimension = 32
```

**不得为了训练 Noise Head 对 backbone hidden states 建立反向图。**

实现必须：

```python
h_t = h_t.detach()
```

---

## 5.2 Entropy

从当前 top-K latent distribution 计算：

\[
H_t
=
-\sum_{i=1}^{K}
p_{t,i}\log(p_{t,i}+\epsilon).
\]

建议额外归一化：

\[
\tilde H_t
=
\frac{H_t}{\log K}.
\]

最终代码优先使用 normalized entropy：

```python
entropy_norm = entropy / math.log(K)
```

范围约为：

\[
[0,1].
\]

---

## 5.3 Top-2 Margin

\[
M_t=p_{t,(1)}-p_{t,(2)}.
\]

解释：

- 大：单个 token/path 占优；
- 小：多个候选接近。

---

## 5.4 Effective Support Size

\[
ESS_t
=
\frac{1}{
\sum_{i=1}^{K}p_{t,i}^2
}.
\]

推荐归一化：

\[
ESS_t^{norm}
=
\frac{ESS_t-1}{K-1}.
\]

最终范围约为：

\[
[0,1].
\]

---

## 5.5 最大概率

\[
p_{\max,t} = \max_i p_{t,i}.
\]

---

## 5.6 Entropy Change

\[
\Delta H_t
=
H_t-H_{t-1}.
\]

第一个 latent step：

```python
delta_entropy = 0.0
```

---

## 5.7 Latent Step Position

\[
q_t=\frac{t}{T_{\max}}.
\]

用于告诉 Noise Head 当前处于：

- reasoning early stage；
- middle stage；
- late stage。

---

# 6. 最终 Noise Head 输入建议

为了 Kaggle 可执行性，第一版采用：

```text
Projected hidden state: 64 dims
+ normalized entropy: 1
+ top2 margin: 1
+ normalized ESS: 1
+ max probability: 1
+ delta entropy: 1
+ normalized step index: 1
--------------------------------
Total: 70 dims
```

Noise Head：

```text
70 -> 128 -> 32 -> 1
```

参数量非常小。

---

# 7. Gumbel Noise：保持原始 Latent-GRPO 分布定义

## 7.1 最重要的实现原则

**不要重新定义作者原来的 Gumbel sampling distribution。**

当前仓库的 sampler 已经实现：

1. 标准 Gumbel sampling；
2. clamp；
3. 可选 one-sided transform；
4. `noise_scale`；
5. top-K Gumbel score；
6. Gumbel-softmax latent mixture。

本文只替换第 4 步中的固定 scale。

原始：

```python
gumbels = sampling_info.noise_scales[0] * gumbels
```

修改为概念上的：

```python
gumbels = adaptive_noise_scale * gumbels
```

其中：

```python
adaptive_noise_scale.shape == [batch_size, 1]
```

而不是单一 scalar。

---

## 7.2 方差比例解释

设原始经过 clamp / one-sided transform 后的 Gumbel random variable 为：

\[
G.
\]

原始 fixed baseline：

\[
\xi^{base}=cG.
\]

本文：

\[
\xi_t=s_tG.
\]

因此：

\[
Var(\xi_t)=s_t^2Var(G).
\]

相对 baseline 的方差比例：

\[
\rho_t
=
\frac{Var(\xi_t)}
{Var(\xi^{base})}
=
\left(\frac{s_t}{c}\right)^2.
\]

如果 baseline：

\[
c=1,
\]

则：

\[
\rho_t=s_t^2.
\]

因此 Noise Head 虽然直接输出的是 **noise scale ratio**，但它可以严格转换成 **variance ratio**。

论文措辞建议：

```text
Noise Head predicts a multiplicative exploration scale;
the corresponding Gumbel-noise variance changes quadratically
with this scale.
```

不要错误地写成：

```text
r_t itself is the variance
```

除非代码显式使用：

\[
s_t=\sqrt{r_t}.
\]

---

# 8. 与原始 One-Sided Gumbel 的关系

Latent-GRPO 代码中存在：

```text
use_one_sided_gumbel_noise=True
```

本文**保留**这个机制。

即：

```text
Original:
Gumbel -> clamp -> one-sided transform -> fixed scale

Ours:
Gumbel -> clamp -> one-sided transform -> Noise-Head scale
```

不修改：

- clamp 范围；
- one-sided shift；
- Gumbel softmax temperature；
- top-K selection rule。

这样可最大限度避免本文创新与原论文其他设计耦合。

---

# 9. 原始仓库中优先检查的代码位置

以下路径来自当前上传的 Latent-GRPO 仓库。

## 9.1 核心 sampler

```text
Latent-GRPO/Latent-GRPO/
sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py
```

目前关键代码附近包含：

```python
gumbels = -torch.empty_like(sampling_log_probs).exponential_().log()
gumbels = gumbels.clamp(-1.5, 3)

if use_one_sided_gumbel_noise:
    ...

gumbels = sampling_info.noise_scales[0] * gumbels
```

**这是本文最重要的修改点。**

注意：

当前代码使用：

```python
sampling_info.noise_scales[0]
```

即只取第一个值。

本文必须改成 batch-wise / step-wise：

```python
adaptive_scale  # shape [B, 1]
gumbels = adaptive_scale * gumbels
```

不能继续使用 `[0]`。

---

## 9.2 SamplingBatchInfo

```text
Latent-GRPO/Latent-GRPO/
sglang_latent_reasoning_pkg/python/sglang/srt/sampling/sampling_batch_info.py
```

当前已经存在：

```python
noise_scales = torch.tensor(
    [r.sampling_params.noise_scale for r in reqs],
    dtype=torch.float
).view(-1, 1)
```

Baseline 保留该变量。

Noise-Head 模式下需要区分：

```text
base / fallback noise_scale
adaptive noise_scale
```

推荐增加开关：

```python
adaptive_noise_head: bool
```

不要破坏原始 fixed baseline。

---

## 9.3 LogitsProcessorOutput hidden states

当前仓库：

```text
sglang_latent_reasoning_pkg/python/sglang/srt/layers/logits_processor.py
```

`LogitsProcessorOutput` 支持：

```python
hidden_states
```

但只有对应 capture mode 打开时才一定存在。

Codex 实现时必须先验证：

```python
logits_output.hidden_states is not None
```

在 latent decode step 是否成立。

如果默认路径不返回 last hidden state，需要启用：

```text
capture last hidden state
```

或在 model runner / logits processor 中增加仅供 Noise Head 使用的 last-state capture。

**目标是只保留当前 step 的 last hidden state，不要保存整条 sequence 的 hidden states。**

原因：

```text
Kaggle 显存约束。
```

---

# 10. Noise Head 推荐代码组织

不要把大量逻辑直接堆入 `sampler.py`。

建议新增：

```text
adaptive_noise/
    __init__.py
    noise_head.py
    features.py
    controller.py
    checkpoint.py
```

---

## 10.1 `noise_head.py`

提供：

```python
class NoiseHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_proj_size: int = 64,
        mlp_hidden_1: int = 128,
        mlp_hidden_2: int = 32,
        min_scale: float = 0.0,
        max_scale: float = 1.0,
    ):
        ...
```

输出：

```python
scale, ratio
```

其中：

```text
ratio in [0, 1]
scale in [min_scale, max_scale]
```

---

## 10.2 `features.py`

函数：

```python
compute_noise_features(
    hidden_state,
    topk_probs,
    step_idx,
    max_latent_steps,
    prev_entropy,
)
```

返回：

```python
{
    "projected_input": ...,
    "entropy": ...,
    "entropy_norm": ...,
    "margin": ...,
    "ess": ...,
    "ess_norm": ...,
    "p_max": ...,
    "delta_entropy": ...,
    "step_ratio": ...,
}
```

注意所有 statistics：

```python
.detach()
```

---

## 10.3 `controller.py`

统一处理：

```python
if adaptive_noise_enabled:
    scale = noise_head(...)
else:
    scale = fixed_noise_scale
```

sampler 不应知道训练细节。

---

# 11. Noise Head 的训练：Kaggle 优先方案

## 11.1 约束

**训练必须适合 Kaggle。**

因此：

```text
Backbone frozen
No backbone backward
No full GRPO
No PPO
No optimizer state for backbone
```

Noise Head 可单独训练。

---

# 12. 首版训练策略：监督 warm start + Head-Only RL

## Stage A：Offline Supervised Warm Start（必须做）

目的：

```text
先得到一个稳定 Noise Head，
避免在 Kaggle 上从随机 controller 直接做高方差 RL。
```

### 12.1 固定 scale 候选集合

推荐：

\[
\mathcal S=
\{0,\ 0.25,\ 0.5,\ 0.75,\ 1.0\}.
\]

如 Kaggle 时间过长，可缩减：

\[
\mathcal S=
\{0,\ 0.5,\ 1.0\}.
\]

### 12.2 数据收集

对每个训练问题：

```text
for scale in candidate_scales:
    run frozen model
    record correctness
    record trajectory statistics
```

至少保存：

```text
problem_id
dataset
seed
fixed_scale
correct
response_length
invalid
runtime_sec
latent_step
entropy_norm
margin
ess_norm
p_max
delta_entropy
step_ratio
hidden_feature / projected_hidden_feature
```

### 12.3 Oracle label

在 validation/training collection split 上，估计每道题不同 fixed scale 的表现。

如果每个 scale 只有一次 rollout，不要直接把单次正确/错误当成可靠 oracle。

推荐：

```text
rollouts_per_scale = 2
```

Kaggle 资源非常紧张时：

```text
rollouts_per_scale = 1
```

但必须在论文限制中说明标签噪声更高。

每题定义：

\[
\hat s_i^*
=
\arg\max_{s\in\mathcal S}
\hat R_i(s).
\]

若多个 scale reward 相同，tie-break：

```text
choose the smaller scale
```

理由：

```text
更保守、减少无必要探索。
```

### 12.4 Head 的监督目标

第一版最容易实现为分类：

```text
Noise Head -> logits over candidate scales
```

但最终架构希望输出 continuous ratio。

因此推荐实现：

### 方案 A（首选，简单）

监督回归：

\[
L=
\operatorname{SmoothL1}
(
s_{pred},s_i^*
).
\]

其中每条 trajectory 的各 latent steps 可共享 sample-level oracle scale 作为 warm-start target。

这只是初始化，不宣称它已经学到了真正 step-optimal policy。

---

## Stage B：Head-Only Policy Gradient Fine-Tuning（首版必须包含）

在 Stage A 与 Kaggle baseline smoke 通过后实施。两阶段共享train题集，Stage B重新生成当前策略的rollout；动作分布和RL具体预算需实施前讨论。

关键：

```text
只更新 Noise Head。
```

不要对 backbone 反向传播。

Noise Head 在每步输出一个 stochastic action distribution，例如：

\[
a_t\sim \operatorname{Beta}(\alpha_t,\beta_t),
\]

再映射：

\[
s_t
=
s_{\min}
+
(s_{\max}-s_{\min})a_t.
\]

最终 reward：

\[
R=
\begin{cases}
1,&\text{final answer correct}\\
0,&\text{otherwise}
\end{cases}
\]

优化：

\[
L_{PG}
=
-
(R-b)
\sum_t
\log \pi_\phi(a_t|x_t).
\]

可增加：

\[
L
=
L_{PG}
-
\lambda_H H(\pi_\phi)
+
\lambda_S L_{smooth}.
\]

其中：

```text
b = moving average reward
```

Noise scale smoothness：

\[
L_{smooth}
=
\frac1{T-1}
\sum_{t=2}^{T}
(s_t-s_{t-1})^2.
\]

### Kaggle 原则

Stage B 只允许在：

```text
Stage A 已稳定
+
单次 frozen inference 在 Kaggle 可运行
```

之后启用。

如果 Kaggle 时间/显存不足，立即停止实验并报告；Stage B标记未完成，不自动退回Stage A-only首版。需要变更范围时先与用户交流。

---

# 13. 最小可行版本（MVP）

为了确保毕业论文可完成，Codex 必须优先实现以下版本：

```text
MVP = Frozen backbone
    + deterministic Noise Head
    + offline supervised warm-start
    + head-only policy-gradient fine-tuning
    + step-wise scale output
    + fixed baseline comparison
```

暂不做：

- end-to-end LLM backward；
- full Latent-GRPO training；
- 多个复杂 baseline；
- adaptive rollout budget；
- dynamic top-K；
- adaptive temperature；
- backbone LoRA；
- 7B 主实验。

---

# 14. Baseline 的最终定义

只比较一个 baseline：

```text
Original Fixed-Gumbel Latent-GRPO inference
```

为了避免人为选择一个弱 baseline，固定 scale 必须通过 validation selection 确定。

例如候选：

\[
c\in\{0.5,1.0\}
\]

或：

\[
c\in\{0,0.25,0.5,0.75,1.0\}.
\]

在 validation split 上：

\[
c^*
=
\arg\max_c
\operatorname{Pass@1}(c).
\]

测试集 baseline 始终使用：

\[
c^*.
\]

这仍然是**一个 Original Fixed-Gumbel baseline**，只是其固定超参数由 validation 选择。

禁止在 test set 上调 baseline scale。

---

# 15. Proposed Method 的最终定义

测试集：

\[
s_t =
f_\phi(x_t).
\]

每个 latent step 重新计算。

模型主干：

```text
same checkpoint
same top-K
same Gumbel transform
same one-sided flag
same temperature
same response limit
same answer verifier
same test examples
same rollout budget
```

唯一核心差异：

```text
fixed scalar scale
vs.
Noise Head predicted step-wise scale
```

---

# 16. Kaggle 实验平台：硬性实现要求

## 16.1 路径规范

Kaggle 输入：

```text
/kaggle/input/<dataset-name>/
/kaggle/input/<model-name>/
/kaggle/input/<code-package>/
```

工作目录：

```text
/kaggle/working/adanoise_lgrpo/
```

所有输出必须写到：

```text
/kaggle/working/adanoise_lgrpo/artifacts/
```

本地开发可以使用相对路径，但代码必须支持：

```python
--platform kaggle
```

或自动探测：

```python
Path("/kaggle/working").exists()
```

---

## 16.2 Kaggle 启动必须先做资源探测

Notebook 第一阶段输出：

```python
torch.cuda.is_available()
torch.cuda.get_device_name(0)
torch.cuda.get_device_properties(0).total_memory
torch.__version__
torch.version.cuda
```

并保存：

```text
artifacts/environment.json
```

不得假定 Kaggle 一定提供某个具体 GPU 型号。

---

## 16.3 单 GPU 优先

主要实验设计必须：

```text
single-GPU runnable
```

不要让主要实验依赖：

- 8×A100；
- multi-node；
- Ray cluster；
- 3 GPU；
- FSDP multi-GPU。

如果 Kaggle 偶尔提供多 GPU，可以用于加速，但实验代码不能依赖它。

---

## 16.4 首先进行 inference smoke test

在任何 Noise Head 训练之前：

```text
1 sample
1 rollout
small max response length
```

确认：

- model 可加载；
- latent mode 可运行；
- Gumbel sampling 可运行；
- hidden state 可获取；
- GPU memory 不溢出。

输出：

```text
artifacts/smoke_test.json
```

必须记录：

```text
peak_gpu_memory_mb
runtime_sec
output_text
num_latent_steps
```

---

## 16.5 OOM 降级顺序

若 Kaggle OOM，按以下顺序降级：

1. batch size -> 1；
2. 关闭不必要的 logprob / hidden-state history；
3. 只保留 last hidden state；
4. 缩短 max response length；
5. 降低 simultaneous rollouts；
6. 使用 fp16 / bf16 中兼容的一种；
7. 若代码路径兼容，再尝试 8-bit / 4-bit inference。

注意：

```text
量化必须先验证是否与当前 latent embedding / custom sampler 路径兼容。
```

不得未经验证直接把量化作为默认方案。

---

## 16.6 Kaggle Session 时间限制

实验必须支持断点续跑。

每处理若干 samples：

```text
flush metrics
save progress
save Noise Head checkpoint
```

推荐每：

```text
25 or 50 samples
```

保存一次。

保存：

```text
artifacts/progress.json
artifacts/results_partial.parquet
artifacts/noise_head_latest.pt
```

---

# 17. Kaggle 数据规模建议

第一阶段不要直接全量。

## Phase 1：Engineering

```text
32~64 train samples
32 validation samples
32 test samples
```

目标：

```text
跑通全流程。
```

---

## Phase 2：Pilot

```text
200~500 train/collection samples
100 validation
200 test
```

目标：

```text
确定算法是否有信号。
```

---

## Phase 3：Final

根据 Kaggle 实际 runtime 决定。

优先：

```text
GSM8K / GSM8K-Aug
```

如资源允许增加 OOD：

```text
SVAMP
MultiArith
GSM-Hard
```

不要把 7B + AIME 作为毕业论文必要条件。

---

# 18. 主要实验数据集

## 主任务

```text
GSM8K / GSM8K-Aug
```

原因：

- 与 Latent-SFT / Latent-GRPO 低难度实验设置一致；
- 1B 模型更适合；
- reward verification 简单；
- Kaggle 成本较低。

---

## OOD（可选）

优先级：

```text
1. SVAMP
2. MultiArith
3. GSM-Hard
```

---

# 19. 实验控制变量

Baseline 与 AdaNoise-LGRPO 必须完全一致：

```text
checkpoint
dataset split
prompt template
top-K
temperature
Gumbel-softmax temperature
one-sided Gumbel setting
max response length
answer extraction
answer verifier
random seed list
number of test questions
rollouts per question
```

唯一主要自变量：

```text
noise-scale policy
```

Baseline：

```text
fixed c*
```

Proposed：

```text
step-wise Noise Head scale
```

---

# 20. 评价指标

## 主指标

### Pass@1

\[
Pass@1
=
\frac{
\# correct
}{
\# samples
}.
\]

---

## 若 Kaggle 预算允许

### Pass@k

推荐：

```text
Pass@4
```

不要默认一定做 Pass@64。

原因：

```text
Kaggle inference cost。
```

---

## Reasoning Length

记录：

```text
num latent steps
explicit answer tokens
total generated tokens
```

---

## Invalid Rate

\[
InvalidRate
=
\frac{
\# invalid/non-terminating/truncated
}{
\# total
}.
\]

---

## Runtime

```text
seconds / question
```

以及：

```text
peak GPU memory
```

---

## Noise Head 专属指标

每题保存：

\[
\bar s=
\frac1T\sum_t s_t
\]

以及：

\[
s_{\min}^{obs},
s_{\max}^{obs},
Std(s_t).
\]

---

# 21. 必须输出的分析图

## Figure A：Noise scale trajectory

横轴：

```text
latent step
```

纵轴：

```text
predicted noise scale
```

至少画：

- 典型正确样本；
- 典型错误样本。

---

## Figure B：Entropy vs Noise Scale

散点：

\[
(H_t,s_t).
\]

目的：

```text
观察 Noise Head 是否学到简单单调关系，
还是更复杂的 state-dependent policy。
```

不要预设必须正相关。

---

## Figure C：Fixed vs Adaptive

至少包含：

```text
Pass@1
Invalid Rate
Average reasoning length
Runtime
```

---

## Figure D：Noise scale 分布

比较：

```text
correct trajectories
incorrect trajectories
```

---

# 22. 统计报告建议

Pass@1 除点估计外，建议报告 bootstrap 置信区间。

例如：

```text
95% bootstrap CI
```

由于同一 test question 会被两种方法同时测试，可以使用 paired bootstrap。

记录每题：

```text
baseline_correct
adaptive_correct
```

可进一步报告：

```text
adaptive-only correct
baseline-only correct
both correct
both wrong
```

如果样本足够，可做 McNemar test 作为补充。

---

# 23. Random Seed 规范

统一：

```text
global_seed
torch_seed
numpy_seed
python_seed
sampling_seed
```

建议最终至少：

```text
3 seeds
```

如果 Kaggle 预算不足：

```text
1 seed pilot
3 seeds final subset
```

不要用不同 seeds 给 baseline 和 proposed。

---

# 24. 日志与 artifact schema

每个生成结果保存一条 JSONL/Parquet record：

```json
{
  "method": "fixed|adanoise",
  "dataset": "gsm8k",
  "split": "test",
  "problem_id": "...",
  "seed": 0,
  "rollout_id": 0,
  "correct": true,
  "invalid": false,
  "response_length": 24,
  "runtime_sec": 1.23,
  "peak_gpu_memory_mb": 0,
  "fixed_scale": 1.0,
  "num_latent_steps": 12,
  "noise_scales": [0.31, 0.28, 0.44],
  "entropies": [0.22, 0.27, 0.50],
  "margins": [0.71, 0.66, 0.31],
  "ess": [1.4, 1.6, 3.1]
}
```

不要只保存最终 aggregate。

必须保存 per-example 数据，方便后续：

- bootstrap；
- error analysis；
- 作图；
- 论文表格重新计算。

---

# 25. 推荐配置文件

新增：

```text
configs/adanoise_kaggle.yaml
```

建议字段：

```yaml
platform: kaggle

model:
  path: /kaggle/input/...
  dtype: auto
  freeze_backbone: true

latent:
  top_k: 10
  gumbel_softmax_temperature: 1.0
  use_one_sided_gumbel_noise: true
  add_noise_gumbel_softmax: true

baseline:
  candidate_fixed_scales: [0.0, 0.25, 0.5, 0.75, 1.0]

noise_head:
  enabled: true
  hidden_proj_dim: 64
  hidden_dim_1: 128
  hidden_dim_2: 32
  min_scale: 0.0
  max_scale: 1.0
  use_entropy: true
  use_margin: true
  use_ess: true
  use_pmax: true
  use_delta_entropy: true
  use_step_ratio: true

training:
  mode: offline_supervised
  batch_size: 64
  lr: 0.001
  epochs: 20
  loss: smooth_l1

experiment:
  dataset: gsm8k
  rollouts_per_scale: 1
  seeds: [0]
  save_every: 25
```

最终数值必须允许命令行 override。

---

# 26. Codex 实现任务顺序

## Task 1：仓库审计

确认：

```text
sampler.py
SamplingBatchInfo
hidden-state path
eval script
current noise_scale flow
```

输出：

```text
docs/adanoise_repo_audit.md
```

---

## Task 2：Baseline 冻结

先不要实现 Noise Head。

确保 Kaggle 可以跑：

```text
fixed-scale inference
```

并得到可复现结果。

---

## Task 3：Hidden State Probe

确认 latent decode 时：

```python
logits_output.hidden_states
```

能否获取当前 last hidden state。

如果不能：

```text
只做最小修改使 last hidden state 可用。
```

禁止缓存所有 token 的 full hidden-state history。

---

## Task 4：Feature Extractor

实现并单元测试：

```text
entropy
normalized entropy
margin
ESS
normalized ESS
pmax
delta entropy
step ratio
```

---

## Task 5：Noise Head

实现：

```text
Frozen backbone + detached input
bounded scale output
checkpoint save/load
```

---

## Task 6：Sampler Integration

将：

```python
sampling_info.noise_scales[0]
```

替换为：

```text
fixed mode -> request noise scale
adaptive mode -> Noise Head scale
```

必须保证 baseline 行为不变。

---

## Task 7：Offline Data Collection

Kaggle 上生成：

```text
fixed-scale sweep dataset
```

支持 resume。

---

## Task 8：Noise Head Training

最好允许在：

```text
GPU or CPU
```

训练，因为 head 很小。

训练结果：

```text
artifacts/noise_head_best.pt
```

---

## Task 9：Adaptive Inference

运行：

```text
method=adanoise
```

保存所有 step-level statistics。

---

## Task 10：Evaluation

生成：

```text
summary.csv
paired_results.parquet
figures/*.pdf
figures/*.png
```

---

# 27. 单元测试要求

至少新增：

```text
test_noise_features.py
test_noise_head.py
test_adaptive_scale_bounds.py
test_fixed_mode_identity.py
test_sampler_adaptive_scale_shape.py
```

---

## 27.1 最关键测试：Fixed Mode Identity

当：

```text
adaptive_noise_enabled=False
```

必须保证行为与原始代码一致。

即相同 seed 下：

```text
old fixed sampler
==
new code fixed sampler
```

至少检查：

```text
topk_indices
topk_probs
next token
```

---

## 27.2 Scale Bound

必须：

\[
s_{\min}\le s_t\le s_{\max}.
\]

---

## 27.3 Zero Scale

当：

```text
scale = 0
```

Gumbel perturbation 应为 0。

---

## 27.4 Batch Shape

禁止：

```python
noise_scales[0]
```

误应用到所有 batch item。

Adaptive scale 必须支持：

```text
[B, 1]
```

---

# 28. 性能测试

Kaggle smoke test 记录：

```text
baseline peak memory
adaptive peak memory
baseline runtime
adaptive runtime
```

目标：

```text
Noise Head 的额外显存与时间相对于 frozen backbone inference 很小。
```

如果 adaptive runtime 明显增加，优先检查：

- 是否重复 forward；
- 是否意外保存 full hidden states；
- 是否触发 CPU/GPU sync；
- 是否每 step 重新加载 head；
- 是否大量 `.item()`。

---

# 29. Acceptance Criteria

MVP 完成必须同时满足：

### Engineering

- [ ] Kaggle 单 GPU 可完成 frozen baseline inference；
- [ ] Noise Head 可加载；
- [ ] Backbone 无梯度；
- [ ] Adaptive scale 每 latent step 可变化；
- [ ] Fixed mode 与原始 sampler 行为一致；
- [ ] 结果可断点续跑；
- [ ] per-example/per-step metrics 被保存。

### Experiment

- [ ] validation 上选出 fixed baseline scale；
- [ ] baseline test 完成；
- [ ] AdaNoise test 完成；
- [ ] 使用相同 test set 与 seeds；
- [ ] 输出 Pass@1；
- [ ] 输出 reasoning length；
- [ ] 输出 invalid rate；
- [ ] 输出 runtime；
- [ ] 至少输出一张 noise trajectory 图。

### Thesis

- [ ] 能回答 Fixed vs Adaptive 是否改善；
- [ ] 能展示 Noise Head 实际输出不是常数；
- [ ] 能分析 noise scale 与 uncertainty 的关系；
- [ ] 能报告 Kaggle 硬件、显存与 runtime。

---

# 30. 失败条件与止损策略

## Case 1：作者 checkpoint 在 Kaggle 连 inference 都 OOM

则本文当前方案无法直接实施。

优先检查：

```text
batch=1
last hidden only
fp16/bf16
short response
single rollout
```

若仍 OOM，再评估量化兼容性。

不要转而尝试完整 RL。

---

## Case 2：Hidden state 获取导致显存暴涨

降级：

```text
只 capture last hidden state
```

若仍困难，允许论文工程版 Noise Head 输入只使用：

```text
top-K probability statistics
```

即：

\[
x_t=
[H_t,M_t,ESS_t,p_{\max,t},\Delta H_t,t/T].
\]

但这属于 fallback，最终论文中必须明确说明 Noise Head 未直接读取 \(h_t\)。

---

## Case 3：Noise Head 学成常数

检查：

- oracle labels 是否严重不平衡；
- fixed scales 是否几乎无性能差异；
- hidden/stat features 是否有 variation；
- `min_scale/max_scale` 是否设置过窄；
- supervision target 是否噪声过大。

如果 fixed-noise sweep 本身显示不同 scale 几乎没有异质性，则该研究假设可能不成立，应如实报告，不应强行制造结果。

---

## Case 4：Adaptive 比 fixed 更差

优先检查：

```text
distribution shift
oracle label noise
head overfit
step-wise application vs sample-level supervision mismatch
```

可将第一版 adaptive 改为：

```text
one scale per question
```

即在第一个 latent state 预测：

\[
s(x)
\]

并整条 trajectory 固定使用该值。

这仍属于 Noise-Head adaptive 方法，而且更容易稳定。

---

# 31. 推荐开发优先级

必须按顺序：

```text
Kaggle baseline inference
        ↓
hidden-state probe
        ↓
feature logging
        ↓
fixed-scale sweep
        ↓
Noise Head supervised training
        ↓
adaptive inference
        ↓
paired evaluation
```

不要先写复杂 RL controller。

---

# 32. 最终论文中两种方法的定义

## Method 1：Original Fixed-Gumbel

\[
G_{t,i}^{base}
=
c^* G_{t,i}.
\]

其中 \(c^*\) 由 validation set 选择后固定。

---

## Method 2：AdaNoise-LGRPO

\[
r_t
=
\operatorname{sigmoid}
\left(
f_\phi(x_t)
\right),
\]

\[
s_t
=
s_{\min}
+
(s_{\max}-s_{\min})r_t,
\]

\[
G_{t,i}^{adaptive}
=
s_tG_{t,i}.
\]

其中：

\[
x_t=
[
\tilde h_t,
\tilde H_t,
M_t,
\widetilde{ESS}_t,
p_{\max,t},
\Delta H_t,
t/T_{\max}
].
\]

最终 latent distribution 的其余构造保持原始代码不变。

---

# 33. 论文创新点最终表述

## 创新点 1：状态自适应 Gumbel exploration

将 Latent-GRPO 的固定 Gumbel scale：

\[
c
\]

改为：

\[
s_t=f_\phi(x_t),
\]

使探索强度依赖当前 latent reasoning state。

---

## 创新点 2：轻量 Noise Head

不更新大模型，仅训练小型 controller：

```text
Frozen backbone
+
Trainable Noise Head
```

因此适合有限计算资源，特别是本文明确采用的 **Kaggle GPU 实验平台**。

---

## 创新点 3：隐状态与统计不确定性联合建模

Noise Head 同时使用：

```text
hidden semantic state
+
entropy
+
margin
+
ESS
+
confidence dynamics
```

而不是单一 entropy threshold。

---

## 创新点 4：Exploration Scale / Variance 的明确解释

Noise Head 输出 scale \(s_t\)，对应 Gumbel perturbation variance：

\[
Var(s_tG)=s_t^2Var(G).
\]

因此：

\[
\frac{Var_t}{Var_{baseline}}
=
\left(
\frac{s_t}{c^*}
\right)^2.
\]

---

# 34. 不做的内容

为了确保项目在 Kaggle 上可完成，以下内容不属于当前毕业论文主线：

```text
完整 Latent-GRPO 再训练
PPO / GRPO backbone update
advantage estimator 修改
dynamic top-K
dynamic rollout budget
adaptive Gumbel temperature
多个 controller ensemble
7B/AIME 必须复现
多 GPU 训练
```

如果以后算力增加，可作为 future work。

---

# 35. 论文实验结论应回答的四个问题

最终只需要清楚回答：

### Q1

在相同 frozen backbone 与相同 Kaggle inference budget 下：

```text
Noise-Head Gumbel 是否优于 validation-tuned Fixed-Gumbel？
```

### Q2

Noise Head 输出的 scale 是否真的随 latent state 变化，而不是退化为常数？

### Q3

scale 与：

```text
entropy
margin
ESS
step position
```

之间有什么关系？

### Q4

自适应探索带来的收益是否值得其非常小的额外：

```text
runtime
memory
controller complexity
```

---

# 36. 最终实验表格模板

| Method | Pass@1 ↑ | Pass@4 ↑ | Avg Latent Steps ↓ | Invalid Rate ↓ | Runtime / Q ↓ | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-Gumbel Latent-GRPO |  |  |  |  |  |  |
| **AdaNoise-LGRPO** |  |  |  |  |  |  |

若 Pass@4 由于 Kaggle 时间限制无法完成：

```text
删除 Pass@4 列，不影响主实验成立。
```

---

# 37. Codex 最终执行原则

1. **先保证 Kaggle 能跑，再增加方法复杂度。**
2. **任何功能不得破坏 fixed baseline。**
3. **Backbone 始终 frozen。**
4. **只保存当前 step 必需的 hidden state。**
5. **每个长实验必须 resume-friendly。**
6. **所有结果保存 per-example 数据。**
7. **Noise Head 输出必须有界。**
8. **保持原始 one-sided Gumbel、top-K 与 temperature 逻辑不变。**
9. **主要比较只保留 Original Fixed-Gumbel vs AdaNoise-LGRPO。**
10. **所有论文主结果必须来自 Kaggle 平台运行记录。**

---

# 38. 给 Codex 的首个执行 Prompt 建议

可以直接将下面内容作为本地 Codex 的第一阶段任务：

```text
Read this specification completely before editing code.

Goal:
Implement the engineering foundation for AdaNoise-LGRPO while preserving
the original Fixed-Gumbel behavior exactly.

Hard constraints:
- Target experiment platform is Kaggle single-GPU.
- Do not train or backpropagate through the LLM backbone.
- Do not implement full GRPO/PPO.
- Preserve current one-sided Gumbel, top-K and Gumbel-softmax logic.
- First make the existing fixed-noise inference path runnable and reproducible.
- Add tests before integrating the adaptive controller.

First tasks:
1. Audit the existing noise_scale data flow from eval arguments through
   SamplingBatchInfo into sampler.py.
2. Audit whether the last hidden state is available during latent decode.
3. Create docs/adanoise_repo_audit.md with exact file/function locations.
4. Add a Kaggle smoke-test script that records GPU model, VRAM, runtime,
   peak allocated memory and one latent inference output.
5. Do not implement the Noise Head until the baseline smoke test passes.
```

---

# 39. 最终摘要

本文最终拟提出 **AdaNoise-LGRPO**。

核心改变不是重新训练 Latent-GRPO，而是在已经训练好的 latent-reasoning model 上增加一个轻量 Noise Head。

原始方法：

\[
\boxed{
s_t=c
}
\]

本文：

\[
\boxed{
s_t
=
f_\phi
(
h_t,
H_t,
M_t,
ESS_t,
p_{\max,t},
\Delta H_t,
t/T
)
}
\]

并使用：

\[
G_{t,i}^{adaptive}
=
s_tG_{t,i}
\]

替代固定 Gumbel perturbation scale。

大模型参数全部冻结，仅训练 Noise Head。

实验平台明确限定为：

\[
\boxed{\textbf{Kaggle GPU Notebook}}
\]

因此整个工程设计必须优先满足：

```text
single-GPU
low-memory
resume-friendly
no-backbone-backward
small-controller-training
```

最终实验只比较：

\[
\boxed{
\text{Original Fixed-Gumbel}
\quad vs \quad
\text{AdaNoise-LGRPO}
}
\]

以验证：

> **在有限 Kaggle 计算预算下，学习式 state-dependent Gumbel exploration 是否能够优于固定噪声探索。**
