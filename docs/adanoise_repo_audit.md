---
id: DOC-AUDIT-001
type: testing
status: archived
title: AdaNoise-LGRPO 源码审计
created: 2026-09-30
updated: 2026-09-30
related_docs:
  - DOC-SPEC-001
---
# AdaNoise-LGRPO 源码审计

返回 [文档索引](README.md)。证据版本：作者仓库 `e70deb8ee1a9dc8908ee473f7cafb1e01fd0de93`。
本审计是静态源码取证，未执行模型推理。

历史记录：用户已将旧 checkout 替换为 minfix。本文件保留旧版证据，不再作为当前 sampler 的实现依据；当前来源见 [minfix 接入审计](minfix_submodule_update.md)。

## 1. 已确认调用链

`eval/eval_low_tasks_sglang.py::persistent_inference_worker` → `sgl.Engine` → schedule/model forward → logits processor → sampler → top-K mixture → 下一个模型 forward。
Llama `LlamaModel.forward` 使用 `embed_tokens.weighted_forward(topk_probs, topk_indices)` 形成 latent embedding，不能用普通 token 采样替代。
作者 evaluator 将 `<think>` 加入 prompt，并由 tokenizer 获取 `</think>` token id；其 bf16、flashinfer、2048 running requests 设置不是 Kaggle smoke 默认值。

## 2. Sampler 数学路径

`layers/sampler.py::Sampler.forward` 在 latent 分支：
full-vocabulary float32 log-softmax → top-p mask（至少保留 K）→ exponential/log Gumbel → clamp [-1.5,3] → 可选 +1.5 单侧平移 → scale → noisy score top-K → temperature softmax → latent mixture。
噪声先在整个 vocabulary 上采样、再选 top-K；不能改成仅对原始 top-K 加噪。
当前固定 scale 是 `sampling_info.noise_scales[0]`；Gumbel temperature 和部分普通 temperature 也用首行。
噪声生效 mask 包含 `latent_modes`、原始 top-1 != 524、`add_noise_gumbel_softmax`。524 是源码硬编码，首轮限定用户的 Llama 1B；tokenizer 必须核验，不擅自推广到 Qwen。

## 3. Batch 与 hidden state

`SamplingBatchInfo.from_schedule_batch` 已建立 [B,1] noise_scales，filter/merge 同步这些字段。
`schedule_batch.py::_get_capture_hidden_mode` 在 enable_latent 时直接返回 LAST；`ForwardBatch` 传递该标志；`LogitsProcessor.forward` 的 LAST 分支返回当前 pruned states。
因此 Spec 对 hidden capture 的未知项在此 commit 上已有静态答案；真实 decode 是否非空、行对齐、无历史保留仍需 Kaggle probe 验证。
新增逐请求 entropy/step 状态必须跟随 filter/merge/reset，不按稳定 batch 行号跨步存储。

## 4. 需要在设计中解决的差异

| 位置 | 差异 | 计划处理 |
| --- | --- | --- |
| Spec 9 | 路径多了一层 Latent-GRPO | 以本项目实际单层路径为准 |
| Spec 5/10 | 扰动后的 p 与 scale 存在循环依赖 | 特征定义为加噪前原始 logits top-K 内归一化概率 |
| Spec 27 | 原版 [0] identity 与异构 batch 修复不能同时成立 | identity 限定同 scale 同 temperature；异构 scale 按请求语义验收 |
| Spec 16.5/30 | OOM 建议降级重试 | 用户当前指令优先，立即停止并报告 |
| docs/AGENTS.md | 含占位符且声明自身为模板 | 原文保留，根 AGENTS.md 补充具体项目入口 |

## 5. 环境与待核验项

只读包元数据：Darwin arm64，Python 3.12.10，torch 2.10.0，transformers 5.12.1，pytest 9.1.1。
作者 README 推荐 Python 3.11、torch 2.6.0、transformers 4.51.1 及 CUDA 专用依赖；本轮未尝试安装或运行这些依赖，不据此宣称本地不兼容。
本地 checkpoint 路径、精确模型配置、数据路径尚未提供；不读取用户其他目录猜测。
`docs/AGENT_USE.md` 由用户补入后已读取；依其 Delegation Gate，本轮主 Agent 顺序完成，不启动子 Agent。
