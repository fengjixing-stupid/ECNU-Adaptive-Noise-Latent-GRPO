---
id: DOC-MINFIX-001
type: testing
status: active
title: Minfix Submodule 接入审计
created: 2026-09-30
updated: 2026-09-30
related_docs:
  - DOC-SPEC-001
  - DOC-AGENT-001
related_code:
  - latent_grpo_minfix/Latent-GRPO-final/Latent-GRPO/sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py
  - latent_grpo_minfix/Latent-GRPO-final/tests/unit/test_streaming_topk_sampler.py
---
# Minfix Submodule 接入审计

返回 [文档索引](README.md)。本轮仅静态审计和仓库引用更新，未运行实验或重新执行历史测试。

## 来源与路径

用户以 `latent_grpo_minfix/` 替换旧作者 checkout。实际独立 Git 根目录为 `latent_grpo_minfix/Latent-GRPO-final/`，其内部源码目录为 `Latent-GRPO/`。
submodule 使用 https://github.com/fengjixing-stupid/Latent-GRPO.git，固定 `0b7e85f15e9859033653964282348e518f9f6291`。只读 `git ls-remote` 确认远端 `Latent-GRPO-Top-K` 指向同一提交；不误用不同的 main HEAD。
外层说明包不是 Git 仓库，不强行移动 .git、重建历史或把它伪装为独立 submodule。

## 当前 sampler 的实际边界

`_streaming_latent_noisy_topk` 先计算 full-vocabulary logsumexp，再以8192 vocabulary chunk抽样 Gumbel、clamp [-1.5,3]、可选+1.5、乘scale、support mask、chunk top-K、running top-K merge；最终返回 noisy scores-log_z、IDs和selected noise。
`Sampler.forward` 保留Top-P support及至少K个候选规则；support构造临时量提前del，helper后del mask；后续K内temperature softmax保持不变。不能将本地生产kernel改回完整[B,V] noisy tables。
固定scale仍传 `sampling_info.noise_scales[0]`；未来batch-wise控制按既定计划修复。当前结束mask从 `_require_latent_end_token_id()` 获取，不再硬编码524。
当前helper只返回K宽结果，但Top-P softmax/sort和普通logprob路径仍有step-local full-V临时量。不能把修复描述为整个推理不存在full-V张量。

## 现有测试与本地修改

`tests/unit/test_streaming_topk_sampler.py` 直接AST抽取实际helper，包含相同noise的monolithic对照、8192尾块、support及workspace宽度检查。
文档中的历史测试结果不作为本轮测试通过证据；本轮没有执行模型或测试。
接入前已有 `Latent-GRPO/cairn/LOG.md` 修改、未跟踪 `docs/TOP_K_MEMORY_OPTIMIZATION_HANDOFF.md`。两者保留，不混入主项目提交；submodule引用仅发布已在远端的commit，克隆者不会得到这些本地文档。
外层 `MINIMAL_FIX_SUMMARY.md` 提到较旧9109896提交，当前运行基准以实际HEAD和源码为准。

## 执行授权

用户已确认本地CPU smoke范围。按新minfix进行reference/边界测试属于工程同步。
涉及后续实验设计必须先交流；checkpoint/data等该阶段必需输入缺失则立即停止。真实模型推理仍在Kaggle单GPU验收。
