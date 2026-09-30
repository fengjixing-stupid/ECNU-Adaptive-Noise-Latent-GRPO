---
id: DOC-SMOKE-001
type: testing
status: active
title: 本地 CPU Sampler Smoke
created: 2026-09-30
updated: 2026-09-30
related_docs:
  - DOC-PLAN-001
  - DOC-MINFIX-001
related_code:
  - adaptive_noise/sampling_kernel.py
  - scripts/smoke_local.py
  - tests/test_fixed_mode_identity.py
  - tests/test_smoke_local_cli.py
  - configs/local_smoke.yaml
---
# 本地 CPU Sampler Smoke

返回 [文档索引](README.md)。这是实现计划 Task 1，尚未验证真实模型或 Kaggle Engine。

## 运行

使用用户指定虚拟环境的 Python：
```sh
python -m pytest -q
python scripts/smoke_local.py --config configs/local_smoke.yaml --device cpu
```
入口可从任意 cwd 运行；默认 config 位于项目 configs，输出位于项目 artifacts。`--output-dir` 可指定其他产物目录。
本命令不下载模型、不导入 SGLang/verl/Ray、不创建 CUDA Engine。依赖是现有 torch、pytest、PyYAML，不执行安装。

## 已验证证据（2026-09-30）

完整项目测试：43 passed（3.17秒）。独立 smoke：38 passed、0 failed、0 errors、0 skipped，运行 1.87 秒。
Python 3.12.10，torch 2.10.0；参考 minfix HEAD为 `0b7e85f15e9859033653964282348e518f9f6291`。
TDD：kernel的37项在NotImplemented状态失败，实现后通过；CLI的2项在入口实现前失败；独立review指出的来源一致性和缺失输入检查由3项RED→GREEN回归覆盖，实现后完整43项通过；最终补充混合active/inactive邻行测试，直接验证非latent行不被邻行噪声改变。

| 范围 | 验收 |
| --- | --- |
| fixed identity | B=1/3、seed=0/1/2、one-sided开关、top-p=.2/.95，indices/probs/latent id逐位相等 |
| chunk边界 | V=8192/8205，scores、IDs、selected noise与实际minfix helper逐位相等 |
| mask与scale | inactive/结束token恢复clean mixture；zero scale消去noise；逐行scale不会误广播首请求 |
| selection | 原始clean top-K外token仍可被Gumbel选入 |
| workspace | noisy helper生成张量的二维宽度不超过8192；Top-P support仍允许step-local full-V |

## 输出与停止行为

`artifacts/local_smoke.json`：schema_version、scope、pass/fail、device、seeds、版本、upstream commit、sampler/kernel SHA256、每个case状态和耗时、pytest exit code。
`local_smoke.junit.xml` 保存测试结果，`local_smoke.log` 保存pytest输出。CPU的peak_gpu_memory_mb=null，不能当作GPU显存证据。
CLI先验证config路径必须指向reference读取的固定submodule，再验证sampler、kernel、reference、测试文件、固定commit和sampler是否被本地修改；失败返回2，停止执行测试。
测试失败返回1，留下报告与日志，不自动重试。环境、网络、设备及OOM问题按用户规则停止。

## 函数边界

`streaming_noisy_topk` 返回(noisy log scores, IDs, selected noise)，保持minfix的8192分块随机顺序。
`sample_latent_kernel` 返回(indices, mixture probs, next_latent_id)，只负责latent mixture边界，不替代explicit token sampler。active_mask由调用方组合latent和add_noise标志，latent_end_token_id显式传入。
可选gumbels是未clamp/未平移/未scale的标准Gumbel样本，只用于受控reference检查；默认路径不创建full-V noise。
固定identity限定同scale、同temperature的原作者请求语义；异构scale以minfix helper支持的逐行输入为参考。

独立review：发现的两处CLI预检问题已修复并复审确认，无残留发现。

## 下一阶段

Task 2是Kaggle单GPU baseline与hidden probe，不在本轮执行。进入该阶段前先与用户确认实验设置，并取得checkpoint/data必需路径；缺失输入立即停止。
Noise Head、真实模型加载、Kaggle吞吐/显存和采样后explicit token路径尚未验证。
