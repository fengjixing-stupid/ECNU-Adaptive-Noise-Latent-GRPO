---
id: DOC-REQUEST-SEED-001
type: testing
status: active
title: Kaggle 持久 Engine 请求级 seed smoke
created: 2026-10-05
updated: 2026-10-05
related_docs:
  - DOC-K0-001
  - DOC-PLAN-001
related_code:
  - scripts/kaggle_request_seed_smoke.py
  - notebooks/ecnu-smoke-adaptive-latent-grpo.ipynb
  - tests/test_request_seed_smoke.py
---
# Kaggle 持久 Engine 请求级 seed smoke

返回 [文档索引](README.md)。用户已提供 K0 PASS 输出并要求直接继续；本阶段 CPU 验证完成，真实 Kaggle GPU 结果待运行。

## 执行入口与输入

交付 Notebook 基于用户的 `ecnu-smoke-adaptive-latent-grpo.ipynb`。保留 git clone、环境准备、安装命令和 Bash heredoc 构建 `/kaggle/working/test_sglang_seed.py` 后执行的逻辑；删除固定 commit checkout 和 git/source SHA256 检测单元格。原附件不修改，历史输出清空。

环境已就绪时运行新 Notebook 的 cell 9（从0计数，第10个单元格）；cell 10 可选，重新运行会创建新产物目录。脚本不自动更新 Git、安装或下载依赖，使用当前已安装的作者代码，在产物目录复制临时 overlay 后追加请求 metadata 与 sampler hook。

| 设置 | 值 |
| --- | --- |
| checkpoint | `/kaggle/input/models/fengjixing/llama3-2-1b-instruct-latent-grpo-top10/other/default/1/LLaMA3.2-1B-Instruct-Latent-GRPO-Top10` |
| 数据 | `/kaggle/input/datasets/fengjixing/datasets-for-latent-grpo/data/GSM8k-Aug-oss-dup-all.parquet`，沿用最新附件挂载路径 |
| 题目 | 固定 train 的 GSM8K source_row=7335，数据文件身份检查保留 |
| Engine | 仅启动1次，T4 GPU0、fp16、tp=1、max_running_requests=1；关闭 radix/prefix cache、CUDA graph、overlap schedule |
| 生成 | temperature=.6、top_p=.95、max_new_tokens=128、top10、Gumbel temperature=1、scale=1、双侧噪声 |

用户要求使用最新脚本，因此本阶段不设置固定上游 commit/source SHA 的相等门禁。实际安装源码、评分文件、模型和 hook 的 hash 仍作为来源记录；轨迹与 hidden hash 仍用于 seed 验收。主项目 submodule 引用保持不变。本次数据挂载路径取代 K0 页的旧路径。

## 请求 seed 与验收范围

同一个 Engine 顺序处理 `same_a`、`same_b`、`different`，前两次 seed=3030731552，第三次 seed=527462139，沿用已批准的 root12345/problem_id/scale1/rollout_id 派生协议。Engine 启动 seed=12345，独立于请求 seed。

请求通过 `custom_params` 携带 run_id/seed。原作者 `SamplingBatchInfo` 在无 custom logit processor 时不保留该字段，因此 overlay 在 from/filter/merge 中附加对齐的请求 metadata。sampler 在每个新 request ID 首次采样前初始化实际 worker 的 CPU 与 logits 所在 CUDA 设备 RNG；同一请求后续 token 不重置 RNG。

验收要求三次请求使用同一个采样 worker、三个不同 request ID；每个请求只重置一次 seed。hidden/clean top10/mixture/token 对齐检查沿用 K0。相同 seed 的 token、trace、hidden 必须一致；不同 seed 的 trace 必须改变，最终答案可以相同。不同 seed trace 未改变时记录 inconclusive 并停止。

本阶段仅验证单卡、单活跃请求、持久 Engine 的顺序请求 seed，不表示并发 batch 隔离通过；不开始正式 probing 或 Noise Head 训练。关闭 prefix cache 的结果不能替代开启缓存时的复现或吞吐证据。

## 产物与验证

产物目录 `/kaggle/working/artifacts/request_seed_smoke_<时间>/`。`input.json` 记录实际输入、模型与源码身份；`prompt.json` 记录实际模板；每个请求目录保存 events、features、trajectory 和 result；`smoke_test.json` 保存三请求综合验收，异常写 `failure.json`。显存包括整个GPU0采样最大值与 worker 从 Engine 启动以来的累计 PyTorch peak；生成耗时包含 hook，不作为正式采集吞吐预算。

CPU 测试覆盖 metadata relay、请求首次重置/后续推进、单 Engine 生命周期与异常关闭、fresh-process overlay 导入、来源文件不修改和 Notebook 环境源码保留。未加载本地 checkpoint、运行 CUDA Engine 或修改虚拟环境。缺失输入、环境/设备错误及 OOM 立即停止，无自动重试。

```sh
python -m pytest tests/test_request_seed_smoke.py -q
```
