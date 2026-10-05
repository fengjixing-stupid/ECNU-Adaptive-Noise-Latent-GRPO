---
id: DOC-K0-001
type: testing
status: active
title: Kaggle GRPO checkpoint latent 与 hidden smoke
created: 2026-10-04
updated: 2026-10-05
related_docs:
  - DOC-PLAN-001
  - DOC-DATA-001
  - DOC-DATA-CPU-001
related_code:
  - scripts/kaggle_latent_smoke.py
  - notebooks/kaggle_latent_smoke.ipynb
  - tests/test_kaggle_latent_smoke.py
---
# Kaggle GRPO checkpoint latent 与 hidden smoke

返回 [文档索引](README.md)。本页记录已获用户确认的 K0 范围；用户于2026-10-05提供真实 GPU 的 K0 PASSED 输出：同 seed 的 token/trace/hidden 一致，不同 seed 的 trace/hidden 改变，均31个混合 latent step、hidden2048、答案3。依据用户粘贴输出验收，未下载或本地读取 raw artifacts。后续使用[持久 Engine 请求级 seed smoke](kaggle_request_seed_smoke.md)，其中已按用户要求移除固定 commit/source SHA 门禁；本页其余内容描述历史 K0 脚本。

## 输入与运行方式

以用户的 `notebook301c1d6406 (1).ipynb` 为基础。环境准备、git clone、依赖安装及版本检查 cells 1–8 的源码完全保留；仅 cell 9 生成的 Python 脚本改为本次 smoke。cell 10 仍使用原命令重新运行脚本；所有旧执行输出与执行计数清空，交付未执行的Notebook，不发布历史运行产物。其余单元格源码（包括旧 seed 检查的 Markdown）原样保留，当前结果含义以生成的 `smoke_test.json` 和本页为准。

用户已跑通环境时，执行 cell 9 即可；无需为了此实验再次运行环境安装单元格。环境准备逻辑由用户保留，smoke 脚本自身不会安装、下载、自动重试或更改虚拟环境。`%%bash` → heredoc 构建 `/kaggle/working/test_sglang_seed.py` → 指定 Python 执行的逻辑保持不变。cell 10 是可选的再次运行，会创建新的产物目录。

| 输入 | 已确认值 |
| --- | --- |
| 模型 | `/kaggle/input/models/fengjixing/llama3-2-1b-instruct-latent-grpo-top10/other/default/1/LLaMA3.2-1B-Instruct-Latent-GRPO-Top10` |
| 数据 | `/kaggle/input/datasets/fengjixing/latent-grpo-data/data/GSM8k-Aug-oss-dup-all.parquet` |
| 设备 | T4 GPU 0、FP16、tp=1、max_running_requests=1 |
| 题目 | 固定候选池 train 的 GSM8K source_row=7335；仅1题，不重新抽样或切分 |
| 生成 | temperature=.6、top_p=.95、max_new_tokens=128、max_topk=10、Gumbel temperature=1、scale=1、双侧噪声启用 |

脚本核验源数据 SHA256 与已固定候选池一致，再读该行的原始 `prompt` 和 `ground_truth`。使用 checkpoint 自身 chat template，加 generation prompt，缺 `<think>` 时追加一次。`</think>` 的第一个 token ID 作为作者约定的退出标记，完整 marker IDs 写入 prompt 产物，不能误换为最后一个 ID。

## 三次运行与采集方式

三次独立 Engine：`same_a`、`same_b` 共享 rollout_id=0 派生 seed，`different` 使用 rollout_id=1 派生 seed。seed 算法沿用 root seed=12345 与已批准 problem_id/scale/rollout_id SHA256 协议；不复用旧 SFT 的诊断 seed。每次 Engine 的启动 seed 传入实际 worker；保持用户已验证的 Engine 创建后再创建 asyncio loop 的顺序。

原作者版本不接受 `generate(seed=...)`。这次只验证独立 Engine 启动 seed，不宣称持久 Engine 的逐请求 seed 已通过。

脚本复制已安装的自定义 SGLang 到产物目录的 `runtime_overlay/`，在副本 sampler 追加采集 hook。主项目不修改作者 checkout 或安装目录。新进程与 Engine spawn workers 继承 overlay 的 PYTHONPATH。复制前核验整个原 SGLang Python 源码树的 SHA256，以及关键文件与评分代码身份，均来自固定参考 commit `0b7e85f`；不因用户 clone 到默认分支就假定源码一致。源码不同即 STOP，不自动切分支或修补环境。

hook 在同一 sampler 调用前读取 LAST hidden 和加噪前 logits 的 clean top10（K 内 softmax 归一化），调用原 sampler 后读取实际 sparse mixture 与代表 token。只立即复制 hidden 向量/K 宽数据到 CPU/磁盘，不保存完整词表或 GPU 历史，不额外消费随机数。退出标记那一步由 scheduler 转为显式 one-hot，不计为实际混合 latent step。

## 验收与输出

输出目录 `/kaggle/working/artifacts/k0_smoke_<时间>/`，已有目录拒绝覆盖。

| 产物 | 用途 |
| --- | --- |
| `input.json` | train题目、输入/模型文件hash、模型身份、源码与hook身份 |
| `<run>/prompt.json` | 实际模板文本、输入token IDs和latent退出marker IDs |
| `<run>/events/*.jsonl` | 每次采样的worker CPU/CUDA seed、token、有限特征及资源记录 |
| `<run>/features.json`、`trajectory.json` | raw-latent-v1 特征与对齐稀疏轨迹，配置包含max_topk=10 |
| `<run>/result.json`、`smoke_test.json` | 单次检查与三次综合结果；错误另写`failure.json` |

结构检查要求：batch=1、至少一个真实 mixture step、hidden 维度符合模型config且有限、top10概率有限非负和为1、step连续、事件token序列等于Engine返回IDs、代表token与mixture首项一致、同一worker且CPU/CUDA初始seed符合请求。退出与预算命中独立记录。

同 seed 两次的 token、mixture trace 和 hidden 特征须一致。不同 seed 的实际 trace 须改变；最终答案文本相同不判失败。不同 seed trace 仍相同时写 `inconclusive` 并停止，不作seed已生效的声明。答案使用固定作者纯评分helpers，正确率只作本题诊断，不作为工程smoke通过门槛。

显存记录来自实际采样worker的PyTorch peak allocated/reserved，以及每0.5秒采样的整个GPU0显存使用；后者是采样最大值，不称精确峰值，也不称单个进程显存。耗时区分启动与带hook的生成；三次诊断运行不是正式probing吞吐测量。

缺输入/依赖、源码不符、设备/网络/环境故障、OOM及模型运行异常均直接停止。不会增加样本、降低参数、安装依赖或自动开始4160条正式probing。

## 本地验证范围

CPU测试验证token/hidden/mixture对齐、错误拒绝、hook不额外改变RNG、fresh进程overlay传递且原源码不变，以及产物配置与现有选择器一致；还核对Notebook嵌入脚本与CLI相同、原环境单元格source hash未变。未在本地加载1B、执行CUDA Engine或更新指定环境。

```sh
python -m pytest tests/test_kaggle_latent_smoke.py -q
```

持久 Engine 请求级seed、完整正式采集器、吞吐预算、Noise Head训练仍为后续任务。本次三个smoke轨迹是诊断产物，不追加进入正式probing记录。
