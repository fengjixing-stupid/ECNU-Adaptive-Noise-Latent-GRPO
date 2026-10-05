---
id: DOC-KAGGLE-PROBING-001
type: guide
status: active
title: Kaggle 正式 probing 与8小时续跑入口
created: 2026-10-05
updated: 2026-10-05
related_docs:
  - DOC-DATA-001
  - DOC-REQUEST-SEED-001
  - DOC-PROBING-PLAN-001
related_code:
  - scripts/kaggle_probe.py
  - adaptive_noise/kaggle_probing.py
  - configs/kaggle_candidate_rows.json
  - notebooks/ecnu-smoke-adaptive-latent-grpo.ipynb
  - tests/test_kaggle_probing.py
  - tests/test_kaggle_probe_runtime.py
---
# Kaggle 正式 probing 与8小时续跑入口

返回[文档索引](README.md)。用户已提供持久Engine请求seed PASS输出，确认跳过单独DAPO smoke、将当前Notebook改为正式入口，并指定单次不得超过8小时。已实现控制与采集入口；正式GPU probing尚未运行，不表示已经选出训练题集。

## 运行方式

`notebooks/ecnu-smoke-adaptive-latent-grpo.ipynb` 保留原 cells 1–8 的 clone/环境准备/安装源码；原固定checkout/source SHA门禁继续删除。cell9在Bash中写入主项目运行bundle，heredoc构建 `/kaggle/working/test_sglang_seed.py`，再使用指定虚拟环境启动正式采集。源码及锁定行号嵌入Notebook，与CLI同步，不依赖clone是否已更新；未嵌入原始题目、数据或权重。旧请求seed诊断另保存在 `notebooks/kaggle_request_seed_smoke.ipynb`。

环境已就绪时只运行cell9。cell10用于同一输出目录续跑，目录缺失会停止。脚本自身不安装/下载/更新环境，不自动重试错误；已有已完成集合会直接报告complete，不重复运行模型。

```sh
ADANOISE_PROJECT_ROOT=/kaggle/working/adanoise_probe_bundle \
/kaggle/working/latent_sglang/bin/python /kaggle/working/test_sglang_seed.py \
  --output-dir /kaggle/working/artifacts/probing_v1 --hours 8
```

Kaggle `/kaggle/working` 不自动跨session保留。新session续跑前必须保存并恢复完整 `probing_v1/` 到该路径，包括candidates、locked_pool、collection_identity及rollouts。只恢复摘要parquet不足以续跑或构建warm-start输入；旧sessions/attempts也建议随目录保存。cell9在目录不存在时创建新集合，因此跨session缺失旧结果时不能把它当续跑。环境与输入错误或OOM发生后立即停止，由用户处理问题再明确运行。

## 已锁定输入与实验设置

| 输入/设置 | 值 |
| --- | --- |
| checkpoint | `/kaggle/input/models/fengjixing/llama3-2-1b-instruct-latent-grpo-top10/other/default/1/LLaMA3.2-1B-Instruct-Latent-GRPO-Top10` |
| 数据目录 | `/kaggle/input/datasets/fengjixing/datasets-for-latent-grpo/data`，需要 `GSM8k-Aug-oss-dup-all.parquet` 与 `DAPO-Math-17k-en-train.parquet` |
| 子池 | 固定GSM/DAPO各120 train、40 validation，共240/80；不读取test作训练、不重新抽样、不去重 |
| Engine | 单次最多1个持久Engine，T4 GPU0、fp16、tp=1、max_running_requests=1、关闭prefix cache/CUDA graph/overlap；static memory fraction=.90 |
| probing | scales `[0,.25,.5,1,2]`，s=0一次，其余累计3→5→7；GSM上限128、DAPO4096，其余作者配置不变 |

`configs/kaggle_candidate_rows.json` 只保存本地已完成子池的source_row/split、源文件hash与原manifest身份。按锁定行号从Kaggle原文件恢复相同prompt和标签，并完整写入私有暂存目录后原子发布candidates；没有根据当前Python随机数实现重新生成分割。

每个请求沿用root12345/problem_id/scale/rollout_id派生seed，通过custom_params/metadata overlay传到实际采样worker，在首次sampler调用设置CPU/CUDA RNG。同一请求后续token推进，不重置。每次采集要求worker保持一致、request ID不同，检查CUDA seed、token与hidden/clean top10/mixture对齐。模型身份、子池和生成配置绑定集合；实际安装源码与评分hash仅记录来源，不比较固定Git/source版本。评分分别使用当前作者low/high纯helpers，不静默改verifier。

## 8小时上限与持久化

从正式Python入口开始计时，准备数据/源码/模型身份、Engine启动/JIT、采集、筛选与导出均受父进程监督。默认8小时，可传更短的 `--hours`，禁止超过8。子进程在截止前300秒停止发新请求；在途请求或准备/关闭阻塞时，父进程在截止前30秒终止整个采集进程组，TERM最多等待10秒后KILL，再等待最多10秒。Engine workers与准备进程属于同一受监督进程组。

逐rollout先写私有attempt中的features/trajectory，再写完成记录，原子rename到 `rollouts/<稳定请求key>/` 才计为完成。中断attempt没有完成记录，不计R=0；续跑只规划缺失身份，已完成s=0永不追加。完成目录发布与临时events删除之间发生中断时，恢复会补清理该请求的events。

每题完整阶段遵循已确认判定：train全对/全错剔除，delta严格>0.4保留，仍不敏感则累计到7封顶剔除并记录ID；validation相同控制但所有终态均保留并标记。最佳scale按q_hat最大、并列较小scale选取，保留该scale全部正确/错误轨迹。终态后删除其他scale本集合生成的features/trajectory/prompt，仅保留轻量完成记录。采集期间不在GPU积累历史，不保存完整中间词表。

缺少必需非空latent特征时按现有warm-start合同STOP；不自动为零latent轨迹发明新的训练规则。环境/设备/网络故障、OOM、评分代码错误也STOP；正常答案错误或不可提取最终答案按批准规则记R=0及invalid分类。

## 产物与结果含义

| 路径（相对 probing_v1） | 含义 |
| --- | --- |
| `locked_pool.json`、`candidates/`、`collection_identity.json` | 固定输入与本集合身份；续跑必需 |
| `rollouts/<key>/completion.json` | 逐条原子完成记录，含reward、seed、终止/invalid、生成耗时及worker累计显存；最佳scale保留features/trajectory |
| `probe_rollouts.parquet`、`decisions.json`、`collection_status.json` | 可重建快照；source/scale正确率、invalid率、实测平均耗时；不完整阶段不作终态决定 |
| `sessions/<id>/`、`watchdog.json`、`failure.json` | 本次安装来源、模型身份、overlay、资源与时间；硬截止见watchdog，异常见failure，不自动换参数 |
| `selected/` | 所有320题终态后原子发布：train_selected、validation_marked、selection_decisions、probe_summary、轻量rollouts及最佳scale完整轨迹 |

未完成时status为budget_stopped/running/failed，不发布最终selected。已完成320题时才发布selection_manifest，记录实际train剩余数量，validation仍80题；不补抽题目或放宽筛选。导出中断留下私有exports暂存目录，恢复可重新完整导出，不将其当成功selected。

显存记录包括采样worker自Engine启动以来的PyTorch累计peak与整个GPU0的0.5秒采样最大值。耗时包含采集hook；首轮JIT与后续生成分开记录。该运行的source/scale统计供后续预算和用户判断s=2表现；脚本不自动切换保守scale集合或调整训练设计。

## CPU验收

测试覆盖锁定池恢复、阶段控制与validation保留、跨运行不重复s=0、失败不计reward、候选/结果/导出原子发布及恢复、非最佳临时数据清理、动态low/high评分与特征身份、单Engine生命周期、OOM直接停止、准备前父监督、真实CPU进程超时和Notebook bundle同步。

```sh
python -m pytest tests/test_kaggle_probing.py tests/test_kaggle_probe_runtime.py -q
```

2026-10-05完整项目测试 **131 passed（10.28秒）**，r-doc strict审计0错误/0警告，Notebook fresh-process bundle导入与环境源码保留检查通过；独立只读复查无剩余阻断问题。进程组超时检查使用忽略SIGTERM的CPU子进程心跳文件，确认超时后停止写入，不依赖本地受限的ps命令。

CPU测试和已有单题seed GPU证据不能代替正式320题GPU运行结果；本次没有在本地加载checkpoint、运行CUDA或修改环境。
