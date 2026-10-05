---
id: DOC-PROBING-PLAN-001
type: plan
status: active
title: Kaggle 正式 probing 入口实施计划
created: 2026-10-05
updated: 2026-10-05
related_docs:
  - DOC-DATA-001
  - DOC-REQUEST-SEED-001
---
# Kaggle 正式 probing 入口实施计划

> 执行：主 Agent 顺序实施（executing-plans），完成后一个只读 review Agent 复核。

**Goal:** 将已通过请求 seed 验收的 Notebook 接入固定子池、自适应 probing、恢复与8小时单次上限。

**Architecture:** 新控制模块复用 seeds/probing/selection；GPU入口复用已验收 metadata/capture overlay。Notebook嵌入所需主项目源码与锁定行号清单，保留原clone/环境/Bash构建运行方式。只携带题目身份元数据，原题与权重仍由Kaggle挂载读取。

**Tech Stack:** Python、PyArrow、作者自定义 SGLang、PyTorch、T4 GPU0。

**Spec:** [已确认数据方案](dataset_construction_design.md)；用户确认正式入口和单次不得超过8小时。

## 约束与验证重点

- 固定240 train/80 validation，不重新抽样、不读test作训练、不改scale或生成长度。
- 一个Engine、单活跃请求、关闭prefix cache；请求首次设置worker seed，同请求后续推进。
- 逐rollout原子提交；恢复保留已完成结果，s=0不追加，未完成不计R=0。
- 单次8小时含准备/启动/采集；提前预留关闭时间，外层进程组超时终止兜底，无环境/OOM重试。
- 终态保留最佳scale全部轨迹；其余仅摘要，删除范围限本运行生成的临时features/events。

## Task 1：固定子池与持久控制

Files: `configs/kaggle_candidate_rows.json`（来源hash/行号/split元数据）、`adaptive_noise/kaggle_probing.py`、`tests/test_kaggle_probing.py`。

- [x] RED：锁定行号重建一致，源文件不符拒绝；全对/全错/敏感/封顶与80题validation保留。
- [x] 实现固定行号载入、逐rollout完成记录、恢复、最佳scale临时记录清理和最终选择。
- [x] 验证中断不计reward、恢复不重复s=0、未到终态不发布最终选择、模型身份一致。

## Task 2：GPU采集与时间上限

Files: `scripts/kaggle_probe.py`、`tests/test_kaggle_probe_runtime.py`。

- [x] RED：多题/多scale动态身份与low/high评分、单Engine、截止停止、超时终止整个进程组。
- [x] 将已验收capture推广为run key；写features/trajectory后原子提交完成记录，按题调用既有控制器。
- [x] 实现8小时总预算、提前停止接收请求、父进程硬超时，CPU fake Engine/时钟验证；不执行GPU。

## Task 3：Notebook与交付

Files: `notebooks/ecnu-smoke-adaptive-latent-grpo.ipynb`、`docs/kaggle_probing.md`、相关索引、`progress/current.md`。

- [x] Notebook保留setup源码，在Bash构建阶段写入独立运行bundle；默认输出固定collection目录，续跑指向已有输出。
- [x] 测试Notebook与源码同步，无固定Git/source SHA相等门禁；记录来源hash与实际输入身份。
- [x] 全套CPU测试、文档strict审计、只读review、选择性commit/push；progress和用户submodule改动不提交。

真实GPU运行由用户执行；单题热启动计时不作为完整DAPO吞吐估计。正式入口保存按来源/scale统计与已完成请求耗时，供后续预算判断。
