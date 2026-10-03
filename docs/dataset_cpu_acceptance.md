---
id: DOC-DATA-CPU-001
type: guide
status: active
title: 数据构建与 probing 控制器 CPU 验收
created: 2026-10-03
updated: 2026-10-03
related_docs:
  - DOC-DATA-001
  - DOC-DATA-PLAN-001
---
# 数据构建与 probing 控制器 CPU 验收

返回 [文档索引](README.md)。设计见 [数据构建方案](dataset_construction_design.md)，实施见 [CPU实施计划](dataset_cpu_implementation_plan.md)。

## 已完成范围

本地实现数据构建、稳定seed、递增probing请求规划、作者答案判定、离线选择与最佳scale轨迹打包。真实四个parquet已完成CPU扫描和候选构建；模型采集器、逐轨迹seed传入GPU引擎、Noise Head训练仍未实现/运行，不将本验收作为方法有效性证据。

主Agent顺序TDD；一个独立只读review Agent检查实现并复查。review发现摘要漏存truncated及轨迹seed未对齐，均通过RED→GREEN回归修复；另补齐既有生成配置对齐要求。复查未发现新增重要缺陷。

## 本轮证据（2026-10-03）

| 检查 | 实际结果 |
| --- | --- |
| 全项目pytest | 95 passed，包含52项新增数据/控制器/评分/选择检查；6.00秒 |
| 既有CPU sampler smoke | 38 passed，0 failed/errors/skipped |
| 真实输入基础检查 | GSM8K-Aug 301852、DAPO 14116、两个test 1319/500；无效记录0；不去重 |
| 真实候选池 | 两来源各120 train/40 validation，共240/80；test仍1319/500 |
| 请求规划与合成选择 | 4160条初始请求，s=0恰320条；合成选择68条rollout→12条最佳scale轨迹，1 train保留/2 validation保留 |

合成选择的model_identity为 `synthetic_cpu_fixture`，只是接口验收。真实候选manifest记录输入哈希、行号、seed、配置、Python/PyArrow版本及输出哈希。初始请求包含3120条train和1040条validation，每条seed核对通过。

本地产物（不提交Git）：

- `artifacts/dataset_candidates_v1/`：train/validation、两个完整test、split/build manifest和rejected。
- `artifacts/probe_requests_v1/`：requests.parquet与request_manifest.json，只规划，没有执行。
- `artifacts/dataset_cpu_smoke/full_suite.junit.xml`：95项完整测试报告。
- `artifacts/dataset_cpu_smoke/synthetic/selected/`：标注合成模型的选择/摘要/12条轨迹与特征；CLI日志在synthetic/。
- `artifacts/local_smoke.json`：原sampler smoke报告。

## 使用入口

在项目根目录，用用户指定的vLLM虚拟环境解释器执行下列命令；下文python代表该解释器。构建和规划已运行，重复验收必须使用新输出目录，不覆盖已有产物。

```sh
python scripts/build_dataset.py --config configs/dataset_build.yaml --output-dir artifacts/dataset_candidates_recheck
python scripts/plan_probe_requests.py --candidate-dir artifacts/dataset_candidates_recheck --output-dir artifacts/probe_requests_recheck
```

已有完成rollout时，规划缺失请求：

```sh
python scripts/plan_probe_requests.py --candidate-dir artifacts/dataset_candidates_v1 --rollouts artifacts/collection/probe_rollouts.parquet --output-dir artifacts/probe_requests_next
```

已有完整终态rollout和必需轨迹/特征时，选择与打包：

```sh
python scripts/select_probe_dataset.py --candidate-dir artifacts/dataset_candidates_v1 --rollouts artifacts/collection/probe_rollouts.parquet --trajectory-root artifacts/collection --output-dir artifacts/probe_selected
```

`artifacts/collection/` 是未来Kaggle采集器的输入接口，本地真实collection尚不存在；缺文件时脚本直接STOP，不采样、不自动下载。选择后Stage A只消费train_selected的最佳scale轨迹；validation_marked完整保留；Stage B使用同题的新on-policy rollout。

## 输入记录与特征契约

规划请求含problem_id、source、split、scale、rollout_id、seed，以及作者生成配置。完成记录至少包含上述身份字段、status=completed、二元reward；轻量永久摘要另保留invalid_reason、truncated、finish_reason、extracted_answer。选择输入还需trajectory_path和features_path，为trajectory_root内的JSON相对路径。

两个JSON均必需schema_version=1、problem_id、scale、rollout_id、seed、model_identity、generation_config；必须与完成记录及批准配置对齐。相同模型身份和hidden维度是本次选择的前置条件。完整轨迹含output_text、explicit_token_ids、reward和steps；每个step含连续step_idx及稀疏top10 IDs/mixture_probs，不存完整词表。特征含feature_schema=raw-latent-v1、feature_source=pre_gumbel及对齐steps，每步为hidden_state和归一化clean topk_probs（10项）。向量必须有限，概率非负且和为1。

选择只复制最佳scale的全部已有轨迹，不删除源输入；其他scale的输入路径不进入永久摘要，保留正确率/invalid诊断。Kaggle采集器之后需按本契约临时保存候选特征，终态后仅发布最佳scale完整轨迹。本地选择器不能从缺失hidden的作者原eval输出自动生成warm-start特征。

## 停止与验收命令

缺必需文件/字段、未知ID或split、seed/生成配置不符、阶段不完整、重复rollout、终态后额外采样、缺最佳scale特征时STOP，不发布complete manifest。已有输出拒绝覆盖；其他写入异常可能留下未完成目录，使用新目录重跑，不自动清理。

```sh
python -m pytest -q
python scripts/smoke_local.py --config configs/local_smoke.yaml --device cpu
python /path/to/r-doc/scripts/audit_docs.py --root . --strict
```

Kaggle执行前还必须验证请求seed到实际引擎随机采样消费者的传递、raw-latent-v1采集对齐和实测吞吐；这三项本地CPU验收不作通过声明。
