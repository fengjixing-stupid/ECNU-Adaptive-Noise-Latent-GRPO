---
id: DOC-DATA-PLAN-001
type: plan
status: active
title: 数据构建与 probing 控制器 CPU 实施计划
created: 2026-10-03
updated: 2026-10-03
related_docs:
  - DOC-DATA-001
---
# 数据构建与 probing 控制器 CPU 实施计划

> **For agentic workers:** 使用 superpowers:executing-plans 在当前会话顺序实施，遵循TDD。用户已明确要求实施计划与CPU验收；保留当前feat/local-cpu-smoke分支，review使用独立只读Agent。

**Goal:** 输出可复现的240/80候选题集，提供可验收的递增probing请求控制、作者评分和最佳scale轨迹选择接口。

**Architecture:** 数据构建不加载模型；probing控制器消费已完成rollout记录，输出待采样请求或终态决定。选择脚本只复制最佳scale的既有完整轨迹及特征，实际GPU采集器留待Kaggle阶段。

**Tech Stack:** 指定Python虚拟环境，标准库、PyArrow、PyYAML、pytest；不安装依赖。

**Spec:** [已批准设计](dataset_construction_design.md)。返回 [文档索引](README.md)。

## Global Constraints

- data/master_seed=42，三个派生seed；GSM8K-Aug与DAPO各抽160，各切120/40。
- probe/root_seed=12345，scale=[0,.25,.5,1,2]；s=0始终一次，非零累计3/5/7，delta严格>0.4。
- train全对/全错剔除，7次仍不敏感剔除并回查；validation所有终态保留且标记。
- 最佳scale正确率并列取较小scale，保留其全部正确/错误轨迹及特征；其他保留轻量摘要。
- 缺输入或环境/网络/OOM立即停止；不改submodule、不加载模型、不运行GPU实验、不提交数据或progress。

## Review Focus

- 来源内部split错误不能让Math-500进入训练。
- 缺失/重复/越过终态的rollout不能算作完整阶段；恢复s=0不重复。
- delta=0.4的数值边界与全对/全错优先级。
- 缺最佳scale特征不得生成可训练产物；来源轨迹文件必须只读。
- 判定异常不可吞成普通奖励0；可评分截断仍保留正确性。

### Task 1: 数据与seed

**Files:** adaptive_noise/seeds.py、dataset_builder.py；scripts/build_dataset.py；configs/dataset_build.yaml；tests/test_dataset_builder.py。
**Interfaces:** derive_dataset_seeds(master_seed)->dict；derive_rollout_seed(problem_id,scale,rollout_id,root_seed=12345)->int；build_dataset(config:dict,root:Path)->dict(manifest)。

- [x] 写fixture验证三路seed固定值、源重复保留、160抽样/120-40切分、test内部train覆盖、输入缺失和已有输出拒绝、同输入可复现。
- [x] 运行目标pytest确认RED（接口未实现）。
- [x] 实现流式字段检查、输入内容hash/稳定ID、抽样和切分、完成manifest及拒绝记录；CLI只处理数据。
- [x] 目标pytest GREEN后运行真实四文件CPU构建，并核对240/80/1319/500及完整统计。

### Task 2: probing控制与作者评分

**Files:** adaptive_noise/probing.py、answer_verifier.py；tests/test_probing.py、test_answer_verifier.py。
**Interfaces:** plan_requests(problem_id,split,source,rows)->list[request]；decide_probe(rows,split)->dict；verify_answer(source,text,ground_truth,finish_reason=None)->dict。

- [x] 写3→5→7累计、s=0不追加、部分阶段恢复、重复/缺失/超阶段拒绝、四类终态/train-val差别、0.4边界及最佳scale平手测试，确认RED。
- [x] 实现纯CPU状态机；seed按已批准公式；评分只加载固定submodule的允许纯函数AST，绕过GPU依赖，不执行作者入口。
- [x] 以实际作者函数对照正确、错误、无答案、boxed/fallback、截断、缺来源与缺必需参考文件，GREEN。

### Task 3: 选择CLI与验收

**Files:** scripts/select_probe_dataset.py、adaptive_noise/probe_selection.py；tests/test_probe_selection.py；docs/dataset_cpu_acceptance.md。
**Interfaces:** select_probe_dataset(candidate_dir:Path,rollouts_path:Path,output_dir:Path,trajectory_root:Path)->dict。

- [x] 用合成完整rollout/feature文件验证：按候选清单路由、所有题有终态、最佳scale全部轨迹复制（含错误）、非最佳仅轻量记录、剔除ID回查、缺特征/seed不符/未知ID/不完整拒绝；确认RED。
- [x] 实现CLI和选择产物；原始轨迹只读，输出拒绝覆盖，异常不发布完成manifest。
- [x] 全pytest、本地sampler smoke、r-doc严格审计、diff检查；记录真实构建与合成控制器验收，清楚区分尚无GPU采集。
- [x] 主Agent集成独立review；修改用RED→GREEN验证，选择性提交并push，progress保持本地。

## 执行结果
2026-10-03：三项任务实现并CPU验收，95 tests通过，真实候选240/80与4160请求通过，sampler38通过；独立review两问题均回归修复且复查通过。新增plan_probe_requests.py用于输出本地状态机请求，接口仍不执行模型。详细证据见 [CPU验收](dataset_cpu_acceptance.md)。
