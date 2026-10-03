# AdaNoise-LGRPO 项目入口

项目目标：在作者已训练的 1B Latent-GRPO checkpoint 上冻结 backbone，仅训练 Noise Head；主要实验使用 Kaggle 单 GPU Notebook。
更新：2026-10-03；当前阶段：CPU sampler及数据构建/probing控制已验收，真实模型尚未运行。

## 读取路线

先读 [项目守则](docs/AGENTS.md)，再读 [正式文档索引](docs/README.md)、[任务规范](docs/AdaNoise_LGRPO_Codex_Kaggle_Spec.md) 和当前任务文档。
长任务恢复再读 `progress/INDEX.md` → `progress/current.md` → 被索引的相关笔记。
实施和协作先读 [子 Agent 协议](docs/AGENT_USE.md)。数据CPU任务由主Agent顺序实施，一个只读review Agent复核；后续按协议决定委派范围。

## 工作边界

- 数据构建、probing请求控制与离线选择已实现，见 [数据CPU验收](docs/dataset_cpu_acceptance.md)；实际GPU采集器仍待Kaggle阶段。
- 本地 CPU sampler smoke 已实现；后续实施按 [smoke 设计与计划](docs/local_smoke_implementation_plan.md) 的前置条件执行。
- 首版训练使用监督warm start＋head-only policy gradient，共享train题集但RL重新采样，validation/test隔离；详见 [训练路线](docs/training_route.md)。
- 只比较 validation 选定的 Fixed-Gumbel 与 AdaNoise；禁止 backbone/LoRA/GRPO/PPO 训练。
- 本地张量级 smoke 使用 CPU；真实 latent inference 先在 Kaggle 单 GPU smoke 验收。
- 后续实验设计必须先与用户交流；不得根据工程判断静默修改实验方案。
- 缺失必需文件或输入立即停止并报告，不继续猜测或绕过；checkpoint 路径由用户提供，禁止自动下载、安装或更改指定虚拟环境。
- 非项目环境/网络/设备错误及 OOM 立即停止实验并报告；不自动按 Spec 的 OOM 降级条目重试。

## 目录与版本控制

`latent_grpo_minfix/Latent-GRPO-final/` 是用户修订仓库，作为固定 commit `0b7e85f` 的 submodule 引用；不更改其 origin，不提交权重、数据、缓存和运行产物。
`docs/` 维护正式事实、规范和计划；`progress/` 保存本地恢复状态，默认不提交，不通过全量 git add 混入。
后续源码修改应形成可复现补丁或独立包，不在作者仓库内制造无法由主项目复现的本地修改。
任何已有用户改动都保留；禁止强推和破坏性 Git 操作。

## 当前可用检查

```sh
python /path/to/r-doc/scripts/audit_docs.py --root . --strict
python -m pytest -q
python scripts/smoke_local.py --config configs/local_smoke.yaml --device cpu
git diff --check -- AGENTS.md README.md docs adaptive_noise scripts tests configs
```
`/path/to/r-doc` 替换为当前安装技能位置。本地 smoke 使用说明见 [Task 1验收](docs/local_cpu_smoke.md)；Kaggle/Noise Head命令仍是待实现接口。
