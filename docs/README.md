# 正式文档索引

更新：2026-10-02；当前阶段：本地 CPU sampler smoke 已实现；真实模型尚未执行。
返回 [项目入口](../AGENTS.md)。

| 阅读顺序 | 文档 | 用途 |
| --- | --- | --- |
| 1 | [项目守则](AGENTS.md) | 用户要求遵守的原始模板，保留全文 |
| 2 | [子 Agent 协议](AGENT_USE.md) | 委派门禁、文件所有权和根 Agent 验证责任 |
| 3 | [任务规范](AdaNoise_LGRPO_Codex_Kaggle_Spec.md) | 方法、预算、实验与验收要求 |
| 4 | [当前 minfix 审计](minfix_submodule_update.md) | 新 submodule、streaming sampler 和本地修改边界 |
| 5 | [本地 smoke 设计与实现计划](local_smoke_implementation_plan.md) | 任务接口、测试与 Kaggle 阶段门禁 |

`docs/AGENT_USE.md` 已读取，当前任务不启用子 Agent。本地 CPU smoke 范围已获确认，后续实验设计先与用户交流；遇到缺失必需文件或输入立即停止。
工作恢复状态在 `progress/`，不属于正式文档树。

[旧作者仓库审计](adanoise_repo_audit.md) 已归档，仅作为历史取证。

[本地 CPU smoke 命令与证据](local_cpu_smoke.md)：Task 1验收和适用边界。

[已确认两阶段训练与共享题集](training_route.md)：Stage B纳入首版；两阶段共享train题目、分别生成轨迹，validation/test隔离。

[数据集构建脚本设计](dataset_construction_design.md)：已确定题源与不再去重边界；切分与配比建议待确认，脚本尚未实现。
