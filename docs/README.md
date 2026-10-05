# 正式文档索引

更新：2026-10-05；当前阶段：用户已提供 GRPO K0 PASS 输出；持久 Engine 请求级 seed smoke 已实现，待 Kaggle GPU 验收。
返回 [项目入口](../AGENTS.md)。

| 阅读顺序 | 文档 | 用途 |
| --- | --- | --- |
| 1 | [项目守则](AGENTS.md) | 用户要求遵守的原始模板，保留全文 |
| 2 | [子 Agent 协议](AGENT_USE.md) | 委派门禁、文件所有权和根 Agent 验证责任 |
| 3 | [任务规范](AdaNoise_LGRPO_Codex_Kaggle_Spec.md) | 方法、预算、实验与验收要求 |
| 4 | [当前 minfix 审计](minfix_submodule_update.md) | 新 submodule、streaming sampler 和本地修改边界 |
| 5 | [本地 smoke 设计与实现计划](local_smoke_implementation_plan.md) | 任务接口、测试与 Kaggle 阶段门禁 |

`docs/AGENT_USE.md` 已读取，本次实现由主 Agent 顺序完成，一个只读 review Agent 复核。本地 CPU smoke 范围已获确认，后续实验设计先与用户交流；遇到缺失必需文件或输入立即停止。
工作恢复状态在 `progress/`，不属于正式文档树。

[旧作者仓库审计](adanoise_repo_audit.md) 已归档，仅作为历史取证。

[本地 CPU smoke 命令与证据](local_cpu_smoke.md)：Task 1验收和适用边界。

[已确认两阶段训练与共享题集](training_route.md)：Stage B纳入首版；两阶段共享train题目、分别生成轨迹，validation/test隔离。

[数据集构建脚本设计](dataset_construction_design.md)：首轮两来源各抽160题、合计240 train / 80 validation，固定切分后probing筛选；本地脚本已实现，模型采集尚未执行。

[数据CPU实施计划](dataset_cpu_implementation_plan.md)：当前会话顺序实施数据构建、probing控制与轨迹选择CPU验收。

[数据CPU验收与命令](dataset_cpu_acceptance.md)：95项全测试、真实240/80候选池、4160请求及合成轨迹选择的证据和边界。

[独立 Engine seed 检测 Notebook](../notebooks/rand_seed_test.ipynb)：`sample_once(prompt, seed)` 使用已确认 Kaggle 模型路径，在 T4 GPU 0 上以 fp16、tp=1 运行；每次通过 `Engine(random_seed=seed)` 重建引擎后生成并关闭。需要预先可用的作者自定义 SGLang/FlashInfer；不安装依赖。保留用户附件的 temperature=1、max_new_tokens=64，仅用于 seed 诊断。两项 CPU mock 测试验证调用传参和异常清理，未执行 GPU。PASS 仅表示独立 Engine 的解码文本复现，不能替代持久 Engine 请求级 seed 和 latent/hidden 对齐验收。

[GRPO checkpoint Kaggle K0 smoke](kaggle_k0_smoke.md)：以用户跑通Notebook为基础，保留环境与clone逻辑；单题三次独立Engine，核验启动seed及逐步hidden/clean top10/mixture对齐。新GRPO路径取代旧SFT模型，用户已提供 GPU PASS 输出。

[持久 Engine 请求级 seed smoke](kaggle_request_seed_smoke.md)：基于最新附件，保留 clone/环境/Bash 构建运行逻辑，删除固定 commit 与源码 SHA 门禁；单 Engine 三个顺序请求，关闭 prefix cache，待 GPU 验收。
