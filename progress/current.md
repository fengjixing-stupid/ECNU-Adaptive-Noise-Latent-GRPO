# 当前恢复点

2026-10-03：DOC-DATA-PLAN-001三任务已实现，95 tests通过（本轮新增52），sampler smoke38通过。正式证据 docs/dataset_cpu_acceptance.md；批准设计 docs/dataset_construction_design.md。
真实CPU四文件完整扫描有效301852/14116/1319/500，无效0、不去重；候选各来源120train/40val，总240/80，保留test1319/500。产物artifacts/dataset_candidates_v1，split在probing前固定。
artifacts/probe_requests_v1含4160初始请求，3120train/1040val，s=0恰320；只规划未执行。合成选择68rollout→12最佳scale轨迹、1train保留/2val完整保留，明确synthetic_cpu_fixture。
本轮TDD任务1 RED→6GREEN，任务2 RED→28GREEN，任务3 RED→10GREEN；review修复truncated丢失与轨迹seed未对齐，及生成配置不匹配6回归RED→GREEN，增加2CLI流程；最终95GREEN。
独立只读review/复查通过，无新增重要缺陷；子仓用户改动、外层包、.serena、progress保留。
未加载checkpoint、未运行模型/GPU/Noise Head训练、未安装依赖。使用用户指定vLLM解释器；真实采集须先Kaggle baseline/hidden/seed传递验收及吞吐预算。
新增接口：build_dataset.py、plan_probe_requests.py、select_probe_dataset.py；控制器/评分/轨迹接口见验收文档。其他模型执行参数与RL头动作/更新仍按分阶段确认，不静默改变。
本轮r-doc严格审计0错误/0警告、diff检查通过；代码文档已提交fa90b71并push到feat/local-cpu-smoke，main未合并；progress只本地。下一步从数据CPU验收读Kaggle前置接口，不重复抽样或覆盖artifacts。

## Kaggle 输入确认（2026-10-03）

用户确认 Kaggle smoke 使用同一 1B checkpoint，挂载路径为 `/kaggle/input/models/fengjixing/llama3-2-1b-instruct-latent-sft-top10/pytorch/latent-sft/1`。
DAPO 与 GSM8K 数据目录为 `/kaggle/input/datasets/fengjixing/latent-grpo-data/data`。
以上为用户提供的路径，尚未在 Kaggle 核验文件存在、内容或模型身份；本轮仅记录，不运行 GPU 实验。
下一步确认 Kaggle GPU 类型，再讨论 K0 smoke 的具体执行范围。旧 Task 2 的 max_new_tokens=64、one-sided=True 属于待讨论方案，不覆盖已批准的作者生成配置（GSM128/DAPO4096、one_sided=False）。代码包挂载/交付方式尚未提供，Notebook 执行前需要补齐。

2026-10-04：用户确认 T4×2；仍使用 GPU 0 / tp=1 的单卡 smoke。附件 `/Users/fengjixing/Downloads/rand_seed_test.ipynb` 实际只测 random.Random 生成整数，其 PASS 不代表模型验收。按用户要求修改 `sample_once(prompt, seed)`，项目副本 `notebooks/rand_seed_test.ipynb`，原附件不修改。已核验固定上游 SamplingParams 没有请求 seed，Engine 支持 random_seed，Scheduler 从 worker 取得 seed 后 set_random_seed；因此每调用重建 Engine、完成/生成报错后关闭，不伪造 generate(seed=...)。固定原模型路径、fp16、GPU0、作者 latent/Gumbel top10；保留附件 temperature1/max_new_tokens64 作为诊断配置。需要 Kaggle 预先提供作者自定义 SGLang/FlashInfer，代码包及依赖状态仍未知，不自动安装。清空旧模拟输出，修正 PASS 范围。两项 CPU mock 测试通过；未加载模型、未运行 Kaggle。该测试不替代持久引擎逐请求 RNG 和 raw-latent-v1 对齐验收。
