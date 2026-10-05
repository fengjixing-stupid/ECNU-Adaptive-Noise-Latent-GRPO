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

用户随后提供真实 Kaggle notebook 与运行输出：SFT 模型在 T4 GPU0、fp16、独立重启 Engine 下，同 seed=123456 的两次文本/hash一致（ee41b1b4d635）；三个不同 seed=111111/222222/333333 的文本/hash不同。独立 Engine 启动 seed 文本检查通过，持久 Engine 请求级 seed 和 latent/hidden 对齐仍未验收。

2026-10-04 模型身份更正：用户确认此前实际为 SFT checkpoint，现已替换为 `/kaggle/input/models/fengjixing/llama3-2-1b-instruct-latent-grpo-top10/other/default/1/LLaMA3.2-1B-Instruct-Latent-GRPO-Top10`。该地址是后续 GRPO 实验模型路径；不再沿用旧 SFT 路径。新 checkpoint 的 seed 复现和 hidden smoke 尚未运行；旧 SFT 结果只作为环境和启动 seed 的历史证据。
下一步拟议 K0：固定 train 候选池的一道 GSM8K 题，单卡GPU0/tp1/fp16，作者 GSM 生成配置及 scale1，三次独立 Engine 运行（同 seed 两次、不同 seed 一次），检查 latent hidden/clean top10/mixture 逐步对齐并记录资源。这是待用户确认的范围，不代表实验或新采集器已获验收。

2026-10-04：用户已确认以上 K0 范围，并要求基于 `/Users/fengjixing/Downloads/notebook301c1d6406 (1).ipynb`，保留clone/环境/通过bash构建并执行py脚本的逻辑，仅替换smoke脚本内容。已实现 `scripts/kaggle_latent_smoke.py`，逐字嵌入 `notebooks/kaggle_latent_smoke.ipynb` 的cell9；其他cell源码完全一致（cell10命令不变），只清空历史输出与执行计数，原附件保留。说明 docs/kaggle_k0_smoke.md，原计划增加当前范围覆盖注释。
固定题为 train候选中的 GSM source_row=7335，source文件hash核对后从Kaggle原数据读取原prompt；使用用户更正的GRPO路径。三次独立Engine启动seed按原root12345/pid/scale1/rollout_id0、0、1派生。临时runtime_overlay复制已安装sglang，整个Python源码树及评分来源与固定参考SHA匹配后追加sampler capture；不修改作者安装或checkout。CPU/CUDA seed、LAST hidden、pre-Gumbel clean top10、实际mixture、全部输出ID对齐检查；结束marker首ID遵循作者约定，退出步不算混合step。实际worker torch显存peak及0.5秒GPU0采样最大值分开记录；生成耗时带hook，不作为正式吞吐预算。
TDD初始4个缺脚本RED后实现，修正test fixture代表token不一致；source/shape与overlay跨进程测试RED→GREEN。一个只读review Agent发现源码核验遗漏embedding/seed消费者、artifact缺max_topk、只看CPU seed；分别修复为整个sglang Python tree hash、effective generation config含max_topk10、观察并校验CUDA initial_seed，完成复查无新增重要问题。8项K0 CPU测试/全套105 tests通过6.20秒；r-doc strict 0 errors/0 warnings，diff检查和CLI help通过。未加载本地模型、未执行GPU、未安装或更改虚拟环境。
Ruling：smoke选source_row7335，不重抽/改变train池；初查row1611题面1500+900+580=2980，但数据标签3290，记录为后续数据标签质量待核实，不在本任务改写标签或增设筛选规则。工程smoke的reward仅诊断，不要求答对。
下一步交付Notebook给用户：已跑通环境时运行cell9，读取 `/kaggle/working/artifacts/k0_smoke_<时间>/smoke_test.json`。源码/输入不符或环境/OOM失败即STOP；不自动进入正式probing。持久Engine请求级seed、正式采集器、预算和Noise Head仍待后续实施。
