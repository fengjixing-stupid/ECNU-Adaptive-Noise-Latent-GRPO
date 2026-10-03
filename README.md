# AdaNoise-LGRPO

冻结作者 1B Latent-GRPO backbone，训练轻量 Noise Head，比较固定与逐 latent step 自适应 Gumbel scale。

当前阶段（2026-10-03）：本地CPU sampler、数据构建和probing控制已验收；尚未运行真实模型实验。

1. 从 [项目入口](AGENTS.md) 读取规则。
2. 查看 [文档索引](docs/README.md)。
3. 审阅 [本地 smoke 实现计划](docs/local_smoke_implementation_plan.md)。

用户修订源码由 `latent_grpo_minfix/Latent-GRPO-final/` submodule 固定提交 `0b7e85f` 引用；远端分支为 `Latent-GRPO-Top-K`。外层说明文件仍留在本地。克隆使用：
```sh
git clone --recurse-submodules https://github.com/fengjixing-stupid/ECNU-Adaptive-Noise-Latent-GRPO.git
```

本地验证：
```sh
python -m pytest -q
python scripts/smoke_local.py --config configs/local_smoke.yaml --device cpu
```
结果和边界见 [CPU smoke 验收](docs/local_cpu_smoke.md)。
第三方许可见 [Third-party notices](THIRD_PARTY_NOTICES.md)。

数据集候选构建与请求规划：
```sh
python scripts/build_dataset.py --config configs/dataset_build.yaml
python scripts/plan_probe_requests.py --candidate-dir artifacts/dataset_candidates_v1 --output-dir artifacts/probe_requests_v1
```
使用指定虚拟环境；已有输出拒绝覆盖。数据文件和产物不提交Git，换机器需用户提供输入。命令、轨迹契约及CPU证据见 [数据CPU验收](docs/dataset_cpu_acceptance.md)。
