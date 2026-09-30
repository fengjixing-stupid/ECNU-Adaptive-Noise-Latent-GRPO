# AdaNoise-LGRPO

冻结作者 1B Latent-GRPO backbone，训练轻量 Noise Head，比较固定与逐 latent step 自适应 Gumbel scale。

当前阶段（2026-09-30）：本地 CPU smoke 范围已确认；尚未实现或运行模型实验。

1. 从 [项目入口](AGENTS.md) 读取规则。
2. 查看 [文档索引](docs/README.md)。
3. 审阅 [本地 smoke 实现计划](docs/local_smoke_implementation_plan.md)。

用户修订源码由 `latent_grpo_minfix/Latent-GRPO-final/` submodule 固定提交 `0b7e85f` 引用；远端分支为 `Latent-GRPO-Top-K`。外层说明文件仍留在本地。克隆使用：
```sh
git clone --recurse-submodules https://github.com/fengjixing-stupid/ECNU-Adaptive-Noise-Latent-GRPO.git
```
