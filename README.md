# AdaNoise-LGRPO

冻结作者 1B Latent-GRPO backbone，训练轻量 Noise Head，比较固定与逐 latent step 自适应 Gumbel scale。

当前阶段（2026-09-30）：本地 smoke 与 Kaggle 实验设计；尚未实现或运行模型实验。

1. 从 [项目入口](AGENTS.md) 读取规则。
2. 查看 [文档索引](docs/README.md)。
3. 审阅 [本地 smoke 实现计划](docs/local_smoke_implementation_plan.md)。

作者源码由 `Latent-GRPO/` submodule 固定版本引用。克隆使用：
```sh
git clone --recurse-submodules https://github.com/fengjixing-stupid/ECNU-Adaptive-Noise-Latent-GRPO.git
```
