---
type: project_topic
status: active
summary: "本地开发时访问三卡经验、运行时打包、规则、日志和设计记录的路径索引。"
tags: [local-development, navigation, 3gpu, cairn]
contains: [procedure, reference]
created: "2026-08-22"
updated: "2026-08-22"
related:
  - "3gpu-deployment-lessons.md"
  - "3gpu-runtime-packaging.md"
  - "Latent-GRPO-final/Latent-GRPO/cairn/top-k-implementation.md"
  - "LOG.md"
  - "../AGENTS.md"
authoring_mode: ai_generated
---
# 本地开发经验文件路径索引

## 仓库根目录

当前 Mac 上的仓库根目录：

```text
/Users/fengjixing/Python_Project/Projects/vLLM/Latent-GRPO/Latent-GRPO
```

下文同时给出仓库相对路径和当前机器绝对路径。仓库迁移到其他目录后，应优先使用相对路径；绝对路径仅用于当前 Mac 本地点击和命令行访问。

## 核心经验与规则

| 用途 | 仓库相对路径 | 当前 Mac 绝对路径 |
|---|---|---|
| 三卡部署、参数、故障、检查方法、解决方案和强制设计检查 | `Latent-GRPO/cairn/3gpu-deployment-lessons.md` | `/Users/fengjixing/Python_Project/Projects/vLLM/Latent-GRPO/Latent-GRPO/Latent-GRPO/cairn/3gpu-deployment-lessons.md` |
| 三卡 Ray/FSDP 拓扑、训练打包和 runtime gate | `Latent-GRPO/cairn/3gpu-runtime-packaging.md` | `/Users/fengjixing/Python_Project/Projects/vLLM/Latent-GRPO/Latent-GRPO/Latent-GRPO/cairn/3gpu-runtime-packaging.md` |
| 经验沉淀与项目进展的时间索引 | `Latent-GRPO/cairn/LOG.md` | `/Users/fengjixing/Python_Project/Projects/vLLM/Latent-GRPO/Latent-GRPO/Latent-GRPO/cairn/LOG.md` |
| 当前路线图和开放事项 | `Latent-GRPO/cairn/ROADMAP.md` | `/Users/fengjixing/Python_Project/Projects/vLLM/Latent-GRPO/Latent-GRPO/Latent-GRPO/cairn/ROADMAP.md` |
| Codex/Claude 项目规则和三卡脚本强制检查入口 | `Latent-GRPO/AGENTS.md` | `/Users/fengjixing/Python_Project/Projects/vLLM/Latent-GRPO/Latent-GRPO/Latent-GRPO/AGENTS.md` |

## 设计与实施记录

| 用途 | 仓库相对路径 | 当前 Mac 绝对路径 |
|---|---|---|
| 三卡部署经验与脚本检查的设计说明 | `docs/superpowers/specs/2026-08-15-3gpu-deployment-lessons-design.md` | `/Users/fengjixing/Python_Project/Projects/vLLM/Latent-GRPO/Latent-GRPO/docs/superpowers/specs/2026-08-15-3gpu-deployment-lessons-design.md` |
| 三卡部署经验文档的实施计划 | `docs/superpowers/plans/2026-08-15-3gpu-deployment-lessons.md` | `/Users/fengjixing/Python_Project/Projects/vLLM/Latent-GRPO/Latent-GRPO/docs/superpowers/plans/2026-08-15-3gpu-deployment-lessons.md` |
| 三卡最终打包实施计划 | `docs/superpowers/plans/2026-08-11-3gpu-final-packaging.md` | `/Users/fengjixing/Python_Project/Projects/vLLM/Latent-GRPO/Latent-GRPO/docs/superpowers/plans/2026-08-11-3gpu-final-packaging.md` |

## 当前最小修复仓库专项记录

| 用途 | 当前工作区相对路径 | 当前 Mac 绝对路径 |
|---|---|---|
| Top-K 计算、传递、内存物化与 checkpoint 边界 | `Latent-GRPO-final/Latent-GRPO/cairn/top-k-implementation.md` | `/Users/fengjixing/Python_Project/Projects/vLLM/Latent-GRPO/latent_grpo_minfix/Latent-GRPO-final/Latent-GRPO/cairn/top-k-implementation.md` |

## 推荐阅读顺序

1. 先读 `Latent-GRPO/AGENTS.md`，确认项目规则和强制检查要求。
2. 再读 `Latent-GRPO/cairn/LOG.md` 顶部，了解最新进展。
3. 涉及任何三卡脚本时，完整阅读 `3gpu-deployment-lessons.md` 和 `3gpu-runtime-packaging.md`。
4. 需要追溯设计理由或实施范围时，再查看 `docs/superpowers/specs/` 与 `docs/superpowers/plans/` 中的对应文件。

## 快速打开

从仓库根目录执行：

```bash
${EDITOR:-code} Latent-GRPO/cairn/local-development-reference.md
${EDITOR:-code} Latent-GRPO/cairn/3gpu-deployment-lessons.md
${EDITOR:-code} Latent-GRPO/cairn/3gpu-runtime-packaging.md
${EDITOR:-code} Latent-GRPO/cairn/LOG.md
${EDITOR:-code} Latent-GRPO/AGENTS.md
```

如果 `$EDITOR` 不支持一次打开多个文件，可直接将上述相对路径传给当前编辑器。
