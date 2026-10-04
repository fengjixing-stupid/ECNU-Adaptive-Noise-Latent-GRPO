# 持续约束

## CON-001 — 本地环境
用户指定 Python：`/Users/fengjixing/Python_Project/Environments/vLLM/bin/python`。
只读元数据确认：Darwin arm64，Python 3.12.10，torch 2.10.0，transformers 5.12.1，pytest 9.1.1。
本轮未导入 torch、未加载模型、未执行实验。checkpoint 的绝对路径未提供，禁止猜测或自动下载。

## CON-002 — SOCKS5 代理异常
仅遇到终端 SOCKS5 代理异常时，在当前命令 shell 临时执行：
```sh
unset http_proxy https_proxy all_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
```
不得修改全局代理配置。发生后在本文件记录操作、结果和时间；不要记录代理凭据。
当前状态：尚未观察到代理异常，未执行重置。

## CON-003 — 非项目故障立即停止
环境、网络、本地设备问题或实验 OOM：立即停止实验，报告位置、原因和已知影响；不得自动安装依赖、换环境、缩短预算或进行 OOM 重试。
用户当前指令优先于 Spec 16.5 的自动降级建议。实验脚本代码错误可在原范围内修复；三次失败后停止并指出可疑假设。
progress 默认不提交；本轮只提交明确列出的正式文档及仓库配置。

## CON-004 — 缺失必需文件或输入立即停止
2026-09-30 用户更正并明确：遇到缺失文件、缺失输入立即停止。不能发出问题后继续依赖该输入的工作。
当前 docs/AGENT_USE.md 已补入并读取；checkpoint/data 路径留作真实模型阶段前置条件，本轮纯设计无需这些输入。

## CON-005 — 实验设计交流
2026-09-30 用户确认本地 CPU smoke 范围；后续实验设计必须先与用户交流。新基线与固定版本见 docs/minfix_submodule_update.md；不把原上游测试历史当本轮证据。

## CON-006 — 两阶段训练与题集边界
2026-10-02用户确认的长期范围：首版包含Stage A监督warm start与Stage B head-only RL；backbone两阶段均冻结。
Canonical: docs/training_route.md（DOC-TRAIN-001）。两阶段共享train题目但RL使用当前head新rollout，validation/test不参与参数更新。
题目池基本筛选即可，不要求复杂难度分层或同prefix分支监督；不强制noise先大后小。
原Spec中Stage B可选/首版只做Stage A已被本次确认取代；不得资源不足时静默省略RL。
尚未确认的随机动作参数化、衔接方式、更新参数、题源配比和预算，在实施前与用户交流。
