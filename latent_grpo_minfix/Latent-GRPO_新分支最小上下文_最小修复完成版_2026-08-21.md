# Latent-GRPO 新分支最小上下文（最小修复完成版，更新至 2026-08-21）

> 用途：将本文件直接作为新会话、Codex 任务或人工交接上下文。不要重新排查已经完成的 CUDA/依赖/三卡基础链路、旧 `2/2` 基线、`32/128`/`32/64` 扫参或 observer-off 根因。
>
> **当前一句话状态：**已基于运行时父提交 `8c9ce4932234844620d4da0b99df982123d35ae7` 完成最小发布修复，形成分支 `fix/text-runtime-dependencies`、HEAD `910989600df1896258a448f3341b010cc265a296`。本提交只修改测试和文档，训练源码、配置 YAML、Hydra overrides 与运行参数均未变化；完整测试为 `241 passed, 1 skipped`，本地 release gate 为 PASS。父提交的 Low `16/32/0.45` 三卡 validation 已全门禁 PASS，但新 HEAD 在启动任何新正式训练前仍必须重新生成 commit-bound GPU acceptance。

---

## 1. 当前代码身份

```text
分支：fix/text-runtime-dependencies
当前 HEAD：910989600df1896258a448f3341b010cc265a296
运行时父提交：8c9ce4932234844620d4da0b99df982123d35ae7
提交标题：Sync 3GPU Low release contract
工作树：clean
```

提交关系：

```text
9109896 Sync 3GPU Low release contract
8c9ce49 Add memory headroom for long training batches
7f81a8f Fix observer-off worker packet emission
ac22be3 Maximize 3GPU low training throughput
98f7a01 Persist static run metadata for output validation
```

现场必须重新确认：

```bash
cd /home/powerop/work/chenwh/latent_grpo/Latent-GRPO-final
git branch --show-current
git rev-parse HEAD
git status --short
git log --oneline -6
```

不要执行：

```text
git reset --hard origin/main
在正在运行的 8c9ce49 正式训练工作树中直接切换分支
把旧 acceptance 用于新 HEAD
覆盖已有正式训练输出目录
```

---

## 2. 本次最小修复边界

本次提交只修改以下 8 个文件：

```text
FINAL_EXPERIMENT_PACKAGE.md
docs/3GPU_ACCEPTANCE_CHECKLIST.md
docs/3GPU_HYPERPARAMETER_DEVIATIONS.md
docs/3GPU_RUNBOOK.md
docs/AUTHOR_HYPERPARAMETER_AUDIT.md
docs/PROJECT_TECHNICAL_HANDOFF.md
tests/unit/test_3gpu_final_package.py
tests/unit/test_upstream_optimizer_patch.py
```

对以下运行目录执行父提交到当前 HEAD 的 diff，结果为空：

```text
configs/
Latent-GRPO/
latent_grpo_runner/
train_latent_grpo.py
tools/
scripts/
requirements/
constraints/
```

因此本次没有修改：

```text
GRPO objective
reward / advantage / FlipGrad
SGLang rollout
FSDP actor/worker/trainer
optimizer 或 scheduler
observer-off 实现
任何 YAML 数值
任何 Hydra runtime override
模型、数据或 checkpoint 格式
```

这保证当前提交与父提交 `8c9ce49` 在训练运行逻辑和配置上等价；变化仅是发布契约、防回归测试和交接说明。

---

## 3. 已修复的问题

### 3.1 陈旧 Low 配置契约测试

原测试仍断言：

```text
Low SGLang gpu_memory_utilization = 0.6
Low formal 与 validation 的 metrics/support/probes 都为 true
Low validation prompt/mini = 3/3
```

当前测试已同步到真实配置：

```text
Low formal + validation：prompt/mini = 48/12
actor_micro_batch_per_gpu = 16
rollout_log_prob_micro_batch_per_gpu = 32
SGLang gpu_memory_utilization = 0.45

Low formal：metrics/support/checkpoint/credit = false
Low validation：metrics/support/checkpoint/credit = true
```

作者真值文件中的原始 Low 值仍保持：

```text
actor micro = 2
rollout logprob micro = 2
SGLang gpu_memory_utilization = 0.6
```

没有把作者参数与当前三卡工程参数混为一谈。

### 3.2 observer-off 直接回归测试

新增测试直接执行：

```python
consume_latent_grpo_observer_facts(observer_disabled_actor) == {}
```

同时确认：

```text
last_update_count 不被消费函数改写
last_update_did_step 不被消费函数改写
```

该测试复用已有 AST standalone loader，不需要导入完整 CUDA/FSDP actor runtime，也没有修改 `dp_actor.py`。

### 3.3 当前参数文档同步

已同步的当前 Low 正式参数：

```text
prompt_batch = 48
rollout_n = 8
mini_prompt_batch = 12
actor_micro = 16
logprob_micro = 32
SGLang gpu_memory_utilization = 0.45
metrics/support/checkpoint/credit = false
val_before_train = false
test_freq = -1
save_freq = 5000
logger = [console]
total_epochs = 10
```

已同步的 Low validation 参数：

```text
prompt_batch = 48
rollout_n = 8
mini_prompt_batch = 12
actor_micro = 16
logprob_micro = 32
SGLang gpu_memory_utilization = 0.45
metrics/support/checkpoint/credit = true
trainer.total_training_steps = 2
save_freq = 1
total_epochs = 1
```

High formal/validation 未被误改；High validation 仍是独立的 `3/3` prompt/mini 几何和 SGLang `0.8`。

### 3.4 旧 release evidence 与打包杂质

新归档只保留与当前 HEAD 匹配的：

```text
release_validation/LOCAL_RELEASE_ACCEPTANCE.json
```

并排除：

```text
.pytest_cache/
__pycache__/
Latent-GRPO/verl-0.4.x/outputs/
Latent-GRPO/verl-0.4.x/wandb/
*.egg-info/
```

不再携带指向服务器绝对路径的 W&B 失效软链接。

---

## 4. 验证结果

### 4.1 定向测试

```text
tests/unit/test_3gpu_final_package.py
tests/unit/test_upstream_optimizer_patch.py

30 passed
```

### 4.2 完整测试

```text
241 passed, 1 skipped
```

唯一 skip：

```text
tests/unit/test_upstream_patch_contract.py::...
upstream CPU dependencies unavailable;
numerical OCP equivalence is target-machine deferred
```

这是原有 target-machine deferred 测试，不是本次修复产生的失败。

### 4.3 本地 release gate

```text
LOCAL_RELEASE_GATE: PASS
TARGET_GPU_RUNTIME: DEFERRED_TO_L20_SERVER
```

全部检查：

```text
required_release_files: PASS
tracked_packaging_cruft: PASS
active_dependency_pin_consistency: PASS
python_source_syntax: PASS
shell_source_syntax: PASS
unit_tests: PASS
四个 final profile dry-run: PASS
git_identity_and_clean_tree: PASS
```

报告身份：

```text
status = PASS
git_commit = 910989600df1896258a448f3341b010cc265a296
python_version = 3.13.5
```

重要边界：该报告是在当前打包环境 Python 3.13.5 下生成的静态/单测报告。服务器正式环境锁定 Python 3.11.15，因此在服务器应用本提交后仍应使用 `.venv-target/bin/python` 重跑 release gate。该本地报告不声称 CUDA/NCCL/SGLang/FSDP 已在新 HEAD 执行。

---

## 5. observer-off 运行时状态

运行时修复仍来自父提交中的 `7f81a8f`：

```python
def consume_latent_grpo_observer_facts(self):
    # Observer-off training must not export worker observer packets.
    # last_update_count / last_update_did_step remain available to the
    # scheduler independently of this optional metrics payload.
    if not self._latent_grpo_observer_enabled:
        return {}
```

调用链：

```text
actor update 完成
→ last_update_count 先供 scheduler 使用
→ observer-off consume 返回 {}
→ FSDP worker 的 if observer_facts: 不写 payload
→ driver 不接收 latent_grpo_worker_observers
→ 不触发 durable coordinator sink RuntimeError
```

本次提交只增加回归测试，没有再次修改该运行逻辑。

---

## 6. 当前 Low 参数与几何

```text
48 prompts × 8 rollout = 384 trajectories / outer batch
384 / 3 GPU = 128 trajectories / GPU
12 mini prompts × 8 / 3 GPU = 32 trajectories / GPU / PPO mini-batch

actor micro 16：32 / 16 = 2 次 FW/BW / local mini-batch
logprob micro 32：128 / 32 = 4 次 forward / local full batch
```

当前不要改回：

```text
16/64/0.5
16/64/0.6
2/2/0.6
Low validation 3/3
```

旧 `16/64` 的单步速度样例不能作为当前 `16/32` 的绝对速度承诺。

---

## 7. 父提交 GPU acceptance 与新 HEAD 的关系

父提交：

```text
8c9ce4932234844620d4da0b99df982123d35ae7
```

其 Low `16/32/0.45` 三卡 validation 已报告：

```text
3GPU_PREFLIGHT_GATE: PASS
3GPU_DISTRIBUTED_RUNTIME_GATE: PASS
CORE_METRICS: 29/29
CUDA_RNG_DEVICE_0/1/2: PASS
CUDA_RNG_ALL_DEVICES: PASS
CHECKPOINT_GATE: PASS
3GPU_FINAL_GATE: PASS
```

新提交 `9109896` 与父提交运行时等价，但 acceptance 规则按 Git commit fail-closed。故：

```text
旧 acceptance 可以说明参数集合和运行路径曾通过
旧 acceptance 不能作为新 HEAD 正式训练的 wrapper 输入
新 HEAD 启动新正式 run 前必须重新跑一次完整 3GPU validation
```

只修测试和文档不要求停止、重启或重跑已经在 `8c9ce49` 上启动的正式 run；该 run 的 manifest、checkpoint 和实验身份必须继续保持 `8c9ce49`。

---

## 8. 当前已启动正式 run 的处理原则

已知旧工作树正式 run：

```text
/home/powerop/work/chenwh/latent_grpo/experiments/runs/
low-speed16-32-sglang045-observeroff-seed17-20260821-112028
```

该 run 若仍在运行：

```text
不要在其工作树切换到 9109896
不要覆盖源码或配置
不要复用其 output root
不要因为发布测试修复而中断训练
只读监控 status、training.log、GPU、内存和磁盘
```

检查：

```bash
export LOW_TRAIN_OUTPUT="/home/powerop/work/chenwh/latent_grpo/experiments/runs/low-speed16-32-sglang045-observeroff-seed17-20260821-112028"
export RUNTIME_LOG="$LOW_TRAIN_OUTPUT/logs/training.log"

cat "$LOW_TRAIN_OUTPUT/status.json" 2>/dev/null || true
cat "$LOW_TRAIN_OUTPUT/latest_checkpointed_iteration.txt" 2>/dev/null || true
tail -n 300 "$RUNTIME_LOG"

grep -nE 'training/global_step|timing_s/(gen|old_log_prob|update_actor|step)|perf/throughput|Traceback|CUDA out of memory|No space left|worker observer packets require' \
  "$RUNTIME_LOG" | tail -n 250
```

---

## 9. 将最小修复应用到服务器

推荐在独立 worktree 操作，不影响正在运行的旧工作树。

### 9.1 使用 Git bundle，保留精确提交

提供文件：

```text
latent_grpo_minfix_9109896.bundle
```

该 bundle：

```text
包含：910989600df1896258a448f3341b010cc265a296
要求父提交：8c9ce4932234844620d4da0b99df982123d35ae7
```

服务器命令：

```bash
cd /home/powerop/work/chenwh/latent_grpo/Latent-GRPO-final

git status --short
git cat-file -e 8c9ce4932234844620d4da0b99df982123d35ae7^{commit}
git bundle verify /path/to/latent_grpo_minfix_9109896.bundle

git fetch /path/to/latent_grpo_minfix_9109896.bundle \
  refs/heads/fix/text-runtime-dependencies:refs/heads/fix/current-low-release-contract

git worktree add \
  /home/powerop/work/chenwh/latent_grpo/Latent-GRPO-release-fix \
  fix/current-low-release-contract

cd /home/powerop/work/chenwh/latent_grpo/Latent-GRPO-release-fix
git rev-parse HEAD
git status --short
```

预期 HEAD：

```text
910989600df1896258a448f3341b010cc265a296
```

### 9.2 使用 patch

提供文件：

```text
latent_grpo_minfix_9109896.patch
```

```bash
cd /home/powerop/work/chenwh/latent_grpo/Latent-GRPO-final

git worktree add \
  /home/powerop/work/chenwh/latent_grpo/Latent-GRPO-release-fix \
  -b fix/current-low-release-contract \
  8c9ce4932234844620d4da0b99df982123d35ae7

cd /home/powerop/work/chenwh/latent_grpo/Latent-GRPO-release-fix
git am /path/to/latent_grpo_minfix_9109896.patch
git rev-parse HEAD
git status --short
```

`git am` 生成的 commit hash 可能因 committer metadata 不同而变化；后续一律使用服务器实际 `git rev-parse HEAD`。

### 9.3 使用完整归档

完整归档已经包含 `.git` 和精确 HEAD `9109896`。不要直接覆盖正在训练的目录；解压到新的目录后核对 Git 身份。

---

## 10. 服务器 Python 3.11 本地门禁

在新的 worktree 中复用已验证 `.venv-target`：

```bash
source /home/powerop/work/anjos/00lib/miniconda3/etc/profile.d/conda.sh
conda activate latent-grpo-bootstrap

export PROJECT_ROOT="/home/powerop/work/chenwh/latent_grpo/Latent-GRPO-release-fix"
export PYTHON_BIN="/home/powerop/work/chenwh/latent_grpo/Latent-GRPO-final/.venv-target/bin/python"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/Latent-GRPO/verl-0.4.x:$PROJECT_ROOT/Latent-GRPO/sglang_latent_reasoning_pkg/python${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" tools/validate_release_package.py
```

必须看到：

```text
241 passed, 1 skipped
LOCAL_RELEASE_GATE: PASS
```

核对报告：

```bash
"$PYTHON_BIN" - <<'PY'
import json
import subprocess
from pathlib import Path

report = json.loads(Path("release_validation/LOCAL_RELEASE_ACCEPTANCE.json").read_text())
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print(report["status"], report["git_commit"], head, report["python_version"])
assert report["status"] == "PASS"
assert report["git_commit"] == head
PY
```

---

## 11. 新 HEAD 的三卡 validation

只有未来准备从新 HEAD 启动新正式 run 时才执行。先恢复原服务器环境和资产变量：

```bash
export PROJECT_ROOT="/home/powerop/work/chenwh/latent_grpo/Latent-GRPO-release-fix"
export ASSET_ROOT="/home/powerop/work/chenwh/latent_grpo/assets"
export LOW_MODEL_PATH="$ASSET_ROOT/models/LLaMA3.2-1B-Instruct-Latent-SFT-Top10"
export LOW_TRAIN_DATA="$ASSET_ROOT/data/GSM8k-Aug-oss-dup-all.parquet"
export LOW_VAL_DATA="$ASSET_ROOT/data/GSM8k-Aug-test.parquet"
export EXPERIMENT_ROOT="/home/powerop/work/chenwh/latent_grpo/experiments"
export RUN_GPUS="4,5,6"

export CUDA_TOOLKIT_ROOT="$CONDA_PREFIX"
export CUDA_HOME="$CUDA_TOOLKIT_ROOT"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_TOOLKIT_ROOT/bin/nvcc"
export PATH="/home/powerop/work/chenwh/latent_grpo/Latent-GRPO-final/.venv-target/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset CUDA_LAUNCH_BLOCKING
unset RAY_ADDRESS
unset CUDA_VISIBLE_DEVICES

cd "$PROJECT_ROOT"
git status --short
"$PYTHON_BIN" -m py_compile Latent-GRPO/verl-0.4.x/verl/workers/actor/dp_actor.py
```

停止本用户旧 Ray 后创建新 validation 目录：

```bash
"/home/powerop/work/chenwh/latent_grpo/Latent-GRPO-final/.venv-target/bin/ray" stop --force

export VAL_TAG="$(date +%Y%m%d-%H%M%S)"
export LOW_VALIDATION_ROOT="$EXPERIMENT_ROOT/validation/low-minfix-16-32-sglang045-seed17-$VAL_TAG"

bash tools/run_3gpu_final_validation.sh \
  --config configs/3gpu-final-validation.yaml \
  --model-path "$LOW_MODEL_PATH" \
  --train-data "$LOW_TRAIN_DATA" \
  --val-data "$LOW_VAL_DATA" \
  --output-root "$LOW_VALIDATION_ROOT" \
  --gpus "$RUN_GPUS" \
  --seed 17
```

必须最终看到：

```text
3GPU_PREFLIGHT_GATE: PASS
3GPU_DISTRIBUTED_RUNTIME_GATE: PASS
CORE_METRICS: 29/29
CUDA_RNG_ALL_DEVICES: PASS
CHECKPOINT_GATE: PASS
3GPU_FINAL_GATE: PASS
```

绑定新报告：

```bash
export LOW_ACCEPTANCE_REPORT="$LOW_VALIDATION_ROOT/acceptance.json"

"$PYTHON_BIN" - <<'PY'
import json
import os
import subprocess
from pathlib import Path

p = Path(os.environ["LOW_ACCEPTANCE_REPORT"])
d = json.loads(p.read_text())
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print(d.get("final_gate"), d.get("profile_name"), d.get("git_commit"), head)
assert d.get("final_gate") == "PASS"
assert d.get("profile_name") == "3gpu-final-validation"
assert d.get("git_commit") == head
PY
```

---

## 12. 新正式训练启动条件

新 HEAD 只有同时满足以下条件才能启动新正式 run：

```text
工作树 clean
服务器 Python 3.11 LOCAL_RELEASE_GATE: PASS
新 HEAD 3GPU_FINAL_GATE: PASS
acceptance.profile_name = 3gpu-final-validation
acceptance.git_commit = 当前 HEAD
新 output root
GPU 仅 4,5,6
```

启动：

```bash
export TRAIN_TAG="$(date +%Y%m%d-%H%M%S)"
export LOW_TRAIN_OUTPUT="$EXPERIMENT_ROOT/runs/low-minfix-16-32-sglang045-observeroff-seed17-$TRAIN_TAG"

bash tools/run_3gpu_training.sh \
  --config configs/3gpu-final-low.yaml \
  --model-path "$LOW_MODEL_PATH" \
  --train-data "$LOW_TRAIN_DATA" \
  --val-data "$LOW_VAL_DATA" \
  --output-root "$LOW_TRAIN_OUTPUT" \
  --gpus "$RUN_GPUS" \
  --seed 17 \
  --acceptance-report "$LOW_ACCEPTANCE_REPORT"
```

正式训练必须放在 tmux 中。

---

## 13. 归档与补丁校验

```text
Git bundle SHA256:
a32252234f6617cca24a1ea883970c3a335522764741631ba62cb87e51850d71

Patch SHA256:
102bec947c9f4bee7ebb3f71a391d57adee28b62a9613dbe98d296e7893b991d
```

服务器上传后可执行：

```bash
sha256sum latent_grpo_minfix_9109896.bundle
sha256sum latent_grpo_minfix_9109896.patch
```

完整归档的 SHA256 以最终下载文件旁提供的值为准。

---

## 14. 当前最短继续路径

```text
1. 不干扰仍可能运行的 8c9ce49 正式 run。
2. 需要发布修复时，在独立 worktree 导入 bundle 或 patch。
3. 使用服务器 Python 3.11.15 跑完整 pytest 与 LOCAL_RELEASE_GATE。
4. 仅在准备启动新正式 run 时，为新 HEAD 重跑 3GPU validation。
5. validation PASS 后使用新的 acceptance 和新的 output root。
6. 不再修改 observer runtime、16/32/0.45 参数或已锁定依赖，除非出现新的第一条真实 traceback。
```

**不要重复：依赖重装、CUDA/NCCL 基础排障、旧 `2/2` baseline、`32/128`、`32/64`、`16/64+0.6` 扫参、重复 observer 修复、把父提交 acceptance 当作新 HEAD acceptance。**
