# 本地实施决定

## PDEC-001 — Task 1执行范围
本轮只执行已经确认的CPU sampler Task 1；不执行Kaggle/Noise Head等后续实验设计。
工作分支feat/local-cpu-smoke；执行记录继续放在项目progress，不另建重复全局状态。
kernel从minfix 0b7e85f的8192 streaming数学路径适配，reference AST加载真实helper；不改子仓，必须保留分块RNG与K宽输出。
已观察37项RED→GREEN和2项CLI RED→GREEN；独立review指出并已修复来源路径一致性和缺失kernel/reference预检；3项回归RED→GREEN，完整测试43通过，独立smoke38通过；复审无残留发现。

## PDEC-002 — 首版Stage B与共享题集
2026-10-02用户已确认：Stage B纳入首版；Stage A/B同一train题集，RL用当前head重新rollout，validation/test隔离；基本筛选，不要求同prefix分支搜索。
Canonical: docs/training_route.md（DOC-TRAIN-001）；实验参数尚未确认，当前仅同步正式文档。

## PDEC-003 — 题源与不再去重
2026-10-02 用户确定训练题源 GSM8K-Aug + DAPO-Math-17k，Math-500与GSM8K-Aug的指定test文件仅作测试。
信任作者已有处理，不重复去重、不要求parent映射；替代PDEC-002所引用旧文档中的parent分组前提。
Canonical: docs/dataset_construction_design.md（DOC-DATA-001）、docs/training_route.md；切分/配比为建议待确认。
PyArrow异常由用户更新后重试解决：25.0.1读取schema和首行成功，未验证全文件读取。

## PDEC-004 — 先抽子池再固定切分，probing在后
Canonical: docs/dataset_construction_design.md（DOC-DATA-001）。用户先确定抽子池再probing筛选，本轮提出320题按240 train / 80 validation分配；每来源各160、各120/40，固定split在probing之前。
替代之前尚未采用的90/10与256/64建议；候选子池不是最终训练题集，模型经验正确率仍用于基本筛选。两个既有test文件保持独立。

## PDEC-005 — 自适应probing次数与noise-sensitive筛选
Canonical: docs/dataset_construction_design.md。首轮scale[0,.25,.5,1,2]，后续视invalid/正确率人工决定是否保守[0,.25,.5,.75,1]；s=0始终1次绝不追加，非零累计3→5→7。
train全错/全对移出本轮选择；delta>0.4保留；非零7次仍有对有错且delta≤0.4剔除并记录可回查problem_id。保留原数据和候选清单。
Supersedes: PDEC-002所指旧文档中全失败暂不剔除的建议；不改split，不自动补抽或放宽阈值。

## PDEC-006 — 数据master seed与三个派生seed
用户确认master_seed=42，三个派生seed分别负责GSM8K-Aug抽样、DAPO抽样与train/validation切分。Canonical: docs/dataset_construction_design.md，包含稳定SHA256-v1派生公式与值；单独的split RNG按固定来源顺序消费。
不把此决定自动扩展为模型rollout seeds或三轮实验；未抽样、未运行模型。

## PDEC-007 — validation终态标记、最佳scale轨迹复用与生成配置
Canonical: docs/dataset_construction_design.md。validation与train同采样控制，但全对/全错/敏感/7次不敏感均保留并标记。
每题仅永久保留最佳经验正确率scale的全部完整轨迹和特征，包括正确/错误；其他scale保留摘要；warm-start复用，Stage B重采样。
用户确认作者配置：GSM8K-Aug low上限128，DAPO high上限4096；共用temp=.6/top_p=.95/K=10/Gumbeltemp=1，单侧False；probing显式开噪声。Supersedes: PDEC-004/005/006对应旧文档中的validation预算、轨迹复用与生成配置待定状态。

## PDEC-008 — 模型逐轨迹seed与invalid判定
2026-10-03用户采用：rollout根seed=12345，按problem_id/scale/rollout_id稳定派生并记录，与数据seed42分离；规范化与SHA256版本见DOC-DATA-001。
沿用作者答案判定；空输出/无法提取标记invalid，有效错误另计；截断单独标记，可判正确仍R=1；硬件和判定执行故障不转奖励0。
Canonical: docs/dataset_construction_design.md、docs/training_route.md。Supersedes: PDEC-007对应旧文档中模型seed与invalid待定状态。

## PDEC-009 — 数据CPU实施与验收
2026-10-03用户授权当前会话实施计划与CPU验收，主Agent顺序TDD，既有工作分支保留，不增加重复worktree/ledger；正式计划DOC-DATA-PLAN-001，正式验收DOC-DATA-CPU-001。
本轮真实文件只CPU数据处理；模型采集器不在本轮，probing请求4160未执行。作者答案函数按固定commit AST加载，raw-latent-v1 sparse Top10特征接口用于离线CPU选择。
独立review要求修复truncated永久摘要丢失与轨迹seed绑定缺失；连同批准生成配置对齐均回归验证；95全测试/38sampler通过，复查无重要缺陷。
Ruling: 初始累计5次fixture错误地把第2个成功放在首3次，会触发敏感终态；改fixture为后续第4次成功，保持真实逐阶段语义。控制器非终态完整阶段保留delta诊断，计划后续请求而不输出选择。
