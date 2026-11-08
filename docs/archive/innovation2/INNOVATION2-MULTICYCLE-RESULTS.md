# 创新2多周期（N1/N2）结果

- 文档状态：**结果归档（DONE）**；对应周期目录 `scientific-20261103-0250`。
- 科学状态：**CLOSED_N2_ALL_FAIL_NO_N3_N4**（`cycle.json`）。N1 六个 fit、N2 十个 fit 均完成；main 冻结 DEV gate 下五个 N2 real 全部未通过，因此 N3/N4 未启动。
- 数据边界：本文件只读取已保存的 `cycle/stage/acceptance/results/exit/selection/initialization` JSON 做数值整理；**未加载 checkpoint、未读取或统计 NPZ、未生成标签、未做模型前向、未重算指标**。
- 代码来源：`65cf98dda0fa87a9f15573884c0cba04da7d226a`（短 SHA `65cf98d`）。该 SHA 是本轮全部 16 个 run 的 `manifest.git.commit`，且 `dirty=false`。

## 1. 结论摘要

1. **N1（3 轮）**：R real 的 DEV accuracy 为 **98.2788%**，Z real 为 **49.9512%**，M real 为 **89.2822%**。Z/M 相对 R 没有出现预期机制改善；Z 基本处于 chance 水平，M 明显低于 R。该结果只说明这三条 N1 路径在冻结 DEV 口径下的差异，不说明真实 key、真实内部状态或任何恢复能力。
2. **N2（6 轮）**：五个 real arm 的 DEV 均未满足冻结 capability；五个 null arm 均无报警。按冻结 DEV gate，main 决策为 **全部 N2 real fail → STOP，不启动 N3/N4，不做 bootstrap/确认统计 cell**。
3. **N2 TEST 观察**：十个 N2 TEST accuracy 约为 chance（0.4961–0.5022），十个 TEST BCE 均差于同一 split 的 TRAIN constant 与 q6 残差常数基线。TEST 已随 run 保存并在此如实列出，但未用于选模或调参；不是 sealed 确认。
4. **边界**：本轮不能据此声称“学到了六轮机制”，也不能据此证明“所有网络/所有方法都会失败”。它只覆盖这 16 个已授权 fit、这一组表示、预算与单一面板。

## 2. 授权、来源与执行边界

- 用户已批准本科学周期；周期授权上限见 `cycle.json`：36 fits、294,912 updates、75,497,472 presentations、real labels/null bits 各 2,506,752、statistics cells 18、retry 0。
- 实际只执行 N1 6 + N2 10 = **16 fits**；N3/N4 未启动（`cycle.json` 中 `N3_started=false`、`N4_started=false`）。
- `source_commit` 与远端 detached HEAD 均为 `65cf98dda0fa87a9f15573884c0cba04da7d226a`；全部 run 的 `manifest.source_sha256` 50/50 与发布值一致（见 `N1-main-acceptance.json`、`N2-main-acceptance.json`）。
- 本报告不修改源码、配置、已有 dirty 文档或 Git 历史；只新增本结果文档。训练产物和权重不进入 Git。

### 2.1 输入证据与 SHA-256

| 文件 | SHA-256（完整） |
|---|---|
| `validation/innovation2-multicycle/scientific-20261103-0250/cycle.json` | `61dac853baaaead6fcac2d0c7e88afdd8c1567a61ed595a108d2327606ec152a` |
| `validation/innovation2-multicycle/scientific-20261103-0250/N1-stage.json` | `fd1a9bfb93ca3496e6f25562441bdbd74d780cb35a494552840e7abea86042f2` |
| `validation/innovation2-multicycle/scientific-20261103-0250/N2-stage.json` | `e73ccf6b5b0caafa44471ed86a95dc414f6607ed25428129c6ea8437050bb1bf` |
| `validation/innovation2-multicycle/scientific-20261103-0250/N1-main-acceptance.json` | `500f4972e765c85dd48f4ec51b8d8ef527d720c2a60326e862a5e5cbeaea63dd` |
| `validation/innovation2-multicycle/scientific-20261103-0250/N2-main-acceptance.json` | `b6b6353d4e7ed95a78926758006c385cc01b92ea591041fd31c592c938c65afa` |

逐 run 的 `results.json`、`exit.json`、`selection.json`、`initialization.json` 位于：

- `runs/received/multicycle-20261103-0250-n1-{r,z,m}-{null,real}/`
- `runs/received/multicycle-20261103-0250-n2-{d,f,r,fr,s}-{null,real}/`

每个 run 的 `results.json.status=completed`、`exit.json.status=completed`、`exit_code=0`、`updates=8192`、`presentations=2,097,152`、`progress.jsonl` 32 行；`selection.json.test_labels_generated=false`，且 `selection.checkpoint_sha256 == results.best_checkpoint_sha256`。以上是对保存 JSON 的只读核对，不是 checkpoint replay。

## 3. 计数、预算与时间

| 项目 | 实际 | 授权/说明 |
|---|---:|---|
| fits | 16 | 36 上限；N1 6 + N2 10 |
| updates | 131,072 | 294,912 上限；每 fit 8,192 |
| presentations | 33,554,432 | 75,497,472 上限；每 fit 2,097,152 |
| real labels | 1,114,112 | 2,506,752 上限；每 fit 139,264 |
| null bits | 1,114,112 | 2,506,752 上限；每 fit 139,264 |
| confirmation statistics cells | 0 | 18 上限；未做 bootstrap/确认统计 |
| fit retries | 0 | 无 fit 重试；传输层故障不计为 retry |
| fit 实际 elapsed 合计 | 1,079.971 s（约 18.0 min） | 只来自 16 个 `results/exit.elapsed_seconds` |
| SSH wall 合计 | 1,181 s | N1 412 s + N2 769 s（`N1-attempts.json`/`N2-attempts.json`） |
| N1/N2 训练总 elapsed | 373.453 s + 706.518 s = 1,079.971 s | 两者相加与 cycle 一致 |

**计时口径**：1,079.971 s 是 16 次 fit 的 run elapsed 之和，约 18 分钟；它不等于整个周期的总 SSH/GPU/人工/传输费用。SSH 墙钟为 1,181 s，另有命令准备、取回校验、本地整理等未计入 fit 的开销。`max_seconds=1800` 是每次 fit 的协作式参数，不是硬进程上限；实际单次 57.7–85.9 s。

### 3.1 DEV 评估次数

- 每个 fit 的 `progress.jsonl` 32 行，且 32 行都做 DEV 评估：16 × 32 = **512 次 online DEV**。
- 每个 fit 在 selected checkpoint 上另有 **1 次 terminal DEV**：16 × 1 = **16 次**。
- 合计 **528 次 DEV 评估**（512 online + 16 terminal）。
- 每个 fit 的 selected checkpoint 还在 TRAIN/DEV/TEST 三个 split 上做 terminal 评估：16 × 3 = **48 次 terminal split 评估**。

这些 online 数值不是本文件的 TRAIN 指标来源。下表的 `TRAIN` 列是 **selected checkpoint 在 TRAIN split 上的保存指标**，不是训练过程中 online batch 的 TRAIN accuracy/BCE；selected epoch 一律取自 `selection.json`。

## 4. 初始化与“同 SHA/同 shape”边界

| 阶段/组 | 初始化 state SHA-256 | 参数量 | 说明 |
|---|---|---|---|
| N1 r-null/r-real | `65cff3249b5727f2326355f714cb095eaf39bcf411903ca91aab3c50e1579a2c` | 24,961 | R 组；null/real 同一初始化 |
| N1 z-null/z-real | `65cff3249b5727f2326355f714cb095eaf39bcf411903ca91aab3c50e1579a2c` | 24,961 | Z 组；与 R 同初始化、同 shape |
| N1 m-null/m-real | `2e214460c45bb1fb0e11cf81740d1183bcb90bbeb8509f4061d367be9c240ab4` | 24,961 | M 组；由同 seed 初态施加确定性的 match-direct 首层 bias 补偿，并非独立随机初始化 |
| N2 d-null/d-real | `65cff3249b5727f2326355f714cb095eaf39bcf411903ca91aab3c50e1579a2c` | 24,961 | D 组；与 N1 R 同初始化值但阶段/面板不同 |
| N2 r-null/r-real | `65cff3249b5727f2326355f714cb095eaf39bcf411903ca91aab3c50e1579a2c` | 24,961 | N2 R 组；与 D 同初始化、同 shape |
| N2 f-null/f-real | `86fa89c36294ed11f5b4f1a9818bf1ea315ff3d86550bb17731efbec399f4647` | 29,057 | F 组；simon-interactions 表示 |
| N2 fr-null/fr-real | `86fa89c36294ed11f5b4f1a9818bf1ea315ff3d86550bb17731efbec399f4647` | 29,057 | FR 组；与 F 同初始化/同 shape |
| N2 s-null/s-real | `60b4957925f08d2e43fb9607e5e6a1e8ba65a19d16ede0805735aa95cc437750` | 97 | S 组；public-linear |

**不能把“同初始化 SHA/同 shape”当成等价函数**：

- 同一组 null 与 real 使用相同初始化，但监督目标不同（null 标签 vs real 标签），训练后参数函数已不同；不能互相替代，也不能把 null 结果当作 real 的重复实验。
- R/Z、D/R 虽然共享初始化，但表示/readout 路径不同；F/FR 同 shape 但 FR 使用 q6 residual readout，函数不等价。
- 相同初始化只提供可配对的起点，不证明两条路径在训练后等价，也不构成本轮“机制改善”的证据。

## 5. N1：六个 fit 的保存指标

`TRAIN`/`DEV`/`TEST` 均为 selected checkpoint 的保存评估值；`sel` 为 `selection.json.epoch`。N1 无 q6 基线；null 报警规则为 DEV acc ≥ train_constant.acc + 0.03 且 DEV BCE ≤ train_constant.bce − 0.005，三条 null 均未触发。

| arm | label | sel | params | TRAIN acc | TRAIN BCE | DEV acc | DEV BCE | TEST acc | TEST BCE | best ckpt SHA-256（前 12） | 验收文件数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| r-null | null | 2 | 24,961 | 0.507477 | 0.692890 | 0.505371 | 0.693185 | 0.500107 | 0.693339 | `b053ae1c17ad` | 17 |
| r-real | cipher | 32 | 24,961 | 0.986298 | 0.070776 | **0.982788** | **0.078328** | 0.981873 | 0.079465 | `6ed46ed3f768` | 17 |
| z-null | null | 2 | 24,961 | 0.515671 | 0.692433 | 0.501099 | 0.693335 | 0.497528 | 0.693526 | `14e6c6505c8d` | 18 |
| z-real | cipher | 3 | 24,961 | 0.529297 | 0.691472 | 0.499512 | 0.693674 | 0.497345 | 0.693619 | `4e6f77a4baf6` | 18 |
| m-null | null | 2 | 24,961 | 0.515076 | 0.692596 | 0.505737 | 0.693244 | 0.498047 | 0.693432 | `a0cac3f67127` | 18 |
| m-real | cipher | 32 | 24,961 | 0.919876 | 0.212956 | **0.892822** | **0.251285** | 0.889816 | 0.254631 | `9253641e5998` | 18 |

N1 DEV 常数基线（用于对照，不是表格中 selected model 的值）：

| label | DEV const acc | DEV const BCE | TEST const acc | TEST const BCE |
|---|---:|---:|---:|---:|
| real | 0.490112 | 0.693214 | 0.499878 | 0.693153 |
| null | 0.501831 | 0.693143 | 0.499603 | 0.693169 |

**N1 解释**：

- R real 在 DEV 上达到 98.28%，明显高于其 DEV constant（49.01%）；这是对该 3 轮公开表达式标签的拟合结果，不是真实 key、真实 state 或恢复能力的声明。
- Z real（49.95%）接近 chance；M real（89.28%）低于 R real 约 9.00 个百分点。按冻结 DEV 口径，**没有观察到 Z/M 相对 R 的预期机制改善**。
- `z-real − r-real` 的探索性 DEV 差为 accuracy −48.33pp、BCE +0.6153；`m-real − r-real` 为 accuracy −9.00pp、BCE +0.1730（`N1-stage.json.exploratory_dev_relative_effects`）。这些只是单轮 DEV 观察，不用于 TEST 因果解释，也不改变后续配置。

## 6. N2：十个 fit 的保存指标

N2 为 6 轮、full plaintext、五个 real/五个 null。下表同样使用 selected checkpoint 的保存 TRAIN/DEV/TEST 指标，`sel` 来自 `selection.json`，不是 online TRAIN 指标。

| arm | label | sel | params | TRAIN acc | TRAIN BCE | DEV acc | DEV BCE | TEST acc | TEST BCE | best ckpt SHA-256（前 12） | 验收文件数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| d-null | null | 1 | 24,961 | 0.503952 | 0.692893 | 0.503296 | 0.693369 | 0.502167 | 0.693421 | `28a47fa21529` | 20 |
| d-real | cipher | 2 | 24,961 | 0.528458 | 0.691776 | 0.504883 | 0.693006 | 0.499878 | 0.693400 | `4aaffbbb5c0d` | 20 |
| f-null | null | 1 | 29,057 | 0.510529 | 0.692634 | 0.512573 | 0.693175 | 0.501160 | 0.693388 | `03e44f688223` | 21 |
| f-real | cipher | 2 | 29,057 | 0.515823 | 0.692296 | 0.505005 | 0.693043 | 0.499985 | 0.693607 | `b744afba8c73` | 21 |
| r-null | null | 2 | 24,961 | 0.524231 | 0.692316 | 0.503296 | 0.693288 | 0.497681 | 0.693338 | `2ffd5e057c23` | 21 |
| r-real | cipher | 2 | 24,961 | 0.515076 | 0.692344 | 0.506714 | 0.693092 | 0.498993 | 0.693709 | `20d879d3024c` | 21 |
| fr-null | null | 2 | 29,057 | 0.512283 | 0.692612 | 0.502563 | 0.693185 | 0.498581 | 0.693401 | `aa1f93ccd386` | 21 |
| fr-real | cipher | 1 | 29,057 | 0.502289 | 0.692994 | 0.506104 | 0.693153 | 0.500259 | 0.693487 | `7dead1c34df9` | 21 |
| s-null | null | 23 | 97 | 0.514832 | 0.692721 | 0.508911 | 0.693275 | 0.496094 | 0.693825 | `548b5097d920` | 21 |
| s-real | cipher | 10 | 97 | 0.512115 | 0.692652 | 0.507446 | 0.693217 | 0.500626 | 0.694111 | `864777240482` | 21 |

### 6.1 N2 共同 DEV 基线（constant 与 q6 residual）

N2 的 `train_constant` 与 `zero_round_constant_reference_constant` 都是 TRAIN 参数应用到对应 split 的常数基线；q6 的 residual bit 为 0。相同 label 分支的基线在各 arm 间相同，下表按 real/null 分组列出。

| baseline | split | accuracy | BCE |
|---|---|---:|---:|
| real train_constant | TRAIN | 0.500900 | 0.693146 |
| real train_constant | DEV | 0.511841 | 0.693106 |
| real train_constant | TEST | 0.497772 | 0.693157 |
| real q6 residual constant | TRAIN | 0.501480 | 0.693143 |
| real q6 residual constant | DEV | 0.507080 | 0.693110 |
| real q6 residual constant | TEST | 0.500107 | 0.693151 |
| null train_constant | TRAIN | 0.500748 | 0.693146 |
| null train_constant | DEV | 0.504272 | 0.693136 |
| null train_constant | TEST | 0.501953 | 0.693142 |
| null q6 residual constant | TRAIN | 0.500107 | 0.693147 |
| null q6 residual constant | DEV | 0.502441 | 0.693146 |
| null q6 residual constant | TEST | 0.499466 | 0.693147 |

### 6.2 N2 real DEV capability（冻结 gate，全部 fail）

capability 定义为同时满足：DEV acc ≥ DEV train_constant.acc + 0.02，DEV BCE ≤ DEV train_constant.BCE − 0.005，且 DEV BCE ≤ DEV q6.BCE − 0.005。

| real arm | ΔDEV acc vs const | ΔDEV BCE vs const | ΔDEV BCE vs q6 | capability |
|---|---:|---:|---:|---|
| d-real | −0.006958 | −0.0000997 | −0.0001032 | false |
| f-real | −0.006836 | −0.0000631 | −0.0000666 | false |
| r-real | −0.005127 | −0.0000145 | −0.0000180 | false |
| fr-real | −0.005737 | +0.0000467 | +0.0000432 | false |
| s-real | −0.004395 | +0.0001111 | +0.0001076 | false |

五个 real arm 的 DEV accuracy 都低于 constant；部分 BCE 略低于 constant/q6，但远未达到 −0.005 门槛，且 accuracy 差距远大于 +0.02 的反向要求，因此按冻结规则全部 fail。

### 6.3 N2 null 报警（全部 false）

| null arm | ΔDEV acc vs const | ΔDEV BCE vs const | ΔDEV BCE vs q6 | capability | +0.03/−0.005 规则 | alarm |
|---|---:|---:|---:|---|---|---|
| d-null | −0.000977 | +0.0002336 | +0.0002230 | false | false | false |
| f-null | +0.008301 | +0.0000395 | +0.0000288 | false | false | false |
| r-null | −0.000977 | +0.0001529 | +0.0001423 | false | false | false |
| fr-null | −0.001709 | +0.0000496 | +0.0000389 | false | false | false |
| s-null | +0.004639 | +0.0001400 | +0.0001293 | false | false | false |

## 7. 六轮 TEST 观察与限制

- 十个 N2 TEST accuracy 范围为 **0.496094–0.502167**，即约 chance；十个 N2 TEST BCE 范围为 **0.693338–0.694111**，全部高于同 split 的 constant 和 q6 基线（基线约 0.69314–0.69316）。
- 五个 N2 real 的 TEST BCE 相对其 constant 的差值均为正（+0.000243 到 +0.000954），相对 q6 也为正；因此本轮没有“六轮 TEST 优于常数基线”的证据。
- 这**不构成“已学到六轮机制”的声明**，也**不能证明所有网络/所有方法失败**：本轮只覆盖 5 种 N2 公开表示、单一 seed/面板、每个 fit 8,192 updates、无 bootstrap/确认统计；N1 的 3 轮 R 已经能拟合其表达式，说明结论不能外推为“该框架无法学习任何密码结构”。
- TEST 指标由既有 CLI 在 selected checkpoint 后保存；选模只用 DEV。TEST 在本轮中**不是 sealed 确认**，训练路径未启用访问强制封存。分支规则在运行前冻结，实际晋级判断只使用 DEV。完整 CLI/results 含 TEST，执行与验收中可能已经读取，因此不声称实际决策先于所有 TEST 接触；TEST 未用于改变门槛、候选、预算或调参。

## 8. 验证、独立确认与异常说明

- `N1-main-acceptance.json` 与 `N2-main-acceptance.json` 均为 `PASS`，`no_checkpoint_replay=true`；文件 SHA 校验数为 N1 17/18、N2 20/21（按 arm，含额外 metadata/log 文件），逐文件与远端清单一致。
- 所有 16 个 run 的 `selection.test_labels_generated=false`，`selection.checkpoint_sha256` 与 `results.best_checkpoint_sha256` 一致；没有使用 checkpoint replay 生成额外指标。
- `N1-stage.json`/`N2-stage.json` 的 provenance 记录：`checkpoint_load_or_replay=0`、`extra_evaluate_calls=0`、`extra_scientific_labels_or_forwards=0`；这两份阶段记录中的 commit/push=0 仅指阶段执行代理本身；整个周期此前已由主模型发布科学源码提交 `65cf98d`，不能把阶段计数冒称全周期零发布。阶段内源码与配置修改为 0。
- 独立确认结果：**confirmation statistics cells = 0、bootstrap = 0**；`cycle.json` 的 `confirmation_statistics_cells=0`，`N2-main-acceptance.json` 决策明确 `NO bootstrap confirmation cells`。
- N3/N4：`cycle.json` 中 `N3_started=false`、`N4_started=false`；本周期没有 N3/N4 面板生成、fit 或评估记录。
- 0 次 fit retry。N1 首次启动时 PowerShell 在解析阶段报 `IncompleteHashLiteral`，**在训练动作之前终止**；随后用多行 stdin 方式重新启动。该事件是**传输/解析失败，不是 fit retry，也没有产生训练计数**，已记录在 `N1-attempts.json` 和 `N1-report.md` §1；相关日志/报告保留，未删除。
- 首次远端发布前曾出现 `git clone` 的 partial 错误：PowerShell 5.1 将 git 进度 stderr 判为 `NativeCommandError`，只留下带 `origin` 的 `.git`；随后在**同一 `.git` 上 `fetch` + detached checkout** 完成，未删除、未 reset/clean/stash。该事件是发布/传输问题，**不是科学 fit 失败**，详见 `report-remote-setup.md` 与 `remote-setup.json`。

## 9. 产物保留与 Git 边界

- 本地保留 16 个完整 run 目录：`runs/received/multicycle-20261103-0250-{n1,n2}-<arm>/`，包括 `results.json`、`exit.json`、`selection.json`、`predictions/`、`remote-logs/` 与 checkpoint 文件。
- 远端保留：`G:\lxy\shared-m-cipher-nd-runs\multicycle-20261103-0250\runs\` 及对应 `logs\train*`；未删除远端原件。
- 权重、预测数据、日志和 `runs/` 不入 Git；本任务只新增本报告，不 commit/push，不改已有 README/current-state 等 dirty 文档。

## 10. 决策与下一方向

**决策**按冻结 DEV gate 执行：五个 N2 real 全部 fail，N1 Z/M 未显示预期改善；周期关闭，不启动 N3/N4，不做 bootstrap/确认统计，不把本轮 TEST 当作 sealed 验证。

**下一方向**（只做方法论证，不承诺新训练）：

- 在考虑任何新 fit 前，先单独论证为什么新方法能在同样公开信息边界内改变六轮表达式的可学习性；需要明确表示、监督、预算与 DEV gate，而不是重复当前网络族。
- 可优先评估代数/统计/积分类公开特征或可证明的代理指标，并说明它们与真实 key/state 的边界；不得把公开标签拟合结果表述为 key recovery。
- 当前证据不足以支持“扩大网络搜索”或“重复训练直到通过”；若未来要新增训练，需新的明确授权、预算和预先冻结的 DEV 判据。

## 11. 追溯索引

| 内容 | 来源 |
|---|---|
| 周期状态/预算/计数 | `validation/innovation2-multicycle/scientific-20261103-0250/cycle.json` |
| N1 全部指标、配对身份、init/selection 元数据 | `.../N1-stage.json`；逐 run `runs/received/.../results.json` |
| N2 全部指标、q6/constant、null/capability | `.../N2-stage.json`；逐 run `runs/received/.../results.json` |
| 逐文件验收与 best SHA | `.../N1-main-acceptance.json`、`.../N2-main-acceptance.json`、`selection.json` |
| fit 预留/开始/完成、SSH 墙钟、传输故障 | `.../N1-attempts.json`、`.../N2-attempts.json`、`.../N1-report.md`、`.../N2-report.md` |
| 远端 setup、clone partial/fetch 处理 | `.../report-remote-setup.md`、`.../remote-setup.json` |
| 面板、无标签预检与跨阶段零重合 | `.../report.md`、`.../panels.json`、`.../verify.json` |

本文件中的所有表格数值均来自上述保存 JSON；完整精度以各 `results.json` 的原始字段为准。没有从 checkpoint 或 NPZ 重新计算任何数值。
