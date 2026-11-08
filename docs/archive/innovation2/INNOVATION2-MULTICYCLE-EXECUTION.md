# 创新2多周期神经预测 — 执行合同（当前已批准范围）

日期：2026-11-02。状态：**本轮已批准，有界科学周期执行说明**。

本文件是版本化（纳入 Git）的执行说明，用来替代原先写在被忽略目录里的
`validation/innovation2-multicycle/neural/READY.md`；干净 checkout 也能读到同一合同。
本文只**描述**已实现的 CLI 入口、固定矩阵、预算与分支规则，**不新建 runner／调度器／注册表**，
也不改动科学算法、标签或训练代码。

科学方法与候选设计的原文依据：
[多周期研究与实现设计](superpowers/specs/2026-11-02-innovation2-multicycle-design.md)（下文简称 spec），
统计以 [spec §7](superpowers/specs/2026-11-02-innovation2-multicycle-design.md#7-预测证据统计与确认比较族) 为准。
本文与 spec 冲突时以 spec 的方法语义为准；当前授权以本轮用户批准及下节为准，不沿用 spec 的历史未授权状态。

## 1. 当前授权状态

本轮用户**明确批准**了完整科学周期：N1 → N2 → 条件性 N3/N4，在下列固定上限内连续推进，
不再每个 run 单独询问；未通过的阶段不释放额度去扩大搜索，也不临时加模型／seed／轮数。

| 项目 | 已批准上限 |
|---|---|
| 正式 fits | 最多 **36**（N1 6 + N2 10 + 条件 N3 最多 8 + N4 最多 12）；是 36 次 fit，不是 36 种架构 |
| updates | **294912**（每 fit 8192） |
| 训练呈现（presentations） | **75497472**（每 fit 2097152） |
| 真实／null 标签 | real 与 null **各 2506752** 次 |
| 单 fit 训练时长 | **1800** 秒协作上限 |
| 全周期训练时长 | 累计 **18 小时**协作上限（不是预计用时，也不是硬 GPU 截止） |
| 配对统计 | 全周期所有 ≤18 格合计 CPU **30 分钟** / wall **30 分钟** 协作上限，**不是每格 30 分钟**；超限标 STATISTICS_INCOMPLETE |
| 正式重试 | 0 |

本轮同时允许：本地已列明任务文件的**精确路径 Git 发布**（commit/push 与远端 fetch/checkout）、
**SSH 调用远端已有 CLI**、固定矩阵内的 **GPU fit**、**SSH/SCP 取回**至
`runs/received/<run-id>/`，以及独立记账的**保存预测的配对统计**。不得经 SSH 传科学源码／任务包，
远端不 commit/push；训练程序不承担同步或远程编排。本次文档修订不执行这些已批准操作。

授权边界同时明确：**不含安装**（不安装 solver／checker／新依赖），**不含积分支线**
（spec 第 6 节 I0–I3 及 solver／checker 预算均未启动），不改代理配置，不用 Git 回传产物。
GPU 串行使用、按 run 独立记账；缺 exit 一律记 UNKNOWN，不得写成 0 消耗或已完成。

**历史授权不继承为当前依据**：Task6／Task7 时期的本地工程批准只作为历史记录保留。
其中 Task7 的 14 个本地 null smoke CLI fits（**28 updates / 112 presentations**）已在当时完成，
属于**已完成工程检查，本轮不重跑**，也不重新申请；它既不构成本轮科学运行的额度，
也不能反过来证明本轮科学周期"未授权"。当前权限判断只看本节。

## 2. 已实现的 CLI 入口（在训练机运行，本机不代为执行）

- `cipher-experiments check --config <config.json>`：加载并校验配置，**不训练**、不生成标签、不评估。
- `cipher-experiments train --config <config.json> --output runs/<new-run-id>`：执行一次正式 fit。
  正式配置使用 `cuda:0`，smoke 配置使用 `cpu`；`--output` 是**必需参数**，且
  **输出目录必须不存在**（CLI 拒绝覆盖已有 run 目录），因此每次运行都要给新的 run id。
- `cipher-experiments evaluate`：用户主动调用的**独立**评估能力入口，与训练分离；
  本文不提供完整参数示例，实际参数以入口实现为准。

训练程序本身不调用 SSH／Git 同步；同步与取回是外部人工步骤。

## 3. 固定矩阵（已落地配置，未新增）

共同协议（K 系列确认块沿用同一协议）：TRAIN 65536 / DEV 8192 / TEST 65536、pool 65536、
batch 256、32 epochs、Adam lr .001、clip 1、每 epoch DEV 最低 BCE 选模（平局最早）、
width 128、`max_seconds` 1800、`record_predictions=true`；每 fit 最多 8192 updates、
2097152 训练呈现、32 次在线 DEV 选模 + 1 次末端 DEV（≤139264 标签，real/null 同量）。
同阶段配对臂共享输入身份、标签、shuffle 与预算；跨形状模型不因同 seed 声称初态相同。

### N1：背景敏感性机制控制，最多 6 fits

`configs/innovation2-multicycle-n1/`：r-real／r-null／z-real／z-null／m-real／m-null。
三轮、标签 `xor2[0,8]`、高半固定 `X0=0xA5A5A5A5`（`plaintext_fixed_value=2779096485`），
明文空间 `high32-zero`。R 用旧 `mlp`；Z／M 用 `public-mlp` + `background-zero`
（`standard` / `match-direct`）。seed：data 196101、null 196102、init 17、key 190502。

### N2：六轮 full 有限结构方法比较，最多 10 fits

`configs/innovation2-multicycle-n2/`：d／f／r／fr／s 各 real/null。
六轮、`bit` 标签（bit0）、`full` 明文。D 用旧 `mlp`；F／R／FR 用 `public-mlp`
（分别为 interactions-direct／identity-residual／interactions-residual）；S 用 `public-linear`。
参数数固定为 F／FR：29057；R：24961；S：97；D：24961。
seed：data 196201、null 196202、init 17、key 190502。

### N3／N4

N3／N4 的正式配置**尚未创建**：候选未知前不伪造。候选确定后按 spec 固定 schema 生成，
完整 seed／init／面板去重结果在首 fit 前冻结；冻结本身不产生额外授权。
smoke 面板在 `configs/innovation2-multicycle-smoke/`（14 个 null／cpu／1 epoch 项），
只用于工程接线，不是科学结果。

## 4. 分支与资格门槛（人工按 spec 执行的期望表）

- **能力资格（DEV gate）**：DEV ACC ≥ TRAIN 多数 +.02，且 DEV BCE ≤ TRAIN 常数 −.005，
  且 ≤ q_r 常数 −.005（两个常数条件都要满足）。
- **方法优势（非 D 候选）**：先满足能力资格，再对同期 D 的 DEV BCE 改善 ≥.005 且 ACC 不下降。
- **N2 选择**：只按 DEV BCE 选，平局（≤1e-8）先取参数少，再按固定 D,S,R,F,FR 顺序；TEST 不参与选型。
- **N3 双层门**：每轮 4 fits（候选 real/null + 同期 D real/null）。八轮先做 4 fits；
  非 D 候选须同时具备方法优势资格才推进十轮，仅具备能力资格则停在八轮；
  八轮失败则不启动十轮，保留六轮候选进 N4；D 候选只判能力资格，不与自己比较、不复制对照。
- **N4 固定 3 块**：K1／K2／K3 各最多 4 fits（候选 real/null + D real/null），每块新面板 + 新 init，
  K3 用第二把 key 从零训练。候选为 D 时去掉重复 D 及对 D 的自比较，节余不得补 seed。
  层级优先：最高**具有方法优势**的轮数；若从无方法优势，则取最高**具有能力资格**的轮数并收缩方法主张。
- **报警与异常**：任一 null 臂触发固定报警即停并核查数据／RNG／split／指标；缺 exit = UNKNOWN，
  停新启动、不跳到更高轮、不自动重试；共用代码错误降级全部受影响结果。N1 全部失败亦不自动扩矩阵。

## 5. 配对统计族（全周期 18 格，alpha=.05/18）

确认块 K1／K2／K3 × 3 个比较对象（TRAIN 常数 / q_r 常数 / 同期 D）× 2 个指标（ACC / BCE）
= **18 个唯一比较格**，每块 6 格。每格 `alpha = .05/18`，整族总计 .05。
无逐块重分配、不进一步拆分；D 不与自己比较，移除该自比较格后保守地**不**把释放的 alpha
分配给其余比较。区间方法按 spec §7 冻结：逐样本配对 IID bootstrap **10000** 次，
行有放回，同时重采候选与比较对象；显式 numpy **PCG64**，seed 为
`196801 + 100*block + 10*comparator + metric`（三个索引均从 0 起）。
固定 percentile 区间，分位数 `alpha_cell/2` 与 `1-alpha_cell/2`，
`numpy.quantile(method="linear")`，每批最多 **64** 次重采样，不切换 basic／BCa。
报告方串行处理全周期所有 ≤18 格，**合计** CPU／wall 各 30 分钟，绝非逐格额度；
超限记 STATISTICS_INCOMPLETE，不得删失败格后仍称全确认。
统计只消费已保存的预测，不重跑模型前向、不重新生成标签。

## 6. TEST 与身份

CLI 在训练结束后照常生成并输出 TEST 指标与逐样本预测，**没有技术封存**，也没有访问隔离。
确认性来自方法本身：先冻结候选／轮数／分布／优化协议与新面板，再一次性收口，事后不据 TEST 改模型、
不加 seed、不换阈值。因此不得声称 TEST 被技术封存或未被访问；结果取回后不自动本地重算指标。

## 7. 接口边界（只描述，不实现）

- **CLI**：解析参数、选择已实现入口、呈现错误；不含科学算法、训练细节、远程编排。
- **配置**：描述并校验实验参数与合法组合；不执行训练、不生成数据、不审批、不操作远程机器。
- **数据层**：采样、划分、输入编码与显式标签构造；秘密留在标签侧，不进模型输入或诊断。
- **模型／公开表示**：参数与前向；公开表示只做无秘密结构变换，不读配置、产物或额外标签。
- **训练层**：组装、优化更新、预算检查、DEV 选模与最终评估时序。
- **评估／checkpoint**：显式指标、身份校验、严格加载；不反向依赖训练器、不改选模规则。
- **统计／报告**：通过显式输入接收保存的预测与聚合量，只读；不参与梯度、选模或标签生成。
- **产物工具**：序列化、进度与身份记录；不含科学决策、同步、调度或自动重试。

本文不新增上述任何模块的实现，也不提供替代 CLI 的 runner；执行只走第 2 节已实现入口。

## 8. 明确不做

- 不安装 solver／checker／新依赖，不启动积分支线（I0–I3）或任何 solver／checker 调用。
- 不改采样、标签、前向、优化、选模或指标语义；不为过测试改科学定义。
- 不重跑已完成的历史工程（含 Task7 的 28 updates / 112 presentations）来"凑"当前额度。
- 不把工程验证（Ruff／mypy／限定 pytest／CPU smoke）当作科学阳性或已部署结论。
- 不作授权精确路径以外的 Git 发布，不用 Git 回传结果；产物只写 `runs/<run-id>/`，
  取回只写 `runs/received/<run-id>/`。已批准的精确发布与远端同步不在禁止项中。
