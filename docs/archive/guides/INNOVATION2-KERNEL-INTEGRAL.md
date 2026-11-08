# SIMON64 核积分：固定有限周期

本入口是选择明文cube的**完整64位输出XOR核分析**，不是随机明文单bit预测、网络训练、密钥恢复或论文高轮复现。协议标识 `simon64-kernel-v1`；`fit=0`、`retry=0`，不创建checkpoint，不支持积分`evaluate`。

## 入口与当前状态

```bash
cipher-experiments check --config configs/innovation2-integral-kernel.json
```

`check`只严格校验三个身份字段 `name/experiment/protocol`，密码调用0，不导入积分执行器或训练模块。JSON拒绝重复、未知、混合及非有限值；不能用配置或CLI调整科学参数。

正式周期的已实现统一入口为：

```bash
cipher-experiments train --config configs/innovation2-integral-kernel.json --output runs/<fresh-cycle-id>
```

`train`在此表示本机执行有限实验，不进行拟合。输出目录必须全新；失败目录不能覆盖、恢复或自动重试。开发源码可用 `PYTHONPATH=src .venv/bin/python -m cipher_experiments.cli ...`。**实现入口不等于授权执行**：Tasks1–3只作合成验证，真实运行仍须控制器独立验收、code-review和Task4门禁。**“尚无真实密码科学结果”是执行前历史状态**：当时本协议尚未运行；正式周期已于2026-11-04执行并关闭，执行后结果见下文“实际结果”。

此协议对应一次性授权窗口，生产时钟检查截止 **2026-11-04T12:57:56.114Z**；到期后停止，不能通过更换配置name延长。开发测试可显式注入fake oracle/collector/clocks，但这些不是另一个CLI科学协议或缩小运行模式。旧prediction/innovation1默认配置、序列化SHA与checkpoint/evaluate语义不变。

## 固定顺序、数据与空间

1. 在任何科学调用前落盘源码/config身份、全部640候选及score、两臂名单和分数/tie/重叠统计。每k=2..6有128个高字cube，池原序前8为R，按 `(score, pool_index)` 取前8为G。唯一评分是公开AND相遇评分，不能读key或密文；全同分时保留原序。
2. 顺序完成真实控制r6/{0,4,10}、r7/{0,2}、r8/{0,2}，再完成匹配r6/r7的两个随机函数null。每面板依次discovery128、screen256、confirm2048行。r6/r7要求e0属于各阶段完整核和high-only空间，**不要求核等于span(e0)**；r8只记录，不要求随机采样必遇已知反例。null发现128行后核非零即报警，停止新增调用，不重抽。
3. 搜索只在八轮、高字k=2..6；按k、R然后G的冻结顺序遍历两臂并集。每唯一cube采128+256=384行。相同cube跨臂实际只计算一次，两个臂槽位仍分别报告固定名义成本；去重节余额度不转投新cube。
4. 全部搜索筛选完成后，每arm/k只从足384行且非平凡高字维数>0的候选中选择最多一个：非平凡维数降序、高字维数降序、池原序号升序。`selection.json`及完整筛选空间先落盘，再采确认；最多10臂槽位，共享cube确认一次。确认失败不补位；无人晋级则跳过搜索确认，仍完成比较。

每行是同一合法主key下完整2^k点的64位密文XOR。key只在数据层；control/search和三个phase独立域，同一role/phase的key索引序列跨cube共享。背景按role/round/cube/phase独立域分离，臂名不进入身份。只保存parity行，不保存key、点级密文或轮trace。

发现求完整右核；后续对**整个空间**取交集，允许两个失败原基的XOR继续存活。high-only是完整核与低32系数为零空间的交集，不是投影或删除混合原基；用同一累积行高字列块直接求核交叉核对。公开缺依赖的高字单位mask空间T须满足T⊆H，主维数为dim(H)−dim(T)，违例停止，不截负数。完整低字/混合关系只作诊断。

## 预算与退出

- 科学密码attempt≤1,308,672，completed单独计；失败批次attempt不退还。工程KAT另计4096，不归本执行器。
- null≤29,184个64位word；真实与null不是同一账本单位。
- 单进程CPU和单调wall各累计≤1800s，含候选生成、评分、采样、消元、身份检查与写盘。
- 每面板/cube的发现+筛选共用60s CPU/wall；确认另有120s。冻结、选择、比较也检查60s及全局/授权限额。达到上限即停止，不把阶段重置当延期。
- 时限是批次/操作边界的**协作检查**，不是进程树硬杀。单个调用/系统I/O可越界，失败收尾也可能超限；保留实际时钟。强杀、磁盘不可写可能无exit；缺exit一律UNKNOWN。

`completed/COMPLETE`仅表示固定周期完整结束；空间通过最多标`EMPIRICAL_SURVIVOR`，不是全key证明。异常、控制失败、partial、计数/时间/身份/数据损坏均停止新增调用，`failed/UNKNOWN`且CLI非零；不默认未完成维数为0。

## 产物

```text
config.json                 # 固定身份配置
manifest.json               # 源码逐文件SHA、聚合SHA、协议/根seed/限额/截止
candidates.json             # 全640候选、score、两臂、tie/重叠/名义成本
sampling.json               # 公开域采样draw数，不含key值
progress.jsonl              # 阶段/行持久化进度、累计counts及clock
stages.json                 # 每阶段起止counts/clock/status
phases/<role>-<source>-r<r>-<cube>/<phase>.rows
                            # packed64，每行16个小写hex及LF
phases/.../<phase>.json     # 行SHA、身份/域、before/after、空间与完整性
controls.json               # 全控制完成后汇总
selection.json              # 确认前冻结的槽位/排名规则及全部筛选空间
comparison.json             # 各k各臂全部8槽位、确认结果、C_R/C_G
results.json, exit.json      # 最终状态、实际counts、时钟、停止原因/exit_code
```

phase的`space`只在完整且检查通过时有效。未完成或失败置null，`observed_space`可保存已确认prefix的诊断维数/基，不能参与晋级；商维数未知时为null。未晋级确认不是0维，比较中为null/NOT_SELECTED。所有mask为完整64位hex，坐标i对应1<<(63−i)。行文件只追加已确认行；元数据用原子replace快照（无断电fsync保证）。

每臂发现/筛选名义成本380,928块，确认最大253,952块；逐k槽位及实际去重总counts分别报告。主终点是确认后非平凡高字正维层数C_G/C_R；只有C_G>C_R且至少一个G成功来自不同cube才称本固定池探索性优势。不称显著、跨seed稳健、新颖或密码预测突破，不自动追加实验。

## 实际结果（2026-11-04 固定周期）

唯一一次正式周期已结束并关闭，结果为**基线控制通过、八轮主目标失败**：八轮 `k=2..6` 共 80 个 arm-slot、76 个唯一 cube，所有 screen 的 `high_dimension` / `T_dimension` / `nontrivial_dimension` 均为 0，无候选晋级，confirm 未运行（不是 confirm 后 0 维）；`C_G=C_R=0`。完整结果、全部槽位表、成本与限制见 [核积分固定周期报告](../result/20261104-innovation2-kernel-integral-cycle/report.md)；原始运行仍在 `runs/kernel-integral-20261104-0935/`。本周期不追加、不重试，不自动改 score / k / seed。
