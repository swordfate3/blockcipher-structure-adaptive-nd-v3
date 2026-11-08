# SIMON64 Kernel Integral Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现并在批准预算内完成SIMON64核积分基线、八轮组合与单一结构评分比较。

**Architecture:** 纯GF2与公开结构不接收key；采样/密码仅数据层；执行器只组装有限阶段；配置/CLI显式分派且不改变旧训练语义。原始产物runs，简报result。

**Tech Stack:** Python3.12、stdlib（GF2/结构/采样）、现有NumPy密码批接口，现有pytest/Ruff/mypy；不安装。

**Spec:** `docs/superpowers/specs/2026-11-04-innovation2-kernel-integral-design.md`（用户已在本轮“是”批准全文，原设计快照保留）。

## Global Constraints

- `/workspace` 原地 main，保留无关dirty；不worktree/commit/push/reset/clean/stash。按项目固定协作方式覆盖通用skill默认。
- 授权起点2026-11-04T06:57:56.114Z，6小时到2026-11-04T12:57:56.114Z；正式合计CPU/墙钟各1800秒，逐stage协作限额见spec；非进程树硬截止。
- 科学密码attempt≤1,308,672；工程真实≤4096；合计≤1,312,768；null64bit-word≤29184；fit=0，retry=0。
- 无GPU/SSH/安装/Git发布/solver/神经训练/旧evaluate；只有Task4在控制器核验后允许真实调用。Tasks1–3只fake/synthetic；不得跑裸全pytest造成越权调用。
- 用`PYTHONPATH=src /workspace/.venv/bin/python`（不resolve）及`/workspace/.venv/bin/ruff,mypy`，作用域仅新模块/指定兼容测试；禁止import备份。
- 每任务TDD RED/GREEN日志存其SDD目录，只有控制器派审查者；代理不派子代理。
- packed向量统一使用自然整数位：坐标i对应1<<(width-1-i)，64bit中bit0为1<<63；GF2内部系数坐标可LSB但接口必须明确。
- 任何被修改现有文件只精确edit。任务报告/brief/测试日志留SDD，不删除无commit的执行证据。

## Review Focus

- 验证两条失败基的XOR可存活：Task1独立穷举验证空间交集，不能只筛原基。
- high-only不是投影：Task1同累积行`kernel(M_H)`与约束完整核对齐；e0成员而非核唯一。
- 批次失败/边界超限：Task2预留全部attempt，不回收；Task3失败exit与partial rows保留，不补位重试。
- 同cube跨臂共享但控制/phase独立：Task2采样身份测试＋Task3共享实际计数、名义臂成本、确认选择先落盘。
- 旧CLI与key泄漏：Task3旧配置字节身份/非积分路由保持，check零密码；Task2/3真实key与trace不序列化，异常不打印oracle参数。

---

### Task 1: 纯GF2及公开cube评分核心

**Files:** 新增`src/cipher_experiments/integral_linear.py`、`src/cipher_experiments/representations/integral_cubes.py`，测试`tests/test_integral_linear.py`、`tests/test_integral_cubes.py`。

**Interfaces:**
- `canonical_basis(vectors: Sequence[int], width: int) -> tuple[int,...]`：规范全RREF行空间（高位pivot优先，降序），输入向量均width以内、bool拒绝。
- `kernel(rows: Sequence[int], width: int) -> tuple[int,...]`：返回右核向量规范基，同packed位序；width1..64，rows≤4096。
- `restrict_basis(basis: Sequence[int], rows: Sequence[int], width: int=64) -> tuple[int,...]`：span(basis)∩ker(rows)，组合而非删基。
- `high_only_basis(basis: Sequence[int]) -> tuple[int,...]`：64位基与低32系数零空间交集。
- `contains(basis: Sequence[int], vector: int, width: int=64) -> bool`。
- `and_encounter_score(active_bits: tuple[int,...], rounds: int=8) -> int`、`trivial_high_masks(active_bits, rounds) -> tuple[int,...]`；只公开结构，输入位0..31，rounds1..8，k1..6（控制k3,k2）。保守依赖按spec，不取秘密/不cipher。

- [ ] 写独立枚举n≤5矩阵核/行空间测试，包含empty/fullrank/zero列/非法dtype；只按点积检查真值集合，不镜像消元。
- [ ] 写组合存活、high-only非投影、规范空间相等、T成员、手算两轮评分与位序测试；RED缺功能后最小实现。
- [ ] 执行`PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_integral_linear.py tests/test_integral_cubes.py`，Ruff/mypy只这4文件；无真实调用。
- [ ] 写任务报告，控制器独立验收并派code-review，修important后标完成；不commit。

### Task 2: 严格配置、域分离与计量采集

**Files:** 新增`integral_config.py`、`data/integral.py`、`configs/innovation2-integral-kernel.json`、`tests/test_integral_config.py`、`tests/test_integral_data.py`。

**Interfaces:**
- 冻结`IntegralConfig`仅字段`name`和`experiment='integral-kernel'`及明示`protocol='simon64-kernel-v1'`，科学值固定于本协议，严格键/重复JSON/类型；提供`to_dict,sha256`和`load_integral_config(path)`，不更改旧ExperimentConfig。
- 域分离固定`derive_seed(root:int, *parts:str) -> int`：SHA256(json.dumps(["simon64-kernel-v1",root,*parts],ensure_ascii=True,separators=(",",":" )).encode("ascii"))转big-endian整数，测试向量在报告及冻结输出保存。
- `candidate_pools() -> dict[int,tuple[tuple[int,...],...]]`：每k独立seed(197101,'pool',str(k))，rng.sample(range(32),k)升序、按抽取顺序去重128个，最多2048draw。
- `cube_points(active_bits,background) -> tuple[int,...]`：MSB组合顺序、清/核背景活跃位。
- `KeyPanels`仅采集层内持有key：control/search×discovery/screen/confirm，control3个面板可共享control phase相同序列；阶段长度128/256/2048，采样root197102，域('keys',role,phase)，全不同域之间唯一；不输出实际key。`backgrounds(rounds,active_bits,phase,count,role)`用197103('backgrounds',role,str(rounds),cube-id,phase)。cube-id=逗号连接升序十进制位置。全部key/背景拒绝重复上限16×count。
- `CallBudget`和`collect_parities`可用明确dataclass API由本任务定，在报告提供给Task3：支持oracle(plaintexts:tuple[int,...],keys:tuple[int,...],rounds:int)->Sequence[int]，每批≤64块；批调用前预留attempt，不因失败退还；completed按实际确认返回的完整合法输出计；partial rows/单调计时检查可传回调。真实oracle适配器只延迟导入现有encrypt_expanded，Task2不执行。
- null按197104('null',role,str(rounds),cube-id,phase,row-index)每点getrandbits(64)，不调密码，计word；保存行但不保存点级密文/key。

- [ ] 写RED严格配置/域向量/不相交key与背景/去重耗尽/fake全密文XOR/位序/批异常计数/非法输出拒绝/截止前检查测试；patch真实cipher为raise以防误调。
- [ ] 实现仅自身文件，API具体化写报告。JSON配置无epochs/model字段，check不能产生数据。
- [ ] 跑仅本任务2测试文件、Ruff/mypy本任务4py文件；不KAT/真实密码。
- [ ] 控制器验收，code-review后进入Task3；不commit。

### Task 3: 有限执行器、报告与统一CLI接入

**Files:** 新增`integral_run.py`、`integral_reports.py`、`tests/test_integral_run.py`、`tests/test_integral_cli.py`，精改`cli.py`，文档`docs/INNOVATION2-KERNEL-INTEGRAL.md`和`ARCHITECTURE.md`相关段；不碰旧training/checkpoints/密码算法。

**Interfaces:** 从Task1/2报告读取实际API；`run(config: IntegralConfig, output: Path, *, oracle=..., clocks=...) -> dict`为唯一有限周期入口。test-only fake放tests；生产只暴露必要依赖注入，不增smoke科学参数。

- [ ] RED fake控制全阶段验证、正控e0成员/允许更多基、null警报停止、筛选无候选跳confirm、不同原基组合存活、同cube跨臂只算一次、阶段partial/未知/预算到期/输出已存在、候选排名tie与择优规则测试。
- [ ] `integral_reports`拥有纯结果对象的空间汇总（接收线性模块计算好的维数/基，不自己cipher）；每phase saved packed rows 16hex/行newline，SHA，rank/nullity、high basis、T及nontrivial dim、before/after，comparison/selection元信息JSON。全stage保存计数、progress、source身份、exit；不序列化key/轮trace，不dump异常oracle参数。
- [ ] 正式执行前保存全部候选640、score、R/G名单、源码/config哈希；search所有cube筛选后先存selection.json再采confirm。控制r6/r7/r8和2null先完成；遵守完整spec，任何科学错误停止，不重试。
- [ ] 动态全局CPU/wall和stage clock覆盖生成/消元/写盘，单线程CPU；数据层扣科学调用，run另check1800s总额/60或120stage。去重科学预算只减实际工作，不释放为新cube；每臂名义成本仍报固定槽位。
- [ ] CLI仅先严格读取一次JSON判断明确experiment；未知/重复/混合字段拒绝。integral check只配置不导入/运行训练；train分派integral_run，无checkpoint/evaluate。旧路径继续原load_config，默认/字节SHA不变。帮助说明integral fit0；不要registry/通用runner。
- [ ] 新CLI fake闭环、旧CLI合成2更新可最多一次（不是真密码），放洁净源码/config副本、不安装、使用原venv、显式PYTHONPATH；或等价mock旧路由测试后将真合成闭环交Task4统做，不能两边重复跑。
- [ ] 仅指定4个新测试+Task1/2测试，Ruff/mypy所有改动py；code-review任务门禁和整体review完成后，控制器再批准真实入口解锁。

### Task 4: 工程KAT、干净CLI、冻结与一次正式CPU周期

**Files:** 仅新`validation/innovation2-kernel-integral/`的验收脚本/计数日志；`runs/<fresh-cycle-id>/`正式输出。不修改科研实现，发现错误交原implementer合成RED修复并复审。

- [ ] 控制器实测新测试+Ruff/mypy，核无无关diff、来源与环境、独立审查；任务验证必须不导入/调用密码。
- [ ] 唯一一次旧合成CPU CLI2updates+新integral check在全源码干净临时副本，结束自身临时资源；新CLI不运行fake之外的缩小科学协议。
- [ ] 独立工程agent核SIMON44 KAT（已知P/K/C），本地scalar/expanded按既有API比对，完整生产加密块总数≤4096，各调用显式计数及身份；不打印key，合成KAT公开向量不保密但不额外展开。只跑此脚本不pytest真实密码集。
- [ ] 工程通过冻结所有相关源码/config/协议SHA后，执行**一次**`PYTHONPATH=src .venv/bin/python -m cipher_experiments.cli train --config configs/innovation2-integral-kernel.json --output runs/<fresh-id>`；已有输出拒绝、正式重试0。experiment agent拥有长跑执行，主模型仍可处理取消；不新造runner。
- [ ] 核所有输出计数/exit/源/config/phase选择先后/SHA/空间，仅已保存parity可GF2复核，不重新加密；无exit UNKNOWN。预算/控制失败即关闭，不追加候选。

### Task 5: 独立结果核查与简单交付

**Files:** 新`result/YYYYMMDD-innovation2-kernel-integral-cycle/{report.md,index.html,source.json}`，精确补`result/README.md,index.html`导航；小文档`docs/INNOVATION2-KERNEL-INTEGRAL.md`仅追加实际状态。

- [ ] 从已保存聚合/行SHA生成结果：各k R/G8槽位、score tie/重叠、筛选维数、晋级/确认维数、C_G/C_R、实际/名义成本、失败/未知/控制，所有候选不只最佳；未晋级不补0确认维度。
- [ ] 代理独立静态研究审查，不称有限经验全key证明/显著/新纪录/单点预测突破；低字/平凡关系不算主收益。
- [ ] 控制器核字节来源SHA、JSON有限值/HTML链接/无远程资源、实际账本及窗口；不重放密码/模型；保留原失败证据，无Git发布。

## Plan self-review / handoff

所有spec段落由Tasks1–5覆盖；Task1定义packed basis，Task2采集API在task报告冻结后Task3消费；仅Task3改cli与ARCHITECTURE，Task5只能追加文档/结果。五条Review Focus分别落Task1/2/3。任务代码无commit步骤，项目规则覆盖通用模板。用户已经明确连续委派执行，不在此再要求选择执行方式或重新批准同一计划。
