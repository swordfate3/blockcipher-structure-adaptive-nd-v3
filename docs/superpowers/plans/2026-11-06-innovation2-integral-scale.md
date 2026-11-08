# Integral Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完整2080维经验积分搜索：k10覆盖SIMON64/128的11—14轮，按门禁进入k16阶梯，独立确认与有限q1认证。

**Architecture:** 新协议独立模块，数据层私有采样并逐点聚合；纯GF2求全空间、公开符号模块认证；专属执行器仅装配阶段/预算。旧协议和训练语义不变，CLI仅严格分派，无通用registry/远程编排。

**Tech Stack:** Python3.12.15、现有NumPy、stdlib、pytest/Ruff/mypy；不安装依赖。

**Spec:** `docs/superpowers/specs/2026-11-06-innovation2-integral-scale-design.md`（用户“推进吧”批准）。

## Global Constraints

- 当前main/HEAD7c66151c780bc279b425183cd88fcd3c2dac99f0；保留全部dirty/历史，no Git mutation/worktree/commit/push/reset/clean/stash。
- `/workspace/.venv/bin/python`，`PYTHONPATH=src`；mypy用模块；不依赖已安装旧wheel。只跑明确fake/synthetic范围，不全仓pytest。
- 真实调用只在Task6通过前置验收后由指定executor一次执行：工程cap131072/science726663168/combined726794240；nullcombined726729728；正式1次/retry0，fit/evaluate/update0。
- 单worker，NumPy线程1，RSS停止线2.5GiB、块≤65536、产物≤512MiB、formal wall10800/CPU14400秒，symbolic CPU1800、baseline600、代表120秒/pair20000000/frontier100000；首次工程真实调用起24h有效。
- 上限/位置/阶段/选择顺序全部遵循Spec；实现者无权减输出族、减少发现行、更换cube或重用确认样本。私有key/种子/状态/点密文不出数据层/不落盘。
- 代理不再派代理；实现默认implement-deep。每task fresh审查→控制器核diff和实测→下一task。独立任务可并行但文件不重叠。
- 用户要求中文一页简版为默认入口，技术计划保留；文档简化不改变科学预算或批准范围。

## Review Focus

1. 中途oracle/写盘/取消失败：已预留调用不退款、不重复采集，partial永不当完整行（Task2/4/5）。
2. 确认集泄露/全空间求交错误：冻结完整核后才C，失败不捞子空间；k16结构只用D/S（Task1/4/5）。
3. 大规模点账本/2048式欠定假核：按组去重/分块，2112发现；完整2080bit特征逐点聚合（Task1/2）。
4. symbolresource/基线遗漏：768单位上限、全部共享额度、BASELINE_PARTIAL中完整dimB/extra=null（Task3/4/5）。
5. 假完成与性能外推：k10/16端到端real/null测速，首正式前与k16前门禁、完成事件和落盘均在累计时钟内（Task4/5/6）。

---

### Task 1: 纯宽GF2与完整经验空间

**Files:** Create `src/cipher_experiments/integral_gf2.py`, `integral_empirical.py`; Modify `integral_symbolic.py`（只提取数学函数并兼容再导出）；Create `tests/test_integral_gf2.py`, `tests/test_integral_empirical.py`。

**Interfaces:**
- `integral_gf2` 保持原 `canonical_basis(vectors,width,*,check=None)`, `kernel(rows,width,*,check=None)`, `contains(basis,vector,width,*,check=None)`, `quotient_basis(space,subspace,width,*,check=None)` 签名/异常/坐标/范围，WIDTH2080/MAX_TERMS33280；旧`integral_symbolic`所有公开名称仍可用。
- `integral_empirical.discover(rows: Sequence[int], *, check=None) -> tuple[int,...]` 固定width2080规范核。
- `screen(basis: Sequence[int],rows: Sequence[int],*,check=None) -> tuple[int,...]` 完整span(basis)∩kernel(rows)。
- `violations(basis,rows,*,check=None) -> tuple[tuple[int,int],...]` 返回所有(rowindex,basisindex)非零，不筛空间。
- `linear_subspace(basis,*,check=None) -> tuple[int,...]` 与前64坐标子空间求交，非投影；`select_geometry(h_dim:int,l_dim:int,h_rank:int,l_rank:int)->str` 遵照Spec。

- [ ] Step1 RED测试小width独立穷举kernel/span/ranknullity；2080高低末bit；空矩阵/fullrank/冗余/非法bool；screen组合相消例：两basis分别违反但其XOR满足；linear子空间同例；全basis确认比只测4代表更严；selection平局H。
- [ ] Step2 用新增测试命令记录真实RED（不得刻意破坏旧源码）。
- [ ] Step3 精确提取原GF2，经验层不读cipher/config/I/O，额外独立数学实现用于测试不复用被测函数。
- [ ] Step4 scoped pytest新增两文件+旧`test_integral_symbolic.py`，Ruff/mypy相关3+2文件；生产encrypt0。
- [ ] Step5 fresh review并核旧API/异常兼容；保存diff/测试日志，不commit。

### Task 2: 私有完整cube数据与2080位聚合

**Files:** Create `src/cipher_experiments/data/integral_scale.py`, `tests/test_integral_scale_data.py`。不改cipher/父包旧行为。

**Interfaces:**
- `ScalePanelBank(*, oracle: BatchOracle|None=None, check:Callable[[],None], real_limit:int, null_limit:int, entropy:Callable[[int],bytes]|None=None)`。oracle接受plaintext/key tuples/rounds，与现expanded adapter兼容；None仅调用时lazy生产导入。
- `collect_group(domain:tuple[str,str,str],index:int,rounds:int,active:tuple[int,...],*,null:bool=False) -> GroupRow`；公开`GroupRow`只含domain/index/2080bit row/attemptedcompleted counts/不可逆摘要，`to_dict()`不含秘密。组唯一(domain,index)，重复拒绝；private entropy仅测试可注入。
- `aggregate_words(words:Sequence[int],*,check=None)->int` 完整2080bit逐点parity，可chunk-XOR组合。
- `cube_chunks(active:tuple[int,...],background:int,*,chunk_size:int=65536)` 数据层私有使用，公开测试可调用，返回有界明文块；在测试可支持k1..16，正式config只k10/16。
- `snapshot()->dict[str,object]` 仅公开attempted/completed/pairunique/null计数。reserve before side effect，失败固定错误码与sanitized snapshot，禁止traceback携私有locals。

- [ ] Step1 RED直接双循环oracle逐项比较64+2016项（0/1/高低bit/重复点/chunk）；cube位序/清active/不重漏；shared密文AND parity≠bitxor product回归。
- [ ] Step2 RED fakeoracle失败/短返回/非法uint64/写回check失败/entropy碰撞/域重复/取消/越cap及不退款，组key精确去重、组内samekey，产物/异常无秘密。
- [ ] Step3 逐块生成/向量化聚合，不建点级dict和N×2080，RSS可控；null独立uniform64，real/null私有域隔离；使用按组key集合确保uniquepairs。fake tests0生产调用。
- [ ] Step4 scoped pytest此文件、Ruff/strictmypy；小数组直接oracle可测fake words，禁止真实加密及旧正控。
- [ ] Step5 independent review/main verification before downstream。

### Task 3: 广义公开support与有限认证

**Files:** Create `src/cipher_experiments/representations/integral_support.py`, `integral_scale_certify.py`, `tests/test_integral_support.py`, `tests/test_integral_scale_certify.py`。旧prefix/suffix不改。

**Interfaces:**
- `SupportBudget(check:Callable[[],None],max_pairs:int=20000000,max_frontier:int=100000)`累计counter；pair在尝试前reserve，超限raise固定resource异常不返回ZERO。
- `SupportOracle(rounds:int,active:tuple[int,...],*,budget:SupportBudget)` p1..13/k1..16；`query(state_mask:int)->SupportEvidence`、`verify(evidence:SupportEvidence)->bool`均使用共享budget/check。support证据携recurrence/method身份。
- `certify_relation(mask:int,oracle:SupportOracle)->RelationCertificate` q1精确GF2重建，只所有残项均完整ZERO才certified；证书状态CERTIFIED/UNRESOLVED/UNKNOWN_RESOURCE，不复用旧flags。
- `baseline_for_cell(basis:Sequence[int],oracle:SupportOracle,*,check=None)->BaselineResult` 64unit、copy/h及公开依赖；完整线性交集来自Task1，逐生成元membership；partial完整dims null。
- 认证层不拥有wall策略，oracle共享callback计量；runner分配baseline600/symbol1800/perrel120。

- [ ] Step1 RED k2新旧判fullmask逐query一致（旧p<=9），k≤4纯人工全支持参考验证maximal-antichain等价，COPY共享误报保持UNKNOWN，不宣称更强。
- [ ] Step2 RED 全部残项+keymask身份准确、不将两不同keyterm消去、groupconstantkey不注真实值、已旧7轮证书当公开fixture对照；边界超限/0mask/非法active/查询截断不签证。
- [ ] Step3 新广义支持实现有pair/frontier检查，证据可重放且不读key/data；baseline最多64单位/cell，不做额外闭包；旧七轮不适用就明确没有该生成元，不伪造十一轮继承。
- [ ] Step4 scoped四项新旧support/suffix tests/Ruff/strictmypy，生产0。
- [ ] Step5 independent review/main verify。

### Task 4: 固定协议配置、阶段装配、预算与CLI

**Files:** Create `integral_scale_config.py`, `integral_scale_run.py`, `integral_scale_reports.py`, `configs/innovation2-integral-scale.json`, `tests/test_integral_scale_config.py`, `test_integral_scale_run.py`, `test_integral_scale_cli.py`; Modify `src/cipher_experiments/cli.py`, `ARCHITECTURE.md`; Create `docs/INNOVATION2-INTEGRAL-SCALE.md`。

**Interfaces:**
- `IntegralScaleConfig(name:str,experiment='integral-scale',protocol='simon64-integral-scale-v1')` strict3field/64KiB/duplicates/nonfinite/bool禁，to_dict/sha256兼容项目风格；check无采样/runner/cipher/training导入。
- 配置模块提供冻结P/H10/L10/H16/L16、D2112/S320/C256、caps；不承载审批。CLI识别新三字段类型，旧行为不变。
- `integral_scale_run.run(config:IntegralScaleConfig,output:Path,*,dependencies:RunDependencies|None=None)->dict[str,object]` 同步新协议；`RunDependencies`仅测试注入fakebank/clock/memory/工程证据读取，不给生产JSON缩规模入口。
- `integral_scale_reports` 标准库public-only JSON/packed rows/freeze/sha/exit writer，exclusive claim先于生产依赖、完整行写成功再承诺完成；异常保留实际completed不重采。
- 工程gate证据从本周期固定路径`validation/innovation2-integral-scale/engineering-execution.json`读，绑定source/config/授权24h/吞吐，身份不符先停止；无该记录不跑生产。

- [ ] Step1 RED阶段顺序/real+null各自早停/2112>2080/完整冻结sha先C/C失败不捞/确认全部basis；k16选结构只用DS(rank/dim/tieH)，k10无论阴性覆盖8cell，k16十一轮自动进入/失败停更高轮。
- [ ] Step2 RED现有output、坏engineeringgate、source漂移、deadline、累计wall/CPU/RSS/size边界、每块扣费与异常；输出保存失败结束后计量仍保留completed；共符号预算与768unit/partialdims null。
- [ ] Step3 装配已验收模块，实时cancel/配额/进度，禁止通用orchestrator；两倍吞吐×real/null最坏组数+600+1800门禁、k16前剩余wall重算；group count/cellids/事件与状态公开。formal计时从run入口（含claim/gate）至exit持久化。
- [ ] Step4 CLI check subprocess隔离证明0productioncalls/no runner，旧配置fixture exactSHA/_integral_config兼容，fake数据完整生命周期和干净源码copyfake CLI；mypy/Ruff/pytest范围记录。
- [ ] Step5 独立审查整个新协议与旧入口diff，主模型实测通过才准真实工程。

### Task 5: 有界工程入口和保存证据验证器

**Files:** Create `validation/innovation2-integral-scale/{engineering_check.py,test_engineering_check.py,verify_saved.py,test_verify_saved.py,clean_cli.py,test_clean_cli.py}`。不读取backup不import旧validation作为生产。

**Interfaces:**
- engineering脚本默认plan/0cipher，`--execute --output <fresh-file>`才真调用；精确66,753real计划+66560null、global工程cap131072；首次真实调用时间原子持久化到本周期授权checkpoint，防止重跑重置24h。
- 固定KAT1+11..14 r16pairs×2=128+旧10r L8groups×4×2=64+k10one1024+k16one65536；k10/16完整real/null聚合与写行时间，禁止求经验核/科学探索；工程artifact不留秘密。输出source/confighash/attemptcomplete/time/memory/throughput/预测/gate。
- `verify_saved.py --run <dir> --output <fresh-dir>` 只stdlib，独立从packed行构建全核/DS交/C所有残差/k16选择/阶段/计数/null/证据/预算，不能import生产模块、生成cipher/label。认证用独立公开重放；原始私有样本不可再生边界保留。
- clean源码copyCLI fake闭环，不调用productioncipher/fit/evaluate；保留日志/失败，不覆写原目录。

- [ ] Step1 fake RED cap/invalidoutput/工程中断/不退款/保存失败/旧首次时间不能reset/已有output零调用拒绝，exactcounts计划。
- [ ] Step2 verifier RED篡改rowSHA/全basis确认最后一basis/改变k16选择/C依赖筛选/伪baselinecomplete/deadline rollback/中途超wallCPU/RSS/漏事件/不全D/非法packed/UNKNOWN冒0；新鲜`python -I`进程no productionimport，poisonfixture真检测。
- [ ] Step3 实现保存验收和工程脚本（只fake运行本task），独立复核不是调用同一kernel；unresolvedsymbol保留不证明nonzero。
- [ ] Step4 scoped validation/test+相关源测试、Ruff/mypy；fakeclean集成全阶段+失败路径，记录未执行真实。
- [ ] Step5 fresh全变更审查/修复波→控制器实际验收后冻结source身份。

### Task 6: 一次工程→有界正式→图文结果

**Files:** New `validation/innovation2-integral-scale/{authorization.json,pre-science-freeze.json,engineering-execution.json,science-invocation.json,closeout.json}`，新runs目录，`result/<cycle-id>/{report.md,index.html,source.json}`；更新本周期progress/plan状态。脚本源码不在执行期间暗改。

- [ ] Step1 核对authorization、首次clock、cap、审查与当前sourcehash，显式指定一executor，其他代理只能只读；工程精确一次，所有真实消耗计账。
- [ ] Step2 正控或吞吐门禁失败：关闭BLOCKED_*，交付实测速度/资源与下一请求，不跑科学、不减族；所有剩余额度失效。
- [ ] Step3 门禁通过：source/config/输出/授权冻结，正式统一CLI一次（train --config configs/innovation2-integral-scale.json --output runs/<fresh>），保存stdout/stderr/exit/时间；k16前再次检查，不重试。
- [ ] Step4 只读saved验收、不重新cipher/evaluate。交付一页通俗摘要+技术appendix、输入流程图+实际轮数/成本/维数/状态图，来源SHA，HTTP200；未启动/未知不能画0。
- [ ] Step5 独立结果核对与控制器新鲜验证、final账本与closeout，明确budget/time剩余失效；不commit/push。

## 实施与审核顺序

Task1→Task2→Task3→Task4→Task5→Task6。可在Task1期间只读规划Task2，但不跨文件并行修改公共接口。所有task终点都有review门禁。实现计划写完需向用户展示简版并确认其捕捉设计，再进入Task1；不重复询问已经确定的代理分工或科学预算。

## Self-review

- 数据/阶段/矩阵/确认/selection/null覆盖Task1/2/4；符号与baseline限制覆盖Task3/4/5；预算/保存/验收/交付覆盖Task4/5/6。
- 所有新公开接口上文逐一命名；跨task有依赖则实现不得另取名。Task4冻结RunDependencies时可补确切typed字段，不改变科学输入。
- 计划不默认真实调用tests；真实集中Task6且gate批准；无后台无界调参。
