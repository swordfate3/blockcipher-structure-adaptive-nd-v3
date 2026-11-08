# Innovation2 Split-and-Cancel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实施已批准的A→B有限符号积分周期并一次执行，明确新认证关系相对已列基线的增量。

**Architecture:** 精确q1后缀、保守prefix oracle、公开矩阵消元彼此分离；数据层独占key，执行器仅组装固定周期、门禁和预算，报告层仅写公开证据。独立新协议不扩宽旧64bit核或旧数据接口；沿用统一CLI，fit=0。

**Tech Stack:** Python3.12、本机`.venv`、标准库、现有pytest/Ruff/mypy，不安装任何依赖。

**Spec:** `docs/superpowers/specs/2026-11-05-innovation2-split-cancel-design.md`（用户已批准末尾完整执行包；设计原历史措辞保留，授权见ledger）。

## Global Constraints

- T0=`2026-11-05T14:00:07.107Z`，截止`2026-11-05T20:00:07.107Z`，epoch deadline=1793908807.107。窗口六小时，协作检查非硬限。
- 科学真实cipher attempts≤7424，工程≤512，合计≤7936；null≤3072个64bitwords；正式一次、重试0、fit/更新/evaluate=0。
- A3/A4仅共享A34面板：A1/A2/A34/A5/A6共5×64×4=1280；B6cell最多6×256×4=6144。
- B q=1，p=7/8/9，H=(0,2)/L=(32,34)，固定2080degree≤2观测；keymixed项身份完整，前缀XOR用union不相消。
- 逐cell≤33280suffix occurrences/unknowns，全局prefix cache-misses≤220000；宽GF2width≤2080/constraint≤33280，旧integral_linear与data/integral不改。
- 主窗口及正式CLI CPU/wall1800s、symbolic cell180s、A symbolic120s、真实cell60s；超限UNKNOWN，不截断证据/丢列。
- 不GPU/SSH/install/solver/Git提交发布/分支worktree/reset/clean/stash；main原位保留全部dirty及历史证据，不import backups/validation/runs。
- 测试只scoped synthetic/fake，禁止全pytest（可能真实训练/密码）。使用字面`/workspace/.venv/bin/python`+`PYTHONPATH=src`，mypy用`python -m mypy`，外部验证`MYPYPATH=src`。
- 代理不派子代理、无正式运行许可；主控制器审核源码和证据后仅委派一次真实执行；无科学训练。
- check0cipher且不导入新执行器/训练；旧配置SHA/算法/模型/checkpoint保持；新积分不支持evaluate。

## Review Focus

1. 不同suffix keymask被误合并/混合项被当常量丢弃：Task1精确身份与故意损坏测试。
2. 不完整prefix/矩阵证据被当ZERO/完整核：Task1 proof replay、缺列/rank completeness测试，Task3停止门禁。
3. A34复用与key全局唯一、失败attempt不退款：Task2 synthetic collision/failure/账本测试。
4. 时间/输出存在/写盘失败发生在科学副作用之后：Task3 exclusive claim、fakeclock及partial测试，Task4工程输出先claim。
5. 正式重复启动/测试误触生产cipher/旧CLI默认改变：Task3惰性导入和兼容测试，Task4一次性记录与冻源验收。

---

### Task 1: 公开符号核心与完整认证空间

**Files:** Create `src/cipher_experiments/representations/simon_suffix.py`, `src/cipher_experiments/integral_prefix.py`, `src/cipher_experiments/integral_symbolic.py`; tests `tests/test_simon_suffix.py`, `tests/test_integral_prefix.py`, `tests/test_integral_symbolic.py`。不得改现有生产文件。

**Interfaces:**
- suffix: `Monomial = tuple[int,int]`; `Polynomial = frozenset[Monomial]`; `observable_terms() -> tuple[tuple[int,...], ...]`（64单位tuple(i,)，再pairs(i,j)）；`suffix_expansions() -> tuple[Polynomial,...]`；`h_mask(j:int)->int`（2080bit组合）；`evaluate_mask(ciphertext:int, mask:int)->int`，只公开末态位函数；`poly_xor`, `poly_multiply`可由实现者命名但report锁定。
- prefix: `PrefixOracle(rounds:int, active:tuple[int,int], *, check:Callable[[],None]|None=None, on_miss:Callable[[],None]|None=None)`，仅公开值；`query(state_mask:int)->PrefixEvidence` frozen dataclass，含rounds/active/state_mask/support(tuple[int,...])/verdict(str)，`to_dict()->dict[str,object]`；`evidence()->tuple[PrefixEvidence,...]`；`verify_evidence(evidence:PrefixEvidence)->bool`从公开递推独立重算/严格校验身份，失败不自动ZERO。
- symbolic: frozen `SymbolicCell` 含rounds_prefix/active/constraints（each packed2080）、unknowns（完整(u,v)固定排序）、K/W/baseline_single/quotient/selected tuple[int,...]、dimensions/rank与所有验证标志；`to_dict()->dict[str,object]`；`solve_cell(prefix:PrefixOracle, *, check:Callable[[],None]|None=None)->SymbolicCell`，selected固定≤4；W>0时完整交叉验证、quotient deterministic；附纯GF2 canonical/kernel/member小接口由report锁定。
- `validate_cell(cell:SymbolicCell, prefix:PrefixOracle)->None` 重新由原始suffix验证所有基/rank/原始矩阵/残项ZERO/证据；不是只检查return字段。

- [ ] Step1 RED：测试2080顺序、q1独立公开单轮真值函数、Boolean平方/相消/keymask保留、错误类型/bool/范围拒绝；穷举人工2活跃小图oracle ZERO健全、constant、未知、证据篡改；小宽GF2全枚举和完整商相关例。先记录预期缺函数失败。
- [ ] Step2 GREEN：实现精确suffix、prefix公开全64state递推（无需复用旧只rootX0图），scope rounds1..9仅H/L k2；querycache计miss/on_miss，check在有界循环调用，资源异常不截断；重放原H/r7/X0得support(0,1,2)。全state leaf支持4种mask不是赋背景值。
- [ ] Step3 RED/GREEN：solve_cell full2080与fake小basis（测试不跑正式6cells搜索）；验证K/W完整性、constraint方向、故意漏unknown/丢列/损坏basis拒绝、canonical-rank-nullity、quotient代表相关性。每个K基重构后允许ZERO残项，不要求空多项式。
- [ ] Step4：scoped tests、Ruff这6files、strict mypy这6files通过；只允许冻结旧X7[0]与工程人工例，不执行完整6cell科学搜索。report列真实命令/RED/GREEN/API/依赖/资源meter归属。
- [ ] Step5：控制器生成新文件完整diff、code-review独立Spec+Quality审查；controller复跑scoped checks才接受。无commit。

### Task 2: 严格配置、私有面板与观测采集

**Files:** Create `src/cipher_experiments/split_cancel_config.py`, `src/cipher_experiments/data/integral_observables.py`, `configs/innovation2-split-cancel.json`; tests `tests/test_split_cancel_config.py`, `tests/test_integral_observables.py`。不得改Task1files或旧data/integral。

**Interfaces:**
- frozen `SplitCancelConfig(name:str,experiment:str='integral-symbolic',protocol:str='simon64-split-cancel-v1')` with strict `to_dict`, `sha256`, `load_config(path:Path)`；JSON仅3字段，duplicates/NaN/unknown/size>64KiB/bool拒绝，name1..100非空。`DEADLINE_UTC/EPOCH`固定上列；identity hash沿旧JSONsort_keys算法不改别支。
- data: `Oracle=Callable[[tuple[int,...],tuple[int,...],int],Sequence[int]]`适配signature同旧batch但返回完整64bits。`PanelBank()`内部全局keys与pairledger；`collect(case_id:str,rounds:int,active:tuple[int,int],phase:str,count:int,masks:tuple[int,...], *,oracle:Oracle, check:Callable[[],None], on_row:Callable[[dict[str,object]],None]) -> dict[str,object]`；bank可构造`attempt_limit:int=7424,null_limit:int=3072`且限定不能超协议；`counts()->dict[str,int]` attempted/completed/null_words/attempt_unique/completed_unique。
- `collect_null(...same public identity/count/masks/check/on_row..., no oracle)`按真实多项式evaluations；`simon_oracle`lazy导入生产cipher，声明用途只受显式科学/工程入口调用。`CollectionError`固定脱敏code，保留已完成聚合，不含key/repr/trace。可调整方法名但report必须完整列新API，供Task3按实际签名调用；不能改变语义。
- A34两case由runner一次collect返回8个mask(X8[0]及7h)；其余panelcase独立。data不实现A/B选择/时间规则，只接check。

- [ ] Step1 RED：strict config/默认旧行为隔离、root198101 ASCII JSON域固定vectors、不同role/case/phase/purpose分离、合法128bits key全局碰撞16×target退出、背景去活跃后唯一、H/L四点顺序、同组keyfake验证。
- [ ] Step2 GREEN：实现私有panel采样，A1/A2/A34/A5/A6 role=A phase=panel；B case_id=`p{p}-{H|L}` role=B phase=check/confirm/null；case_id包含固定protocol identity映射。derive `[protocol,198101,role,case_id,phase,purpose]`，purpose='key'/'background'/'null'；panel共享只由runner一次调用保证。采样只fake测试，不读真实key。
- [ ] Step3 RED/GREEN：一次四点batch先check/reserve4再oracle，输出valid完整64built-in int后才completed+commitrow；submit失败不退attempt，exactpair唯一含域/index/point不含round，截断partial未知/脱敏oracle异常；批复用mask不重复密码调用。null先reserve4独立uniform64words，所有mask从pointwise evaluate计算，不依随机feature猜测。
- [ ] Step4 验证两testfiles+Task1 tests、Ruff/mypy；测试poison `simon_oracle`禁止调用，报告API、固定domainvectors、无secret泄漏。
- [ ] Step5 controller复跑+独立code-review，审计diff和旧源码SHA，无commit。

### Task 3: 有界执行器、公开产物与统一CLI

**Files:** Create `src/cipher_experiments/split_cancel_run.py`, `src/cipher_experiments/split_cancel_reports.py`; Modify `src/cipher_experiments/cli.py`（保留旧_integral_config公共测试兼容，新增strict分派不得反复读同一snapshot）； Create `tests/test_split_cancel_run.py`, `tests/test_split_cancel_cli.py`, `docs/INNOVATION2-SPLIT-CANCEL.md`; Modify `ARCHITECTURE.md`只新增职责。README/currentstate/旧协议不在范围。

**Interfaces:**
- `Clocks(wall=time.monotonic,cpu=time.process_time,utc=time.time)`；`run(config:SplitCancelConfig,output:Path,*,oracle:Oracle|None=None,clocks:Clocks|None=None,...) -> dict[str,object]`，允许显式fake solver/collector测试依赖注入，不加用户CLI科学override。
- task1/2实际报告为输入接口权威（若和本plan命名冲突记ledger ruling不改科学语义）。runner只装配，不重写ANF/GF2/采样。
- reports `claim_output(output)`exclusive mkdir先所有sideeffect；公开json/row/sha写入函数接已算公开记录，无cipher/key；manifest source hashes含新全部modules、ciphers/simon/expanded/public来源及旧query142expectedhash；status/progress/stages/counts/exit一致。partialspace为null，已有证据可诊断但不当完成。

- [ ] Step1 RED fake：existingoutput保旧字节+0oracle；fakeutc越deadline0call；CPU/wall/loop预算；SIGINT/异常写failure不打印secret；strict frozenconfig/selected-before-B-real顺序；A34只collect一次；A4ZERO反例停/UNKNOWN反例继续；A1/2/5/6非零停；row损坏/坏Q/坏W停。
- [ ] Step2 GREEN 固定A9symbolic请求与5panels顺序；A34 mask tuple固定X0,h(j)；A4认证源X6[j]。A2/A5/A6依据已认证A1与精确恒等式，不强求保守oracle重新证出全部扩展。全部B6cells先symbolic并冻结selected；再null全部适用cells、然后B check32/confirm224。NULL K=0 cell记NOT_APPLICABLE，不补cipher成本，不称实测0。
- [ ] Step3 RED/GREEN 产物完整suffix（公共一份）、cell prefix证据/unknown/constraints/K/W/quotient/validation，mask packedhex定长520digits、state16digits/key8digits；逐row记录完整bitparity及selectedparities，SHA/linecount。原有有限budget meter不能被fake注入绕过生产常量。deadline/total1800/cell180/A120/real60在长循环及写盘边界有check。
- [ ] Step4 CLI严格接受experiment=integral-symbolic，check不导入runner/training/data/ciphers，train新分派；仍reject protocol给prediction/innovation1。旧_integral_config调用者兼容、未知/混合/duplicate错；新增CLIconfigdry无actual train测试。所有CLI入口当前实现才写methoddoc命令，fit0/evaluate禁用。
- [ ] Step5 scoped新全部tests＋旧test_integral_cli/test_integral_config（确认fake/safe）和必要旧配置defaultfixture，Ruff/mypy，controller验收、独立任务review；无科学调用/commit。

### Task 4: 冻结、独立工程门禁与唯一正式CPU周期

**Files:** Create `validation/innovation2-split-cancel/engineering_check.py`, `test_engineering_check.py`, `clean_cli.py`, evidence JSON/logs；run全新`runs/split-cancel-<timestamp>/`。本task可修自己harness，经fake RED/GREEN不重跑正式。

**Interfaces:** 单次工程脚本默认planned0cipher，显式`--execute --output <fresh-file>`先exclusive输出claim，attempt/completed409?见下预算401；只此脚本工程生产调用，控制器明确一次执行。

- [ ] Step1 TDD工程：fixed KAT1 + scalar vs expanded(1..10round×10syntheticpairs×2backends)=200 + newdata.simon_oracle vs scalar(1..10×10pairs×2)=200；**总401blocks**，低于512。KAT P=0x656B696C20646E75,K=0x1B1A1918131211100B0A090803020100,C=0x44C8FC20B9DFA07A,r44。syntheticpairs由publicrandom.Random(198111)，每round10(key128/plain64)；比较两路均计calls，工程only permittedcipher，不保存key数据。批adapter可每round同key1point以匹配，预算不变；每调用前reserve/完成后计数，失败无refund。fake tests existingoutput/sanitizedfailure/budget+planned0calls；controller审harness后一次运行。
- [ ] Step2 clean-copy harness：仅copy src/configs到自建临时目录，字面原venv和PYTHONPATH copied/src；integral-symbolic check+fake injected run（可小syntheticcells/rows，不科学），poisonproductioncipher/torchtraining；不旧neuralfit，不evaluate。保存literalmodulepaths/returncodes/exit/rows并清理只owned临时路径。controller运行并保存证据。
- [ ] Step3 主控制器冻结源码/config/design/plan SHA、旧querySHA及deadline、权限与budget；独立code-review final-pre-science审Task1-3接口调用链/计数/科学门禁，无Critical/Important未修后才开始。
- [ ] Step4 只授权experiment代理一次执行已实现CLI：`PYTHONPATH=src /workspace/.venv/bin/python -m cipher_experiments.cli train --config configs/innovation2-split-cancel.json --output runs/<fresh-id>`，capturestdout/stderr/returncode/startend、invocation_count1/retry0/phasecounts。无主模型/其他代理重复执行；deadline前不足运行最坏预算就停止TIME_LIMIT，不借旧窗口。
- [ ] Step5 只读接收核status/exit/attempt-completed/unique/null/clock/6cells/所有来源SHA，不evaluate、重加密或追加run；故障/无exit按UNKNOWN，数据保原位。

### Task 5: 无加密证据复核、独立终审与交付

**Files:** Create `validation/innovation2-split-cancel/verify_saved.py`, test fixtures/evidence/closeout；Create new `result/YYYYMMDD-innovation2-split-cancel-cycle[-n]/{report.md,index.html,source.json}`；Modify resultREADME/index仅navigation与methoddoc actualresultappendix。保留全部原件。

- [ ] Step1 TDD独立stdlibrary verifier，不import生产prefix/suffix/symbolic/cipher/data；独立public q1 ANF及k2support重放，重构全部cell约束并核Kcanonical/rank-nullity/所有basis残差+ZERO、W/quotient/固定selected、rowsSHA/counts/phase/selectionpreconfirmation，标明exactpair计数无法由parity再生、依赖私有ledger与测试。人工小维穷举自检，不重新生成真实cipher。
- [ ] Step2 对唯一savedrun执行verifier，controller独立hash/diff/计数/row/selectedtiming核查；失败报告实际错误，不伪造证书。
- [ ] Step3 3file结果：A全6case各证据/反例；B全6cells dimK/dimW/dextra/单observable空间/组合增量、selected公式/确认及null；成本、时钟、证据等级、未完成/继承/novelty限制。不把data通过作为证明、不把core7派生10轮称新核心突破。
- [ ] Step4 全范围新diff独立code-review（read final-review规则，只一波全范围fix），controller最后scoped tests/Ruff/mypy/全部sourceSHA/localHTMLlinks/JSON identities、deadline消耗和原dirty保留。
- [ ] Step5 closeout与ledger CLOSED/失败状态、权限结束、原件结果路径、实际消耗、无额外进程；无commit/push，不删除SDD证据。

## Self-review / spec coverage

数学/key身份/完整核/W与商→Task1；配置、隐私采样、ledger→Task2；阶段门禁/产物/CLI→Task3；工程512及唯一science7424→Task4；独立重放与报告→Task5。所有ReviewFocus都有对应RED/GREEN例；所有任务在主窗口内停止，工程401实际预案不花满512、节余额度无用途扩展。无神经训练/安装/solver，现有脏状态原位保留。
