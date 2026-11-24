# LSTM 第1–3轮预测矩阵 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development，TDD与独立只读review；代理不再派代理、不Git、不真实科学运行。

**Goal:** 同新未知key/同固定明文上，独立拟合SIMON1/2/3轮LSTM，报告完整/active8/half/整对准确率并排除一轮复制虚高。

**Architecture:** 在cube_compare增加明确low-round矩阵provider和专属low_metrics，仍用一个Meter/RunStore/fit/严格reload/预测流，旧dual/single方法不变；不复制runner或新增远程调度。

**Tech Stack:** 既有本地Python3.12、WindowsPython3.13.3/Torch2.7.1+cu118/NumPy2.4.0、coreSIMONbatch/earlystop、RunStore/Meter、pytest/Ruff/mypy；无安装。

**Spec:** `docs/superpowers/specs/2026-11-24-innovation2-lstm-r1r3-design.md`（APPROVED：用户确认1–3固定矩阵/3h/GPU1/一次发布运行/retry0）。

## Global Constraints

- 3fit r1/r2/r3同一新privatekey、同明文/族/划分/顺序；raw128输入/输出、同216192参数LSTM、seed31新初始化，不能把前轮真实状态作为后轮输入。
- 原source-02三NPZ/run SHA身份固定，仅plaintext/family_ids复用，旧七轮ciphertext不用作输入/低轮标签；新key只留闭包、不保存或交模型。
- 1048576/65536/65536pairs，每轮50max/batch256/10warmup/8patience/delta1e-5/full128BCE/active8DEV最早最低，三模型独立earlystop。
- realblocks7077888/null0，maxupdates614400/maxforward167313408；actualupdates4096Σe/forward1114112Σe+196608。wall10800CPU43200/RSS8GiB/output512MiB/ckpt16MiB，固定3fit任一失败停/retry0，不额外数值预检。
- 三best持久strictrestore均成功后才读源TEST并生成三低轮TEST标签；全部roundscore完成才提交终分，不提前发布单轮终分。
- full128/active8/xhalf64/yhalf64/wholepair128准确率，r1结构baseline复制y半+TRAIN多数x半，非复制x半必须单列；输入/标签MSB0别错半块。
- 开发fake oracle/NPZ/runtime/clock/FFI/tensor结构only，不真cipher/entropy/NNforward/backward/Adam/GPU/RSS/SSH/network/install/Git/旧真实dataweights；guard不得放宽。core/model不改，main/index和旧产物不动。

## Review Focus

1. 三轮secretkey身份相同且仅数据闭包，sourceplaintext同序/去重，不用旧ciphertext标签，不把actualkey写日志/科学模型。
2. r1copy半是y列32:64和96:128复制输入x列0:32和64:96，不错算xhalf或把75%包装学习成功；全128整对必须真实all bits。
3. 各round独立factory/slot/identity/optimizer；最后一份strictreload失败TEST0，后面评分失败无已发布第一轮终分。
4. 一个总时间/计量ledger，exactblocks/maxvsactualepoch，失败attempt不返还，不在报告重算或追加cipher/forward；旧methods仍原值和gates。
5. 新profile/mode/report/源码指纹/PS固定allowlist/globalnativeexit前台完整，purestdlibcheck，guard阻断LSTM aliases，测试名字全局唯一。

## Task 1: 明确low-round矩阵与统计/CLI接入

**状态：COMPLETE。** implement-deep56c6c56a；任务review2e10c575经Fix1放行；whole4758862dStaticApproved，最终P3基线文案小修复审ADDRESSED。主控fresh514人工+core/1科学排除137.05s、fullsafe3702/49排除312.74s（既存ZIPwarning1）、最终文案fresh22通过/forcedRuff全树/mypy387clean。core/fit/model不改。

**Files:**
- Create `cipher-lab/plugins/cipher_lab_plugins/direct_block/cube_compare/low_rounds.py`（fixed矩阵provider与samekey标签闭包）；可Create `low_metrics.py`（专属full/half/pair/copybaseline，不塞training）。
- Modify `.../cube_compare/{config,training,engine,cli,report}.py`最小显式方法/identity/3slot/1ledger装配；若需要`reuse.py`仅拆出plaintext加载职责且旧默认行为保持，不复制其strictNPZ解析。
- Create `configs/direct-block-cube-lstm-simon-r1r3.json`；Modify原`scripts/start-cube-sharedm.ps1`固定allowlist/source3h。
- New globallyunique `.../tests/test_cube_low_{data,metrics,training,cli,report}.py`或等价少量文件，现guard按需fakekey/oracle禁止泄漏。

**Interfaces:**
- `low_rounds`生产factory只一次私有key采样，提供按split/round显式的Panel，不对外暴露secret或encrypt trace；源SavedPanels读取原顺序明文，TEST入口仅在3best gate后，npuint64连续chunk≤65536，meter按每round每block收费。
- 原fit不必改；新明确topology复用其`fit/reload_best/predict_batch`，同order、3best slots，lastreload成功后统一获取TEST；完成epoch数记录明示r1/r2/r3。
- 新metric受显式full128labels/predictions/inputs接收，冻TRAINfull128majority，r1copybaseline纯input，逐列/half/active/wholepair分母真实，至少完整保存每familyrowcounts便核验，禁止伪oldpairCI。
- `config.low_recipe()`/counts(epochs tuple)固定声明samekey-new/plaintext-only-source/rounds[1,2,3]/3fits/caps/指标；CLI purecheck、源outputoverlap前置、统一resource/failure/无freepreflight。
- engine新round-LSTM factory identity明确method/rN，不改模型/Adam/full128训练；report失败无终分、成功三轮全图表保存值。

- [ ] RED：人工三个oracle与私钥替身证明keydraw恰1/rounds1,2,3固定/labelnotold/crossroundsameplaintextorder/TEST迟，cipherchunk计数7077888公式（小人工规模不缩正式config）。
- [ ] RED：fake第一/第二/第三best坏SHA/strictrestore/源TESTfault/后roundscorefault/时间或outputlimit，TEST0或统一无终分，无fallback/no自动fitretry。
- [ ] RED：128标签中只copyhalf正确、active8与full/half/wholepair分母不同，手造一位错整对失败、TRAINtie0、r2/3不套r1copy规则。
- [ ] RED：fixedconfig3fits/budget/actualsume/slotsrepeatbestreplace/3初始化samepublicseed独立params，旧4配方/dualgate/singlegate完整回归。
- [ ] GREEN：最小实现，samecodefit/loader复用不genericregistry/新workflow；严格数据/模型/训练/指标/CLI职责。
- [ ] 验证新owner+现人工套/核心earlystop/artifacts/execution/layout、Ruffforced no-respect-gitignore全树/mypy严格、全部固定配方stdlibcheck/干净闭包人工CLI。default全safe主控fresh一次。
- [ ] 报告task-1-report.md含RED/GREENcommands/output/changeallowlist/actualAPI/启动命令/失败保持/无真实科学；独立review/fix后主控验收。

## Task 2: 主控限定发布/一次GPU矩阵运行/取回

**状态：RELEASING。** 用户已授权3fit/同新key/同明文split/GPU1/总3h/once/retry0，无额外数值预检。

- [ ] 3dae808独立源码archive/允许清单/mainindex/source旧bytes保护，固定spec/plan状态。
- [ ] whole-review＋freshdefaultfullsafe/新owner/forcedRuff/mypy/check过后，一次Gitcommit/push限定files，Windows exactcheckout freshroot，不SSH传科学源码包。
- [ ] GPU1 UUID/freeidle/现解释器user-site/sourceplaintext身份确认；原PS前台globalnativeexit/总3h，无额外数值预检、无GPU0变更。
- [ ] 唯一矩阵1→2→3执行至早停或上限；一key新取，actualcount/三bestreload/TEST/copyhalf指标/nativeexit保存；任一fail停止，retry0。
- [ ] SCP/SFTP取回只实际data/models/result和日志，核hash/identity/actualepochs/nativeexit/SSHexit；不load模型/再生标签/forward/evaluate。
- [ ] 纯saved结果图与方法/复制位图解，报告各roundfull/active/非copy/wholepair与baseline，不称辅助区分已提高轮数；仅服务resultHTTP验证。本矩阵许可关闭，不自动r4/新key/seed。
