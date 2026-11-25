# 单明文→完整64位密文 LSTM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development，TDD和独立review。代理不派代理/不Git/不真实科学；主控独占批准的运行。

**Goal:** 三轮独立LSTM直接P64→C_r64，全64BCE训练/DEV选择，主指标正确位÷所有64位，不再用配对输入或active8。

**Architecture:** 新singleword视图严格只读原1–3轮NPZ，原样labels/划分展平，复用一Meter/RunStore/fit/三best严格reload与完整TEST终分；模型input1/output64、旧128方法默认不变，不复制loader/runner/optimizer。

**Tech Stack:** 既有本地Python3.12和远端WindowsPython3.13.3/Torch2.7.1+cu118/NumPy2.4.0，coreEarlyStopping/Meter/RunStore，pytest/Ruff/mypy；无安装。

**Spec:** `docs/superpowers/specs/2026-11-24-innovation2-direct64-design.md` APPROVED（用户确认单条64位映射/full64口径/1–3轮/50上限早停/GPU1/总3h/一次不重试）。

## Global Constraints

- 单明文64binaryMSB0→seq64×1，2层hidden128单向LSTM每callzeroh/c、finalh→Linear64，207424参数；无P′、族ID、差分/额外card/key/教师/跨batch状态。
- Source `cube-lstm-simon-r1r3-20261124-01` runSHA和9NPZ完整SHA/bytes见spec；每pair两端拆为单条P/C64、familyID重复，原split/顺序保持/跨round对齐。TRAIN2097152/DEVTEST131072单条；0新cipher/key/entropy/labels。
- full64BCE训练和DEV（selectedcolumns=None），主score correctbits/(131072*64)=/8388608；附x32/y32、whole64不替主score；r1copyybaseline与TRAINmajoritytie0。
- 3fit50max/10warmup/8patience/delta1e-5/batch256/seed31/order20261124，actualupdates8192Σe/forward2228224Σe+393216，max1228800/334626816，realnull0。
- 3hwall10800CPU43200/RSS8GiB/output512MiB/ckpt16MiB/GPU1，once/retry0，无额外数值预检/免费warmup或旧模型训练。
- 全部3best严格diskreload后才源TESTread/decode，全部score/dataset完成才统一终分；postfreezereportfailure保留complete但native非零，precommitfault无partial。
- 开发fakeNPZ/runtime/clock/FFI/tensor形状only，禁真cipher/secret/NN数值/Adam/RSS/GPU/SSH/network/install/Git/旧真实dataweights；core/旧models/原5mode不改，globalunique测试/forcedRuff。

## Review Focus

1. 单条展平必须P/C同行及family同步，源声明来自lowround1–3非七轮；不读oldscores/weights，TEST懒，三roundplain/split一致，输出不污染旧目录。
2. 模型64×1→64、无伙伴或hiddenstate泄漏、full64DEV和主accuracy不偷偷沿active8/128shape，strictidentity防错模型宽度。
3. 计量单位单条非pair，earlystopactualepoch/deadline共享，批统计check不漏，最大不是成功必要数，fault不算正常早停。
4. r1copycols32:64=input0:32、x32分母4194304，main64分母8388608，whole64all仅诊断分母131072，不误用旧wholepair65536。
5. 原128方法/loader/optimizer/PS兼容、purestdlibcheck/sourcefingerprint/guard底层recurrent、计划与actual报告区分/globalnativeexitforeground。

## Task 1: 严格源复用、direct64模型和同runner接入

**状态：COMPLETE。** implement8ffa8bed→Fix1 d208bf8c；任务review8443d825经两项边界修复放行，whole910a5dc1StaticApprove。主控fresh559人工/core通过1科学排除131.21s、forcedRuff全树/mypy394clean，最终defaultfullsafe3702/49排除398.45s（既存ZIPwarning1）。core/旧models不改；parentfit仅fullDEV字段真实标注。

**Files:**
- Create `cipher-lab/plugins/cipher_lab_plugins/direct_block/cube_compare/direct64.py`（singlePanel/provider/source9artifacts），可Create`direct_metrics.py`（64指标）。
- Create `.../cube_compare/lstm64.py`。
- Modify `.../cube_compare/{config,engine,training,cli,report}.py`与`reuse.py`仅必要参数化strictdecoder/sourceidentity，旧默认不变。
- Create `configs/direct-block-lstm-direct64-r1r3.json`；Modify原`scripts/start-cube-sharedm.ps1`固定allowlist/source3h。
- New unique `.../tests/test_cube_direct_{model,data,metrics,training,cli}.py`或最小同职责文件，guard不放宽。

**Interfaces:**
- `DirectLSTM64.forward(Tensor[N,64])->Tensor[N,64]`单步一bit、newzero2,N128、207424params，无文件/secret依赖。
- `DirectPanel`有len按单条、batch indices→FP32[N,64]/uint8[N,64]、family_ids[N]，archive可原NPZbytes+明确viewmetadata，不重序列化科学labels。
- 固定源provider buildtrain/dev返回3个对齐roundpanel，label_test只在3newbeststrictreload后访问9pinartifacts中的3test；strictdecoder复用而非复制/monkeypatchglobalconstants。
- 原`TorchEngine`可显式width64/默认128，factory身份 `<directmethod>/rN`；fullBCE/Adam实现唯一，不复制数值runtime。
- 新拓扑`direct64-r1r3`调同fit/3slots、全64DEV设置，singleroworder按2097152，metricinputword64；CLI固定profile/policy/source-root/overlap/actualcounts、纯stdlibcheck，oldprofile值不变。

- [ ] RED：strict9SHA/bytes/sourceidentity/坏NPZ/路径/同roundplainorder/单条展平对齐/labels不来自其它round；TEST无premature/cipher0。
- [ ] RED：fakeLSTMhead shape207424/每step1bit/重置状态/64outputs、runtimewidth64 fullBCE/strictcheckpointidentity；old128defaults依旧。
- [ ] RED：full64DEV变化能选best（原8位不敏感的人工case）、实际单条counts/18stopepoch/50ceiling，r1copybaseline/half/full/whole64分母正确。
- [ ] RED：3best坏SHA/restore/sourceTEST/后roundscore/commit/report预算fault无partial或真实nativefailure；oldmatrix/single/dual/overlap/guard保留。
- [ ] GREEN：最小实现无newrunner/registry/复制loader/secretlogic；scientifictuple边界明确，各层职责。
- [ ] 验证owner人工套+核心earlystop/保存/执行/布局，forcedRuff no-respect-gitignore全树/strictmypy、所有固定配方purestdlibcheck/cleanclosurefakeCLI；defaultfullsafe主控fresh。
- [ ] 写task-1-report.md含RED/GREENcommands输出/changes/allowlist/API/exactstartup/actualcounts与未验证边界，主控独立review＋fresh。

## Task 2: 主控发布、唯一GPU运行与保存值交付

**状态：RELEASING。** 新fixedsource/单明文64→64/full64DEV，3fit总3h、zero新cipher/标签/retry0。

- [ ] 基于2253124隔离archive/baseallowlist/mainindex/sourcebytes保护；old数据只读SHA确认。
- [ ] Task和whole独立review/fresh门禁后一次limitedcommit/push，WindowsGitexactfreshcheckout，不SSH传源码任务包；run新id。
- [ ] GPU1idle/UUID/free16GiB，现Pythonuser-site/原PSglobalexit/foregroundSSH保持/总3h；source-root原lowroundmatrix/work，无新加密/secret/preflight。
- [ ] 三独立freshLSTM1/2/3依次fit至full64DEV早停或50，all3beststrictreload后TEST；实际singlecounts/native0/SSH0证据，任fail停/retry0。
- [ ] SCP只实际data/models/result/logs，核SHA/来源/actualepochs/计量/nativeexit；不load模型/解码科学数据/再生label/forward/evaluate。
- [ ] 纯saved主64位准确率/half/附wholeword/copybaseline及full64loss图，揭示输入总体仍原族；onlyresultHTTP核字节，本许可关闭不自动r4/新key/model/复用预算。
