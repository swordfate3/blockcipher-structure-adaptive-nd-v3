# SIMON32高8位：四个两位MLP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development。TDD、独立只读审查；代理不派代理/不Git/不真实科学。主控执行一次批准的KAT+训练。

**Goal:** 新32位SIMON1/2/3轮上，每轮4个独立32→128→128→2MLP，预测固定高8位后拼接按正确位÷全部8位计分。

**Architecture:** 新纯`simon32`密码模块＋`direct_block/split8/`子owner显式plugin，复用原fit/Meter/RunStore/strictcheckpoint/数值runtime/资源，不再扩cube_compare多mode。原PS唯一启动器最小加config→plugin/无source/3h分派。

**Tech Stack:** 本地Python3.12、WindowsPython3.13.3/Torch2.7.1+cu118/NumPy2.4.0，既有工具；不安装。

**Spec:** `docs/superpowers/specs/2026-11-25-innovation2-simon32-split8-design.md` APPROVED。

## Global Constraints

- round1/2/3 sameoneprivate64key/sameuniqueuint32plaintextsplit，TRAIN262144/DEV32768/TEST65536，no pairs/XOR/difference/cube。高8位MSB0cols0..7预定为(0,1)/(2,3)/(4,5)/(6,7)。
- 每net完整32位P输入，MLP32→128→128→2 ReLU/20994params，12独立fit无sharedparams/grad；seed31+j跨round对齐，orderseed20261124。
- max50/batch256/Adam .001/.9/.999/eps1e-8/clip1/decay0/FP32，各2位DEV BCE earliestmin，corewarmup10patience8delta1e-5，best与stopanchor分开。
- 主科学blocks1081344＋32轮公开scalar/batchKAT各1共2，总1081346；null0；maxupdates614400forward177733632，actualupdates1024Σe/forward294912Σe+786432。只有12fit/no额外模型forward预检/benchmark。
- 所有12newbestsize/SHA/identity/state strictrestore后才TEST标签，全部分数及dataset成功后统一terminal；postfreezereportfailkeepcomplete但native非零，未freeze不partial终分。
- 全一次wall10800CPU43200/RSS8GiB/output512MiB/ckpt16MiB/GPU1UUID/logicalcuda0，失败停retry0，不换卡/减量/加seed/r4/install。
- 开发仅人工oracle/keydraw/NPZ/runtime/clocks/FFI/模型结构，禁真实cipher/秘密熵/NN数值/Adam/RSS/GPU/SSH/network/Git/install/旧dataweights，不放松guard。新增密码32正确门禁只在主控批准批次执行2blockKAT。
- 从0fb6e7隔离archive，main/index/旧work保留。新SIMON32模块与provenance允许增加core文件但不改旧cipher/model/API；newsplit8/tests显式收集/globaluniquenames/forcedRuff。

## Review Focus

1. SIMON32非SIMON64截断，16位rotation/mask/z0/m4keylowfirst/highxlowy KAT一致；scalar/batchno模型/RNG反向依赖，私钥只闭包/不存日志。
2. uint32全域无放回crosssplit全局dedup，同plain同key跨round，生成一次高8labels共享四fit；batch<=65536/attemptcomplete计量，TEST晚。
3. 12factory/slot/roundjidentity均不同、输入完整32输出仅目标2bit、拼接MSB0位序不漏/重位；majority仅TRAIN，bitacc不是2bit全对。
4. 12best最后restore失败无TEST、后round/后chunkscorefailure无partialterminal、actualepochs计账/整3h预算/统计循环每批check，oldcube方法不回归。
5. purestdlibcheck/新sourcefingerprint/provenance/PSconfig映射(globalnativeforeground/无source新路径)/报告saved计划vs执行，guard新增simon32标量批闭合，不执行真KAT或科学测试。

## Task 1: 新纯密码与split8专属方法完整实现

**状态：COMPLETE。** implementac66ef4a，任务reviewc49ea1c1经Fix1关闭两个P2；整体bc450935StaticApproved。主控fresh新owner26pass65.43s/coreplugin禁用、forcedRuff全树/mypy413clean，最终fullsafe3703pass49排除282.17s（既存ZIPwarning1）。真实KAT/GPU未执行。

**Files:**
- Create `cipher-lab/src/cipher_lab/ciphers/simon32.py`：`encrypt_scalar(point:int,key_words:tuple[int,int,int,int],rounds:int)->int`和`encrypt_batch(points:uint32[N],key_words,rounds)->uint32[N]`，native连续/≤65536/no mutation/1..32、16位字、lowwordfirst，独立scalar/batch公式。Update `cipher-lab/docs/provenance.json`新增来源，不改旧条目。
- Create `cipher-lab/plugins/cipher_lab_plugins/direct_block/split8/{__init__,config,data,model,metrics,training,cli,report}.py`；若需纯publicKAT gate模块可用`kat.py`，不得混训练/报告算法。
- Modify `.../cube_compare/engine.py`仅明确允许inputwidth32（旧width64/128默认保持），专属splitfactory构造seed31+j模型后复用TorchEngine、strictcheckpoint（identitymethod/rN/bJ）。
- Create `configs/direct-block-simon32-split8.json`；Modify原`scripts/start-cube-sharedm.ps1`固定allowlist→plugin/sourcerequired/wall mapping，不新增第二启动/调度器。
- Create globallyunique `.../split8/tests/{conftest,test_split8_cipher_structure,test_split8_config,test_split8_data,test_split8_model,test_split8_metrics,test_split8_training,test_split8_cli,test_split8_report}.py`按实际职责可精简；selfcontainedguard/reuseonlytestguardhelper但无兄弟生产依赖漏source。

**Interfaces:**
- `config.recipe()/counts(epochs12)/caps`纯stdlib，明确cols/slices/KAT2/scienceblocks/12fits/资源。
- data显式provider工厂仅一次keydraw，32位plaintext固定publicseed去重/按train-dev-test，r1/2/3labels一次返回且同行对齐；blocklabels full32或高8bytes保存便复用，slice panel仅带两目标列。TESTlabel入口在12strictreload后，source不来自旧cipher。
- model`SplitMLP.forward([N,32])->[N,2]`binaryFP32/同device，不读metadata；独立CPUpublicseedfactory后tocuda复用已实现runtime。
- training同一个RunStore/Meter，roundouter/blockinner12fit，parentsource供应全core/direct_block生产闭包，fit.FitSettings selectedcolsNone/earlystop10/8/1e-5，12slot允许replace；每netDEV选择后从diskstrictrestore全部才TEST，保存epoch和best来源。
- metrics顺序组合4预测为8，主correctbits/524288，pernetwork2/perbit8/labelcounts/TRAINmajoritytie0，附eightallcorrect/65536不替主。no伪complete32bit或额外modelcomparison。
- CLI coreplugin check/run（purecheckstdlib），运行先claim+device资源，同一budget下KAT2通过才私钥采样/训练，CWD/temp/G路径原PS处理，nativeoutput不补假成功。

- [ ] RED：纯shape/边界cipher结构tests，不真实完整encrypt；fakeKAT流程断scalar/batch32round各一次/错误gate不keyfit；字序/rotation公式由review静态核，真实KAT留产线。
- [ ] RED：同keydraw1/sharedlabels/highbitcols/sameplaincrossround/dedup所有split/32bit域无withreplacement漏；chunk/尝试不退款/后restorebadTEST0。
- [ ] RED：12独立模型/seed对应/参数20994/64or2bitwronginputs拒绝、4x2bitconcat顺序、majorityTRAINonly/correctbits和all8diagnostic不同。
- [ ] RED：人工12fit三round18/不同epoch/50case、重复bestreplace/最后restore/score/commit/reportfault/期限与RSS/output故障，不完整不terminal；KATtwo＋dynamiccounts单ledger。
- [ ] GREEN：最小实现复用公共组件不复制runner/Adam/registry，不把新方法塞旧mode大表。
- [ ] 验证独立新owner＋旧cubeowner（inputwidth32兼容影响）＋coreearlystop/artifacts/execution/layout，forcedRuff全树/mypystrict、purecheck和干净closurefakeCLI，真实science0。
- [ ] task-1-report.md含RED/GREENcommands/output/changes/接口/allowlist/启动命令/规范来源/未验证边界；主控独立review/fresh后release。

## Task 2: 主控规范交叉核验、一次发布与真实KAT/训练/取回

**状态：RELEASING。** 唯一批准2block公开KAT＋12fit批次，GPU1总3h/retry0；交付报告直接展示canonical已保存逐网/逐位/实际epoch，不追加生产呈现功能。

- [ ] 只读核z0/keyorder/公开KAT来源，保eprint403限制，原paper公开转换副本免责声明不当primary；provenance写traceable原规范/来源。
- [ ] from0fb6e7archive/mainindex保护、fullsafe+newowner/oldcompat/Ruffmypy/stdcheck/independentreview，限定files一次Gitcommit/push，远端新rootexactcheckout不SSH源码包。
- [ ] GPU1UUID/freeidle/现user-site、固定PS前台/globalnativeexit/总3h；KATscalar/batch2block一次，通过才12fit，不数值预热/额外fit，任失败停retry0。
- [ ] 12best全strictreload后TEST一次labels/4*3forwards账完整终分，nativeexit真证据；SSH缺wait仍UNKNOWN不补造。
- [ ] SCP实际data/models/result/logs核hash/来源/actual12epochs/counts/exit，不loadweights/labels/forward/evaluate；报告只saved8bit拼接主score/pernet/perbit/baseline图、选位非复制/单key/seed/lowround探索边界。
- [ ] report仅result本地HTTP验证，许可关闭，不自动整32位/第4轮/大样本/新key/seed/重试。
