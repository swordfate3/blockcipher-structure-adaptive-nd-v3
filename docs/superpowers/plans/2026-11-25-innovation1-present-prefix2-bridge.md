# PRESENT二轮理想状态桥接区分 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development。按任务TDD与独立审查；代理不派代理、不Git/SSH、不执行真实密码/NN。主控完成一次获准的正式批次。

**Goal:** 用理想S2与最终Cr，PRESENT80总r2..7，六个Cr-only＋六个joint＋一个S2-only共13fit，输出组分类准确率及两种联合增量。

**Architecture:** 新纯PRESENT有界批前缀模块＋innovation1/prefix_bridge子owner。旧Shared-M组件复用，只在本owner扩两时刻输入；公共direct_block.fit加可选完整batch-update接点，复用innovation1.TorchRuntime的Adam/update，旧默认分段Engine路径保持。一个Meter/RunStore，13best全恢复才TEST，不保存原始状态面板。

**Tech Stack:** 本地既有Python3.12/NumPy/Torch/pytest/Ruff/mypy；Windows既有Python3.13.3/Torch2.7.1+cu118/NumPy2.4.0。不安装。

**Spec:** `docs/superpowers/specs/2026-11-25-innovation1-present-prefix2-bridge-design.md` APPROVED。

## Global Constraints

- PRESENT80固定onefreshprivate80key，S2=round2含K3末白化，Cr=roundr含Kr+1末白化且保留每轮pLayer；r2 S2=Cr。非真实预测器、非末轮回退。
- 4pairs/group，同一key跨组/正负/splits/六round；positive P'=P XOR0xD00000，negative两端独立uniform都真加密，不label-dependent前端、不筛正确样本。TRAIN32768/DEV8192/TEST32768组各类恰半，共享panel/ordering/initseed31，dataseed20261125/order20261126。
- canonical6[S2,S2prime,delta2,Cr,Crprime,deltar]，S2-only后3零且不读Cr/round，Cr-only前3零，joint全部。54steminput、96channels/GN8/public13top2/suffix3/5/7/9/11/4pairreadout。全部13模型同构同initialstate独立parameters/Adam；no专用D/MLP额外fit。
- max30epoch/batch128/Adam.001,.9,.999,eps1e-8,clip1,decay0/FP32无AMP/TF32；独立coreearlystop10warmup8pat1e-5及严格minDEVearliestbest。actualΣepochsE，updates256E cap99840；forwards40960E+425984 cap16400384，modelgroup不是pair/bit。
- 科学73728groups×8paths×6outputs=3538944chargedreal，轨迹589824，门禁14（full31官方KATscalar1+batch1、r2..7非零公开vector两路径各1共12），cap3538958real/null0。共享prefix输出一次，不因多fit重复加密，不漏多时刻计费。
- 原始P/S2/Cr/keys全不落盘，内存only；落盘配方/源/计量/panelhash、TESTlabels/logits、权重、报告/必要日志。允许声明idealoracle输入隐含秘密关系，不显式注入key/推导key；不称现实攻击或信息论无秘密。
- 13newbest全size/SHA/metadata/geometry/strictrestore后TEST，全部评分才统一终分，第一次危险best写入前counts持久化；anyfailstopretry0，no额外真实预热/微训练/benchmark/key/seed/r8/扩容量。
- GPU1 UUID GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5/logicalcuda0，一次4hwall/16hCPU/RSS8GiB/输出512MiB/ckpt16MiB；远端只G:\lxy\新root，一次限定Gitcommit/push/SSHrun/SCP取回获准，不install，不新编排/调度器。协作Meter不是硬隔离。
- archivebasecb5e78c55d13cb4153ecbdaa5e07fff5c25dfbcc，仅ownedrepo改，main/index/旧outputs/其它工作保留。代理禁止真实cipher/私密熵/NNforwardbackward/Adam/GPU/RSS/SSH/network/Git/安装，允许人工oracle/keysource/clock/resource/numericFFI替身、模型构造shape/参数检查。
- 独立owner tests与defaultfullsafe分开统计；forcedRuff --no-respect-gitignore --no-cache，mypystrict；不跑science-marked真实原测试，guard下层alias/cachehandles。源identity覆盖实际core+innovation1+direct_block importedclosure。

## Review Focus

1. 同时刻r2重复及S2-only无最终状态/round访问，掩码在canonical后整3ch清零，zero不是合法输入状态的替代采样；仅组ACC不是bitACC。
2. 前缀裸状态推进/white输出clone不可alias，keylowfirst非全0路径拆装、nativeuint64/shape/batchbounds/nonmutation；公开门禁失败零私钥/fit。
3. 完整batchupdate与原分段Engine分支互斥，forward/update计量单一owner；旧fit/旧共享模型/旧PS配置不变。
4. 原始多轮状态不经NPZ/event/checkpointmetadata/traceback泄漏；单keyoracle会包含秘密关系须诚实限定，不实现keyderive。
5. 任一最后restore/后轮TEST/首次model写入故障不提交partial终分/丢counts，SSHexitnativeexit分别记录，缺证UNKNOWN不补0；不重跑修日志。

## Task 1: 纯PRESENT批前缀＋公共fit完整更新接点

**状态：COMPLETE / StaticApproved。** implement9afee4fe；review7d9e77b5经Fix1/Fix2关闭三项testguard P2（cachedredispatch、_VF动态属性teardown、stub双patchstack污染）。主控freshowner57passed1.01s，旧cube/split8/early308passed181.98s，forcedRuff clean/mypy422。产品纯PRESENT算法/fit回调静态通过；真实cipher/KAT/NN/GPU仍0。

**Files:**
- Create `cipher-lab/src/cipher_lab/ciphers/present_prefix.py`。
- Modify `cipher-lab/plugins/cipher_lab_plugins/direct_block/fit.py` only optional batch-update seam and mutually exclusive accounting, no unrelated behavior changes。
- Create `cipher-lab/plugins/cipher_lab_plugins/innovation1/prefix_bridge/{__init__,config,kat}.py`。
- Create `.../prefix_bridge/tests/{conftest,test_prefix_bridge_cipher,test_prefix_bridge_fit,test_prefix_bridge_config,test_prefix_bridge_kat}.py` plus test-only guard helper if needed。
- Append only `cipher-lab/docs/provenance.json` sourceURLs/formulas/boundary verificationstatus。

**Interfaces:**
- `present_prefix.encrypt_prefixes(points: NDArray[np.uint64], key_words: tuple[int,int,int], rounds: tuple[int,...]) -> NDArray[np.uint64]` returns native C-contiguous shape[N,len(rounds)]，1..65536points, strictly increasing distinct rounds1..31，noalias/inputmutation, validated low32/low32/high16。purecipher/noRNGmodelMeter。
- `direct_block.fit.fit(..., *, phase=..., batch_update: Callable[[Floats,Labels,Meter],float] | None=None)` newcallbackcountsforwardinternally with existingruntime，fit owns optimizer reserve/complete only；callbackbranch skips engine.train_forward/backward/update entirely；oldbranch byte-equivalent semantics no extra numerics；record_train_bce要求finite >=0。
- `prefix_bridge.config.recipe()/counts(epochs13)/CAPS`stdlib constants exactspec，SLOTS `s2-only` then `r2-cipher`,`r2-joint` ... `r7-cipher`,`r7-joint`；rowslabelsN×1；counts accepts13completedepochs18..30。
- `prefix_bridge.kat.check(meter, scalar, prefixes)->None` explicitcallables charges14，用full31zero KAT与r2..7nonzero公开K/P，batchsingle-targetrounds每次1；`production(meter)`lazyimports realcipher，开发只fake。

- [ ] RED：publicrecipe caps3538958/99840/16400384、runtime13slot不同identity、wrongcounts rejected；人工scalar/prefixfake记录14调用/strictnonzerokeylowfirst/full31KAT/错误gate无keyfit。
- [ ] RED：批接口malformed dtype/endianness/noncontiguous/keywidth/duprounds/roundbounds/oversized拒绝在轮运算前；用sentinel replacing privatecipherkernel证明dispatchnoinputmutation/matrixcolumns/noalias，禁止真轮循环或KAT。公式单步可静态inspect不运行fullcipher。
- [ ] RED：公共fit人工Engine与人工batchcallback分别走，newcallback不调用原forward/update，meter每trainbatch1update/1forward、DEVsamepredict；callbackfail计attempt不completion、badlossfail、tieearliest/independentstop/firstbestwritefail账保持新值；旧Engine人工测试不改期望。
- [ ] GREEN：最小purecipher算法和fitoptionalhook；无第二runner/Adam。conftest在cipher/NN底层guard生效前不运行科学路径。
- [ ] 验证：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/.venv/bin/python -B -m pytest -p no:cacheprovider -p no:plugin_isolation plugins/cipher_lab_plugins/innovation1/prefix_bridge/tests -q`，仅新人工tests，RED/GREEN命令输出写report。主控后续fresh回归。
- [ ] 报告task-1-report.md含接口/changedpaths/TDD证据/guard范围/未执行KATscience；不Gitcommit，主控从snapshot构diff独立review。

## Task 2: 13fit子owner与最小CLI/既有启动器接线

**状态：COMPLETE / StaticApproved。** implementc9caed8d；taskreview52acfb90及wholea6312ee6StaticApproved，whole唯一补丁为旧config inventory测试按owner静态check分派，scope复审ADDRESSED。主控fresh新owner88pass48.70s/旧cube+split8+early308pass87.36s/defaultfullsafe3704pass49excluded208.97s(既存ZIPwarning1)，forcedRuff clean/mypy440。磁盘fsync慢及64MiB tmpfs容量/pytest提前清理冲突已定位，只改主控人工验证临时目录/清理时序不改产品持久化，失败尝试日志保留。真实科学仍0。

**Files:**
- Create `.../innovation1/prefix_bridge/{data,model,runtime,weights,metrics,training,cli,report,source}.py`。
- Create `.../prefix_bridge/tests/test_prefix_bridge_{data,model,runtime,metrics,training,cli,report,source,guard}.py`；本owner helpers可追加，不写测试专属方法进production。
- Create `cipher-lab/configs/innovation1-present-prefix2-bridge.json`。
- Modify `cipher-lab/scripts/start-cube-sharedm.ps1`仅config固定allowlist/pluginmapping/14400wall/noSourceRoot，现reuse/split8/exit逻辑保持。
- 可调整Task1本owner接口实现以匹配当前任务，但公共fit/purecipher变更先报主控scope，不能暗扩大。

**Interfaces:**
- `data.make_panels(meter, draw, encrypt_prefixes, *, sample=...)` factory onceprivate80draw/同rawplain acrossrounds/splits；`panels.train/view/dev/test`显式：TRAIN/DEV一次生成，TEST延迟；BatchPanel.batch(indices)返回FP32[G,4,6,64]及uint8[G,1]，按mask只提供允许canonical三元组，S2-only无需接收finalround。sample/secrets/cipher注入仅人工tests；productionlazyreal。
- `model.PrefixSharedM`consume[G,4,6,64] finitebinary，twogathers54channels，同stem初态，zeroPRESENTrelation，public13PRESENTroute+suffix/readoutfromcore；无cipher/configfile/datastore依赖。`build(seed)`隔离CPUinit给freshmodel，同参数initialSHA/geometrymask身份。构造/参数计数可测试，无数值forward。
- `runtime.Runtime`inherit innovation1.models.TorchRuntime inference_only，onecurrentslot `add_adam`复用；prepare/predict/dump/restore adapter供fit，`batch_update(inputs,labels,meter)`调用parent.update并仅parent计forward，fit计updates；task/slot/mask严核，FP32predict shape[G,1]，bestload新domain.strictdecodeCPUgeometry buffers全匹配。freshmodel训练及release/diskreconstruct不继续优化。
- `weights.identity` binds method/round or None/mask/S2definition/recipe/source/runid/epoch/bce，encode/decode复用codec，不复用旧fixedfiveidentity。
- `training.run`oneMeter/RunStore：KAT14→onekey/sampling→TRAINDEV→13fit独立stop/slot→所有13diskreload→TESTgenerate→13score→保存labels/logits及pairedmetrics→唯一terminal。BEST_SAVE phase先snapshotcurrentcounts，不含TEST；未terminal任何故障不能partialfinal。为RSS可训练后一fit释放GPU、最后13strictrestore逐一CPU校验再score需何时恢复到GPU交由明确runtime生命周期实现，但不得把仅文件hash叫strictrestore或先TEST后补恢复。
- `metrics.score`组ACC/BCE/每类correct/confusion、jointvsboth pairedcorrect4格、百分点差；S2一次score复用六表，不CI/bootstrap额外runtime、不宣称因果依赖。
- `source.source_identity`core+innovation1+本owner+direct_block真正复用closure（可组装现direct_block identity＋innovation1 identity并explicitnewownerpaths，重复digest必须相同），无遗漏生产imports/测试缓存；claim/devbest/testgatefresh核。
- CLI newexplicitplugin `cipher_lab_plugins.innovation1.prefix_bridge.cli`，core `python -B -m cipher_lab check/run --plugin ... --config ... --root ... --run-id ... --deadline-utc ...`；stdlibcheck无科学import/claim/resources，runlazydevice只GPU1不旧Task1defaultGPU0。配置全部设置在configs exactparse，报告保存值不重新推理。

- [ ] RED：makepanelstestdraw80一次/6outputs只生成一次/各splitlabel恰半/4pairgroups/positiveΔD00000、negativeindependent/输入mask与r2重复/TESTnotbefore13reload；人工oracle literaloutputs，counterexact，有错误node不自动retry。
- [ ] RED：model三个steminput54/后部旧shape/参数sameinitial各slot无alias/publiccardPRESENTonly/不含key或config读取；通过构造state和人工module替身，不运行Conv/Linear/Norm/activation/ATen数值。Guard测试用无害stub验证底层cachedhandle拒绝，零kernel触达。
- [ ] RED：runtime parentupdate被人工stub截获，oneforwardoneupdate，默认五slot未构造、addAdam真实构造poisoned；checkpoint人工state round/mask/bufferwrong/SHAwrong/16MiB拒绝，NN参数不做前向。
- [ ] RED：13fakefit含18/30不同epoch、TESTscoreonly13、S2onlyonce、lastreloadbadTESTzero/后roundscorebad无partialterminal/firstsavebadpaidcounts持久/期限存储source_changed/秘密异常sanitized/noscientificpanelondisk/realrecipe capmatches。
- [ ] RED：stdlibCLI clean-source artificialclosure check/run失败语义、脚本分派只noSourceRoot/4h新config而旧模式不变（无真实PS/SSH，人工进程fixtures）；savedreport有13fit/ACCnotbits/S2shared/oracle/noextend/nativeSSHseparate，不从data生成metrics。
- [ ] GREEN：实现最小owner和fixedconfig/PSmapping，无genericworkflow/registry/no第二runner/no额外真实test。
- [ ] 验证人工newowner、Task1fitcallback＋旧cube/split8人工owner、相关innovation1/core默认fullsafe分别，forcedRuff/mypystrict。完整默认测试按49science排除，不用newowner未收集冒充覆盖。
- [ ] 报告task-2-report.md RED/GREEN/actualcommands/changedpaths/closure/启动命令/参数shape/不落状态/未验证Windows/KAT/GPU，主控taskreview＋全diffreview/freshverification后才正式。

## Task 3: 主控限定发布、一次14block门禁＋13fit正式批次、结果交付

**状态：RELEASING；用户已批准，不逐步骤重复询问。** 工程及静态门槛收口，准备一次限定发布/正式batch；尚未执行真实KAT或GPU训练。

- [ ] 复核Task1/2独立审查闭合、主控fresh新owner/旧owner/defaultfullsafe/Ruff/mypy/staticCLI，科学消耗仍0。
- [ ] 限定allowlist从basecb5e78c独立index构commit/push+核ref，main/index/旧产物不变；远端G新rootGitexactcheckout核runtime源/配置SHA。只用现pythonuser-site，不install/真实预热。
- [ ] GPU1UUID/freeidle/hostname/PSparse/noSourceRoot/4h上限；保持前台SSH，同原生globalexit记录与实际wait，未知不重启、不补退出。
- [ ] 正式唯一batch先14block公开gate，通过后才私钥及13fit，任失败止retry0；13best全strictreload后TEST完整统一终分，不改输入/样本/seed/range。
- [ ] SCP仅实际data/models/result及必要日志，核SHA/实际13fit/epochs/计量/退出，不loadweights/加密/NPZdecode/forward/evaluate；原始状态不应在产物。
- [ ] 保存值做三路轮数表、BCE曲线和方法图，报告joint−Cr/joint−S2描述差/单key探索/oracle/无因果归因/不自动加轮，HTTP仅result验证后交付；独立saved-evidence review按证据层级。
- [ ] 更新ledger/plan并关闭许可/剩余额度，无自动追加实验/重试。
