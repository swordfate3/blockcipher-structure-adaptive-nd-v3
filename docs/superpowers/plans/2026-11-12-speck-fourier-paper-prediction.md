# SPECK论文复现与同轮位预测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在远程A6000物理GPU1、合计20GiB显存、共享8小时内，先论文级复现8轮SPECK32/64教师及Fourier12项，再条件执行同算法同轮绝对位预测。

**Architecture:** 保留统一cipher-lab check/run、一个主owner/RunStore/Meter；新增专属study方法，密码/数据/频谱/教师适配/资源探针各自独立。主进程现有Python3.10+Torch，原TF教师由3.9串行bounded provider服务主进程（仅单机进程、非远程任务系统）。父监督仅资源/退出与关闭证据，不写child产物。

**Tech Stack:** 本地Python3.12、现NumPy/Torch/pytest/Ruff/mypy；远端Windows既有predictor3.10/Torch2.7与py39gpu_new3.9/TF2.5/h5py，禁止安装/升级。源码经本地Git commit/push、远端exactfetch/checkout，所有生成物G:\lxy内。

**Spec:** `/workspace/docs/superpowers/specs/2026-11-12-speck-fourier-paper-prediction-design.md`，用户已确认详细设计并要求继续；本计划待阅核后按项目指定代理驱动实施。

## Global Constraints

- R→条件B各一次、retry0，不重训教师、不NN训练、不多key/seed/阈值矩阵，不用微小科学实验找结果，不恢复旧周期。
- M=10,000,000 pairs/层；前7层完整枚举128prefix无查询；R k8..64，B k8..32；bucket score严格>.005，tau=.1、无beam32；>400保留prefix停止ENGINEERING_LIMIT，不截断。
- R coef1,000,000、teacher一致性10,000,000；classification10,000,000 pairs=20,000,000真实blocks，面板复用teacher/Eq4/提取式评价，不生成第二次标签。teacher上限1,161,000,000 rows，R0公开加密≤512。
- B固定新privatekey8bytes，绝对MSB0bit0、fulluniform32bitPT；500m搜索+1mcoef+10mDEV+条件10mTEST≤521mreal；总real≤541,000,512，policy542,000,000差额不能额外花；optimizer/null0。teacher rows不能写0forward。
- B abs(coef)>.05后最多12项、abs sum降/mask升；DC/intercept固定0，constant0基线；DEVacc≥.501且delta>0只gate；TESTacc≥.501且全样本D Hoeffding lower>0才positive exploratory。
- 采样有放回/阶段publicdomain独立，不承诺样本unique/unseen；重复调用照计，key不publicseed/不落盘。公开PCG64种子Rclassification=19691112、Rsearch=29691112、Rcoef=39691112、Ragreement=49691112、Bsearch=59691112、Bcoef=69691112、Bdev=79691112、Btest=89691112，层domain用SHA256固定stage/depth派生，不随chunk变化；这些是主控本轮配方选择非论文原种子。模型不得真实key/state/otherbit教师，R教师仅R任务。
- physicalGPU1用UUID/PCI绑定核验，内部logical0，不fallback0/不停他人模型。TF8GiB+Torch8GiBallocator，实际所有本任务PID显存合计≤20GiB=21474836480B采样停止，无法观测GPU_UNVERIFIED；非OS硬隔离。
- 首次正式CUDA/模型初始化前父origin统一28,800swall，science28,200s，600s关闭在内；CPU进程树只累计非单核8h门限。GPU/RSS0.5s采样，hostRSS32GiB、总输出4GiB内部64MiBreserve含tmp/IPC/cache/log。
- 正式启动前free≥24GiB/util≤30%；不中止现服务，不无限等待。现场环境不兼容停，不安装/转换权重fallback。OOM停不重试；非OOMbatch可按50k→25k→12.5k预设降档、不重放计费工作/不改总M。
- 生成物全G:\lxy\shared-m-cipher-nd-runs\<freshid>\repo\cipher-lab\{data,models,result}与监督目录，TEMP/TMP/cache同边界；结果SCP取回原生接收子目录，不上传科学任务包，不远端commitpush，不通过Git回传结果。
- main原dirty保留，BASE fd3ed2d451ef5a4c56380cc1dedde2473741640b；无reset/clean/stash/worktree。只本轮闭包明确发布，原有未提交必要依赖逐项列审，不把HEAD大diff冒充本轮diff。
- 开发真实cipher/key/RSS/CUDA/teacher/NN/optimizer0，人工Boolean/toy与假oracle允许；禁止未保护全pytest/旧微训练。先TDD/独立review/主控fresh/cleanfake；实际密码KAT仅正式R0≤512内。

## Review Focus

1. GPU编号不等于CUDA序号、WDDM缺PID显存：Task3必须测UUID错误/合并TF+Torch总量/缺测failclosed；不能allocator冒充actual。
2. 双Python版本导入闭包与IPC超时/部分帧：Task2/3对应解释器synthetic导入，测试partial/oversize/sequence/checksum/provider失联，不跨exec假定父guards。
3. 12项极性/tie/bitorder/classpair两次加密：Task1/4 literal参考与官方保存mask对应，tie0主口径+官方tie1描述，不TEST择方向。
4. 无beam全桶与chunk采样共享：Task1/4测试chunk划分不改各层样本及整数sum，400cap拒绝不topk，None/中断保存attempt/complete。
5. 父/主/provider异常留下活进程或伪COMPLETE：Task3/6 fakeJob/pipes/fsync/lateclock故障；关闭和取回不超8h，缺exit/缺账UNKNOWN非阴性。

---

## Task 1: SPECK公开运算、完整recipe、分块频谱数学与严格模型格式

**Owner:** implement-deep。新增 `src/cipher_lab/ciphers/{speck,speck_tensor}.py`、`speck_study_config.py`、`speck_study/{__init__,parity,search,codec}.py`；tests/test_speck_study_{config,cipher,search,codec}.py。仅这些，旧SIMON/search/schema不改。

**Interfaces:**
- `SpeckStudyRecipe`冻结纯stdlib parse/to_dict，method=`speck-fourier-paper-prediction-v1`；formal与显式synthetic独立，所有尺寸/阈值/gates/seeds/资源规则字段完整严格、未知/duplicate/bool/NaN拒；`estimate_study(recipe)`给real/teacher/parity最大分项。
- `encrypt_scalar(point: int, key_words: tuple[int,int,int,int], rounds: int) -> int`、`encrypt_tensor(points, key_words, rounds)`；32bitblock左右16，公开key顺序与官方完全明确，tensor只Torch算术不读秘密配置/数据/训练。支持R perrowkeys和Bsinglekey，秘密由owner持有，cipher不保存。
- `initial_prefixes(bits: int, depth: int=7) -> tuple[int,...]`；`BucketAccumulator(bits, depth, prefixes, count)`接受 `add(points, paired, signs, paired_signs, *, check)`，`finish(tau=.1, max_prefixes=400)`返回全tested integer sums/count/masks/kept与工程limit；mask完整uint64 MSB前缀，B32域以高32位编码/下32为0。chunk≤50k，内部paritytiles≤4096row触发check；没有beam排序截断。
- `CoefficientAccumulator(masks,count)` chunk整数signsum；`StudyPredictor`纯stdlib结构、R/Bkind/task/source/recipe/terms/sums/count/intercept/tie规则；`encode_predictor`/`decode_predictor(expected_identity,expected_recipe)`≤256KiB strictJSON，不Torchload，duplicate/NaN/超400/零外域mask/sumcount不符拒。R保留全部abs>.05；B筛选后top12/intercept0。

- [ ] **RED:** 配置边界/静态teacher1.161b与pair2blocks；纯literal bitparity/2parityANDsign谱±.5/chunksum invariant；k8/k32/k64、400→401拒绝非截断；符号零分tie；native ndarray拒subclass/MaskedArray；codec expected自模型identity不能自证。
- [ ] **RED:** 密码开发只测公开轮部件整数公式和fakecipher接线，不执行真实多轮KAT/批量加密；tensor用CPU人工单轮数据结构与独立literal round参考（不是私有key/科学样本），正式KAT测试用fakeoracle标记现场R0再执行，测试不得偷偷真encrypt。GPU实现不CUDA初始化。
- [ ] **GREEN:** 最小实现，NumPy/Torch parity分别literal对账，无正式M规模性能跑；来源提取SHA/许可记录交Task5。
- [ ] **VERIFY:** 具名tests/Ruff--no-cache（兼容文件py310）/mypy隔离cache、独立math+qualityreview与主控fresh；完成报告实际接口和7+新文件SHA，no commit。

## Task 2: 原教师3.9-provider与3.10-client（单请求串行）

**Owner:** implement-deep。新增 `speck_study/{teacher_protocol,teacher,teacher_provider,teacher_network}.py`，tests/test_speck_teacher.py/fixtures专用；不改Task1/CLI/RunStore。

**Interfaces:**
- protocol stdlib兼容3.9，fixedmagic/version/seq/count/payloadlen/SHA，输入packeduint64 big endian→[c0L,c0R,c1L,c1R]bitsMSB，replyfloat32prob+hardlabel；最大50krows（400k输入/250k输出），只一个inflight。`read_frame(stream, deadline, max_rows)`/`write_frame`拒partial/overflow/checksum/wrongseq/EOF/超时、无pickle/exec。
- `TeacherClient(python_exe, provider_module, weights_path, expected_sha, gpu_uuid, root, deadline)`启动已Git源码module非stdin代码；`predict(points)->(probs,labels)`无资源退款、调用由主ledgerreserve/complete；`close()`bounded TERM/kill/reap，ready帧包含TF/Python/model参数/weightSHA/GPUUUID，先核再请求。
- `teacher_network.make_official_teacher()`仅lazyTF官方确定架构、weights-only load_weights；TFallocator8GiB/visibleUUID在任何TF/CUDA初始化前设置，GPUidentity核验、禁止tf.function训练/optimizer/未审自定义反序列化。stdout仅帧，诊断stderr有界log在root。

- [ ] **RED:** 全帧malformed/seq/replyrange/p>.5/tie/bitorder/超时/半写/早退/kill；fakeprovider own guards、stderrflood/不存在exe/启动失败不遗livechild。主provider不能得到Bkey/主RunStore。
- [ ] **GREEN:** fake概率与literal公式映射测试；teacher网络层规格用fakeTF对象检查，不loadH5/真实NN/CUDA；不克隆或运行官方notebook。
- [ ] **VERIFY:** 3.9语法静态解析/py39targetRuff及独立mypy配置，主3.10target；对应真实解释器synthetic验证列入Task6pre-science无CUDA步骤，不能3.12通过就声称现场兼容。独立IPCreview+fresh。

## Task 3: Windows资源、进程树所有权与8小时监督契约

**Owner:** implement-deep。新增 `platform_resources/{__init__,windows,gpu}.py`、`speck_study/resources.py`、`tools/speck-study/{supervise,freeze}.py`及`tests/test_speck_resources.py`、`tests/test_speck_supervision.py`。监督入口只本次stage资源控制不做科学算法/SSH/Git。共享 `execution.py`只准等价UTC→timezone.utc以3.10导入、增加显式keyword `enforce_cpu: bool=True`；False仅新study传入，仍采样累计CPU、不放松旧默认语义；相关具名旧Meter测试补默认仍stop/newFalse仅CPU不stop。CLI未接前由Task5使用。

**Interfaces:**
- `GpuBinding.from_snapshot(index=1)`冻结UUID+PCI，`verify_runtime(binding, provider_identity)`failclosed；Windows NVML进程显存若WDDM N/A可用Windows GPUprocessmemory counters等价进程和adapter映射，任何无法证明→GPU_UNVERIFIED，不能降成allocatoronly。实现只标准库ctypes/subprocess现系统接口，不安装pynvml。
- `StudyResources(binding, root, wall_origin, science_deadline, close_deadline, process_tree)`维护所有任务PID创建时间/句柄，`check()`采样0.5s actualGPU合计≤20GiB、RSS≤32GiB、output≤4GiB-reserve；private异常固定码，报告sampled非硬peak；WindowsWorkingSet与Job Object/PID所有权在spawn前准备，任务job不包含既有其他model。
- `StudyMeter(Meter)`或窄adapter只附加teacher/scoring/parityattempt/complete及resourcescheck，不造通用runner；同一计数owner。现forward_rows对应teacherrows，optimizer0。

- [ ] **RED:** UUIDenum不同、TFprovider没继承、physical1逻辑0与PCI错；两个allocator各合规但actual合计>20GiB；missingRSS/GPU/CPU/Jobassignfail；reusePID不杀新进程；deadline28,200/28,800、lateclose/dirfsyncafterrename/部分terminal不能COMPLETE。
- [ ] **GREEN:** 纯fakeWinAPI/fakeGPUclock/假进程Job，Linux开发不读productionRSS/真实nvidia-smi/CUDA；旧MeterCPU默认不变，False读数不归零。
- [ ] **VERIFY:** Ruff/mypy新边界/具名受保护Metertests、独立跨进程review/fresh。Windows实际资源探针只Task6明确现场步骤，不扩大真实科学。

## Task 4: 数据owner、R→B阶段机与保存式报告

**Owner:** implement-deep。新增 `speck_study/{data,method,report}.py`，tests/test_speck_study_method.py、专用fakes；消费Task1–3接口不改其语义。

**Interfaces:**
- `StudyData(recipe, cipher, private_random, public_sampler, meter, store)`唯一secretowner。R1每row新key不保存，保存公开packedcipherpairs/Y/probabilities为boundedimmutable datachunk refs，完整ownermanifest/hash；`classification_chunks()`只读R1文件校验重用，绝不再加密。B `begin_prediction()`仅通过R gate后一次private8bytes，`paired_chunks(stage,depth)`按stage/domainseed+层稳定流，不随计算batch重排抽样；`panel_chunks(coef/dev/test)`显式时序release gates，重复域拒绝，attempt非退款，计费oracle返回验证后complete。
- `run_study(recipe,data,teacher,math,resources,store)`固定R0→R1teacher+Eq4gate→R2→R3→Rgate→Bsearch→Bcoef→strictdurable→DEVgate→条件TEST；每完成chunk保存聚合/尝试complete差，raw秘密不写日志。ArtifactError已sealed不二次finalize；Stopped/UNKNOWN不遗scientific_validationTrue。
- `render_report(saved)`只标准库mapping→三件套，明确reproductionmatched vs predictionpositive，negative coefficient/delta如实，HTMLescape。模型write_model固定slots .ptopaqueJSON与registeredSHA ownerreadstrict，不扩evaluate。

- [ ] **RED:** R公开pair确两次cipher计费、512R0上限；teachers1.161b静态max且earlystopactual；R1immutable复用不再labels；R mismatch不B、不取Bkey；badteacher超概率/NaN/负row fail；B不进入Rprovider；sampledomain不随chunk变；key异常不repr。
- [ ] **RED:** 完整tiny人工场景覆盖Rgate/emptyR/400cap/coefNone/Bbeamempty/DEVfail/TESTpass/stoplate；主tie0与官方tie1只描述；source/task/backend/recipe/registeredSHA/tamper持久失效不fallback；D全N与baseline0/Hoeffding；None与中断工作账保留，不以partial0项报阴性。
- [ ] **GREEN:** 工程所有oracle/key/RSS/CUDA/NNtrap并fake，不以tiny科学结果调参数；save数据消费者/保留规则显式。
- [ ] **VERIFY:** 独立specqualityreview、主控fresh/源码冻结不越权；savedreport零cipher/model依赖、未TEST不填0/50%，singlekey/noise/coverage限制齐全。

## Task 5: 统一CLI、窄3.10兼容、干净副本与可发布闭包

**Owner:** implement（公共导入/兼容复杂时转implement-deep）。可改 `config.py`、`cli.py`、`representations/simon_anf.py`唯一PEP695等价类型别名及确需3.10导入的最小文件；配置/README/AGENTS/docs/provenance；新增configs/speck-study-{paper,example}.json、tests/test_speck_cli.py。不得全包降版/改旧方法范围/扩大旧checkpoint兼容。

**Interfaces:** 新backend=`speck-study`显式dispatch只新recipe，`check`stdlib-only含gpu binding要求/teacherrealrows及4项fit，不读取机器/claim/初始化CUDA；`run`装配StudyData/TeacherClient/resources并claim models。新method typed分支处理，不误走simon/integral。3.10 datetime.fromisoformat不接受Z处显式Z→+00:00、timezone.utc等价，旧py3.12所有接受/拒绝值不变。provider入口不import3.10only共享模块。

- [ ] **RED:** zero-scientificimport check childguard+zerooutput；wrongmethod/backend/flag/exe/pinnedweightsmetadata拒；旧6类needs_models变7最小测试明确；ASTtarget3.10/3.9及命令错误边界。
- [ ] **GREEN:** 配置全部完整正式规模，无silentdemo小规模覆盖；source-onlyimports防未冻结pyc，nativeWindows目录fsync/句柄适配由有职责层小改，不借此降级RunStore身份/owner；此若实际跨文件高风险需implement-deep+专项review。
- [ ] **VERIFY:** 本地clean-sourcefake CLI check→run→read_saved/严格model，childowntraps/fakeentropy、复制仅源码/config/pyproject/gitignore，不旧数据/权重。主控fresh/Ruff/mypy限定包实际兼容文件目标，报告全suite未跑范围。
- [ ] **PUBLISH PREP:** 执行git status/index审查；生成明确本轮filelist+必要未提交importclosure/snapshotmanifest，依赖baseline列表供主控核验。不得stage .learnings/旧report/历史configs/backups/data/weights/cache。若需整旧closure带入先逐文件说明，不用宽泛gitadd .；无关工作原位保留。

## Task 6: 本地发布、Windows现场验证、唯一科学与saved-only交付

**Owner:** 主控核Git；experiment代理远端只执行已批准exactCLI/只读取回，无子代理。消费Task3的`tools/speck-study/{supervise,freeze}.py`及对应人工tests，不另造执行器或改已冻结产品；不得旧远程runner/SSH任务包。

- [ ] **REVIEW/PUBLISH:** whole-chain独立review后主控fresh保护具名tests/cleanfake/diff/职责。明确filelist本地commit/push，用远端refSHA实核发布成功，权重不入Git。远端新目录fetch exactSHA（不是dirty旧目录pull），从官方固定revision获取教师asset/sourceSHA；网络失败停，不SCP科学源码替代。获取权重≤原输出4GiB。
- [ ] **REMOTE SYNTHETIC:** 远端3.10主入口、3.9provider fakeTF/IPC、fakeWin资源/owner闭环，所有生成物root内；此步骤无CUDA/真实NN/cipher/key/RSS且不占正式8h，不称GPU就绪。按冻结source/config/asset exactSHA核，无远端改科学代码。
- [ ] **RELEASE:** 新鲜GPU1 index→UUID/PCI/free≥24GiB/util≤30%快照只检查一次，记录所有权/Job准备；主控门禁通过后明示唯一science释放。first CUDA/model init前统一origin，主child/provider共享wall；R0真实现场检查计入512与8h，实际env/显存不可核不重跑/不安装。
- [ ] **EXECUTE:** CLI本机运行可SSH启动已Git入口但不保证断线存活；WindowsJob终止/所有进程回收与保存exit，必要SSH只读状态；首失败/limit/UNKNOWN停，retry0。层进度/实际查询/forward/RSS/GPU/剩时落盘；科学截止28,200关闭28,800，不耗尽后重新计时。
- [ ] **RECEIVE:** 取回指定data/models/result与监督manifest到新原生接收子目录，核SHA/source/config/weight/exit/counts，不本地loadH5/模型/NPZforward、不重算指标/新标签；未完整取回NOT_RECEIVED不伪成功。远端原件不删。
- [ ] **DELIVER/CLOSE:** saved-only中文report.md/index.html/source.json，方法流程图+实际保存结果图；区别R复现门结果与B持出预测、未访问不补数、GPU20GiB采样非硬隔离、所有partial/overshoot/环境缺口。仅服务本地result/并验证HTTP200/data-modelsescape404，没像素验证直说。独立结果review、主控核账、关余量，不自动evaluate/调阈值/换key/seed。

## 计划自检与交接

Spec§1–4映射Task1/2/4；§5GPU/资源映射Task3/6，§6兼容/IPC映射Task2/3/5；§7–8发布/停止/未知映射Task5/6。五个ReviewFocus均有具名故障类测试；阶段接口依赖顺序1→2→3→4→5→6（不用并行改同文件）。CPU enforce_cpu新keyword默认True只study False保留旧语义；所有大型前向预算逐rows明确。未获安装/teacher重训/fallbackGPU0/fullFWHT/阈值网格授权。

用户已确认本实施计划并要求代理驱动执行，不再询问资源/Git许可或执行方式。所有代理报告/TDD/review/fresh证据放本计划SDDworkspace，主控不把代理自述当验收。

## Task5续补：已批准DXGI证据与实际监督装配

按spec§9逐步实现，仍TDD→独立review→主控fresh，不提前科学或发布。

### Task5A：实际GPU观测与Windows保存证据
**Owner:** fresh implement-deep。
**Files:** 新增`platform_resources/dxgi.py`、`speck_study/gpu_telemetry.py`、`tests/test_speck_dxgi.py`、`tests/test_speck_telemetry.py`；最小修改`platform_resources/gpu.py`和`speck_study/resources.py`及相应既有具名tests；不改CLI/科学数据/阶段机。
**Interfaces:** `DxgiUsage.sample()`返回本PID所有适配器LUID/LOCAL CurrentUsage字节；只lazy ctypes/COM，provider闭包py39safe。`actual_attestor`追加currentcontext CUDA LUID与node mask，UUID/PCI旧检查保留。`TelemetryPublisher`独立线程≤0.5s，标准库bounded原子帧含协议版本/tick/monotonic/PID+creationtime/adapterrows，最大64KiB，输出计入root，不传key/标签；`TelemetryGpu.processes(binding)`由父核Job exact成员与独立创建时间/严格新鲜度/序号，缺PID/缺显式0/错LUID均failclosed。初始化期间所有枚举适配器保守求和，样本仅bootstrap不科学准入；context UUIDPCI→LUID严格收口后只准指定GPU，其他适配器actual>0停止。保存receipt用filefsync/replace/读回长度SHA，Windows显式namespace_powerlossUNVERIFIED，不新增掉电承诺。
- [ ] RED：fakeCOM无CUDAbootstrap显式0、duplicate/malformed LUID/overflow/NaN、COMrelease失败与Query失败；无PID/过期/错seq/creationtime/PIDreuse/错误适配器/总额>20GiB/当前context错GPU；不是missing0。
- [ ] GREEN：最小实现，bounded读写/输入不能自定路径或跨根，独立telemetry线程不取teacher预测锁，故障sticky停止，暂停/closing仍采样直到reap。
- [ ] VERIFY：具名pytest guards、py39/310 Ruff、mypy310隔离cache、对应版本AST仅语法，独立review/fresh；未实际Windows/COM/CUDA验证如实。

### Task5B：父→child→provider真实接线与冻结主闭环
**Owner:** fresh implement-deep，消费5A通过接口再动。
**Files:** `speck_study/application.py`、`closing.py`、`teacher_provider.py`、`tools/speck-study/{supervise,freeze,source}.py`，必要`platform_resources/windows.py`/`cli.py`最小参数装配；新增`tests/test_speck_native_wiring.py`，扩既有具名CLI/source/IPC测试及README/provenance。
**Interfaces:** 单一root/origin由父在现场首次初始化前冻结；suspendedspawn→Jobassign→resume，closing仅指定两event句柄显式handle_list继承，不继承无关句柄/不breakaway；CPU主/provider各telemetry ownPID，父自身也纳预算证据但不自锁Job。主owner装配现固定StudyData/method、TeacherClient trusted source-only wrapper及verify_gpu，正式无fakeoverride；R0后才科学且≤512，teacher close/reaped后B。child消费request后不可逆停科学→原owner保存→ACK；父严格exit/terminal身份/ACK/资源receipt/Jobempty，迟到或任一缺证UNKNOWN。
- [ ] RED：fake native全链、constructor/predict阻塞期间仍采样/closing、事件句柄白名单/spawnrollback、源闭包包含传递依赖/无pyc回退、初始化错GPU/迟样/缺ACK/partialfinalize/晚exit不complete。
- [ ] GREEN：仅既有职责装配不造通用runner；移除无条件ENVIRONMENT_BLOCKED仅在真实接口齐全并逐项门通过时，原环境/资源失败仍阻断不fallback。
- [ ] VERIFY：干净冻结源码childown guards→fake完整CLI/strictownerreadcodec/report，分进程guards测试；fresh/全链独立review→明确Git闭包再Task6。现场py39/310 synthetic验证后唯一formal8h仍需主控release，不初始化CUDA作预探、不安装/改TCC/干扰别的模型。

