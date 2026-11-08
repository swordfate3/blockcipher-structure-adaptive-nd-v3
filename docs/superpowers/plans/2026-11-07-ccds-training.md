# Cipher Lab 阶段2/3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 接入位预测MLP与创新1四任务shared-M，以及不重新加密的显式evaluate，不复制实验专用执行/保存系统。

**Architecture:** 独立类型化配方、秘密隔离的数据提供者、纯模型与公开表示、共享优化/评价/checkpoint能力、两条薄科学组装；沿用唯一Meter/RunStore/报告三文件。积分既有行为回归，不维护未发布旧schema兼容。

**Tech Stack:** Python3.12、现有Torch/NumPy/pytest/Ruff/mypy；不安装、下载或解析新环境。

**Spec:** `docs/superpowers/specs/2026-11-06-ccds-rebuild-design.md` §10（用户已批准），§1—8的职责/产物/安全原则仍适用。

## Global Constraints

- 产品修改仅`/workspace/cipher-lab/`；旧项目/权重/run只读，不Git写/备份/根目录切换。
- 无真实cipher/KAT/科学标签/正式拟合/GPU/SSH/安装/求解器；测试假采集/人工tensor，CPU单线程。每次完整合成闭环位预测≤2更新、shared-M四任务各1共4更新、每种≤60s，不搜索、不多seed重复科研。
- PYTHONDONTWRITEBYTECODE=1，PYTHONPATH=src，`/workspace/.venv/bin/python`；pytest只收新项目，cache临时精确清理。现有Torch2.14.1+cu130不是GPU授权。
- 无旧API/config/checkpoint兼容、通用registry/workflow、空转发层或巨大utils；真实秘密/可恢复私有seed不写公开记录和checkpoint。
- implement-deep按任务串行TDD；fresh code-review与主代理复验，每关完成才下一关，代理不再委派。报告留会话，本计划兼恢复账本，不建task-report/validation树。
- 来源提取只更新现有`cipher-lab/docs/provenance.json`；依赖声明可修改但不执行安装。scope内必要共享边界整理不能暗改积分数学/阶段语义。

## Review Focus

1. 位预测TEST标签在选模严格加载前不可访问；shared-M预建TEST不能被用于DEV选模——Task3/6时序断言。
2. 模型/数据恶意文件：SHA、大小、NPZ压缩炸弹、tensor形状/值/模型族/任务身份——Task2/5/7负例。
3. checkpoint替换失败或metadata提交失败不得配错best身份，终态/已落盘证据不伪造——Task2/6故障注入。
4. shared-M轮数/public13：r8公开变换一致、改rounds只变公开表示不变router输入，评价拒绝任务错配——Task1/4/5。
5. 看报告/check/evaluate不能悄悄训练、重新采集或导入生产cipher；四预算单位各有唯一调用owner——Task1/3/5/7。

## 共享契约

设置三类：现有Integral Recipe不改科学字段；新增`PredictionRecipe`和`SharedMRecipe`，共用`TrainSpec(train:int,dev:int,test:int,batch_size:int,epochs:int,learning_rate:float,init_seed:int,sample_seed:int,shuffle_seed:int)`。精确字段在Task1固定，模型规格也是公开类型不importTorch。

政策明确新`schema=2`，必填原7字段加`optimizer_updates`、`forward_rows`；每种额度非负。不接受缺新增字段的中间JSON；新旧测试fixture统一更新。真实block/nullword仍由数据侧预留，优化步由共享step预留，所有模型前向行（训练/DEV/最终/显式evaluate）由调用forward的共享能力预留，不能多处重复扣账。失败attempt不退款；complete指实际成功，单位分别记录。

保存模型/数据的bytes仍由唯一RunStore写，调用方不直接torch.save(path)。新增refs用`payload/<name>.json`、`dataset/<name>.npz`（data/<id>内）、`model/<name>.pt`（映射models/<id>），元数据固定kind/bytes/sha256。类别/位置分派非任意路径；模型只有best允许同owner替换，写新最佳必须同步提交引用。read_saved报告路径流式校验摘要不解码权重/数据，不importTorch/NumPy；显式loaders读取时再次校验SHA。run.json新schema=2，未发布旧记录不兼容；来源和环境冻结、三报告统一限额等保持。

数据bundle：NPZ无pickle，数组uint8二进制、方法声明/任务元信息为有界UTF8 uint8 JSON数组；位预测x[N,64]/y[N,1]，sharedM每任务x[N,4,2,64]/y[N,1]，public卡从声明校验后重建，不能接受任意卡。单文件/解压总量≤256MiB、≤16条目、禁止重复/路径条目、逐数组shape/dtype/finite/0或1验证，不允许object/fortran异常布局绕过；从ZIP头与NPY头预算预检后再读，拒绝超额与声明不符。保存CPU测试数据一份，不存私有材料。

### Task 1: 独立训练设置与四单位政策

**Files:** Create `src/cipher_lab/training_config.py`, `tests/test_training_config.py`；Modify `config.py`, `execution.py`, `cli.py`仅check分派/估计，`AGENTS.md`, `README.md`策略示例，现有受schema更改影响的测试fixture；Create `configs/{prediction-example,shared-m-example}.json`两份，不批量复制。

**Interfaces:** 公共模型规格`MLPSpec(hidden_width:int)`、`SharedMSpec(stem_width:int,channels:int,norm_groups:int,suffix_kernels:tuple[int,...],head_hidden:int,router_hidden:int)`；`TaskSpec(cipher:str,rounds:int,difference:int)`与四任务顺序/公开结构表单点（具体public张量卡Task4生成）。`PredictionRecipe(schema,method,model:MLPSpec,training:TrainSpec,rounds,target_bit,label_mode)`，label_mode cipher/null；`SharedMRecipe(schema,method,model:SharedMSpec,training:TrainSpec,tasks:tuple[TaskSpec,...])`；均frozen/to_dict。`load_experiment(path)->Recipe|PredictionRecipe|SharedMRecipe`复用严格JSON一次读取、显式method分派，无registry。`load_policy`schema2严格，Meter扩展四单位/固定stop reasons。

- [ ] RED：未知/dup/NaN/bool、两方法跨字段、合法round/bit、四任务且顺序/差分/round验证、模型尺寸合法；Train rows1..2000000，shared每split2..65536偶数，train需batch整除，epochs1..100000，lr正有限≤1，三公开seed0..2^32-1；MLP1..4096，shared widths1..1024且GN整除、odd kernels1..31最多16、router/head1..1024。
- [ ] RED：check毒化Torch/cipher导入仍通过，预计位预测updates=epochs*train/batch、shared乘4；real位预测三split行数、null独立bit不算64bit字可每bit消耗一个null_words；shared real=8*4*(train+dev+test)；forward包括每epochDEV和最终三split，不漏任务因子。积分新政策training预算0仍可check/run demo。
- [ ] 运行指定tests确认RED，实现并全新项目GREEN/Ruff/mypy。只更新受必填schema变化影响的fixture，不为旧值加兼容分支。AGENTS把当前许可改为阶段2/3无cipher合成，保留其余限制。fresh审查/主代理复验。

### Task 2: 统一受控二进制产物

**Files:** Modify `artifacts.py`, `tests/test_artifacts.py`及schema2必要fixture；Create `tests/test_binary_artifacts.py`。若artifacts继续膨胀，仅允许将其现有纯格式校验移至`artifact_schema.py`形成明确职责，不造通用storage框架。

**Interfaces:** `RunStore.write_dataset(name:str,content:bytes)->str`不可覆盖；`write_model(name:str,content:bytes,*,replace:bool=False)->str`仅best模型替换；`read_binary(root:Path,run_id:str,ref:str,expected_sha256:str,max_bytes:int)->bytes`受限类别定位/流式SHA/大小；`read_saved`返回metadata及JSON payload，binary不解码。schema2 fixed artifact kind与ref绑定，模型run必须needs_models=True。

- [ ] RED：binary大小/类别/路径/symlink/occupancy拒绝，共用output临时峰值和错误封存；无models owner不能写模型；report不解码tensor或访问旧项目。
- [ ] RED：best替换的文件rename或metadata提交失败，owner停止而不得成功引用错误SHA；保留明确故障/原完整模型或标记读回不一致，绝不宣称原checkpoint仍可用。中间文件精确清理，数据不复制进result。
- [ ] 运行RED→最小实现→GREEN全tests/Ruff/mypy→独立review/controller；schema2不是迁移系统。

### Task 3: 无秘密的评价数据格式与两个数据提供者

**Files:** Create `data/{bundles,prediction,differential}.py`, `ciphers/{simeck,present,skinny}.py`, `tests/{test_bundles,test_training_data}.py`；Update唯一provenance。生产算法只实现/静态审查，不执行KAT；已有SIMON可复用其纯适配器。无生产cipher eager import。

**Interfaces:** `DatasetBundle(method:str,task_declarations:tuple[TaskSpec,...],panels:Mapping[str,tuple[Tensor,Tensor]])`为公开数据；`encode_bundle(bundle)->bytes`, `decode_bundle(content:bytes)->DatasetBundle`执行共享契约全部预检。`PredictionData(recipe,oracle,private_random,meter)`，`inputs(split)->Tensor`, `labels(split)->Tensor`, `release_test()->None`, `test_bundle()->DatasetBundle`；release_test前labels(test)拒绝，train/dev输入与test全局无重复。`DifferentialData(recipe,oracles,private_random,meter)`，`panel(split,task)->(Tensor,Tensor)`, `test_bundle()`；private RNG不经公共seed恢复，real每组8调用且一key/4pairs；公开采样与label shuffle可复现不决定key。数据对象不写文件/选模，调用采集按chunk/batch协作check。

- [ ] RED：用捕获fake oracle验证固定key位预测/targetMSB，null不oracle，TEST释放/重复输入隔离；四pair共享新key、正例XORΔ/负例独立P加密、平衡与public labels，错误attempt账本和私有异常固定原因。
- [ ] RED：NPZ重复/extra/object/shape/zipbomb/fortran与尺寸溢出拒绝；回环float32训练tensor与uint8保存恢复相等；不保存任何私有key/seed。fake随机源要有明确有限使用，不能死循环生成唯一明文/key。
- [ ] 运行RED后实现，纯cipher sources登记SHA，真实函数测试时poison；全tests/Ruff/mypy、fresh审查与主代理复验。声明真实密码未验证。

### Task 4: 纯模型与public13公开表示

**Files:** Create `models/{__init__,mlp}.py`, `models/shared_m/{__init__,spec,components,router,system}.py`, `representations/{__init__,tasks,neighborhoods,input}.py`, `tests/{test_mlp,test_shared_m}.py`；Updateprovenance/pyproject必要Torch/NumPy依赖声明，不安装。

**Interfaces:** `BitMLP(spec:MLPSpec).forward(inputs:Tensor)->Tensor[N,1]`；`SharedM(spec:SharedMSpec).forward(bits:Tensor,public:Tensor,*,rounds:int)->Tensor[N,1]`；`public_features(cipher:str)->tuple[float,...]`、`public_backstep(bits:Tensor,rounds:int)->Tensor`纯无秘密，`validate_public`四卡白名单。尺寸规格来自Task1（可spec.py消费/校验但不复制类型），不import训练/数据/ciphers/config文件。只实现single/relation/pair/learned，no unused兼容penalty/dual接口。

- [ ] RED：人工tensor shape/finite/gradient、public13 router不读样本标签，top2稳定排序、stem同初值/zero关系投影、默认参数1,087,509；非default合法widths同实现。模型规格严格拒绝。
- [ ] RED：SKINNY r8固定公开向量与独立手算逆MC/SR/常数一致，r1/r7/r8/r9/r32只用公开round数，模型参数数量不随round变；router卡仍13维。无实际加密或真实轮状态标签。
- [ ] RED后最小提取/改造来源登记，forward/backward仅小tensor不optimizer；GREEN全tests/Ruff/mypy、fresh审查/主代理验收。

### Task 5: 共享评价与严格checkpoint

**Files:** Create `evaluation/{__init__,metrics,checkpoints}.py`, `tests/{test_training_metrics,test_checkpoints}.py`。

**Interfaces:** `binary_metrics(logits:Tensor,labels:Tensor,*,majority:int,probability:float)->dict[str,float|int]`，ACC(logit>0)、稳定BCE，TRAIN probability=(ones+.5)/(N+1)。`evaluate_batches(model,panels,model_spec,meter,batch_size)->dict[str,object]`共享窄前向规格，不能回调任意科学workflow；Task6训练与Task7显式evaluate消费。`encode_checkpoint(model,public_model_spec,task_declarations,selection,baseline,recipe_sha)->bytes`与`load_checkpoint(content:bytes,expected_sha:str)->LoadedModel` CPU/weights_only，先摘要/≤256MiB，白名单/safe model spec/state tensor形状dtype/layout/有限值及公开接线校验；不接受full旧config或秘密字段。加载不能生成标签/训练/采样。

- [ ] RED：阈值/平衡/极端logit/BCE，基线只TRAIN；无效NaN/labels/dtype拒绝，forward_rows预留/完成/失败不退款且单位N不是bit数。
- [ ] RED：checkpoint SHA/schema/超大/unknown模型/extra state/错shape/dtype/sparse/NaN/接线错/任务round mismatch拒绝；secret canary字段不接受；两模型CPU新schema往返，在同tensorlogits一致。
- [ ] RED→实现→全tests/Ruff/mypy→freshreview/controller；不保存optimizer/私有RNG、无旧API兼容。

### Task 6: 共享训练基本操作与两条科学方法

**Files:** Create `training/{__init__,steps,prediction,shared_m,report}.py`, `tests/{test_training_steps,test_prediction_training,test_shared_m_training,test_training_report}.py`。

**Interfaces:** `train_step(model,inputs,labels,public_context,optimizer,meter)->float`只做BCE、clip1、step与唯一update/forward计数；同一个Adam创建/step实现。`BestSelection.consider(epoch:int,dev_bce:float)->bool`finite严格下降，Task5写/加载最佳；方法`run_prediction(recipe,data,meter,store,*,backend_identity)`与`run_shared_m(...)`装配不同数据时序/四任务调度，复用step/批评估/checkpoint/owner。methods成功前claim-source/current-source/recipe绑定与fresh时钟检查，固定公开失败。`render_training_report(saved)->dict[str,bytes]`纯渲染已存指标/学习曲线/方法SVG，不读取binary重新前向。

- [ ] RED：使用人工panel/fake loader的无更新流程测试DEV平局最早、选择/保存/严格加载后才release TEST，选中模型而非末模型评分；shared四任务等权DEV/预建TEST但选模不看TEST。不得为测试反复拟合；两次完整合成闭环才执行实际optimizer（位预测≤2步，shared各1共4），其余状态测试用fake窄step，不能多seed跑科学。
- [ ] RED：optimizer预留/step失败、nonfinite、checkpoint失败/错SHA、资源限额、读取或报告失败、最终计数/部分状态/身份保留，unknown不补成功；TEST bundle只在允许时刻写一次，best引用同步。
- [ ] RED：两方法报告分开ACC/BCE/任务与单位、synthetic标记/无科学结论、HTML/Markdown转义，figure只读保存数值，最多必要图不堆附件。
- [ ] RED→实现→scope工程测试及积分回归/Ruff/mypy；freshreview/controller，记录真实合成更新量。报告IO仍原owner，训练模块不出现私有key。

### Task 7: CLI接通两方法与显式evaluate，最终交付

**Files:** Modify `cli.py`, `demo.py`, `README.md`, `AGENTS.md`、必要边界/CLI测试；Create `tests/test_training_cli.py`。若CLI装配需拆，仅一个职责明确`training/application.py`组装文件，禁止三套cli runner/handler registry。

**Interfaces:** check按三方法识别，零Torch/cipher/data采集和输出。run原参数保持，`--backend demo|simon|ciphers`：prediction demo/simon，shared demo/ciphers，integral demo/simon，非法组合拒绝。`evaluate --checkpoint PATH --checkpoint-sha256 SHA --dataset PATH --dataset-sha256 SHA --policy PATH --root PATH --id ID`显式新run，无backend默认/无cipher。加载受控DATA/MODEL再shared评价，不import训练器/密码/生产随机或访问旧runs。来自模型方法的任务声明与bundle精确匹配，记录来源摘要，写一份评价报告，不复制原binary。

- [ ] RED：隔离子进程check毒化Torch/cipher，evaluate毒化training/data生成/cipher/private RNG仍成功；配置非法组合/占用/异常不泄密；claim先生产import，保存runtime身份/metadata与四单位预算，报告失败不改科学终态。
- [ ] RED：完整CLI合成两个run→其best/test-bundle→显式evaluate（无新训练/标签），指标与已存相同、new run identity且无模型副本；不在pytest与controller重复超过单次闭环允许更新。实际重复验收若运行同闭环，仍按独立工程调用如实累计，不称一次科学实验。
- [ ] GREEN新项目全tests、Ruff --no-cache、strictmypy，干净源码临时copy同闭环与报告HTTP只result；HTTP200不称浏览器/手机验证，精确清临时服务器/目录。
- [ ] fresh整项目最终review，主代理复验实际diff/来源SHA/旧src不改，真实密码/RSS/Windows/安装未验收如实标明。README给已实现命令/配置，未做旧备份切换。无Git发布。

## 计划自检与交接

- 两方法同属训练能力，按数据/模型共用契约先建再按方法验收，不额外各建一套计划与工具。依赖T1→T2→T3→T4→T5→T6→T7，串行避免公共文件冲突。
- Spec10覆盖：模型/表示T4，数据/隐私/T时序T3/6，政策T1，artifact T2，checkpoint/evaluate T5/7，报告T6/7。NPZ安全和最佳替换失败已具名测试；共享类型跨task同名。
- 明确新旧schema无兼容；必要已有测试更新不等于科学参数重写。子项目规则文字只将当前阶段1替换为阶段2/3开发及已批准CPU小tensor/≤2+4合成更新，保留0真实cipher限制。
- 用户已以“是”确认本计划，implement-deep+fresh review连续执行，不重复询问已确定方向。
- 计划完成后不自动备份旧项目或正式切换。

## 执行账本

基线HEAD 7c66151c780bc279b425183cd88fcd3c2dac99f0；新项目31文件排序SHA清单摘要ebb313f82df721bbfd2bd6586f8eea99643e403b0c0462f077c163747d595e90；旧src摘要96a3ef759c8f0ff3523f79cc66b278ebf6fd9d8ba290a99fbf7aa94576893431。

Ruling: 在已批准独立cipher-lab子目录实施，不worktree/commit，沿用单计划账本而不生成SDD任务报告目录——尊重用户无Git写与少过程产物要求；风险是需要显式比较文件范围。
Ruling: 两次/四次是每种完整合成闭环的更新上限，回归多次执行每次都记工程更新，不计科研结论；配置/模型/故障测试不得自行跑大量真实优化。其他真实cipher/GPU/安装权限仍为0。
Ruling: 公开训练估计的forward_rows计训练前向+每epoch DEV+最佳模型最终TRAIN/DEV/TEST；shared各split乘4，evaluate按显式数据样本行另扣，batch内4pair不乘成4倍训练样本。

|任务/共享接口|预检与裁定|
|---|---|
|T1内部及T1→T3/4/6/7类型|frozen规格与方法独立，公开seed不派生key；schema2政策全fixture一次更新|
|T1→T2政策/文件schema|政策与run都是未发布schema2，分别校验，不添加迁移分支|
|T2内部及T2→T3/5/6/7 binaryrefs|同owner预算/原子写，report不decode binary；实际readers再验证|
|T3内部及T3→T5/6/7 bundle/data|x/y与公开task声明，私有key不export；TEST时序按method|
|T4内部及T4→T5/6模型|纯model/public representation；SKINNY公开rounds不能改变router卡维数|
|T5内部及T5→T6/7评价/checkpoint|独立evaluation不importtraining/cipher；严格有限值和身份|
|T6内部及T6→T7方法/报告|固定DEV与TEST顺序，唯一step/ioowner；report纯渲染|
|T7内部|显式backend与evaluate数据源，不假承诺真实实测|

进度：Task1 complete。实现`5c3f5dd9-2733-485`、独立审查`aa0a2144-4394-4a8` spec/quality通过；主代理1043tests/58.90s、Ruff、strictmypy27通过。Task2 complete：实现`cba91b89-5e2c-4b5`，审查`ddae2c3e-dd43-4cc`无P0—P3；主代理1119tests/52.18s、Ruff、mypy28通过。Task3实现`efd68e55-9112-465`，主代理1211tests53.53s/Ruff/mypy36通过；审查`32901f91-b992-4a4`指出P2 ZIP真实解压长度未证明（ZipExtFile信任declared截断），fresh修复`ba111ce7-84f2-4aa`仅bundles/test_bundles。修复主代理1224tests69.48s/Ruff/mypy36过；复审`e24db8ce-d998-44a`原P2 ADDRESSED但新P2合法DEFLATE内部待输出在输入用尽时误拒绝，fresh修复`f4e50c21-4a96-456`先精确复现再最小修复。末次修复`f4e50c21-4a96-456`精确合法流RED→GREEN，复审`25f1d367-7f9f-4eb` PASS，两P2均关闭；主代理末次100定向tests1.40s/Ruff过、旧src摘要不变（此前完整1224，末次全套1225为实现方自验，不混称主验）。Task3 complete；Task4 complete：实现`1bdeceb6-25d3-471`、审查`4e283c4d-0322-45d`无P0—P3；主代理1293tests74.28s/Ruff/strictmypy49通过，默认1087509结构核对。Task5实现`181abebb-4b92-47d`；主代理1380tests133.37s/Ruff/mypy54通过。独立审查`14cf9413-8e24-444`发现P1预检pickle OrderedDict映射dict允许非空mapping memo重复复制，schema前内存膨胀；fresh修复`f041ca09-e77b-46c`仅checkpoint/test，先小规模RED拒绝首个非空reducer。修复`f041ca09-e77b-46c`3小载荷RED→GREEN，复审`f9c5a40e-f69c-4db`P1 ADDRESSED无新增；主代理末次87定向tests5.25s/Ruff/mypy54过，全1383为修复方自验。Task5 complete，Task6实现`24c31e40-698e-4dc`；作者5轮2+4小闭环30steps，主代理全1428tests92.06s/Ruff/mypy63通过，另1轮2(1.390s)+4(.325s)=6steps，累计36工程steps。审查`be356239-bb03-492`P2存储失败精简summary时位预测报告误认shared/丢demo，fresh修复`ce6576b7-7758-4f2`仅report/test。Task6末次修复`ce6576b7-7758-4f2`6fail RED→GREEN，复审`b762af86-5a38-49c`P2 ADDRESSED，主代理11报告tests2.56s/Ruff过，无新增optimizer。Task6 complete，Task7实现`0ad894ee-91e0-435`，作者新增48steps累计84。主代理1458tests106.28s/Ruff/mypy66通过（12steps）；clean源码完整配置prediction1+shared4、复评0步指标相同、result-only12HTTP200/raw404、16来源SHA通过并清理。主代理clean首次poison误屏蔽纯training_config在复评前失败（已1step），修正poison不是改产品。累计工程102 Adamsteps。最终审查`e02f77b3-0a59-46c`发现P1 pickle memo tuplehash放大、P2第二设置快照失败方法未知却默认积分报告；并行不重叠fresh fixes `96ffcd7d-6a67-420`checkpoint/test和`30b8cc8c-403f-4fb`cli/test，均只定向测试无新增optimizer。两fix完成；fresh复审`8420dfd8-3582-490`进行中。主代理完整修复后suite为6fail/1478pass93.83s：旧test_cli未知method预期默认积分退出4与新正确exit6冲突（真实工程额外12steps，累计114）；fresh implement `c4508bab-028c-454`只同步必要tests预期保留knownmethod/身份/终态断言，不改产品。末次产品复审`8420dfd8-3582-490`两缺陷ADDRESSED无新增；测试同步`c4508bab-028c-454`和独立复核`275883b9-ccf7-45c`PASS。Task7 complete，阶段2/3状态PHASE2_3_ENGINEERING_COMPLETE。
Ruling: 最初设置写入失败且method从未提交时，报告明确失败REPORT_METHOD_UNKNOWN（exit6）而不是猜积分/训练方法；已保存FAILED字节不变。这是修复方法误标所需的未知报告行为变化。
Ruling: evaluate checkpoint未含来源backend/synthetic，保持UNVERIFIED/synthetic=null，不从文件路径猜身份；独立评价application归评价模块，不能依赖训练组装。不新增checkpoint兼容或伪来源保证。

### 最终主代理验收

- 末次完整全新项目：OMP/MKL/OPENBLAS单线程，pytest **1484 passed in103.93s**；Ruff PASS；strictmypy **66files PASS**。此前一次240s工具超时，没有残留pytest进程，未将超时当通过；单线程480s工具上限重跑通过，非训练预算放宽。
- 末次干净完整源码副本：默认prediction配置1step/3.220s、默认shared-M4steps/3.513s（含启动）；各自显式evaluate与原保存TEST指标完全一致，0更新、无新模型副本。HTTP只服务临时result，12次200、data/model404；服务/副本/output精确清理，无浏览器验证主张。
- 本次可明确计数工程累计131 Adamsteps（84作者历史+12首主控全套+1首clean误poison前+5修正clean+12修复后有失败全套+12末次通过全套+5末次clean）；超时那轮未到闭环打印，未据此推定训练完成。所有已完成小闭环遵守每次2/4上限，未做科学实验；不是总共只6步的说法。
- 旧src SHA仍96a3ef759c8f0ff3523f79cc66b278ebf6fd9d8ba290a99fbf7aa94576893431。16来源SHA逐项复验通过。新项目74文件，无data/models/result演示输出和cache残留；未Git写/发布/旧项目切换。
- 已确认审查发现全部修复，未留已知缺陷。真实cipher/KAT、生产RSS、GPU、Windows、依赖安装/解析、新机器未验证。CPU保存面板复评不能证明密码预测有效或跨key泛化。
- Python私有pickle._Unpickler dispatch用于严格项目checkpoint预检；Python升级需重验。SHA/已存身份不构成外部真实性证明，存储可信无并发root不是OS沙箱。

Ruling: 最佳加载为write_model成功提交的同一immutable encoded bytes+SHA严格load，并非磁盘重读；可信无并发root下成功写契约成立，Task6无新增diskreread保证。不把报告中“重载”说成“从磁盘重读”。
Ruling: DatasetBundle给prediction独立公开target声明（rounds/target_bit/label_mode），不把绝对位目标伪装成差分TaskSpec；evaluate因此可拒绝错位/错轮/错null来源。metadata在bundles模块归属并由后续checkpoint消费，不另造可选配置平台。
Ruling: Task1 shared-M示例用了非默认合法宽度，Task4实现时将唯一shared示例对齐已承诺默认96/96/GN8/kernels3,5,7,9,11/head64/router16，并验证1087509；当前规格支持非default不视作算法变化，不新增多份示例。
