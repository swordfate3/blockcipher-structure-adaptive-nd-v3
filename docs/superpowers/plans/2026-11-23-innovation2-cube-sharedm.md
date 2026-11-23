# 创新2 Cube / Shared-M GPU 对照 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. TDD、任务后只读审查及整体验收；代理不派子代理、不提交或发布。主控完成唯一限定发布。

**Goal:** 在已批准的远端GPU1上，用同一份固定未知密钥的7轮SIMON数据比较MLP和Shared-M位预测版。

**Architecture:** `direct_block/cube_compare`拥有本方法的数据/模型/数值适配与统计；父owner提取具名fit/reload共用件，不复制runner。旧配方行为不变。发布为c7320原样core加限定新owner闭包；主工作区main/index不变。

**Tech Stack:** 现Python3.12本地、Windows Python3.13.3/Torch2.7.1+cu118/NumPy2.4.0；现Meter/RunStore、pytest/Ruff/mypy。无安装。

**Spec:** `docs/superpowers/specs/2026-11-23-innovation2-cube-sharedm-design.md`（APPROVED）

## Global Constraints

- 仅本轮：7轮、一个私有key、两fit、1048576/65536/65536 pairs、5epoch/batch256；active LSB0 `(0,21,42,60)`，完整128输出/训练、8位DEV选模和TEST主ACC。
- 科学counts=2359296 blocks/40960 updates/11272192 forward pairs/null0；唯一预检额外18 blocks/2updates/32forward，总wall7200/CPU28800秒，RSS8GiB/输出512MiB。首次失败即停、retry0。
- 开发仅人工oracle/假runtime/合成张量结构检查，禁止真实cipher/秘密熵/数值网络前后向/真实optimizer/RSS/GPU/SSH/安装/Git写入。主控独立执行已批准真实路径。
- 修改限定于direct_block及本轮configs/一个启动脚本/本设计计划。core和创新1原实现不改；不读私有科学产物或旧权重。禁止自动evaluate。
- 在c7320原样core＋逐文件允许清单的隔离源码副本验证；不import主树或未纳入来源指纹的兄弟插件。代理无提交/发布权限。
- 远端仅G:\lxy下新目录，C:\Python313\python.exe沿用user-site、-B；UUID锁物理GPU1，逻辑cuda:0，无CPU/换卡fallback，不启停其他任务。

## Review Focus

1. 16变体/8互补对与跨split整族互斥；同key跨全split但不进入模型，实际provider身份与recipe逐项一致。
2. 明文canonical直接构造，零关系通道；128输出不裁标签，四槽只打包不均值，原分类器行为不改变。
3. 两臂同数据同epoch顺序；全128 BCE训练/8列DEV最早最低；第二best加载失败绝不构造TEST，第二TEST失败不发布第一臂终分。
4. Windows RSS不是默认Linux procfs，CUDA UUID映射/异步完成/16MiB模型上限/固定整数buffer/真实退出与预算均正确；未知不填0。
5. 发布闭包在原样base完整，既有CPU计量与选模不变；只读saved报告不前向、不生成标签，不将524288位当独立试验数。

## Task 1: 数据、模型、指标和具名fit共用件

**状态：COMPLETE。** implement-deep `8761b2bc`，独立审查 `0a286986` 经fix1放行；后强制Ruff修6项漏查告警，无语义改动。主控fresh受改回归403通过/18排除；最终owner93项归Task2完整闭环。

**Files**（以实现前边界说明为准，新增不制造通用框架）：
- Create `plugins/cipher_lab_plugins/direct_block/fit.py`：无旧mode/loader依赖的epoch fit/reload共用件。
- Modify `plugins/cipher_lab_plugins/direct_block/training.py`：原Trial委托共用件，保留原方法接口与默认科学行为。
- Modify `plugins/cipher_lab_plugins/direct_block/model.py`：最小显式设备支持，原默认CPU校验不变；模型只依赖公开规格，不读实验配置文件。
- Create `plugins/cipher_lab_plugins/direct_block/cube_compare/{__init__,config,data,model,metrics,checkpoint,engine,resources}.py`。
- Create owner tests `plugins/cipher_lab_plugins/direct_block/cube_compare/tests/{conftest,test_data,test_model,test_metrics,test_fit,test_checkpoint,test_resources}.py`。

**Interfaces:**
- `config.recipe() -> dict[str,object]`, `planned_counts() -> dict[str,int]`，只stdlib及public规格，不导入科学依赖。
- `data.CubePanels`接收显式人工oracle或由生产工厂创建的单key标签闭包；按TRAIN/DEV/TEST顺序交付compact `Panel`（[pairs,2] uint64明文/密文＋族ID），TEST标签有明确延迟调用，`Panel.batch(indices)`按批MSB0解码。
- `model.CubeSharedM.forward(inputs: Tensor) -> Tensor`：`[N,128]→[N,128]`，N为4倍数、只public13 SIMON，canonical＋zero relation；dense128 head，1876179参数（实际实例核对）。MLP128为49536。
- `fit`通过显式settings/engine/panel/budget/store完成epoch fit与SHA读回；不导入cube/旧data。旧Trial适配保留1epoch、DEV每128update、32batch；新方法5epoch每epochDEV，128训练/8列选模。
- `metrics`保存逐位及整族配对正确数，主差值/SE/97.5%区间；TRAIN多数不读DEV/TEST。
- `engine`提供train_forward/update/predict/dump/restore接口，数值仅正式工厂调用；CPU state严格加载、fixed buffer逐值，16MiB上限；`resources`读Windows working set/设备身份，无秘密或科学决策。

- [ ] RED：先人工测试16唯一变体/每变体恰一次、两端顺序、mask、族跨split去重、全split相同oracle、全128标签/8列映射、TEST不提前、故障fail closed。
- [ ] RED：共用fit覆盖epoch顺序两臂一致、全128训练与8列DEV的不同、最早同分、checkpoint多次replace/sha/16MiB/第二重载失败门控；旧owner相关回归先运行形成基线。
- [ ] RED：用结构检查和手写tensor替身检验三stem/top2/suffix顺序、zero relation不调用旧representation、四槽逐对独立、输出128/参数形状及数值输入拒绝；真实Conv/Linear前后向、Adam及cipher均由guard禁止。
- [ ] GREEN：实现最小边界与指标；统计手造8192族不必要，使用小人工族表验证协方差/基线/区间，禁止二项独立位sigma。
- [ ] GREEN：运行owner人工测试和旧direct_block/相关core安全回归；Ruff与strict mypy。不跑science标记，不以未运行全项目测试冒称全绿。
- [ ] 输出报告、逐文件允许清单、实际RED/GREEN证据与diff；无Git提交。独立审查Task1，修复复审后才Task2。

## Task 2: 统一CLI、保存值报告、唯一预检及轻量启动入口

**状态：COMPLETE。** implement-deep `9832b36f`，独立任务审查 `9571de0c`、整体审查 `f5f448f3` 经失败路径修复及测试重名修复放行。主控最终发布副本：新owner＋布局100通过/6.87s；默认全safe3702通过/49排除/317.49s，既存frontier重复ZIP警告1；forced Ruff及strict mypy26文件通过。原样base core不改，19生产＋11人工测试＋本设计计划限定发布。

**Files:**
- Create `direct_block/cube_compare/{training,cli,report,preflight}.py`（相对plugins/cipher_lab_plugins）。
- Create `direct_block/cube_compare/tests/{test_training,test_cli,test_report,test_preflight}.py`。
- Create `configs/direct-block-cube-sharedm-simon7.json`。
- Create `scripts/start-cube-sharedm.ps1`：唯一必要启动脚本，仅调用已实现CLI、basic日志/原生退出码；无科学实现/重试/换卡/远程调度。
- 视实际闭包可新增具名source模块或限定复用父source，不复制其它plugin。

**Interfaces:**
- `cli.main(argv:list[str]) -> int`支持core转发的check/run。check使用固定recipe/policy，stdlib零机器/科学import/零claim。run复用RunStore/Meter，具名slots `mlp-best`/`sharedm-best`，返回失败/停止原因不泄露private exception。
- `training.run(...)`装配同一面板两fit，双reload后单次TEST生成并完整两臂评分；所有curve/计量/模型身份保存在run记录，不另写receipt。
- `preflight`显式单次入口：固定18公开密码blocks、两模型2update/32forward人工数据门禁，checkpoint严格保存/加载，失败不启动科学。不得借预检额外微调。
- `report`纯saved值HTML＋必要图解/结果图，不额外科学运行；失败显示实际消耗与无终分。
- 启动脚本从预检开始设7200秒绝对UTC期限，科学沿用剩余wall/CPU。设置UUID在torch import前，CWD/TEMP/TMP/cache为本次G:\lxy新根，使用既有user-site。记录直接CLI退出，不用stdout替代进程退出。

- [ ] RED：CLI零science import、严格固定recipe、rounds/key/width真实身份、占用路径、早期故障无采样、slots replace、budget精确、source变化拒绝。
- [ ] RED：人工完整双臂epoch→DEV→best重载→TEST→保存报告闭环；第二重载/第二评分/存储失败不得泄露partial TEST。
- [ ] RED：预检18/2/32预算与只固定公开key，超时/失败不science、UUID冲突不fallback、未知RSS不填0；Windows quoting及user-site不被-I/NOUSERSITE屏蔽。
- [ ] GREEN：实现CLI/报告/门禁/单启动入口，生成固定小config。人工演练不触真实cipher/NN/RSS/GPU。
- [ ] GREEN：在原样c7320的发布最小闭包源码副本运行`check`和假runtime CLI完整闭环，owner及相关回归、Ruff/strict mypy；核全部允许文件且主树无关SHA不变。
- [ ] 输出Task2报告/最终允许清单；任务审查后全范围独立审查，主控fresh复验。

## Task 3: 主控限定发布、一次远端执行与结果交付

**状态：RELEASING。** 用户许可已确认，不再追加训练/预检或重试。真实消费仍为0，待首次现场门禁。

**Files:** 只修改本计划/设计状态；发布采用审核允许清单，主树main和真实index不变。

- [ ] 冻结最终源码/recipe身份，确认原样c7320 baseline+allowlist完全通过。主控按唯一许可用隔离index/commit-tree形成release、push并核远端ref；不stage主树、不带数据/权重/日志/环境/无关删除。
- [ ] 远端新根精确fetch/checkout，不改正在训练的旧repo；源SHA/现依赖/物理GPU1当前占用一次检查通过，指定卡≥16GiB free，冲突停止，不抢卡。
- [ ] 用一个启动入口执行唯一预检→唯一科学批次；总wall7200/CPU28800、固定count cap，首次失败或未知即停，不自动重跑。
- [ ] 只读观察同任务进度和实际exit；未完成不报告科学阴性，保存SUCCEEDED不冒充进程成功。
- [ ] SCP/SFTP取回本次data/models/result/必要日志，核SHA/退出/计数，不load权重/不forward/不再生成标签或evaluate。
- [ ] 结果图只用保存值，报告全两臂/基线、族聚类区间与单key/seed/减轮限制；默认仅服务result本地HTTP并核响应。周期关闭，不复用剩余预算。
