# CCDS Cipher Lab Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立独立、无旧兼容的积分经验实验闭环，使改轮数/活跃位/组数只改设置，一次运行只保存一套记录并提供一份报告。

**Architecture:** 精简CCDS骨架。纯密码/数学、私有数据采集、固定科学方法、执行限制和产物保存各自负责明确职责；不做workflow/registry。首期无训练/评价/认证占位功能，位预测与创新1只记录后续路线。

**Tech Stack:** Python 3.12、NumPy（生产适配器）、标准库CLI/JSON/HTML/SVG；现有pytest/Ruff/mypy，不安装依赖。

**Spec:** `docs/superpowers/specs/2026-11-06-ccds-rebuild-design.md`（已批准）。

## Global Constraints

- 所有产品文件仅在`/workspace/cipher-lab/`新建；旧项目只读，主代理仅更新本设计/计划状态。不commit/push/stash/worktree/新建Git仓库。
- 不兼容旧配置、checkpoint、import、命令或run schema；禁止import旧项目/备份/历史产物。
- 本轮阶段1；创新2位预测与创新1后续单独启动，必须在README保留三阶段路线，禁止当作已交付。
- 无真实密码/标签/训练/随机科学面板执行，无GPU/SSH/安装/求解器/备份复制/覆盖/批量清理；运行测试只用人工数据和假oracle/假随机源。
- TDD → implement-deep实现 → fresh只读审查 → 主代理范围内验证，每任务一门；代理不再委派。审查留会话，不新增task报告目录。
- 只用`/workspace/.venv/bin/python`及已有工具，`PYTHONDONTWRITEBYTECODE=1`，不污染旧测试缓存；测试临时目录自行清理。
- 旧项目bk备份、根目录替换另有核验门槛，本计划不实施；新目录通过不代表旧项目已备份。
- 共享进度只更新本计划checkbox；临时RED/GREEN日志随会话，不造第二套永久验收文档。

## Review Focus

1. 目录占用/符号链接/路径穿越/失败重跑不得覆盖他人资料——Task 4输出申领与CLI测试。
2. 小样本必有经验核，不能报证明或显著性；确认失败不救援挑选——Task 2/5测试与报告限定。
3. 失败调用必须先扣预算，秘密及私有异常不得落盘——Task 3/4伪oracle抛错/故障注入。
4. 科学终态成功后报告失败，不得改成科学失败或改写终态——Task 4/5故障测试。
5. check不采集/加密/建run；演示与生产明确区分，演示成功不宣称真实可用——Task 1/6隔离CLI测试。

## 科学方法与接口约定

方法固定`integral-empirical-v1`。Cell轮数1..44，活跃位2..16且全在同一32位半字，MSB0。2080项固定单位64+不同位二次2016；feature i对应整数位`1 << (2079-i)`。每组完整cube内固定同key/背景，不同组域分离且私有key不重复；随机函数null逐点均匀64位输出，不用均匀2080bit假行冒充null。

每cell：D采集→求K_D→S采集并求K_DS→冻结完整规范基→C验证全部冻结基（失败whole-cell，不删基）→null使用独立D/S/C同样方法；null确认全通过且空间非空则alarm，停止新增cell。真实K_DS空则C记NOT_APPLICABLE、不伪造通过，null仍处理。任意科学调用失败立即结束run并保留已完成进度。real/null每阶段请求组数记录，空空间跳过确认的实际消耗与计划分开。首期按cell清单固定顺序，禁止自适应几何选择；报告至多显示4个代表，但载荷保存完整基。

运行前工程控制是无真实密码的数学/流程自检：人工预定关系应通过、单行翻转应失败、null样本构造边界测试；只证明软件接线，不称真实密码正控。实现者不能把工程控制替代未来科学周期的正/负控制。

共享核心接口由各任务拥有者实现；后任务如需调整必须先回报，不另造平行类型。

### Task 1: 严格设置与可检查的独立项目

**Files:** Create `cipher-lab/{AGENTS.md,README.md,pyproject.toml,.gitignore,configs/integral-example.json}`、`src/cipher_lab/{__init__.py,config.py,cli.py}`、`tests/test_config.py`。配置示例可check，不授权run。

**Interfaces:** `Cell(rounds:int, active:tuple[int,...])`；`StageSizes(discovery:int, screen:int, confirm:int)`；`Recipe(schema:int, method:str, cells:tuple[Cell,...], groups:StageSizes, chunk_points:int)`；`Policy(wall_seconds:float,cpu_seconds:float,deadline_utc:str,real_blocks:int,null_words:int,rss_stop_bytes:int,output_bytes:int)`；`load_recipe(path:Path)->Recipe`、`load_policy(path:Path)->Policy`、`to_dict()`仅公开字段；`cli.main(argv:Sequence[str]|None=None)->int`。具体JSON字段即上述字段，deadline需UTC带时区，预算非负/时间正有限，CPUonly；chunk1..65536，groups各1..1000000，cells1..64且无重复；布尔不能冒充整数，拒绝未知/缺失/重复字段、NaN、超64KiB输入。Recipe与Policy不绑定历史时间/研究轮数。

- [ ] RED：配置参数化测试rounds1/11/44、H2/L10/H16；非法跨半字/重复位/非法数/字段/大文件拒绝。`check --config <file> --policy <file>`输出预算估计与参数，不建目录，毒化数据/密码import仍通过。
- [ ] 执行RED：`cd /workspace/cipher-lab && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /workspace/.venv/bin/python -m pytest -p no:cacheprovider tests/test_config.py`；记录预期缺模块失败而非依赖故障。
- [ ] 实现类型、严格loader、仅check入口及最小包元数据；不注册未实现run/evaluate。配置示例一份，执行策略测试fixture不变永久文件。
- [ ] GREEN同命令；Ruff/config strict mypy；fresh范围审查与主代理复验。现有Python版本记录，不生成依赖锁文件或假称可重复安装已验证。

### Task 2: 2080观测及宽GF(2)纯数学

**Files:** Create `src/cipher_lab/integral/{__init__.py,observables.py,linear.py}`、`tests/test_integral_math.py`；首次提取维护`docs/provenance.json`一份，后任务复用。

**Interfaces:** `aggregate_words(words:Iterable[int])->int`（逐点求项再XOR，不能用位XOR乘积代替）；`canonical_basis(rows:Iterable[int],width:int=2080)->tuple[int,...]`；`kernel(rows:Iterable[int],width:int=2080)->tuple[int,...]`；`survivor_basis(discovery:Sequence[int],screen:Sequence[int])->tuple[int,...]`；`confirms(basis:Sequence[int],rows:Sequence[int])->bool`。纯函数无config/cipher/I/O/RNG。

- [ ] RED：两点二次项反例、64单位/2016pairs顺序、2080边界与非法word；小宽度穷举独立真值表比对kernel/规范基/交集，重复行/零行/生成器；D=0约束维数2080、1约束至少2079，小样本不是科学发现；C失败原basis不变。
- [ ] 运行`tests/test_integral_math.py`确认RED。
- [ ] 实现从现有源码只读参考的数学能力；若提取任何旧算法记录原路径、实际SHA、改动，不引入旧import/2080以下降维捷径。
- [ ] GREEN数学范围pytest、Ruff、mypy；独立审查验证位序/二次项/核维度，主代理复验。

### Task 3: 私有cube采集与执行账本

**Files:** Create `src/cipher_lab/{execution.py,data/__init__.py,data/cubes.py,ciphers/__init__.py,ciphers/simon.py}`、`tests/{test_execution.py,test_collection.py}`；Update唯一`docs/provenance.json`。

**Interfaces:** `Clocks(wall,cpu,utc)`可注入；`Meter(policy:Policy,clocks:Clocks,rss_bytes:Callable[[],int])`提供`check()->None`、`reserve(kind:Literal['real_blocks','null_words'],amount:int)->None`、`complete(kind,amount)->None`、`snapshot()->dict[str,object]`；预留先于外部调用，失败不退回。`Stopped(reason:str)`只公开固定reason。

`Oracle = Callable[[tuple[int,...],tuple[int,...],int],Sequence[int]]`只在data层使用（输入points/私有128bitkey字/rounds），`RandomSource`私有随机字节接口由实现者明确；生产用secrets，测试用有限确定fixture。`CubeCollector(meter,oracle,private_random,chunk_points)`，`row(cell:Cell,domain:tuple[str,str,int])->int`返回仅公开聚合2080bit行，阶段/组身份禁止复用；`counts()->dict[str,int]`保守attempted/completed block与group，key/background不外泄。不声称全局unique plaintext；不同key同plaintext合法。

生产SIMON适配器`encrypt_many(points:tuple[int,...],key_words:tuple[int,...],rounds:int)->tuple[int,...]`，密码只依赖标准库/NumPy；未经实际KAT调用标明未验证。不为封装而暴露实际状态。生产RSS监测采用平台明确实现，平台不支持就返回固定UNSUPPORTED并停止真实后端，禁止默认为0假装监测有效；demo测试注入固定RSS。所有内存检查仍是协作式，不称硬隔离。

- [ ] RED：用伪oracle验证完整cube构造、chunk1/3/65536边界、固定背景、组内同key、组间新私有key；null伪随机字逐点聚合与数学真值相等；控制捕获调用确认没有真实cipher访问。
- [ ] RED：假钟截止/wall/CPU/注入RSS超限，超预算零新oracle调用，oracle抛含秘密异常时attempt保留/completed不增加，非法响应长度/值拒绝，私有随机源耗尽固定公开错误。
- [ ] 执行两测试文件确认失败后实现。科学预算仅data owner单点计数，runner不重新维护另一套ledger；Meter是通用额度预留而非混淆组/块统计。
- [ ] GREEN范围pytest/Ruff/mypy，断言tests不调用生产encrypt_many；审查源码密码接线，但不声称实际密码正确。主代理复验与来源记录核对。

### Task 4: 单一运行记录与安全产物保存

**Files:** Create `src/cipher_lab/artifacts.py`、`tests/test_artifacts.py`。

**Interfaces:** `RunStore.claim(root:Path,run_id:str,*,output_bytes:int,needs_models:bool=False)->RunStore`；`write_snapshot(value:Mapping[str,object])->None`；`append_event(value:Mapping[str,object])->None`；`write_payload(name:str,value:object)->str`返回受约束相对引用；`finalize(value:Mapping[str,object])->None`终态后拒改；`source_identity()->dict[str,str]`仅新包py；`read_saved(root:Path,run_id:str)->dict[str,object]`校验schema/来源引用/摘要类型。根root由显式CLI参数提供，所有内部路径都限制root下，不用cwd隐式寻找旧产物。

- [ ] RED：run_id拒绝点/斜线/绝对路径，data/result占用与符号链接拒绝；跨目录claim中途失败不删除旧内容，只清本次新建空目录；积分不创建models目录。
- [ ] RED：原子snapshot损坏不覆盖既有完整文件；终态不可再写；事件不重复完整snapshot；payload路径与symlink逃逸拒绝；保存失败保留已经落盘的数据。
- [ ] RED：输出字节限制在写入前检查，临时文件纳入限制，失败尽力写固定公开终态（实际存储失败无法保证）；秘密异常文本不进入JSON/HTML。不可用通用黑名单正则代替数据边界。
- [ ] 运行RED后实现唯一JSON/事件/SHA保存逻辑。run.json含设置/策略/源码身份/状态/计数/摘要/引用，终态后冻结；报告不得重写它。
- [ ] GREEN/Ruff/mypy，fresh审查与主代理复验。文件写入语义由新项目定义，不复刻旧兼容行为。

### Task 5: 固定经验流程与保存结果报告

**Files:** Create `src/cipher_lab/integral/{method.py,report.py}`、`tests/{test_integral_method.py,test_report.py}`。

**Interfaces:** `run_integral(recipe:Recipe,collector:CubeCollector,meter:Meter,store:RunStore)->dict[str,object]`；依赖可替换为符合相同窄接口的fake，不用动态registry。`render_report(saved:Mapping[str,object])->dict[str,bytes]`纯渲染返回report.md/index.html/source.json三份内容；调用方保留已finalize的原store并调用`write_report_files`，所有数据来源于run身份，不计算新科学指标。首期不提供跨进程报告owner重开/重绘命令。

- [ ] RED：人工小行验证固定D→S→freeze→C；完整基持久化再确认，故意一条确认行破坏关系时whole-cell fail不救援；real/null空空间均NOT_APPLICABLE，不称通过；随机函数null存活确认alarm阻止下一cell。
- [ ] RED：冻结前后配方/源码身份变化停止；阶段/组请求无复用，单一bank账本，attempted/completed实际计数；失败不继续阶段，冻结载荷留存；假钟终态时限检查。
- [ ] RED：报告至多4代表但结果保留完整空间；经验/工程/未真实验证醒目标识，不输出证明或p值；方法SVG+依据已保存维数的结果SVG，人工结果明确合成、不画假科学成绩；HTML escape所有设置/错误文字，source指向run SHA不提供data下载链接。
- [ ] RED：生成报告故障在科学终态之后返回report_failed，不篡改run科学状态；再渲染前核原始SHA不改数据；毒化cipher/model进口渲染仍通过。
- [ ] 运行RED后实现方法及薄报告；逐cell释放点数据/无跨cell逐点缓存。工程控制仅无cipher自检，未来科研控制未获准不得隐式执行。
- [ ] GREEN/Ruff/mypy、独立审查阶段语义/报告边界，主代理复验。

### Task 6: 真实入口、干净源码假数据闭环与交付

**Files:** Update `cli.py`、`README.md`、`pyproject.toml`；Create `src/cipher_lab/{__main__.py,demo.py}`、`tests/{test_cli.py,test_boundaries.py}`。`demo.py`明确工程后端，用人工确定公开函数、无密码；不冒充SIMON实验。

**Interfaces:** `check --config PATH --policy PATH`与`run --config PATH --policy PATH --root PATH --id ID --backend {demo,simon}`。`run`必须显式backend无默认；demo只写合成工程身份，simon为真实适配器入口，本次验证不得调用。科学许可仍由操作纪律管理，不让设置文件伪装授权令牌；README准确区分已实现/已实测。`evaluate`尚不注册。

- [ ] RED：check无采集/密码import/磁盘产物；demo成功输出可读报告路径；simon dry check仅配置校验；run占用/错误/输出失败非零，科学成功报告失败独立返回，缺终态UNKNOWN；参数穿越/非法backend拒绝。
- [ ] RED：AST边界测试禁止旧包与data目录读入算法，ciphers无CLI/训练，数学无I/O，report无密码/数学/训练；打包只发现新包；README不承诺未实现位预测/创新1/evaluate、KAT/Windows/安装。
- [ ] 实现分派、显式demo后端和文档；demo不调用私有生产随机/实际cipher，预算计数注明模拟。
- [ ] 运行所有新项目测试（不收集旧tests），Ruff/strict mypy，使用本任务新建临时目录复制新源码并执行check→demo run→保存结果render；不安装/连接模型、不修改旧环境。
- [ ] 对临时demo报告启动仅服务其result/的回环HTTP，确认200与内容，停止本任务服务并清本任务临时目录；永久展示如用户需要另指定结果位置，不堆永久验收副本。
- [ ] fresh整项目code-review；主代理核diff/实际依赖方向/无越界修改，再执行同样fresh命令。记录实际测试数和未验证边界在本计划末尾，README使用说明简短不塞日志。

## 新子项目AGENTS.md全文（用户已随本计划确认）

```markdown
# Cipher Lab

仅适用于本目录。新项目采用精简CCDS结构，不兼容旧项目；旧项目保持原状。
- 入口为已实现的cipher-lab check/run；evaluate随训练功能以后接入。不适用父旧项目的train、runs/、旧来源文件和旧报告绝对路径契约。
- 实验设置只在configs；原始记录data/<id>，权重models/<id>，报告result/<id>。来源提取记录只用docs/provenance.json。无runs/outputs/validation平行产物。
- 密码/数据/模型/训练/积分数学/评价/执行限制/保存各守职责，不import旧项目或历史产物，不造workflow/registry/巨大utils。秘密留在数据生成侧，不进入模型、符号层或公开产物。
- 每轮配方改参数，不复制执行器；新增真正算法才新增算法代码。每种永久文件须有消费者与保留规则，测试临时产物自行清理。
- 保留父规则的明确权限、代理分工、TDD、独立审查与fresh验证纪律。当前仅阶段1开发和无真实密码工程测试；真实加密/训练/GPU/SSH/安装/Git/清理均另行授权。
- 仅服务本项目result/提供报告；不暴露data/models，不重新采样/计算指标来美化报告。依赖/Windows/密码实测未完成须直说。
```

## 执行门槛与计划自检

- 用户已在本会话以“是”确认本计划和局部规则，按Task1→2→3→4→5→6连续执行，不重复确认。
- 设计范围全覆盖：Task1设置/隔离/规则，2纯数学，3采集/计数/真实适配器未实测限制，4唯一产物，5方法/图报告，6CLI/独立闭环。旧备份切换非本期。
- 自检：同一Cell/Recipe/Policy跨任务；聚合固定2080；无模型训练/认证占位；Review Focus五项均落入明确测试；总6门、无额外永久审查文档。
## 执行账本

**最终状态：PHASE1_ENGINEERING_COMPLETE；真实密码/生产RSS/安装/Windows/科学验证未执行。**

- 六任务均完成（下方原任务步骤checkbox保留原计划，不伪造逐步时间戳）；独立审查与fix见任务记录。
- 主代理最终全测试：803 passed/68.27s；Ruff --no-cache通过；strict mypy25files通过。首次110s协作超时仅到98%，未声称通过，确认无残留pytest后以240s命令完整重跑。
- 修复后干净源码副本fresh闭环：check0且无run，tiny demo ALARM5、12模拟blocks/12nullwords、environment/source SHA正确、仅result HTTP200；production cipher/RNG/RSS毒化禁止。临时服务/目录精确清理，无永久demo副本。
- 3个提取来源SHA已实核，31个新项目文件；旧src排序SHA仍96a3ef759c8f0ff3523f79cc66b278ebf6fd9d8ba290a99fbf7aa94576893431。
- 未Git提交/发布，未备份或切换旧项目；新目录原位交付，无旧兼容层。位预测/创新1仍后续独立阶段。


基线：main@7c66151c780bc279b425183cd88fcd3c2dac99f0；旧src全部py排序SHA清单的SHA为96a3ef759c8f0ff3523f79cc66b278ebf6fd9d8ba290a99fbf7aa94576893431。启动前新目录不存在，Python3.12.15及pytest/Ruff/mypy/NumPy可用。

Ruling: 按用户确认的新子目录隔离实施，不创建Git worktree/commit；只用本计划作恢复账本、不运行SDD工作目录生成器——用户明确要求避免新增过程产物，独立review从新文件/前后字节快照核对而非不存在的提交差异。成本是主代理需要显式核对路径范围。
Ruling: 不跑旧仓全pytest；只跑新项目测试——旧测试可能生成真实科学标签，超出本轮权限。

|共享任务/接口|预检结果|
|---|---|
|1↔2/3/5/6，Cell/Recipe/Policy/CLI|统一不可变类型，后续只消费；Task6才加run入口|
|2↔3/5，2080观测/GF2|严格统一MSB0和整数位序，data只聚合不选关系|
|3↔4/5/6，Meter与调用账本|data单点计数、store独立输出字节限额；不强行合并科学单位|
|4↔5/6，RunStore/报告引用|终态冻结后渲染；报告失败与科学状态分开|
|1↔6，pyproject/README|Task1只check，Task6补run；不增加未实现evaluate|
|2↔3，docs/provenance.json|串行更新同一来源文件，不再创建多份provenance|
|Task1内部|配置边界明确，严格JSON解析；check零科学侧效应|
|Task2内部|数学真值表独立验证，不运行真实密码|
|Task3内部|真实适配器仅实现静态审查，fake测试不得调用cipher|
|Task4内部|claim/write/finalize对失败与越界均有测试|
|Task5内部|固定经验方法不等于证明，C失败不可挑幸存者|
|Task6内部|demo显式合成，报告HTTP与安装/Windows/密码实测分开|

进度：Task1 complete。实现`42c0b43e-96d1-4fb`，独立审查`bbf69a0f-5a58-442` spec/quality通过；主代理实测159 passed/0.48s、Ruff通过、strict mypy4文件通过；旧src排序SHA未变。Task2 complete：implement-deep `dcde076b-d81a-4ca`，审查`afb2e6a5-61c0-428` spec/quality通过；主代理503 tests/5.52s、Ruff、mypy8files通过，2来源SHA实核通过。Ruling: survivor_basis第一参数是已发现关系基，Task5必须kernel(D)→survivor_basis(K_D,S)，不得直接传D行。Task3 complete：实现`13d1bf15-513a-48c`、审查`dfa1b1e7-e70a-4e0` spec/quality通过，主代理548tests/5.46s、Ruff、mypy15files通过；生产cipher和RSS未运行。Task4首轮review `90d53e34-e2aa-475`指出2个P2待修：source walk扫描错误静默遗漏、Task5报告签名与唯一owner衔接。主代理首轮624tests/6.40s、Ruff、mypy17files通过但不因此过审。Ruling: 改Task5为纯渲染返回三份bytes，由原store受预算写，避免二套IO；跨进程重绘不纳入首期。Task4 complete after fix round1/5：修复`5583bc80-72c2-4b8`，复审`ba6d659c-84c4-400`两项ADDRESSED无新阻断；主代理83产物tests/0.80s、Ruff、mypy17通过（累计631由代理报告，最终会全测）。Task5首版主代理659tests/36.39s、Ruff、mypy21通过；review `8ffade01-0d26-455`指出claim→method来源身份gap及Markdown耗时字段转义P2，进入fix round1/5。Ruling: RunStore新增source_snapshot()->dict[str,str]公开只读副本，method沿用claim身份而非自己重冻；只为明确跨模块契约调整Task4文件/测试，不读私有字段。Task5 complete fixround1/5：`543dd159-f160-49b`修复，`1e45319f-b3f9-41a`两项ADDRESSED无新阻断；主代理相关120tests/30.18s、Ruff、mypy21通过。累计668待最终全测。Task6首版主代理738tests/48.42s、Ruff、mypy25通过；review `a78e2222-3e80-4d0`发现execution导入异常边界外、storage失败清掉backend元数据两个P2，fixround1/5已由`bfb01699-d00e-426`修复、`52fe0090-0a60-470`复审两项ADDRESSED；主代理相关144tests/14.35s、Ruff、mypy25通过。Task6 complete，整项目审查`1d769e1f-d8de-488`发现.gitignore误忽略src/data(P1)、run缺环境身份(P2)，唯一最终修复波由`b5fd9b49-4586-466`完成，`77ef1d4e-2580-42b`两项ADDRESSED无新缺陷，最终审查门禁通过。Ruling: 根锚定/data /models /result；run schema1尚未发布，直接纳入公开environment小映射(Python实现/版本/OS/架构)，不做旧schema兼容，不采集路径/主机名/env。主代理最终762tests/46.78s、Ruff、mypy25通过；干净源码tiny demo D/S/C各1、L2/r3/chunk3得到ALARM exit5、12模拟blocks/12nullwords、SHA一致，result-only HTTP200与raw-data404，临时服务/目录已清。第一次控制器误用示例D/S/C各2并断言tinyALARM，实际SUCCEEDED导致验收脚本失败；已纠正为独立tiny fixture复验，未改产品代码、未把两份配方结果混淆。3来源SHA全核对通过。Ruling: 唯一owner失败封存只保留已有公开backend/validation身份并固定reason，CLI不得补写终态，允许最小修改artifacts与其回归测试。Ruling: run_integral新增显式keyword backend_identity限demo/simon/UNVERIFIED、默认UNVERIFIED；只标后端来源，scientific_validation仍False，不声明未做的密码/科学验证。policy由CLI预写不覆盖。Ruling: 确认关系失败是cell科学否定非进程异常，可完成null继续固定清单；执行/预算/身份错误才停止run。阶段聚合行需流式留存而不是一组一文件。Ruling: 产物owner需提供受限报告写入接口，将Task5三个报告文件也计入同一output_bytes，避免后续报告绕开额度；算法仍不进保存模块。Ruling: collector domain=(side,phase,index)，内部去重含cell身份，避免不同cell组index误冲突；null不生成真实key。用户已确认连续执行，后续不重复询问。
