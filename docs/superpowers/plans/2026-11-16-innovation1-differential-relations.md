# 创新1多位差分关系预测 Implementation Plan

> **发布时状态**：实现与合成/static验收已完成，原任务步骤作为实施记录保留；本次科学KAT、私有标签、神经前向和优化更新尚未执行。用户已批准必要Git发布、SSH exact同步、指定GPU1五fit/8小时周期和saved-only取回，不安装、不换卡、零正式重试。远端现有Python3.13.3/Torch2.7.1+cu118/NumPy2.4.0及CUDA卡身份已核验。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已批准的SIMON六轮、四关系、shared-M预测变体接入统一CLI，完成受保护的本地合成验收，再准备固定五fit周期。

**Architecture:** 核心仅新增纯预测模型与通用CLI委派、通用存储能力；外置插件拥有配置、数据、身份、五fit、评价、资源适配与报告。插件单向依赖core，旧任务与checkpoint保持原样。

**Tech Stack:** Python 3.12+、现有NumPy/Torch、pytest、Ruff、mypy；开发不安装。

**Spec:** `docs/superpowers/specs/2026-11-16-innovation1-differential-relations-design.md`

## Global Constraints

- 用户已经要求“开启实验”“继续推进”；本计划落实原设计，默认委派，不重选任务。
- SIMON64/128 r6，alpha=0x40；R1..R4掩码分别0000000008040201、0000000080402010、0804020100000000、8040201000000000，全字LSB0；模型位序MSB0。
- 一个私有随机K，TRAIN/DEV/TEST和五fit共用；模型与公开产物无key、key哈希、内部状态、额外密文监督。
- small262144/large1048576/dev65536/test262144对；orbit代表bit6=0，划分互斥，small是large前缀。
- batch256组=1024对；32768 updates/fit；每1024步DEV；Adam(lr 1e-3或3e-4, betas=.9/.999, eps1e-8, wd0)，clip1，FP32，无AMP/日程/dropout。
- F1 small/lr.001/A31，F2 small/lr.0003/A31；DEV择lr平局.001；F3 large主结果/A31，F4 large/B37，F5 large整行置乱/A31。
- 最多2752768 real blocks（含门禁256）、163840 updates、183566336 pair forwards；独立orbit1376256，训练呈现167772160；零正式重试。
- 总科学wall28800s，每fit5400s含其最终评价；GPU20GiB、RSS32GiB、产物2GiB。总限优先，未完成记PARTIAL。
- data/<id>/run.json与一个合并日志、必要紧凑数据；models/<id>/最多五best；result/<id>/一个自包含report.html。
- 测试使用假oracle/人工输出/假优化器；实际cipher、真实优化器step、模型forward、GPU、SSH、安装、Git发布均不由开发代理执行。
- 本地现有pytest/mypy模块可用；不跑可能触发旧科学的full suite。远端Python3.10历史环境不作为已兼容；不降项目3.12最低版本。
- 隔离源码副本 `/tmp/cipher-relations-owned-7ir9jcyd/cipher-lab` 基于当前dirty内容；逐文件核对后精确回迁，禁止reset/clean/stash/commit/push。

## Review Focus

1. 插件名看似不存在但实际为插件内部缺依赖：必须分别呈现NOT_FOUND/LOAD_FAILED，禁止回退core；Task1测试。
2. 同config/seed但不同固定key数据：evaluate必须拒绝其他experiment_id/内容SHA，Task3/4测试。
3. 训练完成但最终评价将到期：共享截止保留收尾时间，逐fit计时跨最终评价累计，不超时继续，Task4测试。
4. Windows资源读数缺失/错误、源文件被换、checkpoint读回失败：停后续并保留UNKNOWN/PARTIAL，Task1/4测试。
5. 四标签缺类、所有标签常量、损失差零方差、极端点支配：保留undefined/uncertain，禁止换掩码或临时主模型，Task3/4测试。

## 文件与职责

- `src/cipher_lab/cli.py`：main第一步识别并转发显式plugin；不放实验逻辑。
- `src/cipher_lab/plugin_cli.py`：纯stdlib参数清理、一次importlib调用、main协议与固定错误边界。
- `src/cipher_lab/artifacts.py`：最窄通用扩展：claim显式声明可覆盖的模型槽；独立单HTML写入，原三件套API原样。
- `src/cipher_lab/models/shared_m_relations.py`：纯模型，复用既有公开邻域、PublicRouter、Residual；不依赖plugin、训练或cipher。
- `plugins/cipher_lab_plugins/differential_relations/{config,data,metrics,checkpoint,training,application,evaluate,resources,report,cli}.py`：各自承担本实验同名职责；不造通用runner。
- 插件包`__init__.py`空/文档；配置`configs/differential-relations-simon6.json`；唯一启动`plugins/cipher_lab_plugins/differential_relations/start.ps1`。
- 对应测试`tests/test_plugin_cli.py`、`test_plugin_artifacts.py`、`test_shared_m_relations.py`、`test_differential_relations_*.py`；精确修改AST边界例外，不放宽其他目录。

## Task 1：通用委派与持久化边界

**Files:** create `src/cipher_lab/plugin_cli.py`, `tests/test_plugin_cli.py`, `tests/test_plugin_artifacts.py`; modify `src/cipher_lab/cli.py`, `src/cipher_lab/artifacts.py`, `tests/test_boundaries.py`。

**Interfaces:**
- `plugin_cli.dispatch(argv: Sequence[str]) -> int | None`，无plugin返回None；有plugin调用指定模块`main`，传入剔除plugin参数后的其余argv。
- `RunStore.claim(..., model_slots: tuple[str, ...] = ())`，仅该owner显式声明的合法token槽新增replace权限；旧默认槽行为不变，不写实验名单。
- `RunStore.write_html_report(content: bytes) -> None`：仅终态owner、核冻结run.json、单report.html、预算/临时峰值/所有权保护与现有方法相同。

- [ ] RED：非法/空/重复/非点分plugin返回2；不存在返回PLUGIN_NOT_FOUND/2；内部缺依赖PLUGIN_LOAD_FAILED/2；main缺失/坏返回值（含bool）PLUGIN_PROTOCOL_ERROR/2；执行异常PLUGIN_EXECUTION_FAILED/4；插件自己SystemExit(0/2)正常处理；无plugin旧路径不变；不输出原始异常。
- [ ] RED：五自定槽可反复写读，未声明槽不可覆盖，原best槽回归；HTML只能终态owner写，run被篡改或预算不足拒绝，原三件套API回归。
- [ ] 运行 `PYTHONPATH=src /workspace/.venv/bin/python -m pytest tests/test_plugin_cli.py tests/test_plugin_artifacts.py`，确认缺功能失败。
- [ ] 实现上述签名；只允许plugin_cli.py::dispatch的一处importlib.import_module调用，AST测试枚举并限制该精确边界；main在旧argparse前委派。
- [ ] GREEN并Ruff/mypy本任务文件；交付实际命令、失败/通过证据和变更文件，不提交。

## Task 2：独立纯shared-M关系模型

**Files:** create `src/cipher_lab/models/shared_m_relations.py`, `tests/test_shared_m_relations.py`；不改旧模型。

**Interface:** `SharedMRelations(nn.Module)`无参固定构造；`forward(pairs: Tensor, public: Tensor) -> Tensor`，[B,4,2,64]和public13→[B,4,4]；参数1103547。

- [ ] RED：构造参数数目1103547，三stem同初始权重；无relation_projection/input_view/backstep依赖，head残差末层零；输入形状/dtype/device拒绝。
- [ ] 测试允许实例化/静态检查，禁止执行真实forward或optimizer；形状、逐对独立与同值语义通过AST和注入fake组件检查，真实数值forward留后续明确有界门禁。
- [ ] 实现3×Conv27→96→96+GN/ReLU，既有local/word/nibble索引；两端及XOR三通道；public top2混合；五Residual3/5/7/9/11；逐对6144→4 base + 6144→64→4 residual，逐对无均值。
- [ ] GREEN：本任务pytest/Ruff/mypy；旧SharedM源码SHA保持不变。不提交。

## Task 3：插件配置、数据、统计与checkpoint

**Files:** create插件`__init__.py`, `config.py`, `data.py`, `metrics.py`, `checkpoint.py`、配置JSON及对应测试。

**Interfaces:** 本任务代理在写实现前将具体dataclass/API写到独立接口说明供Task4使用；公共输入必须能注入假oracle和meter，无任何run-on-import。config.load_config(Path)纯stdlib，返回冻结配方并提供to_dict；data管理规范代表、分割、标签和延迟TEST；metrics接收数值/聚合量、不触发前向/采样；checkpoint只保存/严格加载state_dict与公开身份。

- [ ] RED：掩码位序双路toy结果；反向orbit/重复跨集拒绝；small嵌套；假oracle每对2块先reserve再调用；私有key不在输出元数据；TEST调用需所有五fit冻结令牌。
- [ ] RED：整行四标签置乱保留联合频数、train/dev独立；TRAIN-only常量/63位树桩Laplace与平局；四标签先orbit均值再CI；零方差/极端支配标uncertain；逐关系缺类undefined；精确McNemar+Holm8项。
- [ ] RED：checkpoint绑定experiment_id、编码、掩码、train/dev和real/null目标SHA；严格keys/shape/dtype/有限值；另一实验同config/seed拒绝；保存数据格式无pickle；unsafe torch.load不允许。
- [ ] 实现固定配方校验（配置不得自增预算），紧凑uint64 orbit+uint8标签，科学数据不入Git；TEST绑定原run selected SHA；raw key仅数据侧内存。
- [ ] GREEN pytest/Ruff/mypy指定文件；测试全fake，不执行SIMON/真实模型前向/真实优化器。不提交。

## Task 4：固定五fit、资源、统一CLI和单HTML

**分工：**4a资源模块与对应测试由`implement`独立完成；4b纯报告与对应测试由另一`implement`完成；4c训练/应用/evaluate/CLI/启动脚本由`implement-deep`在Task3接口交付后接线。三者文件不重叠，最终统一审查。报告只读取保存的比较量，次级准确率8项与主BCE分别展示；TEST基线不得从DEV/TRAIN回退推断。

**Files:** create插件`training.py`, `application.py`, `evaluate.py`, `resources.py`, `report.py`, `cli.py`, `start.ps1`及对应测试。

**Interfaces:** plugin `cli.main(argv: Sequence[str]) -> int`；config/check纯stdlib；claim后才导入numpy/torch/真实oracle/RNG/资源，evaluate独立且不import training/ciphers/data生成。

- [ ] RED：假trainer验证F1/F2择lr平局规则、F3固定主模型、F4不择优、F5独立目标、32DEV机会及首个min、非有限/OOM/异常停后续、无重试；TEST在五fit冻结后；全部TRAIN常量零fit/零TEST。
- [ ] RED：fakeclock总8h与逐fit90min累计（训练/周期DEV/selected TRAIN/DEV/TEST均计入）；开新fit前保留明确收尾窗口，总限优先；完成不足步数标PARTIAL，缺F3/TEST不补指标。
- [ ] RED：Windows/Linux RSS可注入且缺值停；GPU内存需实际可用读数，unsupported不当0；CPU模式禁止绕过正式GPU配方；python<3.12启动拒绝且不安装。
- [ ] RED：参数校验/设备绑定失败不取私有熵或标签；运行前KAT与独立标量实现对照总计≤256block且计real_blocks（正式门禁代码只写不执行）；日志异常不泄私有值。
- [ ] 实现单次私有key、KAT→采样/标签→基线→F1..F5→全部冻结→TEST→最终评价→释放K；reuse Meter与RunStore，model_slots声明F1-best..F5-best；每次DEV改善耐久写best，并按摘要从磁盘严格读回。
- [ ] 训练/DEV流式batch记pair forwards，总上限183566336；最终TRAIN/DEV可以训练结束就做但TEST等全部冻结；同fit跨阶段复用同一CumulativeBudget，不能重置额度。
- [ ] 报告纯saved-only：单自包含HTML+内嵌SVG示意/曲线/对比，保存值缺失显示未完成，不美化重算指标；report由原RunStore.write_html_report落盘。
- [ ] start.ps1只本地既有源码/环境与GPU/路径/覆盖检查+调用统一CLI，合并日志内记版本/退出；不SSH/Git/安装，不另生成取回/监控脚本。
- [ ] GREEN指定fake测试；干净源码副本check/help；fake组装CLI闭环保存→严格读回→最终报告；不启动科学。

## Task 5：独立验收、精确回迁与远端准备

- [ ] code-review检查Task1、Task2各自交付；Task3/4完成后整体独立审查。主控核对实际diff与依赖方向，scope外变更拒绝。
- [ ] 主控fresh运行指定pytest、Ruff、mypy和干净源码CLI check/help；无Torch/cipher导入的check探针；不使用fullsuite。
- [ ] 对比baseline.json与/workspace当前文件：被他人改动则停该文件回迁；新文件必须不存在；目标保留其余全部dirty。
- [ ] 实施通过后统一列出需要的Git发布、SSH同步、实际GPU/解释器选择；缺批准项一次集中确认，不把工程就绪称科学运行。
- [ ] 获明确所需权限后才Git发布、远端现场验证和固定科学周期；不传源码包、不追加fit。取回saved-only、直接SCP相关目录。

## 当前执行记账

- 设计批准依据：用户“开启实验吧”及本轮“继续推进他设计的实验”。
- deep-research已完成静态审阅：无科学阻断；配置文件路由为astra，工具回包未提供实际模型字段，尚不声称核验了运行模型身份。
- 本地现有pytest/mypy/torch/numpy可定位；尚未执行新科学。
- 采用当前dirty源码隔离副本开发，不创建忽略未提交内容的HEAD worktree；其不是Git worktree，回迁前逐文件对比baseline。
- Task1/2可并行（不同文件）；Task3/4c顺序交接明确API后实施，4a/4b不共享修改文件。
- 所有实现已回迁；主控338项指定fake/static测试、Ruff和mypy通过，独立最终审查覆盖范围内无新增确定P1/P2。共享1800秒、逐fit5400秒、总28800秒分别记账。发布候选仅含本实验文件和最窄通用依赖，不夹带其他dirty实验；本轮真实科学尚未启动。
