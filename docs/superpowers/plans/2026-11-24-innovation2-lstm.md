# 创新2 LSTM 单臂预测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development，TDD/独立审查；代理不派代理、不Git、不真实实验。

**Goal:** 用2层hidden128单向LSTM，在同一固定百万对上进行一次50epoch上限/DEV早停位预测。

**Architecture:** 在既有cube_compare方法owner中加入明确LSTM单臂拓扑与模型工厂，复用SavedPanels/fit/strictreload/earlystop/计量与报告；不再拟合MLP/Shared-M，不复制runner/启动器，不破坏原双重载门禁。

**Tech Stack:** 既有Python3.12本地、WindowsPython3.13.3/Torch2.7.1+cu118/NumPy2.4.0，原Meter/RunStore，coreEarlyStopping，pytest/Ruff/mypy；无安装。

**Spec:** `docs/superpowers/specs/2026-11-24-innovation2-lstm-design.md`（APPROVED）。

## Global Constraints

- 新模型仅标准单向2层hidden128，input_size2/sequence64/bias/无dropout和projection，末层finalh→Linear128；216192参数；每次零h0/c0不跨sample/batch状态。
- 输入[P||P′]128位按MSB0取(N,64,2)对应同位两块，输出128、full128BCE、active8 DEV/TEST不变；不加卡/状态/key/真实cipherbit教师。
- 复用source-02精确run/NPZ SHAbytes与旧split；TRAIN/DEV先读，唯一lstm-best严格diskreload成功后才readTEST。原双臂mode仍强制两best，不用duplicateengines冒充。
- 1fit、max50/10warmup/8patience/delta1e-5、batch256，seed31/order20261124、Adamlr.001FP32clip1；cap204800updates55771136forwards/blocks0/null0，按实际e*4096/e*1114112+65536核账。
- wall10800CPU43200/RSS8GiB/output512MiB/ckpt16MiB、GPU1、once/retry0。无实际数值预检/免费额外warmup/旧模型更新/新数据。
- 开发人工tensor结构/fakeNPZ/runtime/clock/FFI，真cipher/entropy/NN数值/Adam/GPU/RSS/SSH/network/Git/install禁止。guard显式封LSTMforward/RNN路径。旧真实产物仅主控hash/生产只读复用，不许代理decode旧data/loadweights。
- 只本owner/fit必要接口/config/script/tests/docs；core、旧模型、其它owner不改；新tests唯一名，forcedRuff，来源指纹完整。

## Review Focus

1. stack(Pbit,P′bit)而非直接reshape配相邻位；2层finalh选对，全部128输出；初态每次新零、无状态跨batch，不能teacher forcing。
2. 单slot/lstm身份和单fit计量真实，旧双reload gate仍需两份不同best；TEST在唯一newbest持久restore后读取，failure无终分。
3. 早停和best分离/最早18，实际epoch预算，resourcefault不是早停；既有双臂及5epochmode不能被globalARMS monkeypatch改变。
4. 固定sourceNPZ不重采样/不预读旧TEST、sourceoutputoverlap不漏；失败报告不混作SharedM/生成版或伪历史pairedCI。
5. LSTM数值guard补齐，checkpurestdlib；WindowsUUID/user-site/globalnativeexit/SSHforeground，原退出UNKNOWN不重写。

## Task 1: LSTM 模型、明确单臂装配、计量与报告接入

**状态：COMPLETE。** implement-deepaa43bc65；任务审查9c2d9925经Fix1放行；整体审查10aa33ccApproved。主控fresh独立owner198通过46.33s/forcedRuff全树clean/mypy382clean，默认fullsafe3702pass49排除313.60s，既存ZIPwarning1。未实际数值LSTM/GPU/采样。

**Files:**
- Create `cipher-lab/plugins/cipher_lab_plugins/direct_block/cube_compare/lstm.py`。
- Modify `.../cube_compare/{config,engine,training,metrics,cli,report}.py`，以及父`fit.py`仅若需显式复用gate；保留旧接口默认行为。
- Modify `cipher-lab/scripts/start-cube-sharedm.ps1`只加固定LSTMconfig；create `configs/direct-block-cube-lstm-simon7.json`。
- New/modify `.../cube_compare/tests/test_cube_{lstm,cli,training,metrics,report,fit}.py`与conftestguard。

**Interfaces:**
- `CubeLSTM.forward(inputs:Tensor)->Tensor`: (N,128)binaryFP32→(N,128)，内部(N,64,2)、zero(2,N,128)和h_n[-1]，216192params。
- `config.lstm_recipe()`或等价固定method声明模型shape/source/1fit/caps；`lstm_counts(epochs)`返回actual；旧recipe/reuse_recipe原值不变。
- `engine.make_lstm_engine()`返回同TorchEngine/identity `<lstm-method>/lstm`，无初始化其它模型/旧best，不改科学loss。
- 训练`run`显式plan/topology LSTM单臂，不改变旧默认names/slots/gates；共用fit/reload_best。单臂score保存lstm/majority逐位/逐族与97.5%pairedinterval，历史仅savedreference不load模型/再计算。
- CLI固定三配方stdlib分派，source-root必需/lstm-best唯一slot/10800/actualepochs；报告、script固定source参数正确，no unknown fallback pretendingknownmethod。

- [ ] RED：结构实例与参数shape，64步位对映射/顺序/zeroh0c0每次、相同输入不受上批人工状态影响，取最后一层h_n，不误用整sequence/第一层；fakeLSTM/head验证无数值。
- [ ] RED：显式LSTMguard实际数值调用在fake底层被触达前失败；旧Conv/Linear/Adam/cipher/entropyguard不放松。
- [ ] RED：纯check/固定recipe/counts/slot/identity/earlystop和单臂仅1fit及一次predict；源TEST在reload后、hash/restore/scorefailed不终分；旧双臂缺第二best依然失败。
- [ ] RED：人工计数/基线/完整族区间/不伪旧pairedresults；sourceoverlap/failurewrite/reportbudget/globalnativeexit旧回归保留。
- [ ] GREEN：最小实现；不genericregistry/workflow/copyrunner、无同一个结果伪双臂；schema/mode真实显式。
- [ ] 验证owner人工套＋core早停/保存/执行/布局；forcedRuff/strictmypy、两旧配方与新配方纯stdlibcheck/人工闭环、最小sourceclosure，报告真实范围，不科学。
- [ ] 报告task-1-report.md含RED/GREEN命令输出、修改/allowlist/运行参数/实际单臂预算；主控独立review＋fresh，必要fix复审。

## Task 2: 主控限定发布与一次前台GPU运行/取回

**状态：RELEASING。** 仅本轮允许清单一次发布，singlefit/3h/retry0，无旧模型fit或新数据。

- [ ] 从791145ba独立源码archive，base/allowlist/source/main/index保护，无无关工作覆盖。原数据read-onlyhash匹配，旧报告保持。
- [ ] 最终whole code-review/freshdefaultfullsafe＋显式新owner/forcedRuff/mypy/check，scope允许清单一次commit/push，WindowsGit exactcheckout新root；不传源码任务包。
- [ ] source-root原-02/work、GPU1当前UUID/free/idle一次准入，既有Pythonuser-site，不安装不动其它服务。
- [ ] 以同启动脚本新config/新id/10800UTC期限前台执行，保持SSH并wait真实退出；一次LSTMfit/失败即停，无额外数值探测/重跑。
- [ ] 取回实际data/models/result/logs，核SHA/actualepochs/counts/source/nativeexit0/SSHexit，无model load/label/forward/evaluate。图表只saved值，与原多数/历史保存分数注明探索限制。
- [ ] 报告仅服务result/本地HTTP核字节；本轮关闭，剩余预算不复用，不自动双向/Transformer/换seed。
