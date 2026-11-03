# 创新2神经主线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为N1背景机制控制、N2五方法、N3升轮及N4确认提供可严格加载、保存逐样本证据的本地实验能力，不在开发中启动正式科学实验。

**Architecture:** 公开变换、研究模型、证据统计分别归独立小模块；旧MLP／采样／标签不变。training只接线，evaluation在原末端forward中可选捕获预测，统计只消费保存数组。外层研究决策由主模型按设计执行，不造通用调度器。

**Tech Stack:** Python3.12、现有Torch／NumPy／pytest／Ruff／mypy，不安装。

**Spec:** `docs/superpowers/specs/2026-11-02-innovation2-multicycle-design.md`（先完整阅读，尤其§3—5、7—11）。

## Global Constraints

- 用户已选代理执行：核心implement-deep→auto/strong，常规implement→auto/glm-flash；code-review独立只读；不再派子代理，不修改代理配置。
- 留main、保留全部既有dirty，不reset/clean/stash/stage-all。不commit/push；任务中的“提交”门槛由精确diff交付代替，发布另批。
- 本轮仅本地人工／fake／null工程，生产encrypt=0，正式标签／fit／GPU／SSH／安装／主动旧checkpoint evaluate=0。
- 工程优化累计≤64updates、≤512presentations：核心12/96、外围12/96、主验收12/96、干净CLI28/112；重复测试也扣额度。
- 纯前向／人工oracleCPU和wall各≤30分钟，开发验证wall≤4小时（不含设计讨论等待）。每次执行先登记计数，不以pytest通过豁免。
- 正式矩阵最多36fits／294912updates／75497472presentations；这不是开发许可。1152在线DEV另36末端DEV，共1188；正式retry0。
- 默认序列化／旧state键／RNG／checkpoint schema／采样／标签／DEV最小BCE／strict load后TEST时序不变。
- 用`PYTHONPATH=src .venv/bin/python`，不resolve解释器symlink，不安装旧wheel替代源码。
- 累计工程账本用`validation/innovation2-multicycle/neural/engineering-ledger.json`，源码／测试／命令／attempt状态／updates／presentations逐次追加；失败和未知保留。账本不是授权或通用runner。

## Review Focus

1. 新参数被旧model_kind静默忽略：非法组合必须构造时拒绝（Task2）。
2. 背景补偿在strict load后发生第二次：同一checkpoint重复加载与原模型输出必须一致（Task3）。
3. 捕获预测改变forward次数或提前请求TEST：计数模型／假标签时序钉住（Task4）。
4. 配对数组顺序错、SHA混用、恶意NPZ／NaN：拒绝，不只比较labels分布（Task5）。
5. 末端DEV被漏当选模或确认D与自身比较：预算矩阵与冻结决策fixture分别钉住（Task6）。

## 文件责任与所有者

| 文件 | 归属 | 可依赖／不负责 |
|---|---|---|
| `src/cipher_experiments/representations/prediction_public.py` 新 | implement-deep | 标准库／torch；公开phi、Psi、q，不读config/key/data |
| `src/cipher_experiments/models/public_prediction.py` 新 | implement-deep | 上述表示／torch；研究MLP／linear、初始化、L1，不写报告 |
| `src/cipher_experiments/{config.py,models/prediction.py,checkpoints.py}` | 同一个implement-deep | 严格参数接线；旧语义不重构 |
| `src/cipher_experiments/prediction_reports.py` | 同一核心代理 | 纯metadata，不forward或写盘 |
| `src/cipher_experiments/prediction_evidence.py` 新 | 另一个implement-deep，Task3完成后 | numpy／torch捕获／保存检查，不import模型／训练／cipher |
| `src/cipher_experiments/paired_statistics.py` 新 | 同证据核心代理 | numpy纯统计，不读文件／重新生成标签 |
| `src/cipher_experiments/{evaluation.py,training.py}` | 证据核心代理独占Task4 | 可选收集器／显式L1组装，不内置特征和bootstrap |
| `tests/test_public_prediction_*.py` 新 | 各任务所有者 | 人工／fake，0真实密码 |
| `tests/test_prediction_evidence.py`, `tests/test_paired_statistics.py` 新 | 证据代理 | 人工数组／fake前向，0科学fit |
| `configs/innovation2-multicycle-*`, `tests/test_multicycle_configs.py` 新 | implement | 接口冻结后独立小配置／矩阵算术 |
| `validation/innovation2-multicycle/neural/{READY.md,engineering-ledger.json,verification.json}` 新 | implement外围／主模型验收 | 精确执行记录，不搭服务和新训练入口 |

## Task1：公开变换与研究模型核心

**Owner:** implement-deep。**Dependencies:** 无。不得改shared config／training接线，先交付可独立人工测试核心。

**Interfaces**（types取Tensor、nn.Module，均只接受显式公开参数）：

- `public_coordinates(inputs: Tensor) -> Tensor`，返回[N,64] float32；与旧phi一致。
- `public_interactions(inputs: Tensor) -> Tensor`，返回[N,96]，前64是phi，后32是设计规定的AND；输入bool／float32二值。
- `zero_round_constant_reference(inputs: Tensor, *, rounds: int, target_bit: int = 0) -> Tensor`，rounds仅6/8/10，bit0，返回[N,1] float32。
- `zero_fixed_background(inputs: Tensor, *, plaintext_space: str, fixed_value: int) -> Tensor`，本期只high32-zero，对高半实际值与公开fixed_value不符直接拒绝，不静默丢未知信息。
- `PublicPredictor(hidden_width: int=128, *, model_kind: str, public_input: str, public_readout: str, background_init: str, rounds: int, target_bit: int=0, plaintext_space: str="full", plaintext_fixed_value: int=0, l1_penalty: float=0.0)`；`.forward(Tensor)->Tensor`、`.regularization_loss()->Tensor`、`.indices`非持久arange(64)、`.checkpoint_config_sha256`。
- MLP `.layers`按旧Linear/ReLU/Linear/ReLU/Linear注册顺序；linear用单一`.layers=nn.Sequential(nn.Linear(96,1))`；不随机采额外节点embedding。新模型保存字段区分public_input/readout/init。

- [ ] **Step1：写RED测试** `tests/test_public_prediction_core.py`。独立uint32 scalar旋转oracle验证phi/Psi/q6/8/10；逐列对应；错误dtype/shape/bool参数/float轮数拒绝；全零轮常量与零主密钥命名不同。测试N1 P不被in-place修改。
- [ ] **Step2：执行RED** `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -o addopts= -q tests/test_public_prediction_core.py`；应因新模块/符号不存在失败，不因torch缺失失败。无fit。
- [ ] **Step3：最小实现** 新表示和模型。MLP输入64或96，hidden_width严格int1..4096（formal128），linear不伪造hidden层。残差最后符号恢复；L1只linear weight、bias除外。无真实key、config或labels依赖。
- [ ] **Step4：GREEN断言** 独立FP64 affine oracle验证M bias补偿误差≤1e-12；CPU FP32 R/M logits≤1e-5；R/Z state和RNG完全一致；F/FR、D/R同形状权重／RNG一致；参数24961/29057/97；正则手算一致且不改变参数。FP64只测独立仿射oracle，不放宽生产FP32输入约束。
- [ ] **Step5：验证与交付** 同命令PASS；Ruff两新模块／测试，mypy限定三文件；交精确diff、测试命令和0优化计数，独立review后才进入Task2。

## Task2：配置、factory、metadata和历史身份

**Owner:** 同核心implement-deep，独占共享文件。**Consumes:** Task1接口。**Produces:** 现有`build_predictor`增加keyword-only研究参数，默认全部兼容。

- `public_input="identity"`, `public_readout="direct"`, `background_init="standard"`, `l1_penalty=0.0`, `record_predictions=False`加入ExperimentConfig；factory只接前四项及公开plaintext_space／fixed_value，不接record开关／key_seed／整个config。
- Public新字段组合严格按Spec§5.1。N1 public-mlp/background-zero仅r3/bit0/xor2 bits[0,8]、high32-zero；非full旧限制只增加这个显式例外。N2 public-mlp：full、bit0、6/8/10、input identity/interactions、readout direct/residual，但identity+direct可作人工oracle，formal D仍旧mlp。public-linear固定interactions/residual/1e-4。
- record_predictions严格bool、仅prediction；旧model_kind只允许其余研究字段为默认。default字段从to_dict删除；非默认科研字段都入SHA。

- [ ] **Step1：RED** 新`tests/test_public_prediction_config.py`测试默认config字典/SHA与固定当前4596264d已有代表fixture相等，不用移动HEAD；所有新非法组合、NaN/inf/bool L1、伪字符串拒绝；所有旧model_kind不接受研究字段静默忽略。
- [ ] **Step2：跑RED** 指定此测试，确认失败来自缺字段／合法新模型暂拒绝，不执行旧cipher测试。
- [ ] **Step3：实现最小接线** config校验/序列化；factory新分支；旧路径拒绝非默认研究字段；old phi包装调用Task1公开实现但默认输出逐元素一致。metadata显式写实际表示／reference解释／无额外监督。
- [ ] **Step4：GREEN** 同形状工厂参数SHA／RNG一致；metadata不forward、不改mode/RNG/梯度；旧model_kind、旧direct/public-coordinate/public-residual的默认行为不变；创新1不接受研究开关。
- [ ] **Step5：限定验证与review** Ruff/mypy改动模块和两新测试，pytest仅新测试；交接口清单和diff，不推进训练接线。

## Task3：strict checkpoint与候选目标函数接线

**Owner:** 核心implement-deep。**Consumes:** Task2 factory。**Produces:** strict重建public模型的旧schema checkpoint；training传公开选项／L1但不实现数学。

- [ ] **Step1：RED** `tests/test_public_prediction_checkpoint.py`用人工state，覆盖new save/load、R/M重复加载不累加bias、config换背景/表示/readout身份拒绝、state缺键/shape/dtype/NaN篡改；保留至少一旧稀疏config checkpoint可加载。旧schema不改、不放松state校验。
- [ ] **Step2：跑RED** 只此文件（0fit），应在factory参数重建或类型不支持失败。
- [ ] **Step3：实现** checkpoints仅类型和factory参数透传；若无persistent新buffer不新增canonical wiring钩子，公开固定行为由严格配置+固定实现定义；若代理认为必须buffer，先回报接口变动，不能暗加兼容分支。training build_predictor透传公开参数；仅public-linear时显式取regularization_loss加入objective，public-mlp及原分支只BCE，不因加一个零正则张量改变默认图和日志。
- [ ] **Step4：GREEN集成** 新`tests/test_public_prediction_training.py`人工labels，monkeypatch生产encrypt使任意调用立即失败；只允许一次2updates/8presentations core fake fit。Spy原y和优化目标，S日志train_bce保持原BCE、train_objective才含L1；旧run不新增objective字段。new selected strictload后才请求TEST。记录此fit后重复执行需扣core12/96额度。
- [ ] **Step5：验收** 新三文件限定pytest（checkpoint和core可无fit重复，training须计量）；逐项Ruff/mypy通过；独立review和主模型核diff。Task3结束核心代理共享源码所有权交给证据代理，不并行再改factory/config。

## Task4：末端捕获和独立公开参考基线

**Owner:** 证据implement-deep。**Consumes:** 公共变换、研究config、strict模型。**Produces:** 可选收集器，不多forward。

**Interfaces:**

- `PredictionCapture(expected_rows: int)`，`.append(logits: Tensor, targets: Tensor)->None`、`.finish()->tuple[NDArray[np.float32], NDArray[np.uint8]]`；detach/cpu/copy，拒绝溢出／少行／非有限，固定原行顺序。
- `evaluate_model(..., capture: PredictionCapture | None=None, zero_constant_reference: Tensor | None=None, zero_constant_majority: int=0, zero_constant_probability: float=.5) -> dict[str, Any]`；不删除或重命名旧reference参数。
- 新基线输出键`zero_round_constant_reference_constant`；旧`public_reference_constant`不改。TRAIN拟合复用`baseline_parameters`的规则（reference与原y残差），不从DEV重拟合。

- [ ] **Step1：RED** 新`tests/test_prediction_capture.py` CountingModel和手算数组检查append顺序／同一次评估calls；捕获开/关原metrics完全相同；q基线独立手算概率／ACC；非法ref形状／非binary拒绝；capture禁止多finish或额外append。
- [ ] **Step2：跑RED** 仅人工eval，无优化/密码，预期接口未定义失败。
- [ ] **Step3：最小实现** `prediction_evidence.py`实现capture；evaluation只在batch已有logit/target后append。q参考只在本周期record_predictions且r6/8/10 full/bit0时组装，保持D与public模型皆可比较；N1表达式不错误加q基线。training缓存每split公开q面板、只TRAIN拟合概率；在selected末端三split创建收集器，不捕获在线DEV。
- [ ] **Step4：GREEN集成** fake labels/CountingModel验证32在线DEV＋1末端DEV计量fixture可用较短epochs显式等式；实际只一次2updates/8presentations外围fake fit，严格load前无TEST标签。默认record=false不新增证据文件／返回字段或额外q计算。旧initialization与结果默认格式不变。
- [ ] **Step5：验证** 无fit测试可重复、fit测试登记剩余额度，Ruff/mypy限定；独立review不将保存标签称科学新生成。

## Task5：证据存储与确认纯统计

**Owner:** 同证据代理核心；常规implement仅专属测试／文档。**Consumes:** Task4 capture；不import模型／training／cipher。

**Interfaces:**

- `write_prediction_evidence(output: Path, *, split: str, logits: NDArray[np.float32], labels: NDArray[np.uint8], x_sha256: str, y_sha256: str, config_sha256: str, checkpoint_sha256: str, reference_bits: NDArray[np.uint8] | None=None, reference_majority: int=0, reference_probability: float=.5) -> dict[str, Any]`。
- `load_prediction_evidence(manifest_path: Path) -> PredictionEvidence`；dataclass持logits/labels/optional ref＋身份；先file size/SHA/schema，再np.load(allow_pickle=False)；不得读绝对／../逃逸路径；拒绝zip重复键/额外数组/畸形dtypes/非有限。
- `paired_interval(candidate: NDArray[np.float64], comparator: NDArray[np.float64], *, block: int, comparator_id: int, metric_id: int) -> dict[str,float|int|str]`。输入是按正方向定义的逐行贡献，返回point/lower/upper/resamples/seed/method；固定10000 bootstrap、alpha=.05/18、percentile、quantile linear、PCG64 seed=196801+100*block+10*comparator_id+metric_id，每chunk≤64。
- `paired_comparison(candidate: PredictionEvidence, comparator: PredictionEvidence | ConstantBaseline, *, block: int, comparator_id: int) -> dict[str,Any]`放`paired_statistics.py`；具体ConstantBaseline冻结dataclass含bit/probability/optional ref bits以及x_sha256、y_sha256、rows；先验证同x／label／rows身份（不要求不同方法的config/checkpoint SHA相等），模型BCE用FP64 logaddexp不裁剪；ACC仅对候选／D的模型logits使用>0。TRAIN常数分类直接用保存majority bit；q基线分类逐行用reference_bits XOR reference_majority，不能由概率转logit再判>0；概率只参与BCE。调用paired_interval时ACC传candidate_correct与baseline_correct，BCE传baseline_loss与candidate_loss，因此底层统一算第一个数组减第二个数组；不得依metric_id暗中再翻符号。

- [ ] **Step1：RED** `tests/test_prediction_evidence.py`三种split保存／读取／元信息；prediction NPZ仅logits/labels；ref有独立`<split>-reference.npz`且仅uint8 bits，概率在manifest；整个manifest SHA写入results证据索引。数组hash使用实际存储dtype字节，不把uint8 label hash冒称旧float32 y_sha。
- [ ] **Step2：跑RED** 仅文件人工数组；缺模块／函数应失败。
- [ ] **Step3：实现存储** 文件exist_ok=False，先临时文件同目录原子落盘，失败不覆盖旧证据；manifest列每数组shape/dtype/SHA与file SHA，旧input x/y身份单列；整run失败由training现异常路径收口。NPZ磁盘≤32MiB/文件、总解压数组≤16MiB/证据、rows≤1m；人工fixture只小数组。
- [ ] **Step4：RED/GREEN统计** `tests/test_paired_statistics.py`用常值差/零差/方向反转、独立少量参考重采样验证seed／分位；重复调用bitwise deterministic；NaN/错序身份拒绝；18格比较族固定，D自比较跳过但不收回alpha。10000参数不暴露为科学调参，测试可小N但同10000计数。增加TRAIN残差恰好一半0一半1、reference同时含0和1的人工fixture：residual_majority=0且probability=.5；在线评估与保存证据后处理逐行q预测均等于reference，correct数组完全一致，不能全部判0。TRAIN常数平局分类全0另测，避免与q平局混淆。
- [ ] **Step5：末端接线与核验** 将capture结果一次保存、results只加索引；原fp32 BCE与保存数组FP64差>1e-5报错且不覆盖原metrics。接收SHA验收与报告统计分开命令／记录；不得运行load_checkpoint或labels_for。
- [ ] **Step6：审查交付** Ruff/mypy三个新模块／测试，pytest纯数组限定通过；bootstrap正式30分钟预算不在开发中用65536科学预测跑满。此任务不启动模型fit。

## Task6：固定配置矩阵、预算与停止分支fixtures

**Owner:** implement，禁止改核心源码。**Consumes:** Task2最终配置接口＋Task5证据索引schema。**Produces:** 配置／小的阶段决策说明，不产生自动调度器。

**Files:** `configs/innovation2-multicycle-n1/`六正式config；`configs/innovation2-multicycle-n2/`十正式config；`configs/innovation2-multicycle-smoke/`十四null config（覆盖N1三模式、N2五候选及r8/r10三种研究路径组合；不必为每个formal配置重复smoke）；`tests/test_multicycle_configs.py`；`validation/innovation2-multicycle/neural/READY.md`。

- [ ] **Step1：RED** 读config文件的测试：N1 fixed seed196101/null196102/init17/key190502、xor2[0,8]和B=2779096485；N2 seed196201/null196202、bit0/full/r6；每formal8192updates、32在线DEV、1selected DEV、record_predictions=true。文件不存在应FAIL。
- [ ] **Step2：创建最小矩阵** 精确模板对应Spec，不增加参数网格。smoke每8/4/4、batch4/epoch1/2updates/8presentations/cpu/null/60秒。formal cuda:0/1800秒只存配置，不调用CLI。
- [ ] **Step3：预算／分支测试** 纯测试fixture，不新增生产workflow：N2全败停N3/N4；D选中去重复；r8能力但无方法优势且r6有优势则确认r6，二者都无优势则确认r8；null报警停；缺exit UNKNOWN；36最大计数；N4三块alpha固定。阶段规则用test-local纯函数＋文档预期核对，不让它成为正式权限／自动任务执行器。
- [ ] **Step4：GREEN** `PYTHONPATH=src .venv/bin/python -m pytest -o addopts= -q tests/test_multicycle_configs.py`，只解析／算术0fit；Ruff/mypy限定test。
- [ ] **Step5：交付说明** 明确N3/N4配置由已批准条件选择后按Spec固定schema生成、完整参数在正式前冻结；不得生成不存在的候选结果或“已通过”标记。未获科学权限时停在ready-for-approval。

## Task7：主验收、干净源码小闭环和最终交接

**Owner:** 主模型验收（可执行命令，不接管核心编码）；code-review只读整体范围。

- [ ] **Step1：核实际diff** 每个Task输出的文件／依赖方向／无关dirty保留；检查所有新增生产import不触及backups／历史runs／cipher模型反依赖。记录起始HEAD及每文件SHA，不用HEAD推定dirty已发布。
- [ ] **Step2：限定回归** 新测试＋已知无fit／无cipher旧兼容测试；先collect并审调用链，再运行。任何可能fit的测试先预算，主验收最多12updates/96presentations，不全仓盲跑pytest。Ruff/mypy可全tracked源码范围但不收集backups，未跑项目全pytest如实报告。
- [ ] **Step3：干净源码** 复制当前所需tracked源码＋本任务新文件到精确新临时目录（不是备份，不复制runs／环境），依赖沿当前venv，显式PYTHONPATH该副本src；14个null config各check/train，28updates/112presentations，cipher guard可单独fake测试证明null路径；严格核best SHA/load时序和exit，不做未授权旧科学evaluate。副本输出只在其自建路径，完成只清自身路径。
- [ ] **Step4：独立review** 按代码正确性／科学语义／计数／兼容／数据访问审核；修复后仅重跑必要无fit测试，若fit余额不足停申请而非续跑。
- [ ] **Step5：交付** READY/verification含实际执行范围、累计updates/presentations/cipher0、源码/config身份、未测GPU/Windows/科学性能。正式入口命令仅列已实现check/train，**不实际启动**；一次集中申请36fit条件矩阵／GPU／SSH／最多3精确发布／取回权限。原计划标task done和blocked，不自动commit。

## Self-review／计划审阅门槛

Spec§3/4对应Task6科学配置／条件；§5对应Task1—3；§7对应Task4—5；§8/9对应Task7和工程计数；积分§6由独立积分计划覆盖；正式权限§11未在本计划授予。

文件接口、错误测试和预算已自检；独立审查的三项契约缺口修订后，用户回复“是”确认实施。执行方式为已选代理驱动，当前Task1开始；按任务TDD→独立review→主验收连续推进，不再重复总路线讨论。正式科学／GPU／SSH／发布权限仍未授予。
