# 5轮硬布尔SAT推进：一次求解UNKNOWN，批次关闭

## 结果先行

固定SIMON64/128目标bit0，512条真实TRAIN标签，本机CPU单次Glucose4求解：
**UNKNOWN，无可用参数赋值，无预测器，无DEV/TEST预测指标。**
未生成TEST标签；按预定门槛停止，r6与r5-null均未启动，没有重试/换后端/增加预算。

这不是“50%预测”、UNSAT、5轮不可预测或硬布尔路线无效的证据。
之前新后端r4硬参数辨识TEST100%继续成立；历史旧代数r5持出100%另有证据，
但其算法与本次SAT合同不同，不能据此把本次标记成功。

## 方案与实际执行

运行前合同见[SAT推进方案](INNOVATION2-SAT-PLAN.md)。

- TRAIN512 / DEV1024 / TEST16384，新的data191901、固定key190502；只用TRAIN约束拟合。
- 5轮目标锥44个符号轮效应。每个TRAIN样本展开独立AND/XOR电路，各行共享未知常量；
  明文和目标位作为单位子句，真实key/轮密钥/trace未提供给求解器。
- 编码113708个总SAT变量（含电路内部变量，并非113708个秘密参数）、361984条子句。
- 现成python-sat1.8.dev24、Glucose4，无依赖安装。
- 完成建模后单次solve_limited(expect_interrupt=True)，预算中断，无重启/第二调用。
- 拟合返回assignment=null。保存status=unknown，exit_code=0表示能力正常返回未知，**不表示找到了预测器**。
- TRAIN/DEV标签已按配置生成，TEST明文已生成但没有标签，data.json的TEST项没有y_sha256。
- 没有best.pt/selection.json，没有本地checkpoint replay或补算预测。

## 时间边界

配置max_seconds=120为**协作预算，不是OS硬墙钟**。
保存结果elapsed_seconds=`132.1835962680052`，exit elapsed=`132.18368831899716`；
拟合函数elapsed=`132.14911361399572`，包含编码/求解/关闭所用时间，不是独立测量的纯native求解时间。
总结果超过协作目标约12.184秒。Timer调用interrupt不保证原生代码即时返回；
本次没有阶段级interrupt时间记录，不能进一步声称已定位超出都发生在哪一阶段。
没有把这12秒冒充事先授权扩时，也没有因UNKNOWN主动再运行。

门槛检查脚本在读取UNKNOWN后抛AssertionError，发生在CLI正常返回并保存完整exit之后，
它表示r5推进门槛未通过，不是训练程序崩溃。后续目录已核实不存在。

## 可以推断与不能推断

这是一次“朴素全TRAIN电路CNF在指定协作预算下未给出解”的负结果。
硬结构模型仍包含真实目标；模型表达正确不等于此通用CNF求解过程能在预算内找到模型。
没有结果不能归咎为数据完全无信号或参数学习违规。

下一项更合理的设计工作是保留公开XOR/布尔代数结构、共享/化简可抵消项或分解未知效应，
而不是直接增加这个CNF的时间、样本或轮数。本批没有执行这些新候选。
最初不含key值的保守单项式展开在4轮中间集合已超过8192项，明确停止，未盲目扩上限；
不代表所有代数展开方法都需要如此大，也不否定更紧凑精确化简或旧r5代数结果。

## 工程和证据

新增可选`model_kind=simon-hard-sat`，继续统一check/train/evaluate；
HardSimonPredictor在SAT赋值保存并严格加载后执行独立TRAIN硬前向复核，之后才可生成TEST。
无assignment时保留UNSAT/UNKNOWN差别，不伪造指标。模型推理不依赖SAT库。

独立只读审查发现并修复：
1. 两类拟合结果的局部类型冲突：显式联合类型，SAT类型仅TYPE_CHECKING导入；
2. 可选SAT后端的测试不能强制默认安装：真实后端测试importorskip，纯CNF/输入校验继续运行，
   加缺依赖明确失败测试。无新建无SAT虚拟环境，不声称全新安装已验证。

749tests、Ruff、mypy64files通过；干净源码副本完成null check/train/evaluate。
正式拟合后只做保存JSON/字节SHA核验，无额外真实密码拟合。
基础HEAD111adab，当前实现未提交，manifest如实记录dirty和实际源码SHA。
本轮无SSH/GPU/依赖安装/commit/push；旧未提交工作完整保留。

```text
runs/innovation2-sat-20261020-073150-r5/
validation/sat-simon/verification.json
validation/sat-simon/clean-path.txt
validation/sat-simon/r5.log
```

7份原始文件：backend/config/data/exit/fit/manifest/results。
configSHA=`33823e71d0711a1064262490686d83e17805bbe0a79b3ff09510a55d968e467d`。
fit文件SHA=`600e679c95e0f1c366f93057dee4cde71ef0c274288ad5f2ef8e249b15d6d225`。
全部保存文件字节长度/SHA、配置身份、源码身份、无模型/无TEST标签及未启动后续路径均已核验。
批次关闭，无剩余预算自动授权。
