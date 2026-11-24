# 创新2 LSTM：第1–3轮预测能力固定矩阵

**APPROVED。** 用户在1/2/3轮、同新未知key/同明文划分、3个独立LSTM、各50上限DEV早停、GPU1总3h/一次限定发布运行、任一失败即停不重试提案后回复“是”。仅本矩阵许可，无第4轮/新架构/额外key或seed。

## 目标与方法

回答当前LSTM从低轮开始能预测多准确，为以后“预测中间信息辅助区分”提供先决证据；本轮**不实现后端区分器、不把轮数相加、不做密钥恢复**。公开SIMON64/128缩减1/2/3轮、本机合成一次性key/明文，无真实系统凭据或第三方目标。

模型原样复用：两层单向LSTM hidden128、input2×64步、finalh_n[-1]→128，216192参数；每call零态、不跨sample/batch记忆，无结构卡/真实状态/教师。三个模型都seed31全新初始化、互不共享训练权重/optimizer，无旧权重加载。相同epoch顺序seed20261124、batch256、FP32 Adam .001/.9/.999/eps1e-8/clip1/无decay，全128BCE训练，仍用指定8位DEV BCE最早最低选模。max50，warmup10/patience8/min_delta1e-5各自独立，最早18，best选择独立于早停阈值。

这是预定3fit矩阵，不是根据看到TEST决定加轮；即使第1轮不理想，也按原矩阵完成2/3轮，除非工程/预算/存储故障（立即停）。不改原LSTM来追分，不自动扩大预算。

## 明文与新低轮标签

以source `cube-sharedm-simon7-20261123-02` 原TRAIN/DEV/TEST的**明文与族ID/顺序**作为固定输入。源runSHA=`bc2c87a5e2ab919021c7f4cdf556830f99b59380b858352ae72d30c93669f523`，三NPZ SHA/bytes仍沿用已发布config.source_recipe：train41943816/e01ca03e…、dev2622216/116ebebf…、test2622216/f00dfe78…。代码使用完整SHA，不短写校验。TRAIN1048576对，DEV/TEST各65536对，原16变体8互补对/族，划分不变。

旧7轮标签不能当低轮标签，旧key也未保存。生产数据工厂仅新取**一个128位私有key**，以闭包固定到三轮全部split，使用core `encrypt_batch(points,key_words,rounds)`，nativeuint64连续块、每批≤65536。每个round每split每个块只加密一次，共7077888blocks。真实秘密只在标签生成闭包，不在模型/诊断/日志/产物保存；不从已保存旧标签反求key。

可以复用SavedPanels的严格源字节/格式/族校验后仅取plaintext/family_ids，旧ciphertext不得作为新模型输入或新标签；新dataset保存真实对应1/2/3轮ciphertext并声明source仅明文复用、key新生成。TRAIN/DEV读取一次，三轮对齐，不能按轮重采样/改序。原source所有文件只读，输出root提前拒落入旧owneddata/models/result目录。

依次拟合r1/r2/r3（同进程/同Meter，三slot `r1-best/r2-best/r3-best`）。**三份best都从磁盘size/SHA/strictrestore成功后，才读取源TEST明文和生成三轮TEST标签**，不因一份best成功就提前采TEST。评测全部完整后一次性提交3轮终分；任一TEST或**终分提交前**写入失败，只保留无终分的实际进度/已付计数，不单报第一轮为完整结果。若完整三轮终分已冻结后才发生可选报告读取/渲染/写入失败，遵循原RunStore不可撤销契约：保留已提交的完整终分，返回非零并标明端到端验收未通过；不撤销/重开/改写终态，不自动补跑。此时不是部分终分泄漏，不能从保存SUCCEEDED冒充原生exit0。

## 指标与基线

逐位预测正负以logits>=0，标签为两块绝对密文C||C′128位，不求XOR。

每轮保存：
- 原active8正确数/denom524288和逐位成绩（LSB0(0,21,42,60)×2，MSB0列63,42,21,3,127,106,85,67）。
- **全128位**正确数/denom8388608及128列正确数/标签0/1分布。
- 每块的x高32/y低32分别聚合：两份cipher的x半共64位分母4194304，y半同分母；同时可保存各64位块准确率。
- **128位整对全正确数/65536**，不把逐位均值当整个中间状态恢复成功。
- 每轮只用TRAIN冻结128位多数串（同票0）作为同分母对照；全/active/半块/整对同一规则评测。

SIMON一轮为(x1,y1)=(y0 XOR f(x0) XOR k0, x0)，所以r1的y半是**直接复制**：输出MSB0列32:64=P输入0:32，96:128=P′输入64:96。r1额外结构基线为“y半复制对应输入，x半TRAIN逐位多数”；无真实key或状态。r2/r3不声称同样复制规律。报告r1**非直接复制的x半64位**准确率，避免复制半块撑出约75%却被说成学会一轮。

为本轮描述性指标，不新增未批准的拟合/阈值调参。保存模型和基线每族正确数可用现paired_interval扩明确分母；默认只准确率/误差与整对结果，不必须为每项多重测试造显著性。任何置信区间须按族而不是独立bit，不能把r1的copy成功当神经贡献。

## 预算与执行

固定加密7077888blocks/null0；最多614400updates/167313408forward pairs，成功按sum(e1,e2,e3)核算updates=4096×Σe，forwards=1114112×Σe+196608。每round每split标签不重复，TEST只一次，不能为报告重加密/前向。

全矩阵从device/provider到report/exit wall10800s/CPU43200s、RSS8GiB、产物512MiB、checkpoint16MiB；GPU1 UUID `GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5`，单logicalcuda0/free≥16GiB。3fit共一绝对UTC期限，按实际早停正常结束；无法跑满/超预算则停止，不延时、不缩水/换卡、retry0。无额外真实warmup/数值预检/benchmark、安装或服务变更。

一次限定Git发布基于3dae80835666a391d09bf281804e505bf9127984，新root/source old-02/work readonly；前台SSH始终保持，脚本global LASTEXITCODE真捕获，取回只实际data/models/result/必要日志。主控独占真实运行；开发代理仅人工cipher替身/NPZ/runtime/clock/FFI/张量形状，不调用实际cipher/秘密熵/NN数值/Adam/GPU/RSS/SSH/Git/旧真实产物，不派子代理。

## 最小模块归属

- 新`low_rounds.py`负责固定矩阵数据装配/显式轮次/共享secret标签闭包与同split，不改core密码；可再拆`low_metrics.py`专属128位/half/pair/copybaseline统计，训练不塞这类数学。
- 原`training.run`共享fit/reload/predict；增加明确r1r3拓扑或小方法层调用同fit，**不复制runner或改globalARMS**，不变旧1arm/dual行为。各round独立identity与slot不可误加载。
- config新固定配方含rounds[1,2,3]/新key协议/source plaintext-only/全部budget/method；CLI显式分派同Meter/claim/3slot/失败边界，纯stdlibcheck，旧方法仍值不变。
- model/LSTM/engine计算不改，仅factory identity允许明确round label；report纯saved显示三轮全部指标/复制基线/整对成功，不拼旧singleCI或伪历史分数。
- 只扩已有启动脚本固定config allowlist/新scope/source10800；不第二PS/远程workflow/调度器。

TDD、ownerguard含底层recurrent别名、独立审查/forcedRuff/mypy/显式owner与全safe/干净closurecheck通过才发布。不以工程测试代替科学结果。

## 交付限制

低轮结果属于同新key/seed/固定面板的探索，不是整个明文空间证明；即便某8位100%也不等于64位全状态已恢复。128位整对准确率与非复制位成绩用于评估是否值得下一步桥接，不在本轮宣称提高区分轮数。原source/历史run保持，不自动第4轮/新key/新模型。运行关闭后未用预算不复用。
