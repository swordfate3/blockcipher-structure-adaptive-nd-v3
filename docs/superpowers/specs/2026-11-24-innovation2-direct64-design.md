# 创新2：单明文→对应轮数完整64位密文

**APPROVED。** 用户确认本轮修正任务：单条64位P→64位绝对C_r，r=1/2/3；正确比特总数÷全部64位，完整64位BCE训练与DEV选模，不再配对输入、不用指定8位/差分。沿用50轮上限/DEV早停/GPU1总3小时、一次限定发布运行、失败不自动重试。

## 1. 语义与模型

本轮独立拟合3个标准单向LSTM。每样本输入MSB0 `[N,64]`→`[N,64,1]`；`nn.LSTM(input_size=1,hidden_size=128,num_layers=2,bias=True,batch_first=True,dropout=0,bidirectional=False,proj_size=0)`；每次forward新建两份`[2,N,128]`零h/c，不跨样本/batch留状态。`h_n[-1]→Linear(128,64)`产生64logits，参数**207424**，不得默默用2特征配对模型。

三模型均seed31独立新初始化/独立Adam，epoch顺序seed20261124；不加载旧权重、不输入真实key/其它cipherbit/状态/结构卡/差分或族ID。模型只读单条P，目标直接C_r，不计算C XOR C′。语境仅公开SIMON减1–3轮、程序合成固定key/数据学术研究，无真实系统/凭据/第三方目标/密钥恢复。

## 2. 原低轮数据只读复用

source run=`cube-lstm-simon-r1r3-20261124-01`，runSHA=`e967e9cdc18f164f0f98c28e3eda391d6c3664c0b46d305ffc60795a9391b65d`。同一新key的三低轮标签已保存；本轮不加密/不新key、不重新采样或划分。

| artifact | bytes | SHA256 |
|---|---:|---|
|dataset/r1-train.npz|41943816|322dbf480f8ab2f282737ad69d1259e357824dcd80301c9a5bb7f9b41a53b9bb|
|dataset/r1-dev.npz|2622216|de7c8ee2e1b4b175f6fb36a1a014e3e42b5198ae5407f5a9dfb643959a09309a|
|dataset/r1-test.npz|2622216|2fea353b0d6e1a9c59cc721658a5fa2806be4c117b7ae4a3382dd06283ed1d58|
|dataset/r2-train.npz|41943816|4917189d1fdb8da0a3b259944c8441719e5b049bb2556f416ef0494b0d178202|
|dataset/r2-dev.npz|2622216|644f64f52804798dddf33075ec54c2c56c85b96b131c782aafb2c67c76327bc5|
|dataset/r2-test.npz|2622216|b48b23afcd5007b64e0e1e80e545efd7aa5c09f0f66107d9cb597a488202b942|
|dataset/r3-train.npz|41943816|d0b099ec4580d8ff943c80881cc64d4470ff685827c96f5903def95706285e55|
|dataset/r3-dev.npz|2622216|716b395534868e188b6cadeaae39df3982fab3b264a5b8d8cea53d826e3eebe6|
|dataset/r3-test.npz|2622216|eecdc2a84ca0c05fc74b93f209d64090a4373afee5db0cb4f1995aefcd991972|

源NPZ仍保存[P,P′]两个字，但仅在数据层展平为(P,C_r)单字行：依原row顺序逐个取左、右，同步展平标签，族ID重复两次。模型不可看到伙伴、配对关系或族ID。TRAIN **2097152单条**，DEV/TEST各 **131072单条**，原族131072/8192/8192不变，每族16个单条。

这是真实单明文输入/绝对标签，不是差分任务；但复用面板原先由16变体族产生，**不能声称变为重新独立均匀采样的全新总体**。原整族划分保留防泄漏；族仅统计/校验，不入模型。无需因本轮语义修正再生成新数据。

复用现严格NPZ解析/路径/源记录sizeSHA/regular/symlink/junction拒绝及族完整互斥校验，必要提取内部显式源描述参数，不复制loader、不修改旧默认描述。新source读取仅9个明确dataset、SHA前后固定run，不扫描旧模型或使用旧TEST成绩。各split三轮明文和族ID逐值对齐，出现不一致失败不fallback。

TRAIN/DEV可先读；**全部3份本轮best严格磁盘恢复成功后，才读取/解码任何源rN-TEST文件**，之后一次最终TEST。可以向新run写原NPZ bytes并明确记flatten view/provenance，而不另序列化/重采样；旧run所有文件只读，输出目录在claim前保护不落入source owneddata/models/result。

## 3. 优化、选模与主指标

- full64 BCE训练；**full64 DEV BCE**最低且最早选择best，取消active8选列（selected_columns=None）。max50/batch256，warmup10/patience8/min_delta1e-5每round独立早停，最早18；best严格最低与显著改善anchor分离。
- FP32 Adam lr.001/betas(.9,.999)/eps1e-8/clip1/weight_decay0、无AMP/TF32，CUDA确定性仍不作逐位复现承诺。3fit顺序r1→r2→r3、共用同一epoch行排列函数，不跨轮warmstart。
- **主准确率＝正确位数÷(131072×64)=正确位数/8388608**。每轮报告64列correct/标签0与1、总体full64、x高32/y低32半块（各4194304位）及TRAIN全64多数基线，同票0。不再以active8为主或进行active8DEV选择；旧八位指标不必保存。
- 每条64位全部正确数/131072作为附加诊断，不作为主准确率或训练/选模依据，不要求全串正确才算样本得分。可保存族聚类计数，不把16条×64位当独立抛硬币。首轮copy结构基线y1=x0，y列32:64复制输入0:32，x用TRAIN多数，明确75%左右来自复制非神经贡献；r2/r3不套copy。
- 三轮TEST完整计分后统一终分，不提前发布r1单轮终分。终分前故障无partial；完整3轮冻结后可选report失败保留不可撤销完整记录/native非零，不能用保存SUCCEEDED替退出成功。

## 4. 单次总预算

**0 real_blocks / 0 null_words / 0秘密熵 / 0数值预检**，仅3个新fit。最大updates=**1228800**，最大forward_rows=**334626816**（单条明文为一row，不是旧pair）。实际成功`updates=8192×Σe`，`forward_rows=2228224×Σe+393216`；3模型正常停止18..50，最大仅cap不要求用满。

全矩阵wall10800s/CPU43200s、RSS8GiB、artifact512MiB、checkpoint16MiB；同一Meter/UTC从device到报告/exit共用，batch256固定。仅物理GPU1 UUID `GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5`、singlevisible logicalcuda0/free≥16GiB；不动GPU0/其它任务、不装依赖。一次run任一FAILED/STOPPED/UNKNOWN停，retry0、不换卡/缩水/加轮/key/seed。

用户许可包含本次实现/人工验证/审查、一次限定Git发布/远端exact checkout、唯一GPU运行及取回。源码不经SSH传包，程序本身不做SSH/Git。前台SSH保持到真实退出、global LASTEXITCODE立即捕获，缺证据UNKNOWN；旧UNKNOWN不回填。

## 5. 模块范围与兼容

基于2253124bfcdc0eaddc256d350a78d5b414c4c356隔离archive，main/index/旧产物不动，core/原模型/旧方法行为保留。

- 新`direct64.py`：单条Panel视图/固定9artifact读取provider（可再拆direct_metrics.py统计）。严格解析复用reuse内部函数；不得对旧source常量做全局monkeypatch或复制loader。
- 新`lstm64.py`：单输入/64输出纯模型，不读文件/训练/配置。
- `engine.py`：显式64宽runtime参数或专用factory使用原TorchEngine优化实现，原128宽默认不变，不复制Adam逻辑；新round身份/3slot不能与旧128checkpoint混用。
- `training.py`：新明确direct64-r1r3拓扑复用原fit/reload/predict/统一三臂终分，full64 DEV/单条count；必要小方法函数归属清楚，不复制runner或构建registry。
- `config/cli/report`：固定direct64新配方/源描述/3h/identity/primarymetric/expectedcount、纯stdlibcheck与preclaimoverlap保持。失败报告不得把计划写作事实、无终分不声称逐族结果；只saved值制图。
- 原PS只扩fixedconfig/source allowlist，前台/globalnativecapture不变。开发和测试禁止真实cipher/熵/NN数值/Adam/RSS/GPU/SSH/network/install/Git或旧真实dataweights读取，沿用补齐LSTM底层别名/ATen缓存guard。

TDD→任务review/fix→主控fresh/whole review/forcedRuff no-respect-gitignore/mypy/显式owner+全safe/干净closurecheck后才发布。新tests名称全局唯一。结果取回只核原始SHA/身份/计账/真实exit，不load模型/解码数据/前向/再生标签/evaluate。

## 6. 结论边界

本轮同时修正输入输出样本单位与DEV目标，故不能把新旧分数差仅归因LSTM层或单一变量。旧TEST已报告，仍是固定同key/seed/面板的适应性探索，不是独立确认/全域证明；本轮无差分输入或标签，但面板来源结构须如实披露。不因第1轮75%复制基线宣称密码被削弱。完成即关闭，不把剩预算自动给第4轮/新模型。
