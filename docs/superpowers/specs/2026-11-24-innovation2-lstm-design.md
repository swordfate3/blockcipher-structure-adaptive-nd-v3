# 创新2：固定面板上的单向LSTM位预测

**状态：APPROVED。** 用户对标准LSTM固定提案回复“是”：64步每步(P_i,P′_i)、2层hidden128、最终隐藏态→128输出、零初态且不跨样本记忆；复用原数据/50上限和早停、GPU1/3小时、一次限定发布运行/失败不自动重试。

## 1. 研究问题与方法选择

把明文的位位置当作序列，检验LSTM的门控递归表示能否在同一单key、SIMON64/128七轮、原百万对固定数据上产生超越TRAIN多数基线的位预测。不是预测真实时间，也不是跨样本下一条预测。公开算法、本地合成数据、无真实系统/凭据/攻击目标，不做密钥恢复。

比较过：①标准单向LSTM（用户选定，最小序列基线）；②双向LSTM（额外方向和参数，后续才可单独申请）；③Transformer（非本轮用户选择）。本轮只有1个新LSTM fit，不额外拟合MLP/Shared-M；旧模型已保存成绩只作描述性参考，不冒充同期参数/算力匹配消融。

## 2. 模型合同

- 每条输入仍是MSB0 `P || P′`，FP32 binary `[N,128]`。显式转换为 `[N,64,2]`，第i步两特征是P_i/P′_i，不能误reshape成相邻两位配对。
- `nn.LSTM(input_size=2,hidden_size=128,num_layers=2,bias=True,batch_first=True,dropout=0,bidirectional=False,proj_size=0)`；不添加结构卡、其他通道、公开回退、历史真实密文、轮状态或key。
- 每次forward显式新建两层全零h0/c0 `[2,N,128]`，不保存输出hidden作为下批初态，样本互相独立。取`h_n[-1]`经过`Linear(128,128)`得到完整`C || C′` logits。
- 参数总数**216192**，由实现结构实例核对。不是已有Shared-M与LSTM拼接。只接收输入与可学习参数，不读数据/配置文件/产物/训练器。
- 新初始化seed31，CPU隔离初始化后转指定GPU；epoch shuffle seed20261124。数值策略仍FP32/无AMP或TF32、Adam lr.001/betas(.9,.999)/eps1e-8/clip1/weight_decay0，不加载旧权重。

依据：PyTorch LSTM官方API及已安装`torch/nn/modules/rnn.py`中batch_first/input/h_n/zero初态约定。官方网页抓取失败，改只读本地源码；未进行网络数值测试。LSTM门控提供组合与记忆归纳偏置，不保证适合密码或超50%，也不把语言/时序建模成功外推本任务。

## 3. 数据、选模与TEST

直接复用先前已校验的 `cube-sharedm-simon7-20261123-02` 三NPZ及原划分，源runSHA=`bc2c87a5e2ab919021c7f4cdf556830f99b59380b858352ae72d30c93669f523`。TRAIN41943816bytes SHA=`e01ca03e68a3feaaa98553cbb2d7662bdd943a31e304cc7abcec4ccaaaa1cf8a`；DEV2622216bytes SHA=`116ebebf97f4a1df3685f53bc9aa02d98f8d8d58ab202d3e85477f0ba5b6f2a3`；TEST2622216bytes SHA=`f00dfe78821c28024b826eb8ad8444ebf1acd37e9525001061fe4b9c4f990ddc`。

沿用已发布 `SavedPanels`严格路径/字节SHA/NPZ header/uint64/C-order/族完整和划分互斥读取。TRAIN1048576对(131072族)，DEV/TEST各65536对(8192族)，激活LSB0(0,21,42,60)，16变体8互补对/族。数据key不变、不读取key或采样熵、不重新加密或生成标签。

完整128位BCE训练；DEV指定8列MSB0(63,42,21,3,127,106,85,67)每epoch一次最早最低选模。max50、batch256、warmup10/patience8/min_delta1e-5复用core单版本EarlyStopping，最早18轮停止；best选择独立于min_delta。LSTM唯一best从磁盘size/SHA/严格加载成功后才读取/解码源TEST。没有第二模型或伪造双臂来满足旧接口。

评分分母65536×8=524288位；TRAIN冻结多数同票0。保存LSTM/多数逐位正确数和每族0..64正确数；主要对比LSTM−多数按8192族配对，沿用既有paired_interval的**97.5%区间**（单项比较保守口径，不伪称有两项新主检验）。历史MLP49.9969%/Shared-M49.8363%只作保存参考，来源run `cube-sharedm-simon7-50epoch-20261124-01` SHA=`befe14e5fb467412965d98cc0f121a827370d82860d9399e8cf4df145ff9ef72`；不重新前向或从aggregate捏造对它们的新配对CI。

旧TEST已见，本轮是适应性探索，不能称独立确认。正结果也需另授权新面板确认；阴性只否定本配方，不声称所有LSTM/网络不可能。

## 4. 单次预算与许可

- **1个新fit**，最多50epoch；max updates204800、max forward_pairs55771136(50×TRAIN+50×DEV+TEST)；blocks0/null0/秘密熵0/数值预检0。
- 实际成功按e完整epoch：updates=e×4096；forwards=e×1114112+65536；e最小正常早停18，最大50。不要求用满cap，不把资源失败当早停成功。
- wall10800秒、CPU43200秒、RSS8GiB、产物512MiB、checkpoint16MiB；固定batch256。新run，新目录；首次FAILED/STOPPED/UNKNOWN停止，retry0，不换卡/缩batch/加模型/seed或加时。
- 只用物理GPU1 UUID `GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5`（单可见logicalcuda0），保留已有device/free>=16GiB准入/WindowsRSS/预算。CUDA/GPU数值只在批准的正式fit中发生，不额外跑免费warmup/benchmark/微训练。
- 本轮允许本地实现/人工测试/独立review、一次限定Git发布/远端exact checkout、前台SSH持续保持直至真实退出、取回保存结果。无安装/新数据/旧模型训练或evaluate。只读检查GPU冲突则停，不停其它任务。

## 5. 实现边界与兼容

从已发布791145ba734ef8de8d33b9b25036ca1e85a9709e隔离源码开发，main/index/旧数据权重报告不动。core和旧Shared-M/MLP模型不改，旧5epoch/双臂50reuse配方/行为保留。

- `cube_compare/lstm.py`新：唯一LSTM模型及序列适配，纯参数/forward。
- `config.py`：新增具名LSTM-reuse配方（描述模型/单臂/source/earlystop/caps），旧recipe不变。
- `engine.py`：新增明确LSTM工厂/identity，复用已有TorchEngine的128BCE/Adam/预测/模型保存；不把LSTM伪装mlp/sharedm。
- `training.py`与父`fit.py`：最小显式单臂拓扑，复用同fit/reload/TEST流程；旧双臂默认两best门禁保持。若公用gate支持1臂，必须只在显式本模式允许，不能让原两臂静默少一份重载。
- `metrics.py`：单臂与多数的真实计数，不把LSTM结果塞进旧mlp/sharedm字段、不用重复预测伪装第二臂；共用现有确定性多数与族区间数学，不新增通用registry。
- `cli.py/report.py`：固定recipe分派/单slot `lstm-best`/实际epoch计量/TEST及mode正确展示，失败不发布partial或伪造历史比较，保持purestdlibcheck、source-output overlap检查与报告预算。
- `scripts/start-cube-sharedm.ps1`：最小新增固定config allowlist/同3h/source-root，global LASTEXITCODE真值捕获、前台运行；不复制启动脚本、远程调度器或runner。

开发代理只能人工oracle/NPZ/engine/clock/FFI/形状测试，无真cipher/entropy/NNforward/backward/Adam/GPU/RSS/SSH/Git/install/旧真实数据权重。新guard要显式封`nn.LSTM.forward`及RNN数值路径，不能沿用只封Conv/Linear却让LSTM漏跑。先TDD，独立review，forcedRuff no-respect-gitignore/strictmypy/owner+相关回归/干净闭包CLI，主控fresh后发布。

## 6. 结果交付

取回本run实际data/models/result及必要日志，核原始SHA/源码和配置身份/实际epoch与计量/原生exit/SSHexit；不重算模型指标/再生标签/加载权重/forward/evaluate。只用保存曲线、scores/CI制图与报告，明确探索及单key/seed/位顺序和长序列压缩风险。无信号或正常早停后关闭许可，不把剩余额度给新模型。
