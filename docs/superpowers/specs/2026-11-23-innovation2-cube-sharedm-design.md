# 创新2：固定密钥、16变体大数据与 Shared-M 位预测对照

**状态：APPROVED / 用户在本设计与预算、发布范围确认问题后回复“是”（2026-11-23 UTC）。**

已确认方向及本轮许可：远程物理 GPU 1、两臂各5epoch/batch256、预检至进程退出合计2小时、失败不重试、一次限定代码Git提交发布/远端同步、唯一小预检、正式运行与结果取回。只读环境核对已完成；批准不是已实现、已发布或已开始训练，也不扩大为安装或其他实验许可。

## 1. 目标、范围与备选方案

研究问题：固定未知密钥的 SIMON64/128 **7轮**，模型仅输入 `P || P′`，输出完整 `C || C′` 的128个 logits。只以预定8个输出位置的逐位正确率作为主准确率；不做密文异或求和、差分分类或密钥恢复。

已讨论三种方案：①只放大 MLP 数据（不能判断 Shared-M 效果）；②只换网络、保留小数据（不能检验用户的大数据建议）；③两网络在同一大数据上对照。用户“测试下”及随后“可以，你推进使用吧”按③推进准备；以下数值/细节由主控提案并随本设计获用户确认，不冒充用户最初逐项指定。

本轮只比较 MLP 与新初始化的 Shared-M 位预测版本，共2个科学 fit。不自动追加8轮、key、seed、损失函数消融或独立确认。只研究程序生成的合成样本和一次性私有随机密钥，不涉及真实部署/凭据/第三方目标；单密钥减轮结果不外推为现实密码被破解。

## 2. 明文族、标签与密钥

- 全64位整数的激活位置为 **LSB0 `(0,21,42,60)`**，mask=`0x1000040000200001`。输入/标签位表使用 MSB0；评分列为 `[63,42,21,3,127,106,85,67]`，报告同时标注两种编号，禁止混用。
- 每个随机64位底明文清掉上述4位，得到族标识 `base`；枚举 `t=0..15`，将 `t` 的4位依次填入上述位置，真实构造16个不同明文。
- 取 `t=0..7`，与 `t XOR 15` 配成8个互补明文对。每个变体只出现在本族的一对中；不添加反向重复行。每对均满足 `P XOR P′=mask`。
- 族标识跨 TRAIN/DEV/TEST 全局唯一。同一族只能归一个划分；输入、标签、族ID整行同步打乱，训练每epoch按固定seed派生顺序，两臂使用完全相同顺序。
- TRAIN **1,048,576对=131,072族**；DEV、TEST各 **65,536对=8,192族**。TEST评分分母每臂 **65,536×8=524,288位**。
- 同一次随机生成的128位私有主密钥用于全部划分和两臂；不每千对换钥。密钥只保留在数据标签提供者闭包内，模型/诊断不接触，不写公开产物、日志或配置。
- 两臂共用同一份数据及标签，不为各臂重采样。保存紧凑明文/密文 uint64 数据与族身份/hash，以后相同数据契约的网络对照可复用；私钥不保存，不能靠再生成假称复现。
- TRAIN/DEV标签在拟合前生成；TEST明文可提前准备，但**两臂 DEV best 均从磁盘严格重载成功后才生成 TEST 密文标签**。公开报告必须承认最终 TEST 会被保存/看到，不称强制密封确认。

## 3. 模型和公开表示

### MLP基线

沿用现 `DirectBlockMLP(128,128)`，两隐藏层宽128、ReLU，约49,536参数；新初始化，不加载历史权重。

### Shared-M位预测版

保持创新1原模块不变。复用其三个邻域 stem（local/word/nibble）、仅由 SIMON public13 卡决定的 learned top2 router、96通道 GroupNorm 和5个 residual blocks（kernel 3/5/7/9/11）。

专属适配包括：

1. 输入是明文，不是密文；canonical通道仅为 `[P,P′,P XOR P′]`，无真实中间状态。原密文两轮回退关系不用于此处：关系通道全零，不偷偷引入额外算法特征、密钥或教师。
2. 四pair形状只用于现有逐行组件的计算打包，不能平均/混合四对或整族以形成一个判决；逐对特征完整保留。
3. 原单分类输出替换为**逐对128位 dense head**：`Linear(96×64,128)` 加 `Linear(96×64,64)→ReLU→Linear(64,128)` 残差；最后残差层零初始化。输出的前后64位分别对应 C 和 C′。
4. 新模型独立初始化seed31；保留原stem相同初始网络状态的规则。MLP同样seed31，但不同架构不声称逐参数相同。

静态算术估计：Shared-M适配后约 **1,876,179参数**；约44.1M MAC/对的前向（不含norm/激活、反向/临时workspace），相比MLP容量及计算量都显著更大。因此结果比较是“架构+容量”的候选比较，不把任何增益归因于routing一项。精确参数数由实现验收核对，显存和速度尚未实测。

## 4. 训练、选模与统计（本轮固定配方）

- 两臂各 **5 epochs，batch256**，均为FP32、Adam `lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=0`、grad clip1，固定学习率。数据seed20261123、epoch排列seed20261124、初始化seed31；这些是主控建议，不冒充用户逐项指定。关闭AMP/TF32，不强称跨Torch版本逐位可复现；记录实际精度策略及确定性设置。
- **全128位 BCE 训练**；DEV以预定8位 BCE 选最低，完全相同则取最早epoch。全128位监督与只训练8位不是同一方法，本轮不混用，也不擅自改成8位输出。
- 每epoch仅一次完整DEV，共5次选择机会/臂，无额外 early stopping。先MLP后Shared-M顺序执行，单GPU，不并行抢占另一张卡；两臂拟合完成且均持久重载后统一TEST。
- 保存训练loss、DEV active/full BCE、选中epoch、严格重载身份；TEST保存8位正确数/多数基线正确数/标签0与1个数，以及每族两模型和基线的正确数。辅助128位指标不得替换8位主指标。
- 多数基线仅用TRAIN逐位置统计，同票0；固定8位串，与模型用同一524,288位分母。
- 置信区间以**整族作为聚类单位**，使用保存的8,192个TEST族分数的配对差和标准误；不把族内8对×8位视为独立Bernoulli试验。两个预定比较为Shared-M减多数基线、Shared-M减MLP，各报97.5%正态近似双侧区间（z=2.241402727604947），对这两个比较作Bonferroni族错误率5%控制；其余模型/逐位置结果描述性呈现。本轮只是单key/seed探索，不宣称独立确认或跨key结论。
- 报告两网络对基线以及Shared-M对MLP的全部差值；正结果需预定族聚类差值区间支持，仍只属探索；无增益则仅否定本配方，不声称所有网络不可能。TRAIN拟合和DEV/TEST泛化分开呈现。

## 5. 预算、工程门禁与停止

### 科学批次精确计数

- 真加密：`2×(1,048,576+65,536+65,536)=2,359,296 blocks`，共享标签只生成一次。
- 更新：每臂 `5×1,048,576/256=20,480`；合计 **40,960 updates**。
- 前向：每臂 `5×TRAIN+5×DEV+TEST=5,636,096 pairs`；合计 **11,272,192 forward pairs**。不额外重算TRAIN选中模型指标，不自动evaluate。

### 唯一真实工程预检（已随预算批准）

本地开发仅人工oracle/假runtime/合成张量结构测试，不调用真实SIMON、秘密熵或科学拟合。远端已发布代码后允许一次小预检，固定公开KAT及算术对照：44轮官方KAT scalar/batch各1块，7轮固定公开key下8块scalar/batch各一次，共 **18 blocks**；不使用私有科学key。

GPU数值预检使用确定性人工标签、8对输入，两模型各1次update并checkpoint读回后各一次predict，共 **2 updates / 32 forward pairs**，不是科学fit、不报告其ACC。任何错误即停止，不自动补跑。

含该预检的总上限：**2,359,314 blocks / 40,962 updates / 11,272,224 forward pairs / null0**。

### 时间、资源和失败

建议预检至科学进程退出合计wall上限 **2小时**，CPU累计上限 **8小时**（至多4线程，interop1），主存RSS协作停止阈值8GiB、产物总额512MiB，固定batch256。启动时指定GPU1须至少16GiB空闲；实际GPU显存峰值未实测，记录本进程allocated/reserved峰值但不冒充物理显存硬限。OOM停止，不减batch/换卡再跑。

使用既有Meter/RunStore的进程内预算与UTC deadline；保存SUCCEEDED不替代真实进程退出。远端只用一个必要的启动脚本记录CLI原生退出码及基本日志，仍有进程/退出证据缺失时标UNKNOWN。不新增远程调度器/看门狗、多套预检与重复证明文件，不把SSH协作超时说成OS硬限。

正式重试 **0次**；首次FAILED/STOPPED/UNKNOWN、身份不符或资源冲突即停止本轮，不自动改参数、换卡、追加训练或补TEST。用户取消时只处理本次确认的进程，绝不按python镜像名批杀。未使用额度关闭，不转给新实验。

## 6. 模块归属与实现范围（架构只读审查后）

新增 direct_block 下职责明确的 `cube_compare/` 子包，显式插件CLI接入当前 `python -B -m cipher_lab check/run --plugin …`。不复制core runner、Meter/RunStore，不改原CPU固定配方或创新1生产行为。父层fit/reload的必要共用逻辑提取成具名小模块，只接受显式epoch/batch/DEV选列/checkpoint上限与依赖，不导入旧loader/mode分派；现旧training通过它保持原默认行为和测试。新子包只导入这个共用件，不导入绑有旧loader/mode的training.py；不再用额外loader的isinstance分支识别本任务，不建立通用registry。新入口直接装配GPU，不经旧CPU CLI设置CUDA_VISIBLE_DEVICES=-1。

- config：本轮固定配方、数据/模型身份和预算，无采样/运行。保存的rounds=7、single-key协议、input/target宽度及族划分必须与实际provider逐项一致；人工测试显式拦截旧PAIR4里recipe仍显示rounds6/单key、实际loader算r7r8/换key的误登记，旧产物不追改。
- data：16变体、互补配对、族级划分、紧凑保存与同key标签提供者；只此层持有私钥。
- model/representation：仅公开输入与参数/前向；复用core Shared-M组件，不依赖训练/CLI/配置文件或产物。
- runtime/checkpoint：设备适配、Adam与严格身份/shape/dtype/finiteness/fixed-buffer加载；不能生成标签或调度SSH。每个checkpoint上限16MiB，保存CPU model-only state（含固定整数buffer），不用旧256KiB/仅浮点validator；包括best替换时新旧并存的产物预算。来源指纹递归覆盖core及direct_block本owner，不直接import full_difference/innovation1等兄弟插件形成漏记依赖。Windows RSS使用本owner明确的OS采样适配（参考既有GetProcessMemoryInfo），不调用只支持Linux的默认production_rss_bytes，也不把不支持当0。
- training：相同数据/顺序的双臂fit、DEV最早最低、双重载与TEST时序；不塞16变体数学/统计报告。
- metrics/report：显式输入的逐位/逐族统计，报告只读保存值；不生成标签/前向或更改选模。
- CLI：参数、claim、装配、错误与退出呈现，无算法或远程编排。

实现默认由 implement-deep 代理承担，禁止再派子代理；先TDD、人工全闭环、Ruff/strict mypy、owner及相关回归、独立code-review与主控fresh验收。已有实际多mode接线错误作为负面用例，须覆盖128输出、计数、权重槽replace和重载失败不访问TEST。

## 7. 发布、远程运行和交付权限

本轮已许可：方案实现与人工验证、只读SSH、物理GPU1、上述5epoch/2小时配方、唯一数值预检，以及**一次本地限定文件Git commit/push→远端fetch/精确checkout→启动→取回**。用户确认后实施；不自动重试或额外发布，不改远端其他服务，不安装依赖。

拟基于既有已发布提交 `c7320f25aee8fa39700ca20b98c05bb14029c993` 创建本次专用release，只加入审查通过的本轮最小文件/必要依赖。该base有core/Shared-M但没有direct_block；验收对象明确为“c7320原样core＋逐文件允许清单”，仅增加新cube_compare子包及其实际需要的父包入口/fit共用件/模型或固定规格/来源模块，不顺带发布旧loader、旧CLI或全部mode。新owner不能import绑有所有旧loader的父training.py。以该base为准的隔离源码副本验证完整导入闭包、双best槽replace/reload和SIMON批接口，不允许PYTHONPATH回落主树。主工作区main HEAD与真实index保留；不把数百项无关未提交改动、core删除或历史结果一起发布。若该base缺必要代码，先列最小依赖，不默默改发布范围。

Windows只能在 `G:\lxy\shared-m-cipher-nd-runs\<本次新id>\` 下新建检出/产物/TEMP/cache，不修改运行中的旧检出。复用 `C:\Python313\python.exe` 的现有user-site依赖（`-I`会隐藏这些包，不能用于实际启动），禁字节码，不安装或升级。GPU1用UUID `GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5` 绑定，进程内成为`cuda:0`，不把逻辑0当物理0；启动前核一次当前占用，不停GPUStack/现有训练。

结果只取本次data/models/result与必要日志，不传科学源码任务包、不在远端commit/push、不用Git回传产物、不自动evaluate。遵守轻量交付约定；用保存值生成必要图解/结果图及报告，并只服务结果目录，不新开放公网入口。

## 8. 当前实测与下一门槛

2026-11-23只读检查：主机DESKTOP-BBLPACJ，GPU0忙、GPU1为170MiB used/48,402MiB free/0%；现Python3.13.3、Torch2.7.1+cu118、NumPy2.4.0，包在user-site；G盘约4.2TB空闲。最近旧检出在c7320f25且git status干净。包metadata不等于CUDA/本次模型已验收，GPU空闲不等于独占或预约。

只读architect审查已完成：支持明文canonical＋零关系通道＋逐对dense128头；核实四槽仅打包不汇总、checkpoint/Windows RSS/源码闭包风险。进一步builder check要求收窄到“原样base core＋逐文件允许清单”，已纳入§6–7；不发布全部旧mode。参数手算主控按原1087509参数标量模型交叉核对，新头整网1876179，仍待实例验收。

**当前门槛：writing-plans → 委派实现 → 独立审查与主控fresh人工验收 → 限定发布/同步 → 唯一预检与正式运行 → 保存值交付。** 用户许可已取得；此状态更新时尚未改产品代码、发布、数值预检或启动本轮训练。
