# Cipher Lab

独立、无旧接口兼容层的 Python 3.12 CPU 实验后端，核心保留普通积分、位预测 MLP、四任务 shared-M 和显式 evaluate；实验通过下表显式插件入口调用，科学状态与历史结果见各节。积分包含完整
2080 项观测（64 单位项 + 2016 二次项）、完整 cube 采集、固定 D/S/冻结/C 与 null
流程、协作式预算、唯一运行记录及保存结果报告。**工程演示不是实际 SIMON 实验，经验不是证明。**

## 多位差分关系预测插件（2026-11-16）

已新增独立 `shared-m-differential-relations-v1`：SIMON64/128六轮、固定输入差分0x40、四条预固定四位奇偶关系；shared-M预测模型逐对输出4个logits，原四密码分类模型保持原样。实验配置与五fit矩阵见 `configs/differential-relations-simon6.json`，设计和实施计划在仓库根目录 `docs/superpowers/`。

源码入口（本目录执行，Python≥3.12；check只校验冻结配置，不探测设备）：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins /workspace/.venv/bin/python -m cipher_lab.cli check \
  --plugin cipher_lab_plugins.differential_relations.cli \
  --config configs/differential-relations-simon6.json
```

正式 `run` 另须显式 `--root --id --device cuda:0 --gpu-uuid --gpu-pci`；一个启动脚本位于 `plugins/cipher_lab_plugins/differential_relations/start.ps1`。它要求已有兼容Python、明确物理卡和G:\\lxy范围内的新输出；不安装、不同步源码、不自动选其他GPU。科学总wall8小时、每fit90分钟、共享数据/收尾30分钟均为合作式停止检查；原生调用和最后终态IO可能迟到，需结合退出码与 `CLOSING_OK/CLOSING_FAILED`。显存为间隔采样，非连续峰值硬限制。

产物为 `data/<id>/run.json`、一个合并事件日志及紧凑NPZ，`models/<id>/`最多五best，`result/<id>/report.html`单文件。插件 `evaluate` 显式读取原run/checkpoint/dataset并核实际身份，不再生标签；取回默认不调用它。

该差分关系实验的批准周期已结束，结果为负；结果仅适用于该批准条件，不能扩大为跨 key 结论。本次仅迁移与人工/static验证，不重新运行、评价或改写历史结果。七插件源码现已分离；整合后的主控 fresh、独立审查与干净源码闭环验收另行记录，不以目录迁移宣称现场验证。

## 入口与权限

已实现 `cipher-lab check/run/evaluate`（打包入口）；未安装时在本目录使用下列源码入口，二者参数相同。
本轮只验证已有环境下源码运行，**没有安装验证**，无需为使用源码命令执行安装。

### 显式插件入口（无自动发现）

在本目录执行源码命令，统一设置 `PYTHONPATH=src:plugins`；Windows PowerShell 用
`$env:PYTHONPATH='src;plugins'`。从其他 cwd 或独立 child 调用时须传两个**绝对源码根**，
不能依赖 pytest 的 `pythonpath`（只改变测试父进程的 `sys.path`，不自动传给 child）。
已安装的 `cipher-lab` 使用相同参数，但核心 wheel **只含 src/cipher_lab**，插件根仍须显式可 import；
不增加安装、entry-point 扫描、registry 或按 method 自动选插件。插件是可信 Python 代码，不是沙箱。

|`--plugin` 模块|实际 CLI 能力|配置/额外条件|
|---|---|---|
|`cipher_lab_plugins.differential_relations.cli`|check / run / evaluate|`configs/differential-relations-simon6.json`；run 显式物理 GPU 身份|
|`cipher_lab_plugins.residual.cli`|check / run / evaluate|`configs/residual-paired-example.json` 或冻结正式配方|
|`cipher_lab_plugins.feature_probe.cli`|check / run / evaluate|`configs/feature-probe-example.json` / `feature-probe-r3.json`|
|`cipher_lab_plugins.frontier.cli`|check / run / evaluate|`configs/frontier-example.json` / r4/r5；check/run 必须 `--public-plan`|
|`cipher_lab_plugins.fourier.cli`|check / run；无 evaluate 子命令|`configs/fourier-example.json`（synthetic/demo）或 r6（simon）|
|`cipher_lab_plugins.speck_study.cli`|check / run；无 evaluate 子命令|`configs/speck-study-paper.json`；run 仅 speck-study，须现场门禁|
|`cipher_lab_plugins.integral_search.cli`|check / run；无 evaluate 子命令|`configs/integral-search.json` / `integral-search-k18k20.json`；demo/simon|
|`cipher_lab_plugins.innovation1.cli`|check / run|`configs/innovation1-fresh.json` 和 `configs/innovation1-skinny-layout-early-stop.json`，见下文创新1；run 显式 GPU UUID 与新 ID|

统一形式：`python -m cipher_lab check|run|evaluate --plugin <上表模块> <该插件参数>`。
只有列出的命令已实现，core 的统一 evaluate 委派能力不表示每个插件都有 evaluate。
无 `--plugin` 时仅普通积分、基础 prediction/shared-M 与基础格式 evaluate；旧扩展无插件命令
及 core 扩展 Python import 有意不兼容，拒绝而不 fallback。插件独立严格加载其 schema2/3/4
扩展格式，基础 evaluate 不再接收这些格式。旧 data/models/result 原位只读，不自动重算。


先自行建立完整执行策略 `/tmp/cipher-policy.json`（路径可自选，示例不是授权）：

```sh
cat > /tmp/cipher-policy.json <<'JSON'
{
  "schema": 2,
  "optimizer_updates": 0,
  "forward_rows": 0,
  "wall_seconds": 120,
  "cpu_seconds": 120,
  "deadline_utc": "2099-01-01T00:00:00Z",
  "real_blocks": 24,
  "null_words": 24,
  "rss_stop_bytes": 268435456,
  "output_bytes": 20000000
}
JSON
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins /workspace/.venv/bin/python -m cipher_lab check \
  --config configs/integral-example.json --policy /tmp/cipher-policy.json
```

基础 `check` 仅支持普通积分、基础位预测与四任务 shared-M；实验须按表显式选择插件。
`check` 仅读配置并输出静态预算估计；公开组合前沿的 check/run 另须 `--public-plan PATH`，
有界读取并核对公开计划身份，不建 adapter、不导入科学包。基础入口及非 frontier 入口拒绝该 flag。
不创建 run、不导入Torch/NumPy/采集/密码/模型，不启动 validity 或任何科学控制。成功只表示配置合法，不表示截止时间尚未到达、资源足够、
获得实验许可或科学有效。不存在 `--dry-run`，检查生产配方也使用上述同一个 `check`。

获得相应操作许可后，可运行无密码 demo（输出位置必须显式给出，编号不得占用）：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins /workspace/.venv/bin/python -m cipher_lab run \
  --config configs/integral-example.json --policy /tmp/cipher-policy.json \
  --root /tmp/cipher-demo-output --id engineering-001 --backend demo
```

`--backend demo|simon|ciphers` **必须显式给出，没有默认值**。demo 使用公开确定函数与明确无秘密的工程字节
序列；不同组 fake key 字节不重复，所有 blocks/words/groups 都是**模拟计数**，RSS 模拟为 0，
不是生产 RSS 验证。仍按配方采完整 cube、固定 2080 项，绝不为演示降低维数。
小样本有巨大经验核，null 可报警；`ALARM` 非零退出是正确结果，不会伪造演示科学成功。
工程数学自检只验证软件接线，不能替代未来科学正/负控制。

生产入口语法与上面一致，只将显式参数改为 `--backend simon`；积分分支会使用
`secrets.token_bytes`、`encrypt_batch` 和 `production_rss_bytes`，**是真实加密入口，不是演示**。
训练/标签侧仍用原 `encrypt_many`，不随积分批量接线切换。
已有单次 §11.6 工程门禁验证了 SIMON 官方44轮 KAT、标量/批量及完整2080聚合一致性，
并保存了有限整cube采集与当前RSS观察；该许可已经关闭，不可重跑或花剩余额度。
这不代表所有密码家族已有 KAT，也不代表正式搜索、长journal/GF2预算通过或科学有效；
不能由人工数组或少量整cube耗时外推全规模可运行性。
RSS 实现仅支持 Linux procfs，不支持的平台/读取失败 fail closed；**Windows 暂 unsupported**，
不承诺可用。检查是协作式的，不是 OS 硬资源隔离。真实加密/科学实验、正式训练、GPU、SSH、安装与
发布仍须单独明确授权；策略文件、命令示例和历史批准均不是许可令牌。

## A1 批量能力与当前边界

当前本地开发为 §11.9 高激活位18/20支持与无密码人工验证，**无科学采样或训练更新许可**；
不继承历史阶段2/3的小更新许可或已关闭的 §11.6 工程额度。
已接通的位预测/shared-M/evaluate 是产品能力，不是本次可运行授权。

- `aggregate_batch` 接收严格一维、native uint64、C连续的0..65536点数组，仍计算完整
  64+2016=2080项/MSB0观测；64个bit列先packbits，二次项逐列AND后求parity，绝不先XOR再相乘。
  不建立点数×2080矩阵；原标量聚合保留作小人工数组独立参考。
- `CubeCollector` 显式选用 batch oracle，按 `chunk_points` 完整枚举cube，私有null按块读取，
  每个输出word仍扣一个单位。预算先reserve，只有有效完成块才complete，失败attempt不退款。
  demo使用明确的fake batch/工程字节适配，不调用真实密码或私有随机。
- 求核、空间交与完整basis确认接入数学协作检查，回调中断原样传播，不当作空核/False。
  数学循环与资源检查都是协作式，**不是 OS 硬隔离或硬实时限额**。
- 仍是同一个积分method、Meter和同一个 RunStore；普通 Recipe 的 D/S/冻结/C/null 行为不变。
  新 SearchRecipe 的 D空跳S/C、S空跳C 属于明确搜索语义，不改变普通方法。
  source SHA、ownership/tamper校验、durable行保存与临时峰值输出预算均未降级。
- 人工成本检查只记录wall/process CPU、数组nbytes/形状；512MiB仅设计内存参考，未读实际RSS，
  不是实测峰值或硬内存保证。正确性unit回归也不是吞吐benchmark。

A1工程门禁不是11轮或11—14轮科学结果；A2首轮正式搜索已完成并关闭，A3认证未接入。
本次开发实际科学采样为0，真实cipher/科学RNG/当前RSS/optimizer更新均为0，无新性能测量。
正式科学执行仍须另获一次完整有界授权；配置和静态预算不授予实际采样额度。

## A2 两档积分搜索（首轮周期已关闭）

`configs/integral-search.json` 是完整独立 SearchRecipe，method=`integral-search-v1`：
rounds=11/12/13/14；patterns 为 MSB0 的 0,2,…,30 和整体+32；
discovery_active=10、expanded_active=16、D/S/C=2112/320/256、chunk_points=65536。
不从名称猜参数，不降低完整64+2016=2080项观测。

```sh
# 仅读配置；上述示例小策略会显示预算不足，但不会采样或写run。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins /workspace/.venv/bin/python -m cipher_lab check --plugin cipher_lab_plugins.integral_search.cli \
  --config configs/integral-search.json --policy /tmp/cipher-policy.json
```

`run` 须同样给出 `--plugin cipher_lab_plugins.integral_search.cli`，其余参数与普通积分相同，仍显式 `--backend demo|simon`、`--root`、`--id`，不创建 models；
**本说明不授权运行正式配方**。demo是人工工程计数，simon会触发真实密码、私有采样及RSS。
静态最坏计划为12 cells（8 discovery + 4 conditional expanded），real/null各32256组、
各726663168 blocks/words，optimizer_updates/forward_rows=0。
D空或S空早停、null报警与门控只影响保存的实际计数，不能作为预算降低保证。

先按轮次和pattern顺序完成8个k10候选。每轮仅用real DS维数选最大者；正维平局固定pattern0，
两者都0时取较小D rank，再平局固定0。不读C/null作选择；这是探索内预定选档，不是独立确认。
即便全部k10阴性也从11轮k16启动；每个expanded仍做real与独立null。
real非CONFIRMED停止后续高轮，记NOT_STARTED_GATE；任何非空null全通过即ALARM停全局。
报告展示保存的tier、selection维数/ranks/规则、阶段实际组数和跳过状态，不重新选档或重求数学；
NOT_STARTED_GATE/未访问/partial不是阴性，不补0。SUCCEEDED仅表示执行完成，不保证存在关系，
经验不是证明、显著性或认证。正式全规模含长journal/GF2可行性仍UNKNOWN。

首轮已关闭的旧几何搜索只访问r11—14 H10/L10与r11 H16，real/null发现核均空；
更高轮k16未启动，不称阴性，不重用旧许可或剩余额度。

### 高激活位新几何（单次科学周期已完成关闭）

`configs/integral-search-k18k20.json` 是新增完整 schema1 SearchRecipe，仍用同一method/Meter/RunStore：
rounds=[11]，升序patterns为H=(0,…,19)、L=(32,…,51)，discovery_active=18、expanded_active=20，
D/S/C=2112/320/256，chunk_points=65536，完整2080项不变。
这是**连续位置新几何族**，不是旧P16的自然嵌套，也不是只改变位数的纯对照；不使用真值选位置，
不声称文献最优或旧关系可继承。

先完成H18/L18及各自独立null的必要D/S/C；只用real DS维数、两者空时D rank、平局pattern0
选择一侧20。即便18全空仍启动选中的20（null报警/资源/身份/错误停止除外）。共3个cell，
不补另一侧20、不升至12—14轮。k18每组262144点/4块，k20每组1048576点/16块，
每块最多65536点，惰性枚举，不缓存整cube或构造2^k×2080矩阵。

|每个real/null侧|组数|blocks/words|
|---|---:|---:|
|三cell完整D+S+C最坏上界|8064|4227858432|
|三cell均仅D时|6336|3321888768|

以上是配方静态成本，不是自动执行许可。随后用户明确批准的单次周期
`integral-k18k20-20261109T033827Z` 已正常exit0：H18/L18/H20各real/null完成D2112，
保存rank2080/核0，S/C因空核跳过；每侧实际3321888768单位，保存wall940.745240s。
L20及更高轮未测，不能称阴性；未用额度不能复用，详见同run的result三报告及spec§11.9。
人工验证中k18用完整聚合、k20只测结构/计账，不替代生产密码认证；本次科学运行也不是全key证明。
可用现有静态入口检查（不采样/写run）：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins /workspace/.venv/bin/python -m cipher_lab check --plugin cipher_lab_plugins.integral_search.cli \
  --config configs/integral-search-k18k20.json --policy /tmp/cipher-policy.json
```

## 设置、计数与退出

配置严格完整，拒绝未知/重复字段、布尔整数、NaN/Infinity、超过 64 KiB 的 JSON；不继承旧配置。
配方 schema=1，method=integral-empirical-v1；1..64 个互异 cell，rounds=1..44；active 为
同一个 32 位半字内的 2..16 个互异 MSB0 位（**普通Recipe仍拒绝17..20**）；
采集合法类型Cell与SearchRecipe的pattern支持同半区2..20位，21位仍拒绝。
SearchRecipe保持两个不同、各自严格升序的tuple pattern，档位须满足2≤discovery<expanded≤两pattern长度，
严格JSON/类型/同半区规则不变。chunk_points=1..65536；D/S/C 各 1..1000000 组。
每组 `2 ** len(active)` 点；real_blocks 为明文块次数（不乘轮数），null_words 为逐点 64 位字数。
`check` 估计完整 D+S+C；空空间可跳过确认，实际 attempted/completed 由采集器唯一计数。
Policy独立使用 **schema=2**：原7字段加schema、optimizer_updates、forward_rows，共10字段全部必填，
不接受旧/中间策略JSON。时间限制正有限、UTC截止带Z/+00:00，其余额度非负；
输出还须足够保存完整基及三份报告。积分不消耗optimizer_updates/forward_rows，可设为0。

### 训练配方与执行

`configs/prediction-example.json`、`configs/residual-paired-example.json`、
`configs/feature-probe-example.json` 与
`configs/shared-m-example.json` 均为独立recipe schema=1，与Policy版本无关。可将上述check命令的
config分别替换为这些路径；残差与 probe 必须同时按表加对应 `--plugin`，不能只换 config；示例策略中的训练预算0会如实显示训练额度不足，但check仍成功，不审批执行。
四种方法 run 已接通，必须另给足够的执行策略。示例不是训练或科研授权。

位预测method=bit-prediction-v1：SIMON64/128，rounds1..44、MSB0 target_bit0..63、
label_mode=cipher|null；MLP两层等宽隐藏层hidden_width1..4096。
四任务method=shared-m-four-task-v1：任务声明固定顺序SIMECK/PRESENT/SIMON/SKINNY，
各自轮数上限44/31/44/32，keybits128/80/128/64，均64bit块；rounds与非零uint64差分可配置。
shared模型stem_width/channels/norm_groups/head_hidden/router_hidden为1..1024，
norm_groups整除stem_width和channels；suffix_kernels为1..16个1..31奇数。

TrainSpec统一包含train/dev/test/batch_size/epochs/learning_rate/init_seed/sample_seed/shuffle_seed：
各split 1..2000000行，train必须被正整数batch_size整除；shared各split另限2..65536偶数组；
epochs1..100000、learning_rate正有限且≤1。三个公开seed均0..2^32-1，仅用于初始化/采样/shuffle，
不包含私有key seed，也不得派生密钥。纯typed模型/TaskSpec由模型/公开表示消费，不读取配置文件。

静态预算：令T/D/E为train/dev/test行数、k为epochs、b为batch_size。
位预测updates=k*T/b，forward_rows=k*T+k*D+T+D+E；cipher分支real_blocks=T+D+E，
null分支每bit标签预留一个null_words（不是64个bit合用一字），不加密。
shared的updates和forward_rows均乘4任务，real_blocks=8*4*(T+D+E)，null_words=0；
一group含4pair/8块，但模型前向行不额外乘4pair。
Meter四单位分别保留attempted/completed，失败attempt不退款，超额使用固定公开停止原因；
check不调用Meter、不采当前时钟/RSS，也不判断deadline是否过期。

`run` 输出 JSON，分开列 `scientific_status`、`report_status`、`report_failed` 与 `exit_code`。
SUCCEEDED 仅表示流程完成，包含科学否定的 cell，不代表真实验证成功；backend 身份也不认证实现。

|退出码|含义|
|---|---|
|0|保存终态 SUCCEEDED，报告保存成功|
|2|参数/配置错误或 claim 失败（不覆盖旧 run；无新终态为 UNKNOWN）|
|3|保存终态 STOPPED，报告保存成功|
|4|保存终态 FAILED，报告保存成功|
|5|保存终态 ALARM，报告保存成功|
|6|已知科学终态，但报告读取后的渲染/保存失败；JSON 保留原科学状态|
|7|无法确认终态/缺终态为 UNKNOWN；读取失败时 report_failed=true，否则报告未尝试|

配置/参数错误以固定公开错误或 argparse 用法呈现；运行异常不回显私有异常内容。
报告失败不改变科学终态；初始化失败尽力保存公开终态；缺终态不事后补成功，不自动恢复/重试。

## 公开参考＋残差配对预测（首轮已关闭：未见明确优势）

`configs/residual-paired-example.json` 是 T/D/S=2/2/2、batch2、epoch1 的小示例，只用于
check/demo 接线，不是训练或科研授权。三份正式配方
`configs/residual-paired-r3.json`、`configs/residual-paired-r4.json`、`configs/residual-paired-r6.json`
分别为 T65536 / D8192 / S65536、batch256、epochs32、lr0.001、MLP 64→128→128→1（24961参数）、
MSB0 target_bit0、cipher、init_seed17、sample_seed196901/2/3、shuffle_seed296901/2/3、rounds3/4/6。
它们只是完整严格配置，不授予运行、GPU、SSH、安装或发布权限；确认配方要在候选冻结时才写。

method=`bit-residual-paired-v1`，reference=`simon64-zero-master-v1`，固定 direct/residual 两臂，
没有自由 arms 列表或通用 registry。两臂使用同一私有 key、同一 TRAIN/DEV/TEST 明文面板、
相同初值和 shuffle，各自按最低 DEV 绝对 bit BCE（平局取最早）选模并持久保存；两臂都严格 SHA
加载成功后才释放 TEST。residual 臂训练目标是 y XOR h(x)，但最终 TRAIN/DEV/TEST 指标始终按
绝对 y 评价，不把残差 r 的准确率冒充 y。模型输入仍只有 64 明文 bit，不读取真实 key、h 历史、
其他密文 bit 或对方预测。h(x) 使用零 MASTER key（128bit，不是“零轮密钥”捷径）走同一 SIMON
装配，并如实计 real_blocks；demo 只注入公开 fake，不调用生产密码。

check 的静态估计（T/D/S，k epochs，b batch）：
- cipher：real_blocks=2*(T+D+S)，null_words=0；
- null label_mode：real_blocks=(T+D+S)，null_words=(T+D+S)（公开参考仍需加密）；
- 两臂 optimizer_updates=2*k*T/b，forward_rows=2*(k*(T+D)+(T+D+S))；参考不计模型前向。
示例为 12 real_blocks / 2 updates / 20 forward_rows；r3/4/6 每对为 278528 real_blocks /
16384 updates / 4997120 forward_rows。旧积分、位预测与 shared-M 公式不变。

保存 summary 的 `arms`（每臂 selected/learning_curve/metrics）、`paired`（TRAIN/DEV/TEST 四格表、
单侧精确条件 p、loss 差）、`diagnostics/reference_constant`（TRAIN 冻结的 fixed_flip 与 Jeffreys
残差概率）和 `pairing`（两臂初值 SHA、面板 x/y/h SHA、shuffle_seed）；不保存私有 key 或 key 指纹。
顶层 `summary.metrics/learning_curve/selected` 仍是旧公共空占位，报告不当作结果。fixed_flip 的 BCE
使用 TRAIN 多数硬翻转配套的固定单位幅度 logit（概率 sigmoid(±1)），residual_probability 使用 TRAIN
Jeffreys 概率 (ones+0.5)/(rows+1)；两者概率定义不同，均在 TRAIN 冻结，不重新拟合。

报告只读保存值：两臂绝对 bit ACC/BCE、配对差与精确 p、全部学习曲线、固定翻转诊断、初值/面板身份，
内联 SVG 不安装绘图库，不重新训练、前向、加密或重算指标。配对精确检验只回答固定 key 与固定训练
模型下的新明文；DEV 筛选 p 只作预定探索，不是确认 p 值或多重比较校正。约 50% 的微小差异不算
有用密码预测；低轮结果不冒充 6 轮进展。TEST 在双严格加载后自动释放并评分，报告明确说明它不是
技术 sealed。首轮三配方已各运行一个配对、共6fit，全部正常wait4 exit0，scientific_validation=false。
保存TEST direct/residual ACC：3轮50.0626%/50.2060%、4轮49.7116%/50.0000%、6轮50.1846%/50.0717%；
三项DEV均未达预定候选条件，确认未启动，周期CLOSED_NEGATIVE，剩余额度不可复用。
实际835584real_blocks（含417792公开参考）/49152updates/14991360forward_rows/12582912TRAIN呈现，
无重试/额外科学evaluate。汇总见`result/public-residual-20261109-summary/`，完整身份与偏差在对应spec/plan。
工程曾实际前向148行超批准128行，已披露；随后全部人工输出验证，不宣称完成真实工程小训练闭环。
本轮仅否定指定零主密钥参考＋MLP/配方下的预期优势，不否定所有残差或结构预测方法。

```sh
# 只读配置的静态检查（示例小策略会显示训练预算不足，但不会执行训练）。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins /workspace/.venv/bin/python -m cipher_lab check --plugin cipher_lab_plugins.residual.cli \
  --config configs/residual-paired-r6.json --policy /tmp/cipher-policy.json
# 仅工程接线且另获执行许可时，才可显式给出 root/id 与 --backend demo|simon。
```

## 三轮公开组合输入对照（唯一批次已执行并关闭）

2026-11-09 UTC 唯一批准批次 `public-feature-probe-20261109T182455752379Z-r3` 已正常退出，
同一私有密钥的16384条留出测试：原始输入直接预测49.9634%、原始输入学习纠错62.8601%、
13个公开组合学习纠错100%。这是固定三轮/单密钥/单初始化的正向探索，不是全部明文域、
跨密钥、六轮或新颖性证明。许可已关闭，不自动继续或复用时间。
中文汇总：[结果说明](result/public-feature-probe-20261109-summary/report.md)／
[可视化](result/public-feature-probe-20261109-summary/index.html)。

`configs/feature-probe-example.json` 是 T/D/S=2/2/2、batch2、epoch1、width2 的小示例，
只用于 check/demo 接线，不是训练或科研授权。正式配方 `configs/feature-probe-r3.json`
为 T16384 / D4096 / S16384、batch256、epochs32、lr0.001、rounds3、MSB0 target_bit0、cipher、
init_seed31、sample_seed196905、shuffle_seed296905；它是完整严格配置，不代表已经运行或授权运行。

method=`bit-public-feature-probe-v1`，reference=`simon64-zero-master-v1`，
representation=`simon-r3-residual13-v1`，固定 A/B/C 三臂，没有自由 arms 列表：
A 用64位原始明文直接预测绝对目标位；B 用64位原始明文预测 y XOR h 再合并 h；
C 用13个公开组合预测 y XOR h 再合并 h。三臂使用同一私有 key、同一 TRAIN/DEV/TEST 面板、
相同 shuffle，各自按最低 DEV 绝对 bit BCE（平局取最早）选模并持久保存；三个选中 checkpoint
都严格校验 SHA/spec/recipe/baseline 并成功加载后才释放 TEST 一次。13个公开组合只从明文与
公开接线派生（常数1、l、m、n、p、q、s、t、lm、mn、mq、pm、ln⊕lq⊕pn），不读真实 key、
轮密钥、实际中间状态或其他密文位，模型也不会拿到 h、标签或私有 key。C 首层13维、A/B 首层64维，
参数总量不同，不能声称完全参数匹配；每层用公开种子+层名+fan-in 派生独立初始化流，
A/B 初始参数逐字相同、C 的共享隐藏/输出层与 A/B 相同，输入层按自身维数用同一规则生成。

check 的静态估计（T/D/S 行数，k epochs，b batch）：
- cipher：real_blocks=2*(T+D+S)，null_words=0；
- null label_mode：real_blocks=(T+D+S)，null_words=(T+D+S)（公开参考仍需加密）；
- 三臂 optimizer_updates=3*k*T/b，forward_rows=3*(k*(T+D)+(T+D+S))；参考不计模型前向。
示例为 12 real_blocks / 3 updates / 30 forward_rows；r3 为 73728 real_blocks /
6144 updates / 2076672 forward_rows。旧积分、位预测、残差配对与 shared-M 的公式不变。

保存 summary 的 `arms[A/B/C]`（arm/input representation/output mode/model spec/参数数/
selected/完整 curve/metrics）、`comparisons[C-A/C-B][train/dev/test]`（复用 paired_accuracy，
方向明确写为 C-A 或 C-B）、`pairing`（面板 x/y/h SHA、各臂初态全态与共有层 SHA、初始化规则）
与 `arm_resources`（每臂累计墙钟/CPU）；不保存私有 key 或 key 指纹。固定主比较是最终 TEST 的
C−A 绝对 bit 猜对比例，C−B 只用于解释；一次固定 key/初始化下的留出观察不是跨 key、四/六轮
或新颖性结论，也不要求超过历史任何成绩才算有价值。

中文报告只读保存值：先讲问题、三种做法差别、测试集猜对比例卡片与 0–100 对比图，再给完整
学习曲线、主/次比较与结论边界；技术缩写词典、四格表、参数/成本、初始化与身份 SHA、原始保存值
放在折叠附录。正文不使用未解释的缩写，缺失值显示“未保存（不补 0）”，不重新训练、前向、加密、
解码模型/数据或重算指标；source.json 绑定原 run.json 校验值。

```sh
# 只读配置的静态检查；示例小策略会显示训练额度不足，但不会执行训练。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins /workspace/.venv/bin/python -m cipher_lab check --plugin cipher_lab_plugins.feature_probe.cli \
  --config configs/feature-probe-r3.json --policy /tmp/cipher-policy.json
```

## 公开组合前沿（四轮有信号；五轮预算停止；周期已关闭）

2026-11-10唯一周期已结束，禁止复用剩余预算或自动补跑。四轮最终新样本16384条：公开组合C **68.44%**、原始输入A **50.08%**、原始输入纠错B **50.17%**；仅单key/单面板探索性，不是全域、跨key或参数匹配结论。五轮C活动600.197s略超600s触发协作停止，没有最终TEST，不能称阴性；六轮表示未准入未启动，非理论最高轮数。汇总见 `result/feature-frontier-20261110-summary/index.html`，精确退出、来源、费用与限制见同目录source.json。以下配置描述接口，不代表新运行授权。

method=`bit-public-feature-frontier-v1`，新 schema4 checkpoint/bundle 自包含规范化公开计划，
旧 schema1/2/3 字段、字节与拒绝范围不放宽。正式配置 `configs/frontier-r4.json`、
`configs/frontier-r5.json` 只由已冻结的结构审计身份产生（r4 residual89 维、r5 factored8098 维），
包含 T16384 / D4096 / S16384、batch256、epochs32、lr0.001、init_seed31、
sample_seed196905+r、shuffle_seed296905+r、cipher、MSB0 target_bit0，且不含 `key_seed`；
**六轮没有通过结构准入，因此不生成 `frontier-r6*.json`**，也不虚构可运行配置。
`configs/frontier-example.json` 是 r4 审计身份、width2、T/D/S=2/2/2、batch2、epoch1 的
工程小示例，只用于 check/demo/显式 evaluate 接线，不是训练或科研授权。
测试用公开计划字节固定在 `plugins/cipher_lab_plugins/frontier/tests/fixtures/frontier/r4-public-plan.json` 与
`r5-public-plan.json`（仅已审计公开 math 字节，不含数据/权重/密钥/源码备份）；
测试按 `__file__` 相对路径读取，先核对固定 SHA-256 再 decode，不依赖任何会话私有目录。

固定 A/B/C 三臂：A 用64位原始明文直接预测绝对目标位；B 用64位原始明文学习 y XOR h 再合并 h；
C 用该轮公开组合学习 y XOR h 再合并 h。公开组合只从明文位与公开的异或/与运算派生，
不读真实密钥、轮密钥、实际中间状态或其他密文位；h 不进网络，只在输出端用于恢复绝对 y。
每条路线每轮以最低 DEV 绝对 bit BCE（平局取最早）选模并原子替换 `A-best.pt`/`B-best.pt`/`C-best.pt`，
保存成功才提交 selection；三个持久 selected 都严格校验 SHA/spec/recipe/baseline 并成功加载后
才释放一次 TEST。保存 summary 的 `arms`/`comparisons[C-A/C-B]`/`pairing`/`arm_resources`，
以及 `plan_identity`（轮/bit/表示/维数/orderSHA/planSHA）与 `output_bound`；
`plan-identity` payload 另记公共 plan 字节 SHA 与配置/plan 的逻辑名（不含用户绝对路径）。

工程拒绝在科学采集之前：`run` 在 claim 前只用标准库校验公共 plan，claim 后先调用
`output_bound(recipe, plan)` 再构造 `PredictionData`，超限记 `ENGINEERING_LIMIT` 未启动，
不补 0、不当作训练阴性。每轮最坏三份 best 加一份最大临时副本、三份 plan 副本、
TEST rawbundle 与16MiB元数据/报告预留须≤128MiB；模型权重仍由 RunStore 实限128MiB门控。
`check` 的静态估计（k epochs、b batch）与三轮公开组合一致：
cipher 时 real_blocks=2*(T+D+S)、null_words=0；null 时 real_blocks=(T+D+S)、
null_words=(T+D+S)；三臂 optimizer_updates=3*k*T/b，
forward_rows=3*(k*(T+D)+(T+D+S))。示例为 12 real_blocks / 3 updates / 30 forward_rows；
r4/r5 为 73728 real_blocks / 6144 updates / 2076672 forward_rows。旧方法公式不变。

中文报告只读保存值：先讲问题、三种做法、测试集猜对比例 0–100 卡片与保存曲线，
再区分**最终 TEST 的 C−A 主比较（探索性，正差且 p≤0.025/3）**与
**DEV 升轮门槛（C≥55%、C−A>0、精确单侧尾≤0.025/3，是资源门槛而非校准显著性）**，
C−B 只作描述；p 显示 0 时注明浮点下溢不是概率零；TRAIN 基线是多数类常数，不是 h；
不声称跨密钥确认、全部明文域或新颖性。正文缺失值写“未保存（不补 0）”，
初始化校验值不全时不比较“相同”。专属 `plugins/cipher_lab_plugins/frontier/report.py` 由
显式 frontier 插件的 saved 分派调用，旧三轮 renderer 不改。

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins /workspace/.venv/bin/python -m cipher_lab check --plugin cipher_lab_plugins.frontier.cli \
  --config configs/frontier-r4.json --policy /tmp/cipher-policy.json \
  --public-plan <冻结计划目录>/r4-public-plan.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/.venv/bin/python -m cipher_lab run --plugin cipher_lab_plugins.frontier.cli \
  --config configs/frontier-example.json --policy /tmp/cipher-policy.json \
  --public-plan plans/r4-public-plan.json \
  --backend demo --root /tmp/cipher-frontier-output --id frontier-demo
```

示例 `run` 只用于工程接线；正式 r4/r5 科学运行、GPU、SSH、安装与发布均须单独明确授权。
`evaluate` 自含 checkpoint 内的公共 plan 与 bundle，不需要也不接受 `--public-plan`。

## 六轮目标位 Fourier 频谱搜索（仅接线，科学未启动）

`configs/fourier-r6.json` 是完整独立正式配方：method=`bit-fourier-search-v1`、synthetic=false、
SIMON64/128 六轮、MSB0 绝对密文 bit0、64 层每层 65536 pairs、beam32、tau=0.1、
coef/DEV/TEST 各 65536、四个公开采样 prefix seed 19691111/29691111/39691111/49691111。
静态最坏 real/null 各 8388608+3×65536=8585216、sample-bucket products 536870912、
optimizer_updates/forward_rows=0；这是配方上限与门禁计数，不是执行授权、性能预测或科学结果。
`configs/fourier-example.json` 是**完全独立**的 tiny synthetic 配方（input_bits6、rounds0、
每层/面板256、beam8、tau0.5，独立 seed），只用于 check/demo 工程接线；它不是低轮 SIMON，
demo 的人工 oracle 输出也绝不算真实密文结果。正式/示例均须显式给出所有字段，不能只改默认值冒充。

`check` 只读配置并用标准库复核计数，不导入 NumPy/密码/数据/执行/存储，不创建 run 或文件；
`run` 在 claim 之后才装配科学依赖：synthetic+demo 使用明确公共 fake oracle 与工程字节，
formal+simon 使用现 `encrypt_batch`/`secrets.token_bytes`，两者共用同一 Meter/RunStore，
不会落入积分或训练分支。Fourier 插件 claim 显式申请模型目录（本方法无优化器更新但需要保存 predictor）。
方法先做 64 层配对 bucket 搜索（bucket 分数为系数平方和估计，保留 score>tau²/2、超 beam 按
score 降/mask 升截断），再用独立 coef 面板拟合均值系数（abs(coef)≥tau/2，intercept=mean），
保存纯 JSON opaque `models/<id>/spectral.pt`（独立 `fourier-predictor-v1` codec，不是 Torch
checkpoint），owner-read+SHA+严格校验后才释放 DEV；DEV 只是资源门（准确率≥0.505 且比 coef
常量基线有正差），通过后才在固定模型上评分一次 TEST。首异常停止、retry0，不追加 key/beam/tau/KAT。

```sh
# 只读静态检查；示例策略会显示 real/null 预算不足，但不会采样、加密或写 run。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins /workspace/.venv/bin/python -m cipher_lab check --plugin cipher_lab_plugins.fourier.cli \
  --config configs/fourier-r6.json --policy /tmp/cipher-policy.json
# 仅在另获工程执行许可后，可用 tiny synthetic demo 验证接线（不是科学实验）。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins /workspace/.venv/bin/python -m cipher_lab run --plugin cipher_lab_plugins.fourier.cli \
  --config configs/fourier-example.json --policy /tmp/cipher-policy.json \
  --root /tmp/cipher-fourier-demo --id fourier-demo --backend demo
```

中文报告由 `plugins/cipher_lab_plugins/fourier/report.py` 直接从 `read_saved` 的保存值渲染（`report.md`/`index.html`/
`source.json`）：展示方法流程、tau/beam/截断与每层候选、保存的模型项与系数、DEV/TEST 全样本
统计（TEST 含 D、下界、positive_exploratory）、real/null 查询与评分计数、来源身份及边界。
未生成/未访问的 DEV/TEST 明确写未启动，绝不补 0 或半数；demo 身份明确标注不是 SIMON。
报告不访问 data/models、不重新加密/采样/前向/拟合或重算指标。Fourier 插件没有 `evaluate` 子命令；
基础 checkpoint loader 仍拒绝本方法的 JSON 模型。唯一真实科学周期仍须 Task4 主控在独立审查、
fresh 验证和资源监督都完成后另行放行；配方、示例命令和历史许可都不是令牌。

## 一套记录与报告

显式 `--root` 下独占创建 `data/<id>` 和 `result/<id>`；积分不创建 models，训练/残差配对创建 models。
`data/<id>/run.json` 保存配方、实际策略、claim 源码身份、状态、计数与摘要；environment 仅在 claim 时采集 Python 实现/版本、OS 类型/架构，不含主机名、用户名、路径或环境变量，报告不重新探测。
events.jsonl 流式进度、
payload 保存聚合行及完整规范基，不保存 key/背景/逐点密文。终态冻结后由 `read_saved` 校验结构/引用/
SHA，再纯渲染；原 store 在同一 output_bytes 内写 `result/<id>/report.md`、`index.html`、`source.json`。
报告包含方法图与基于保存值的结果图，最多展示 4 个代表；残差配对报告只消费已保存的 arms/paired/diagnostics，不把旧顶层空 metrics 当结果，不再加密、采样、重算科学指标或改写终态；三轮公开组合对照报告只消费已保存的 arms/comparisons/pairing/arm_resources，先讲普通读者可读的问题、三种做法与猜对比例，技术缩写/四格表/身份 SHA 放折叠附录。
source.json 引用 run SHA，不是数据下载链接。仅服务 result/，不要将 data/models 或整个 root 暴露为网页。
无跨进程重绘、resume、旧包兼容或 runtrain 命令。

## 三阶段路线

1. **本期**：普通积分及两档搜索入口、fake engineering 闭环；SIMON有限工程门禁与首轮旧几何科学搜索均已完成并关闭。18/20新几何仅开发，真实周期未批，符号认证未执行。
2. **已接通工程能力**：位预测MLP、TRAIN/DEV/延后TEST标签、严格DEV选模与显式evaluate；公开参考＋残差配对同面板两臂及配对精确检验（首轮3/4/6轮已运行关闭，未见明确优势）；三轮公开组合输入固定 A/B/C 三臂唯一科学批次已执行关闭，公开组合在同key16384条留出全对、原始纠错62.86%、直接49.96%；只支持该固定条件的探索证据。
3. **已接通工程能力**：四任务shared-M、公开表示、joint训练与显式evaluate；不混用协议或标签。

## 各训练方法 demo 与显式 evaluate

现有两份**完整配置**为 `configs/prediction-example.json` 和 `configs/shared-m-example.json`，
不需要拼接旧配置。前者 train/dev/test=2/2/2、batch2、epoch1、MLP width32（1次更新）；
后者同样每任务2/2/2、batch2、epoch1，默认 shared-M 1,087,509 参数（总4次更新）。
两方法均 BCE + clip1 + Adam，DEV BCE 严格下降才保存 best，平局保最早；最终使用严格SHA加载的
最佳模型。位预测加载后才释放 TEST 标签；shared-M 预建 TEST，但选模不看 TEST。
最终 TRAIN/DEV/TEST 的 ACC/BCE、TRAIN 平滑基线与四单位成本分别保存，不合并两方法准确率。

backend 合法组合：积分/位预测/残差配对/三轮公开组合/公开组合前沿 `demo|simon`，shared-M `demo|ciphers`；Fourier 的 synthetic demo 配方只允许 `demo`、正式六轮配方只允许 `simon`，不合法组合在claim前拒绝。
demo 使用公开 fake oracle/确定字节（含 null 的交替bit）、模拟 RSS=0，**不使用生产秘密随机**。
null demo 也不是独立随机标签科学对照。生产 null 配方使用秘密随机bit但不调用密码；其生产RSS仍未验证。
`ciphers` 会接入四密码适配器；训练分支这里只验证接线，未做四家族真实训练或全家族 KAT。
已有的 SIMON 单次工程门禁不能替代其余家族的核验，也不认证本分支科学有效性。

建立完整 schema2 策略（以下仅展示已有训练入口，**不在当前 A2 本地开发执行范围**；
须另获训练许可，截止时间应按实际许可另定）：

```sh
cat > /tmp/cipher-training-policy.json <<'JSON'
{
  "schema": 2,
  "wall_seconds": 60,
  "cpu_seconds": 60,
  "deadline_utc": "2099-01-01T00:00:00Z",
  "real_blocks": 192,
  "null_words": 8,
  "optimizer_updates": 4,
  "forward_rows": 100,
  "rss_stop_bytes": 2147483648,
  "output_bytes": 20000000
}
JSON
# 在本目录；已有环境源码运行，不进行安装。每条run是独立工程调用。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/.venv/bin/python -m cipher_lab run \
  --config configs/prediction-example.json --policy /tmp/cipher-training-policy.json \
  --backend demo --root /tmp/cipher-training-output --id prediction-demo
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/.venv/bin/python -m cipher_lab run \
  --config configs/shared-m-example.json --policy /tmp/cipher-training-policy.json \
  --backend demo --root /tmp/cipher-training-output --id shared-demo
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/.venv/bin/python -m cipher_lab run --plugin cipher_lab_plugins.residual.cli \
  --config configs/residual-paired-example.json --policy /tmp/cipher-training-policy.json \
  --backend demo --root /tmp/cipher-training-output --id residual-demo
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/.venv/bin/python -m cipher_lab run --plugin cipher_lab_plugins.feature_probe.cli \
  --config configs/feature-probe-example.json --policy /tmp/cipher-training-policy.json \
  --backend demo --root /tmp/cipher-training-output --id probe-demo
```

预测/shared-M 训练 owner 独占 `models/<id>`，只保存 `best.pt`；残差配对保存两臂独立的
`direct-e<epoch>.pt` 与 `residual-e<epoch>.pt`，三轮公开组合对照保存三臂独立的
`A-e<epoch>.pt`/`B-e<epoch>.pt`/`C-e<epoch>.pt`，公开组合前沿保存原地原子替换的三槽
`A-best.pt`/`B-best.pt`/`C-best.pt`，清单记录最终 selected 引用。TEST bundle 为
`data/<id>/dataset/test.npz`（残差配对还包含公开 h；公开组合对照同样含公开 h，但面板仍是
原始64位），不存TRAIN/DEV副本、optimizer 或私有key/RNG状态。
本组三组 `check/run/evaluate` 均须使用 `PYTHONPATH=src:plugins` 及显式
`--plugin cipher_lab_plugins.residual.cli`、`cipher_lab_plugins.feature_probe.cli` 或
`cipher_lab_plugins.frontier.cli`；基础入口拒绝这三种 method/格式和 `--public-plan`，
不按 method 自动发现插件。已安装核心 wheel 不自带这些插件，须显式使插件根可 import。
独立 evaluate 只加载显式保存文件，不 import 训练装配、密码或标签生成；每个入口拒绝其他 method。

残差配对的显式评价使用 `models/<id>/direct-e<epoch>.pt` 或 `residual-e<epoch>.pt` 与含公开 h 的
`data/<id>/dataset/test.npz`；checkpoint 的 output_mode 决定翻转，evaluate 不调用密码。
公开 JSON refs 绑定 kind/bytes/SHA。`read_saved` 流式校验引用，不解码二进制；
报告只用已存指标画图，不前向、不生成标签。报告始终由原 RunStore 在相同 output_bytes 额度内写三文件。

显式评价为**新的run**，不覆盖训练，也不复制原模型或数据。四项source参数均必填，SHA须小写64位hex：

```sh
checkpoint=/tmp/cipher-training-output/models/prediction-demo/best.pt
dataset=/tmp/cipher-training-output/data/prediction-demo/dataset/test.npz
checkpoint_sha=$(sha256sum "$checkpoint" | cut -d' ' -f1)
dataset_sha=$(sha256sum "$dataset" | cut -d' ' -f1)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/.venv/bin/python -m cipher_lab evaluate \
  --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
  --dataset "$dataset" --dataset-sha256 "$dataset_sha" \
  --policy /tmp/cipher-training-policy.json --root /tmp/cipher-training-output --id reevaluate
```

每文件先 stat/regular-file/≤256MiB、有界读取、SHA，再严格CPU模型与NPZ加载。
只接受明确schema与完整任务声明：method、顺序、rounds、target_bit/label_mode或difference必须精确一致；
不明/缺失声明拒绝。同任务的其他已保存面板可评价，统一标 `explicit-saved-panel`，不冒充原TEST。
新记录仅保留逻辑路径 `explicit/checkpoint`、`explicit/dataset` 与原SHA/bytes，不暴露用户源绝对路径/文件名。
当前 checkpoint schema **没有 backend/synthetic 身份**；因此 evaluate 记
`backend_identity=UNVERIFIED`、`synthetic=null`，不因路径或本次参数伪称真实/demo。
评价只扣 forward_rows，不优化、不采集、不使用训练器/密码/生产随机；会使用生产RSS探针，生产RSS仍未验。
Linux不支持/失败即停止，Windows未验且RSS unsupported。工程测试仅注入模拟探针。

限制：同数据复评使用相同模型/CPU算法；极大面板分batch浮点运算不承诺跨版本逐bit一致。
Torch/NumPy依赖已经声明，**本轮没有安装、依赖解析或新机器验证**；没有切换旧项目root，旧入口/产物不迁移。
评价包现有初始化会加载Torch；纯评价报告只展示保存值，但其独立import并非零Torch保证。


## 开发验证

源码验证入口使用 `PYTHONPATH=src:plugins`；Windows PowerShell 对应
`$env:PYTHONPATH='src;plugins'`。pytest 配置另包含现有测试 helper 路径，不扩大 core wheel 范围。

pytest 显式覆盖 core 及七插件 owner 路径，不收集归档；差分关系既有测试仍位于
`tests/test_differential_relations_*.py`（插件 tests 路径为最终 owner 预留，未复制测试）。
mypy 的 `files` 与 `mypy_path` 均覆盖实际 `src/tests/plugins`，不静默跳过插件。

默认配置 `-m 'not science'` 保留但排除真实科学测试：不只几项微训练，还包括真实 MLP/shared-M
前向/反向、checkpoint 的数值前向 roundtrip、训练 step 的 Adam/反向、多个真实 evaluate、
生产 RSS 与纯 cipher/KAT 套件。精确函数分类在 `tests/plugin_isolation.py`，原断言不删不弱化；
纯密码 suite（含 `test_speck_study_cipher.py` 的单轮/假包装四项）保留 core，默认也排除；
独立公开表示/人工 fake 与 CPU 格式/参数初始化可保留。新增科学用例须显式标记
`@pytest.mark.science`；**选择 science marker 不表示获得授权**。

测试专用 poison 由 pytest 显式加载，对各 owner 的安全用例阻断实际 cipher、秘密熵、真实
NN 前向/反向、optimizer、RSS/GPU 与真实 TensorFlow/权重；NPZ/Torch 格式检查仅容许
内存人工字节或测试专属临时文件。人工 fixture 可绑定明确 fake 输出，不能把科学用例
改成人工输出造绿。子进程不继承 monkeypatch，现有 child helper 须自己装 guard；
这里不是通用科学测试运行平台，也不是 OS 沙箱。源码环境必须显式设置，尤其独立 child：

```sh
# 先安全收集；-m science 在此仅列出保留用例，绝不执行。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins /workspace/.venv/bin/python -m pytest -p no:cacheprovider --collect-only
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins /workspace/.venv/bin/python -m pytest -p no:cacheprovider --collect-only -m science
# 在各 owner 稳定、已核对人工/child guard 后执行默认安全测试；不另加 -m science。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins /workspace/.venv/bin/python -m pytest -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 /workspace/.venv/bin/python -m ruff check --no-cache .
cache_dir=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 /workspace/.venv/bin/python -m mypy --strict --cache-dir "$cache_dir"
rm -r -- "$cache_dir"
```

本次整合仅运行新增测试、稳定 owner 人工测试与 collect-only；T4 活跃期间未执行全 suite。
主控还须 fresh 整体安全验证、干净源码闭环及独立审查；本说明不把这些待验收项说成已通过。

包元数据只发现 src/cipher_lab；SIMON标量路径用标准库，批量路径按需导入已有NumPy，
不破坏check零科学依赖。干净源码副本（含.gitignore）demo与本地回环报告 HTTP
可用于工程验收，不能代表 wheel 安装、新机器/Windows、真实密码正确性或科学实验已验证。

## SPECK论文复现→绝对位预测（Task5B native接线，待审查/现场门禁）

新增正式方法 `speck-fourier-paper-prediction-v1`，唯一backend为显式
`--backend speck-study`。`configs/speck-study-paper.json` 与
`configs/speck-study-example.json` **都是完整正式规模、完全相同的固定配方**；example不是
小规模科学试跑，不允许CLI隐式缩样本、demo替代或通过参数覆盖门槛。
目前已实现Windows独立父Job/CPU-RSS/DXGI观测、child原owner及trusted py39 provider装配；
**只通过本机fake工程验证，尚未完成独立review、主控fresh和Windows现场门禁，不得科学release**。
非Windows/直接未监督child仍由原owner保存STOPPED/ENVIRONMENT_BLOCKED；现场任一身份、
测量、runtime或保存证明失败仍停止，不以fake测试宣称GPU就绪。

```sh
# 已实现入口：仅静态检查；不claim、不读机器、不import科学包、不写产物。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:plugins /workspace/.venv/bin/python -m cipher_lab.cli check --plugin cipher_lab_plugins.speck_study.cli \
  --config configs/speck-study-paper.json --policy /absolute/path/study-policy.json
```

策略仍是schema2完整11字段，正式算术预算real_blocks至少541000512（批准policy542000000的
差额不可花）、forward_rows至少1161000000、null_words/optimizer_updates均0；
wall28800、CPU仅记录而非单核8h门控、RSS32GiB、output4GiB（内含64MiB终态reserve）。
`check`报告全部四项fit、teacher分项及UUID+PCI要求，但不证明策略尚有效、GPU可用或获准运行。
本新study的CPU规则不修改旧Meter默认 `enforce_cpu=True`；新native显式使用False，但没有现场生效证据。

窄Python3.10兼容只覆盖明确主源码闭包，pyproject仍要求Python>=3.12，**不宣称整包/旧方法支持3.10**。
独立provider源码闭包以Python3.9语法为目标；本机静态检查不是Windows或实际3.10/3.9执行证据。
主入口以 **core 根 `src` + plugin 根 `plugins` 双根**冻结源码字节后 compile，不执行项目 pyc，
闭包外 project import 拒绝。旧 `tools/speck-study` 路径及单根/旧命名 manifest **不兼容**，
必须按新源码重新冻结四份清单并独立核对 Git 身份；生成清单本身不是发布核验或授权。
main3.10/provider3.9 闭包仅做人工 tree/fake 验证，本轮未在实际解释器/Windows现场验证：

```sh
# 仅生成hash清单，不等于发布核验/授权；须与独立审查的Git清单逐项比对。
python -B plugins/cipher_lab_plugins/speck_study/runtime/source.py freeze /absolute/path/main-sources.json src plugins
python -B plugins/cipher_lab_plugins/speck_study/runtime/source.py run /absolute/path/main-sources.json src plugins check \
  --plugin cipher_lab_plugins.speck_study.cli \
  --config configs/speck-study-paper.json --policy /absolute/path/study-policy.json
# 同一次freeze另生成main-sources.json.provider.json / .bootstrap.json / .package.json。
# 四份清单均须独立核Git身份；provider不得导入main config/execution。
```

以下是**未执行的Windows正式入口模板，非科学release许可**；必须先通过review/fresh与现场synthetic。
主Python须3.10、provider须3.9；root预建且专属，四份manifest及工具身份需独立核验。
一旦正式启动，统一原始Windows QPC origin在首次Torch/TF初始化之前冻结，不是开发预探。
路径、GPU UUID/PCI和解释器必须来自现场独立核验；不能把GPU序号1当logical1：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONPATH='src;plugins'
python -B plugins/cipher_lab_plugins/speck_study/runtime/source.py run G:\lxy\main-sources.json src plugins run `
  --plugin cipher_lab_plugins.speck_study.cli `
  --config configs/speck-study-paper.json --policy G:\lxy\study-policy.json `
  --root G:\lxy\fresh-study-output --id fresh-id --backend speck-study `
  --teacher-python G:\absolute\py39gpu_new\python.exe `
  --teacher-weights G:\lxy\official\core\net8_small.h5 `
  --weights-revision 96344da116ff624be30f503038b22f26c7f7f693 `
  --weights-sha256 15c58fff86855e882be1f977e42006761eeb5b9af28739d7a7fc8f44834902c0 `
  --gpu-uuid GPU-00000000-0000-0000-0000-000000000000 --gpu-pci 0000:00:00.0 `
  --source-manifest G:\lxy\main-sources.json
```

所有运行参数只属新方法，`check`与旧方法拒绝它们；路径/固定revision/SHA只做元数据验证，
正式native仅在父Job登记/独立资源监督已建立后读取并加载权重；不把有效参数当现场证明。公开权重只可从官方
`YuanYufeiISCAS/FourierNeuralDistinguishers`上述固定revision取得，禁止纳入本Git；本地仅hash raw asset，
从未H5 load。来源/许可缺口见 `docs/provenance.json`；快照未找到LICENSE，不猜测授予许可。

**观测和关闭信任边界**：每PID独立线程实际查询DXGI LOCAL CurrentUsage，CPU-only的0也须实测。
父以Job、独立handle创建时间、完整adapter/node inventory、tick/TTL及物理UUID/PCI→LUID核验；
证据类别`dxgi-self-per-process+parent-luid`，不等同NVML或完全独立监测，不用allocator替actual。
首次frame有固定5s bootstrap期限，缺测显式unknown且不能科学准入；后续TTL1.5s，PID重用、
错GPU/错LUID/CPU身份、失联立即fail closed。父自身不入自锁Job但纳GPU/RSS/CPU合计，不cuInit。
R需main Torch与teacher TF均context已绑定；teacher close/reaped且父注销后才B。
只向主child继承两个manual-reset request/ACK event，显式handle_list及close_fds=True。
child不可逆停科学→单owner保存→requested ACK；父独立严格saved/config/全package/source/modelcodec、
exit/reap/Jobempty及receipt读回核验。缺证、迟到、partialsave/kill回收不证实均UNKNOWN。
Windows保存沿file-fsync+replace+target长度/SHA读回，namespace_powerloss=UNVERIFIED；不承诺断电耐久。
本机没有实际验证Windows/DLL/COM/TF/H5/CUDA/生产RSS/真实显存或8h可完成性。
真实22轮KAT及CPU/GPU公开对照hook仅允许正式8h origin之后，由StudyData账≤256/实现、合计≤512；
本地开发从未调用真实完整加密/KAT，仅literal单轮与fake公开输出。

R1公开面板使用RunStore opaque `.npz`槽，实际为`SPKR1-v1-public-only`有界binary（**不是NumPy NPZ，
不可np.load**）；含公开cipherpairs/Y/float32概率/hardlabels，无私有key/PT。结果仍仅在原生
`data/<id>`、`models/<id>`、`result/<id>`及专属监督目录，owner/registeredSHA/严格JSON模型身份不降级。
工程fake闭环不是科研结果，不产生新的key/seed/阈值或许可。

## 创新1：四密码 M/D 与 SKINNY 行扫／列扫

唯一入口是 `cipher-lab --plugin cipher_lab_plugins.innovation1.cli {check,run}`。
`check` 只用标准库读取和校验完整配方，输出静态预算；不申请输出目录、探测设备或
导入 NumPy/Torch。它不是运行授权或 CUDA 可用性证明。只接受以下两个冻结配方：

| 配方 | 方法 | 更新上限 | 加密块上限 |
|---|---|---:|---:|
| `configs/innovation1-fresh.json` | 四密码共享 M vs 每任务专用 D，24×1024 cycles，同批对照 | 196608 | 413138944 |
| `configs/innovation1-skinny-layout-early-stop.json` | SKINNY64/64 8轮单对行扫 vs 列扫，100轮上限，24轮预热后两臂各自检查耐心12轮 | 204800 | 105512960 |

四密码以固定 DEV 选出五份 best；SKINNY 两臂各自按 DEV BCE 选 best（并列取早），
早停与选模独立。两种方法均须从磁盘严格重载所有 best 后才构造共同 TEST。
SKINNY 配方中的 `run_id: null` 表示尚未绑定，`run --id` 把新编号写入配方、权重、
面板和报告身份，不改公开种子或采样命名空间。旧固定训练编号不接受，无续训／覆盖。

```sh
PYTHONPATH=src:plugins python -B -m cipher_lab.cli --plugin cipher_lab_plugins.innovation1.cli check --config configs/innovation1-fresh.json
PYTHONPATH=src:plugins python -B -m cipher_lab.cli --plugin cipher_lab_plugins.innovation1.cli check --config configs/innovation1-skinny-layout-early-stop.json
```

现场运行须另行授权。在 Windows 新进程中以 `-B` 启动，绝对输出根必须位于 `G:\lxy` 下。
示例（将 UUID 占位符替换为明确指定的完整 `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`）：

```powershell
$env:PYTHONPATH = 'src;plugins'
python -B -m cipher_lab.cli --plugin cipher_lab_plugins.innovation1.cli run `
  --config configs/innovation1-skinny-layout-early-stop.json `
  --root G:\lxy\cipher-lab-output --id skinny-layout-new-01 --gpu-uuid '<GPU-UUID>'
```

UUID 在首次导入 Torch 前写入 `CUDA_VISIBLE_DEVICES`；只接受一张可见卡、逻辑 `cuda:0`。
训练使用 FP32，关闭 TF32，按配方限制 PyTorch 进程分配器显存（**不是整卡显存上限**）；
不探测驱动或采样整卡显存，不自动重试、换卡或回退 CPU。单个 Meter 检查协作式时间、
计数、进程 RSS，单个 RunStore 管理产物上限和所有权；不是 OS 硬隔离。

保存至 `data/<id>`、`models/<id>`、`result/<id>`。报告只从已封存且已核验的文件生成；
报告失败不改写 `run.json`。命令分别给出保存的科学状态、报告状态和请求退出码
（0 成功、2 参数／claim 拒绝、3 停止、4 训练失败、6 结果核验／报告失败、7 未知终态）。
实际退出须由调用者记录。介质故障时只能依赖最后成功落盘的计数和已注册产物；当前 main
的 RunStore 存储失败封存不保留尚未登记为 payload 的 `pending` 明细，也不会用推测补齐。

旧独立 `evaluate`、嵌套训练入口、24轮横／竖扫配方、独立 M/D 单对实验、预检与设备诊断
均不再提供。历史发布和记录不动；新版不读取旧科学产物。此替换只经人工替身验证，未证明
现场 CUDA、Windows、性能、真实训练或科学结论。
