# Cipher Lab

独立、无旧接口兼容层的 Python 3.12 CPU 实验后端，已接通积分、位预测 MLP、四任务 shared-M 和显式 evaluate。积分包含完整
2080 项观测（64 单位项 + 2016 二次项）、完整 cube 采集、固定 D/S/冻结/C 与 null
流程、协作式预算、唯一运行记录及保存结果报告。**工程演示不是实际 SIMON 实验，经验不是证明。**

## 入口与权限

已实现 `cipher-lab check/run/evaluate`（打包入口）；未安装时在本目录使用下列源码入口，二者参数相同。
本轮只验证已有环境下源码运行，**没有安装验证**，无需为使用源码命令执行安装。

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
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /workspace/.venv/bin/python -m cipher_lab check \
  --config configs/integral-example.json --policy /tmp/cipher-policy.json
```

`check` 支持积分、位预测与四任务shared-M，仅读配置并输出参数、四单位静态估计及各预算是否足够；
不创建 run、不导入Torch/NumPy/采集/密码/模型，不启动 validity 或任何科学控制。成功只表示配置合法，不表示截止时间尚未到达、资源足够、
获得实验许可或科学有效。不存在 `--dry-run`，检查生产配方也使用上述同一个 `check`。

获得相应操作许可后，可运行无密码 demo（输出位置必须显式给出，编号不得占用）：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /workspace/.venv/bin/python -m cipher_lab run \
  --config configs/integral-example.json --policy /tmp/cipher-policy.json \
  --root /tmp/cipher-demo-output --id engineering-001 --backend demo
```

`--backend demo|simon|ciphers` **必须显式给出，没有默认值**。demo 使用公开确定函数与明确无秘密的工程字节
序列；不同组 fake key 字节不重复，所有 blocks/words/groups 都是**模拟计数**，RSS 模拟为 0，
不是生产 RSS 验证。仍按配方采完整 cube、固定 2080 项，绝不为演示降低维数。
小样本有巨大经验核，null 可报警；`ALARM` 非零退出是正确结果，不会伪造演示科学成功。
工程数学自检只验证软件接线，不能替代未来科学正/负控制。

生产入口语法与上面一致，只将显式参数改为 `--backend simon`；这会使用
`secrets.token_bytes`、`encrypt_many` 和 `production_rss_bytes`，**是真实加密入口，不是演示**。
生产适配器虽已实现，但本轮没有调用 SIMON、KAT 或生产 RSS，均**未实测、未经科学验证**。
RSS 实现仅支持 Linux procfs，不支持的平台/读取失败 fail closed；**Windows 暂 unsupported**，
不承诺可用。检查是协作式的，不是 OS 硬资源隔离。真实加密/科学实验、正式训练、GPU、SSH、安装与
发布仍须单独明确授权；策略文件、命令示例和历史批准均不是许可令牌。

## 设置、计数与退出

配置严格完整，拒绝未知/重复字段、布尔整数、NaN/Infinity、超过 64 KiB 的 JSON；不继承旧配置。
配方 schema=1，method=integral-empirical-v1；1..64 个互异 cell，rounds=1..44；active 为
同一个 32 位半字内的 2..16 个互异 MSB0 位；chunk_points=1..65536；D/S/C 各 1..1000000 组。
每组 `2 ** len(active)` 点；real_blocks 为明文块次数（不乘轮数），null_words 为逐点 64 位字数。
`check` 估计完整 D+S+C；空空间可跳过确认，实际 attempted/completed 由采集器唯一计数。
Policy独立使用 **schema=2**：原7字段加schema、optimizer_updates、forward_rows，共10字段全部必填，
不接受旧/中间策略JSON。时间限制正有限、UTC截止带Z/+00:00，其余额度非负；
输出还须足够保存完整基及三份报告。积分不消耗optimizer_updates/forward_rows，可设为0。

### 训练配方与执行

`configs/prediction-example.json` 与 `configs/shared-m-example.json` 均为独立recipe schema=1，
与Policy版本无关。可将上述check命令的config分别替换为这两个路径；示例策略中的训练预算0
会如实显示训练额度不足，但check仍成功，不审批执行。两方法 run 已接通，必须另给足够的执行策略。两示例不是训练或科研授权。

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

## 一套记录与报告

显式 `--root` 下独占创建 `data/<id>` 和 `result/<id>`；积分不创建 models。
`data/<id>/run.json` 保存配方、实际策略、claim 源码身份、状态、计数与摘要；environment 仅在 claim 时采集 Python 实现/版本、OS 类型/架构，不含主机名、用户名、路径或环境变量，报告不重新探测。
events.jsonl 流式进度、
payload 保存聚合行及完整规范基，不保存 key/背景/逐点密文。终态冻结后由 `read_saved` 校验结构/引用/
SHA，再纯渲染；原 store 在同一 output_bytes 内写 `result/<id>/report.md`、`index.html`、`source.json`。
报告包含方法图与基于保存维数的结果图，最多展示 4 个代表；不再加密、采样、重算科学指标或改写终态。
source.json 引用 run SHA，不是数据下载链接。仅服务 result/，不要将 data/models 或整个 root 暴露为网页。
无跨进程重绘、resume、旧包兼容或 runtrain 命令。

## 三阶段路线

1. **本期**：积分经验流程及 fake engineering 闭环；真实适配器已实现但尚未真实验证，符号认证未实现。
2. **已接通工程能力**：位预测MLP、TRAIN/DEV/延后TEST标签、严格DEV选模与显式evaluate。
3. **已接通工程能力**：四任务shared-M、公开表示、joint训练与显式evaluate；不混用协议或标签。

## 两方法 demo 与显式 evaluate

现有两份**完整配置**为 `configs/prediction-example.json` 和 `configs/shared-m-example.json`，
不需要拼接旧配置。前者 train/dev/test=2/2/2、batch2、epoch1、MLP width32（1次更新）；
后者同样每任务2/2/2、batch2、epoch1，默认 shared-M 1,087,509 参数（总4次更新）。
两方法均 BCE + clip1 + Adam，DEV BCE 严格下降才保存 best，平局保最早；最终使用严格SHA加载的
最佳模型。位预测加载后才释放 TEST 标签；shared-M 预建 TEST，但选模不看 TEST。
最终 TRAIN/DEV/TEST 的 ACC/BCE、TRAIN 平滑基线与四单位成本分别保存，不合并两方法准确率。

backend 合法组合：积分/位预测 `demo|simon`，shared-M `demo|ciphers`；不合法组合在claim前拒绝。
demo 使用公开 fake oracle/确定字节（含 null 的交替bit）、模拟 RSS=0，**不使用生产秘密随机**。
null demo 也不是独立随机标签科学对照。生产 null 配方使用秘密随机bit但不调用密码；其生产RSS仍未验证。
`ciphers` 会接入四密码适配器；这里只实现接线，**没有执行真实密码或 KAT**。

建立完整 schema2 策略（以下仍不是科学或生产运行授权；截止时间应按实际许可另定）：

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
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/.venv/bin/python -m cipher_lab run \
  --config configs/prediction-example.json --policy /tmp/cipher-training-policy.json \
  --backend demo --root /tmp/cipher-training-output --id prediction-demo
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/.venv/bin/python -m cipher_lab run \
  --config configs/shared-m-example.json --policy /tmp/cipher-training-policy.json \
  --backend demo --root /tmp/cipher-training-output --id shared-demo
```

训练 owner 独占 `models/<id>`，只保存 `best.pt`；TEST bundle 为
`data/<id>/dataset/test.npz`，不存TRAIN/DEV副本、optimizer或私有key/RNG状态。
公开 JSON refs 绑定 kind/bytes/SHA。`read_saved` 流式校验引用，不解码二进制；
报告只用已存指标画图，不前向、不生成标签。报告始终由原 RunStore 在相同 output_bytes 额度内写三文件。

显式评价为**新的run**，不覆盖训练，也不复制原模型或数据。四项source参数均必填，SHA须小写64位hex：

```sh
checkpoint=/tmp/cipher-training-output/models/prediction-demo/best.pt
dataset=/tmp/cipher-training-output/data/prediction-demo/dataset/test.npz
checkpoint_sha=$(sha256sum "$checkpoint" | cut -d' ' -f1)
dataset_sha=$(sha256sum "$dataset" | cut -d' ' -f1)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
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

只收集本新项目测试。本轮已批准阶段2/3开发和无密码CPU合成验证：小tensor；
每种完整合成闭环≤60s，位预测≤2次更新、shared-M四任务各1共4次，不搜索/多seed科研。
真实密码/KAT/科学标签/正式拟合/GPU/生产RSS/SSH/安装/Git/备份切换仍未授权。
测试只用公开配置、人工数据/fake oracle；完整工程回归会执行微型真实 Adam 更新，必须逐次记录，不算科学实验：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /workspace/.venv/bin/python -m pytest -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 /workspace/.venv/bin/python -m ruff check --no-cache .
cache_dir=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 /workspace/.venv/bin/python -m mypy --strict --cache-dir "$cache_dir"
rm -r -- "$cache_dir"
```

包元数据只发现 src/cipher_lab；当前 SIMON 适配器仅用标准库。干净源码副本与本地回环报告 HTTP
可用于工程验收，不能代表 wheel 安装、新机器/Windows、真实密码正确性或科学实验已验证。
