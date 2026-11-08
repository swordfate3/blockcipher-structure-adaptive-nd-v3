# 创新2：明文到单密文位的直接/公开残差预测

## 当前完成的能力

统一使用 `cipher-experiments check/train/evaluate`。输入始终是一个明文的64个二进制位，输出一个**绝对密文位logit**。
数据生成层独占固定128-bit key；模型没有key、真实轮状态、密文、其他位标签或教师模型输入。
这条路径不包含SAT、密钥恢复、差分分类、历史checkpoint或远程启动器。

配置新增字段（旧配置的默认行为不变）：

| 字段 | 取值 | 含义 |
|---|---|---|
| `model_features` | `full` / `support` | 原始64位 / 公开轮函数依赖集合对应的明文位 |
| `prediction_mode` | `direct` / `public-residual` | 直接预测 / 相对公开零主密钥参考的残差预测 |
| `label_mode` | `cipher` / `null` | 真实密码标签 / 与明文无关的独立随机标签 |
| `null_seed` | uint32 | 独立null流；按TRAIN/DEV/TEST做域分隔 |

`support`仅适用于SIMON。依赖集合沿公开轮函数做集合传播，**是保守上界而非最小代数支撑**；
不消去重复XOR路径，不能将其大小当作未知系数个数或可学习性证明。
MSB编号0..31属于左字，32..63属于右字。目标bit0：r3保守支撑20位、r4为33位。

`public-residual`目前限定SIMON 1..4轮。令 `g(P)` 为同轮数零**主密钥**加密的目标位；
SIMON64/128前四个轮密钥均为零，第五个以后不能继续忽略密钥调度常数。
模型学习残差 `r=y XOR g(P)` 的logit `z`，返回绝对logit `(1-2g(P))*z`。
训练仍使用原标签 `y` 的BCE，等价于对 `r` 用残差BCE。
它是明确引入公开算法先验的模型，不是“只给明文不提供算法”的纯黑盒；也不是参数free的真实密钥预测器。
logit恰为0统一判绝对位0，不用残差硬判再XOR来改变平局语义。

## 基线、TEST时序与结果

r1..4不论direct还是residual，均额外报告 `public_reference_constant`：
**仅用TRAIN**拟合 `y XOR g(P)` 的多数位及Jeffreys平滑概率，再恢复到绝对位计算ACC/BCE。
原有 `train_constant` 仍在，使用原始TRAIN标签。模型须同时对比这两种基线，不能只超过50%就宣称学习到了密码。

预测训练先生成三组不重叠明文身份；只生成TRAIN和DEV标签，按DEV BCE最小值（平局取最早）选模，
保存并严格加载best checkpoint后写 `selection.json`，此后才请求实际TEST标签。
模型参考函数可提前处理所有**明文**，不是实际TEST标签。API本身没有权限隔离，源码/配置也可见key_seed，
因此这不是sealed confirm。显式 `evaluate` 会重建配置的数据/标签和TRAIN基线，绝不用于默认取回远端结果。
创新1仍预建split标签，这次不改变其训练合同。

新增 `initialization.json` 记录初始state SHA、输入索引和参数量；
配对direct/residual使用相同seed、特征和宽度时初始state应相同。
旧checkpoint仍按保存时原始配置映射核验配置SHA；显式评估分别记录原 `config_sha256`
和补默认字段后的 `resolved_config_sha256`，避免混淆身份。

## 可运行的CPU功能配置

`configs/innovation2-smoke.json` 使用8/4/4行、batch4、epoch1、CPU、最多60秒协作时间；
训练恰2更新，只生成独立随机标签，不生成真实密码标签。它验证新残差路径，而不是密码拟合。

```bash
uv run cipher-experiments check --config configs/innovation2-smoke.json
uv run cipher-experiments train --config configs/innovation2-smoke.json --output runs/i2-null-smoke-001
```

## 首批配方（后续已执行）

`configs/innovation2/` 中五份完整配置：

1. `r3-direct-positive.json`：低轮数信号探针，不保证必然成功；
2. `r4-direct.json` 与 `r4-residual.json`：同数据/初始化的机制配对；
3. `r4-direct-null.json` 与 `r4-residual-null.json`：同明文、独立随机标签负对照。

每份：固定key_seed=190502（null不使用该key）、data_seed=191401、init_seed=17；
TRAIN 65536 / DEV 8192 / TEST 65536，目标bit0、support、hidden114、batch256、epoch50，
理论12800更新/3276800训练呈现，`cuda:0`、协作1800秒。
r3为15619参数，r4两机制均17101参数。各run必须是新目录，不自动重试/续训。
五份合计64000更新，最多9000秒**协作**时间，不是硬GPU预算。

实现阶段这些是未执行候选，不代表训练授权。随后用户明确同意五项范围，并澄清可用SSH启动Git同步后的CLI。
2026-10-19五项GPU运行已完成，见[首批GPU结果](INNOVATION2-FIRST-GPU-RESULTS.md)。
这不授权再次执行或因50%自动加seed/epoch；没有依赖安装。

已准备环境中，查看一份配置的实际设备可用性：

```bash
uv run --no-sync cipher-experiments check --config configs/innovation2/r4-residual.json
```

没有CUDA时明确失败，不回退CPU。若由用户在训练机主动执行，示例入口为：

```bash
uv run cipher-experiments train --config configs/innovation2/r4-residual.json --output runs/i2-r4-residual-001
```

本批已现场执行，但不承诺未来GPU空闲、任意长训练稳定性或断线存活；SSH可调用已Git同步CLI，不传任务包或新增编排框架。

## 实际验证与结论边界

单元测试用小随机明文比较公开参考与标量SIMON零密钥输出，检验支撑外翻转不改目标位，
并验证残差BCE/梯度等价、配对初始化、null不调用真实加密、checkpoint兼容与TEST标签请求顺序。
训练集成测试使用**调用方任意标签**而非密码标签；干净源码CLI闭环使用每split独立随机标签。
这些都是工程验证，不是r3/r4拟合结果，不说明跨key泛化或超过历史神经/代数成绩。

当前仅一个候选key/seed，未设计独立多key统计确认，不能凭一组TEST ACC声称普遍有效。
来源与算法范围见 [provenance-prediction.json](provenance-prediction.json)。
