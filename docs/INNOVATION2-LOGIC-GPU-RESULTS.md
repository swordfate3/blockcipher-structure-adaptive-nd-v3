# 创新2 logic16 GPU试验：正控未通过，停止后续

## 结论

2026-10-20，用户要求“推进试试”后，发布代码、Git同步到Windows，并通过SSH调用远端已同步的CLI。
**r3逻辑网络完整训练成功，但科学正控失败。按预定门槛停止，后三项r4未启动。**
这不是传输故障、缺失exit、训练被中断或只完成了准备，也不是证明所有逻辑门网络不能学习SIMON。

## 冻结范围与执行

- 代码：`2a969f7491193a4124fcac16be4f9835baa6c67e`；GitHub远端ref已核对，Windows检出同提交且干净。
- 配置：`configs/innovation2-logic-proposal/r3-logic-positive.json`。
- 配置SHA：`865afb49dd94946f56307ada8df4fea42792e4a792e05dbe991d44df7a798319`。
- Windows主机：`DESKTOP-BBLPACJ`；物理GPU1 RTX A6000，UUID `GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5`。
- GPU起始快照空闲48410MiB、利用率0%，满足空闲≥24000MiB/利用率≤30%；不切GPU、不回退CPU。
- 映射`CUDA_VISIBLE_DEVICES=1`，程序`cuda:0`。Python与依赖复用既有环境，无安装升级。
- 模型：logic16、3层软门、support、width114、5587参数、direct。
- 固定SIMON64/128 r3、MSB bit0、key_seed190502、init_seed17、data_seed191501。
- TRAIN65536 / DEV8192 / TEST65536；batch256、50epochs、Adam.001、clip1。
- 实际12800更新、3276800训练呈现，保存elapsed331秒（约5分31秒），exit0。
- 每项最多1800秒协作预算；四项最多7200秒，但未使用预算不构成自动追加许可。
- 更新前本地Ruff/mypy通过、154项测试通过。训练后未改科学代码或配置。

同步先后两次HTTPS失败（传输低速和443连接失败），随后Windows既有GitHub SSH认证正常完成Git fetch。
未修改凭据，不通过SCP传源码、模型worker或任务包。训练经SSH指令直接调用现有环境适配器和CLI，
无WMI、Job模板、服务、远程调度框架；失败同步不是训练重试，实际只有一次科学fit。

## 最佳模型保存指标

按DEV绝对BCE最小值选择epoch5，完整50epochs结束后严格加载best再生成TEST标签。
以下直接摘录训练机保存指标，不在本地重算：

| 划分 | 行数 | ACC | BCE |
|---|---:|---:|---:|
| TRAIN | 65536 | 50.2777% | 0.693161 |
| DEV | 8192 | 50.6714% | 0.693202 |
| TEST | 65536 | **49.7238%** | **0.693403** |

TEST原始TRAIN常量基线：49.9863% / BCE0.693151。
TEST公开参考加TRAIN残差常量基线：62.4863% / BCE0.661646。

推进门槛预定为TEST ACC≥70%，且BCE优于两种TRAIN-only基线；**未满足**。
最后epoch训练在线BCE0.693050、DEV BCE0.693474，未出现可用的训练/DEV学习信号。
在线训练BCE混合不同step参数，不等同最后checkpoint的静态TRAIN BCE。

| 后续配置 | 实际状态 |
|---|---|
| r4-mlp-direct | 未启动 |
| r4-logic-direct | 未启动 |
| r4-logic-null | 未启动 |

未加epoch、换seed、试新接线、硬化门或追加拟合。随机标签对照因正控停止而未执行，不能称本次null通过。
与前一批r3 MLP81.94%使用不同明文划分，因此只能说明方向不同，不能作为本批同面板配对比较。

## 证据与核验

远端原始run保留：

```text
G:\lxy\shared-m-cipher-nd-runs\git-validate-20261019-060000\repo\runs\
  innovation2-logic-2a969f7-20261020-r3-logic-positive\
```

完整取回本地：`runs/received/innovation2-logic-2a969f7-20261020/`。
11个原始产物与远端清单逐一核对字节数/SHA，另取回`artifact-inventory.json`。
核对源码身份（8个既有文件仅LF/CRLF差异）、配置SHA、50条epoch记录、更新/呈现/exit、选模epoch及best文件SHA。

- best checkpoint SHA：`c5c250e2421f3b94f5927cfdb974859063d459d1be613d988ef7be0592ace0f5`。
- 初始state SHA：`7938611c36235268e77bfdc379f6a6ded0b9259f5fd48e412b6cf90ec0b90f94`。
- 核验：`runs/received/innovation2-logic-2a969f7-20261020/verification.json`。
- SSH日志/关闭记录：`validation/innovation2-logic-gpu/`。

只读保存的JSON和文件字节，**未在本地加载checkpoint、再生密码标签、执行模型前向或重算指标**。
远端原件未删除，结果不纳入Git，不在Windowscommit/push。

## 下一步应诊断什么

这次未通过的是“该固定接线/初始化/宽度/训练日程能学到r3”的候选推进条件。
门真值表、XOR优化和手工parity见证通过，不保证随机接线中存在或容易发现SIMON电路。
当前证据不能区分接线表达限制、软门优化、初始化或配方问题；没有同面板MLP，因此也不能逐项归因。
先做公开结构与接线的分析，再决定是否值得提出新的有界诊断；不凭剩余预算自行追加GPU试验。
