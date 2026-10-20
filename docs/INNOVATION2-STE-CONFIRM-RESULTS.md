# STE四轮确认未通过：五轮探针未启动

## 完整结果，不挑赢家

用户在4轮确认＋5轮探针方案后要求“推进试试”。
执行前已固定[六项合同](INNOVATION2-STE-CONFIRM-PLAN.md)，本轮实际完成四项4轮确认，
预定r4-panel门槛未过，按规则停止，5轮真实/null均未启动。

**上一批74.90%的TEST成绩未在新面板上复现。当前STE训练方法不能称稳定的4轮能力。**

| 项目 | r4-panel | r4-init18 | r4-init19 | r4-newkey |
|---|---:|---:|---:|---:|
| data_seed | 192101 | 192101 | 192101 | 192101 |
| init_seed | 17 | 18 | 19 | 17 |
| key_seed | 190502 | 190502 | 190502 | 190503 |
| 最佳DEV epoch | 15 | 9 | 8 | 36 |
| TRAIN ACC | 49.8291% | 51.0742% | 50.3906% | 61.5723% |
| DEV ACC | 54.9805% | 53.0273% | 56.1523% | 58.8867% |
| **TEST ACC** | **50.4395%** | **49.7681%** | **50.2197%** | **61.9141%** |
| TEST正确数 | 4132/8192 | 4077/8192 | 4114/8192 | 5072/8192 |
| TEST BCE | 1.11813913 | 1.13156687 | 1.12253366 | 0.88864691 |
| TEST常量概率基线BCE | 0.69331236 | 0.69331236 | 0.69331236 | 0.69342058 |
| TEST公开参考残差常量BCE | 0.69336515 | 0.69336515 | 0.69336515 | 0.69344704 |
| 保存elapsed_seconds | 2.853493 | 2.694261 | 2.715755 | 2.487837 |
| 更新 / presentations | 800 / 204800 | 800 / 204800 | 800 / 204800 | 800 / 204800 |
| exit_code | 0 | 0 | 0 | 0 |

固定明文输入、目标bit0、公开硬AND/XOR前向、sigmoid叶子STE反向、Adam.01、clip1，
TRAIN4096/DEV1024/TEST8192、batch256、50epochs，每项120秒协作预算，CPU。
新key项是独立从头训练，不是同一模型直接跨key。前三项同标签，新key臂标签不同。
合计3200梯度更新、819200presentations、保存elapsed相加10.751346秒；没有重试或扩大预算。

## 与上一批74.90%的关系

已核对新r4-panel与旧`innovation2-ste-20261020-080758-r4`：
- 所有manifest源码SHA一致；本轮没有修改模型或训练实现。
- 初始化文件逐字节一致。
- 配置只有name和data_seed不同。
- TRAIN/DEV/TEST明文SHA不同。

data_seed既控制明文面板，又控制训练shuffle种子，因此**不能把差异唯一归因于“明文分布”**；
目前只能说对新面板及其训练顺序所形成的优化轨迹不稳定。
随机明文仍来自相同总体，旧74.90%不是因此被证伪成伪造，但未能成为可复现的普遍效果。
另外两个初始化在同一新面板仍接近50%，并未解决问题。
新key有61.91%的准确率信号，但固定±2输出使错误代价较高，BCE仍明显差于常量概率基线，
不满足本批≥70%且BCE优于基线的通过条件，不挑这项替代原门槛继续推进。

旧数据面板、旧key、不同初始化与新key当前没有完整正交矩阵；只覆盖声明的四项。
没有5轮梯度结果，不能据本批推断5轮好坏。旧5轮SAT UNKNOWN另行记录，不混进神经结果。

## 诊断边界与下一步

保存的最终在线TRAIN BCE约为1.1333、1.1465、1.1289、1.1084；
最终DEV ACC分别49.80%、49.61%、52.34%、48.54%，没有稳定保持最佳epoch的信号。
最佳checkpoint的TRAIN也没有明显高于DEV/TEST，因此不是高TRAIN准确率、低TEST的典型记忆过拟合。

本轮没有新增逐层梯度、参数符号轨迹或逐batch诊断，**不能据曲线断言唯一原因是梯度消失或符号震荡**。
最值得先做的下一项独立任务是记录梯度/符号翻转，并把“换明文”和“换shuffle”解耦，
区分优化不稳定与样本问题；本批未运行该诊断，不直接再冲5轮或换多个超参数碰运气。
差分预测继续搁置，SAT/枚举不替代神经训练主线。

## 验证与证据

```text
runs/innovation2-ste-confirm-20261020-084816-r4-panel/
runs/innovation2-ste-confirm-20261020-084816-r4-init18/
runs/innovation2-ste-confirm-20261020-084816-r4-init19/
runs/innovation2-ste-confirm-20261020-084816-r4-newkey/
validation/ste-confirm/verification.json
validation/ste-confirm/frozen-configs.json
validation/ste-confirm/r4-gate.json
```

48份原始产物核验文件SHA/字节数、配置/source身份、初始化/同面板标签身份、50行progress、
DEV checkpoint选择、800updates/204800presentations、exit及门槛。只读保存JSON/字节，
未对正式checkpoint追加加载、生成标签、模型前向或重算预测指标。
拟合前774tests、Ruff、mypy68files通过，干净源码null check/train/evaluate通过；
后补六份配置的冻结合同回归，不改变科学源码或重跑拟合。最终775tests、Ruff、mypy69files通过。

| arm | best.pt SHA256 |
|---|---|
| r4-panel | `6db93815f6e6306e84c24d7e18f0613ef8b03efad1a080f56107dbb5028e1bd2` |
| r4-init18 | `ede5e9d99af1af14b384f021929f5ae6ae7ac1c54dd0af870fd726a7c150dc87` |
| r4-init19 | `d161b70b6ace82daaa44ccad3259b457221f7c5f07e9fe8a4e435cd492c7090a` |
| r4-newkey | `0cfe920f4e35091d689202de436faae0795feb347f54e1808fe721d54f441044` |

一个准备命令在所有检查通过后误引用不存在的`cipher_dummy`模块，尚未开始拟合；
随后删除无关探测，正常完成配置冻结与干净源码闭环。该错误不影响已执行训练，未重复任何正式run。
当前实现为未提交工作树，基础HEAD111adab，运行manifest如实记dirty及源码SHA。
无GPU/SSH/安装/commit/push，原文件保留，批次关闭，无剩余槽位自动授权。
