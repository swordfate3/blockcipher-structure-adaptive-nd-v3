# 两轮四通道对照：扩宽局部输出未改善本次学习

## 结果和结论

按用户“继续推进”和[运行前合同](INNOVATION2-R2-VECTOR-PLAN.md)，完成固定四项，
批次 **`innovation2-r2-vector-20261024-092011`**，均exit0。没有自动重试、续训或升轮。
任务仍是SIMON64/128两轮、固定未知key、明文→绝对密文bit0，不是差分分类、整块预测或完整key恢复。

**四通道候选本次更差：TEST85.9863%，同期标量模型96.9482%。**
DEV差值vector−scalar为 **−12.6953个百分点**，BCE差值 **+0.13613949**。
不支持本设置下“将1通道扩为4通道即可补齐误差”；两种模型均未达99%及BCE<0.05目标。
只是一组key/init/面板的探索性观察，不能证明通道扩展普遍无效，也不能证明瓶颈与优化无关。

## 架构与固定条件

| 项目 | scalar | vector |
|---|---|---|
| model_kind | simon-r2-local | simon-r2-vector |
| 局部模块 | 三个独立4→16→1 | 三个独立4→16→4 |
| 局部隐层激活 | Tanh | Tanh |
| 传递 | sigmoid/center的3标量 | 逐通道sigmoid/center的12标量 |
| 末端组合器 | 3→8→1 Tanh | 12→8→1 Tanh |
| 参数 | 332 | 557=3×148+113 |

两模型相同公开局部接线和输出`(1−2X[j])`符号，不输入U/真实key/状态/其他密文bit，不用旧权重或中间教师。
现有table/scalar/base类AST与上批干净副本一致，旧scalar参数/state名保持；证据`source-delta.json`。
通道数、参数量、后端输入维度及随机抽样同时改变，因此不是唯一归因的单因素实验。
实数标量可以理论区分多个角点，未证明旧模型信息论容量不足。

共同配置：CPU，TRAIN4096/DEV1024/TEST8192，data193301/null193302/key190502/init17，
full/direct/hidden_width16、rounds2/target0，batch256/epochs50/Adam lr.01/clip1。
每项800更新/204800次TRAIN呈现，共3200/819200；每项120秒协作上限不是硬截止。
训练elapsed合计 **7.172977秒**，不是含开发/测试的端到端耗时。

固定顺序scalar-real、scalar-null、vector-real、vector-null。新明文内部13312条唯一，
与前五相关面板data192801/192901/193001/193101/193201无交集。
同模型real/null初态相同，跨模型形状不同不称同初态；所有模型共享明文与shuffle，real/null分别标签配对。
两个真实臂各13312次真实标量加密标签构造，合计26624调用、13312个不同明文；两个null无真实加密。
公开零key参考和无标签角点报告不算秘密oracle标签查询。

## 全部科学分数

指标为最低DEV BCE选中的best；平局最早，不按TEST选配置。

| 项目 | 参数 | best epoch | TRAIN ACC | DEV ACC | TEST正确/8192 | TEST ACC | TEST BCE | elapsed秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scalar-real | 332 | 49 | 96.8262% | 95.9961% | 7942 | **96.9482%** | .08390925 | 1.813390 |
| scalar-null | 332 | 1 | 51.0986% | 48.4375% | 4098 | 50.0244% | .69336108 | 1.760516 |
| vector-real | 557 | 49 | 85.5225% | 83.3008% | 7044 | **85.9863%** | .19442642 | 1.845491 |
| vector-null | 557 | 15 | 54.0039% | 49.8047% | 4123 | 50.3296% | .69506327 | 1.753580 |

scalar真实TRAIN/DEV BCE为 **.08450231/.09685582**；vector为 **.19803370/.23299531**。
两真实模型各split均优于TRAIN-only常量及公开参考常量BCE，但绝对目标仍失败；null正常不抵消失败。
TEST错误scalar250条，vector1148条；不作逐样本配对误差检验，不能宣称同一批错误被替换。

| 真实标签基线 | TRAIN ACC / BCE | DEV ACC / BCE | TEST ACC / BCE |
|---|---|---|---|
| TRAIN-only常量 | 51.0498% / .69292675 | 49.2188% / .69369557 | 49.8779% / .69341879 |
| TRAIN-only公开零key参考常量 | 50.3174% / .69312703 | 45.2148% / .69377467 | 50.6470% / .69308520 |

选模后strict load→写selection→仅TRAIN/DEV诊断→生成TEST标签。
外层先保存`dev-decision.json`再展开TEST。CLI自动保存TEST，不称访问强制封存或无人提前读取的审计证明。
没有重放任何旧正式checkpoint/TEST，也未用本次TEST改方案或延长训练。

## TRAIN/DEV错误定位

同样的10bit模型输入签名`[0,2,3,4,9,10,16,33,34,40]`，TRAIN覆盖1003种（最多1024种），
真实标签同签名无冲突。DEV1000条签名seen、24条unseen；这不是64bit明文重复。

| 模型 | TRAIN错误/4096 | DEV seen行数/错误 | DEV unseen行数/错误 |
|---|---:|---:|---:|
| scalar | 130 | 1000 / 39 | 24 / 2 |
| vector | 593 | 1000 / 167 | 24 / 4 |

大多数错误仍在已见输入签名上，不能单用未知签名解释。两模型均有未见行错误，
本批不保留上批“所有DEV错误都seen”的结论作为普遍规律。

公开`U=Y XOR F(X)`的三位顺序`U[j+1],U[j+8],U[j+2]`：

| U组 | TRAIN行数 | scalar错 | vector错 | DEV行数 | scalar错 | vector错 |
|---|---:|---:|---:|---:|---:|---:|
| 000 | 550 | 0 | 0 | 109 | 0 | 0 |
| 001 | 514 | 0 | 0 | 110 | 0 | 0 |
| 010 | 476 | 0 | 0 | 124 | 0 | 0 |
| 011 | 521 | 0 | 0 | 120 | 0 | 0 |
| 100 | 492 | 0 | 115 | 155 | 0 | 47 |
| 101 | 512 | 0 | 414 | 140 | 0 | 106 |
| 110 | 522 | 64 | 64 | 133 | 18 | 18 |
| 111 | 509 | 66 | 0 | 133 | 23 | 0 |

本批scalar错误集中在U前两位11，与上批相同init17在另一面板上集中00不同。
数据seed也改变shuffle，因此不能当纯数据因果隔离；错误区域不是已证明固定的某个真实中间bit缺陷。
vector的更广错误说明扩宽未消除此问题，尚未区分局部映射/连续压缩/末端组合/优化路径。
相同分组错误数不能证明相同行出错，报告不保存逐条样本。

局部16个公开角点报告的每通道sigmoid均有变化，不是严格常数；vector第3模块四通道约min .129–.168/max .847–.857，
仅为动态范围描述，不能据此断言塌缩、冗余或真实状态恢复。完整无标签角点汇总在各`diagnostics.json`中。
诊断不进入梯度、初始化或选模，无真实状态/极性教师，无TEST诊断。

## 工程与保存核验

- Ruff通过，mypy **73个源码文件**通过，pytest **832 passed**。
- 干净源码副本scalar/vector各null check/train/evaluate各两更新；保存TEST与显式evaluate相同：
  `validation/r2-vector-clean-20261024-091710/`，仅工程，不是额外科学臂。
- 执行prepare成功后附带显示日志的`tail -1`命令语法失败；仅修正显示为`tail -n 1`，
  没有重跑工程或科学训练。成功退出记录在`engineering.json`。
- 实现静态审查`3b99143c-3cab-4b6`、合同/config/driver审查`c823a6b7-36be-460`指定范围无确认缺陷。
- 正式验收核52文件字节SHA、41科学源码身份、配置、200条进度、选模、配对、计数/退出及诊断汇总；
  不加载正式.pt、不重新生成标签、不额外科学前向。
- 保留main、全部dirty及他人分支/保存引用。未安装、GPU、SSH、commit/push。

结果目录：`runs/innovation2-r2-vector-20261024-092011-{scalar-real,scalar-null,vector-real,vector-null}/`。
证据：`validation/r2-vector/{frozen-plan,dev-decision,verification,engineering,panel-preflight,source-delta}.json`。

best SHA256：

```text
scalar-real 326165eb869c93cef8a3555088e75e09024c719f646ccb3ae7a74f8efd2dd336
scalar-null 94950234ef48c245bd1520d0bdf0bc3c787b8731015328f2953e1b373c5d8307
vector-real 1190ca479c94b504133b0e09ecd70d001420bf74f4bc2975cd4857b1fa4b6663
vector-null 1148694512bcca2453b98a59cc8e2f4b62e76baeac084d5fb5b3953e942e271c
```

**后续仅建议，未实施：** 暂不继续盲目扩宽或升轮；若另授权，可单独检验优化或取消连续sigmoid压缩，
必须预先固定新预算与新面板；不拿本批TEST追改到99%。本批无论结果都关闭。
