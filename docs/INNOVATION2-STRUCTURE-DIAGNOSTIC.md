# 结构接线诊断：先分清接线、表示与优化

## 问题与冻结的小实验

logic16的r3正控TEST49.72%，说明当前固定随机接线/初始化/配方没有学到低轮信号。
用户提出“学习密码结构推进预测轮数”后，本轮先执行一个CPU人工目标对照，**不新增密码拟合或GPU工作**。

目标：前8个输入位上的

```text
y = (x0 AND x1) XOR (x2 AND x3) XOR (x4 AND x5) XOR (x6 AND x7)
```

两臂共用已有`SoftGateLayer`，网络8→4→2→1，7个16真值表软门和一个标量线性读出，共114可训练参数。
没有手工设AND/XOR参数作为训练初始化：所有门沿用透传A logit偏置+4与小随机扰动；
读出weight4/bias-2。**只有接线干预不同**：

- aligned：第一层相邻配对，第二层组合相邻门，第三层合并；按已知目标结构给出拓扑先验。
- random：每层随机perfect matching，所有输入恰好使用一次，保持同层宽度与门数。

同init_seed17时两臂可训练参数逐字节一致；wire buffers不同且持久化。
初始化文件分别记录全部state SHA（包含接线）和仅可训练参数SHA，避免把不同拓扑误称完全相同初态。
同一人工函数未进入模型forward或训练初始化，不给训练中间层标签；但aligned拓扑显然由该已知目标设计。
这是带oracle结构先验的机制诊断，不是公平未知结构发现赛，也不是新颖性证据。

## 数据与预算

- 256个8-bit赋值先无放回打乱，再切TRAIN128/DEV64/TEST64，data_seed191601。
- 嵌入64维输入，其余56位全部清零；相关8位也跨split不重复，不靠无关padding制造“新样本”。
- 不按标签筛选/分层；目标为确定性人工布尔标签，不调用密码算法。`label_mode=cipher`只是原字段默认值，
  此dataset分支并不生成密码标签；manifest明确标为`synthetic-engineering`。
- CPU，每项batch128、300epochs=300更新、38400训练呈现，Adam.05/clip1，协作60秒。
- DEV BCE最小选best；加载后再生成TEST标签，TEST64只评分；API无权限隔离，不称sealed confirm。
- 两次独立训练，合计600更新、76800训练呈现，没有观察结果后追加seed/epoch。

配置：`configs/structure-diagnostic/aligned.json`与`random.json`，独立完整。

```bash
uv run cipher-experiments check --config configs/structure-diagnostic/aligned.json
uv run cipher-experiments train --config configs/structure-diagnostic/aligned.json --output runs/structure-aligned-001
uv run cipher-experiments train --config configs/structure-diagnostic/random.json --output runs/structure-random-001
```

当前已准备环境，开发源码可用`PYTHONPATH=src .venv/bin/python -m cipher_experiments.cli`。
这是人工任务专用模型；配置禁止拿`circuit-aligned/random`直接切到真实SIMON数据。
`hidden_width=64/full/direct`在此为固定兼容字段，不控制实际8→4→2→1结构，参数量以initialization记录为准。

## 这次实际观察

| 臂 | 选中epoch | TRAIN ACC | DEV ACC | TEST ACC | TEST BCE | elapsed秒 |
|---|---:|---:|---:|---:|---:|---:|
| aligned | 300 | 100% | 100% | **100%** | 0.010309 | 1.409 |
| random | 292 | 63.2813% | 53.1250% | **48.4375%** | 0.725041 | 1.349 |

两项均300更新、exit0；相同data.json，相同初始可训练参数SHA：
`243a5c4cdbe1837a4a4a41bf5f03c554b041c4ea1e71d6876d6be9ddb3686a06`。
DEV/TEST均未参与更新，显式evaluate另行执行，重放的本地TEST指标一致。
本次是本地自生成诊断数据，不涉及默认重算取回的远端密码结果。

证据在`validation/structure-clean-20261020-014747/`，完整run/evaluation日志保留。
此路径仅源码副本和已有环境，无backups依赖，未做新安装。实际记录见`validation/innovation2-structure-closeout.json`。
独立只读审查确认一项状态绑定缺陷：aligned checkpoint若换成其他合法matching可通过加载。
已增加aligned canonical接线校验和SHA正确但接线不符的回归测试；这不改变训练算法或两项实际接线。
没有因此追加训练；修复后另行显式评估原本地checkpoint，验证合法状态仍可用。

## 可以说什么，不能说什么

**支持：**在这一个人为设计、单seed、固定预算下，结构接线让模型学到新输入的目标关系，而该随机接线没有。
说明“单门可训练但随机接线整体学不动”不矛盾，接线是值得单独控制的变量。

**不能说：**已证明随机臂只存在优化问题。接线也改变可表达函数族，随机结构可能根本没有合适表示。
也不能说任何结构网络都会优于通用网络、已学出SIMON结构、已预测更多轮或达到跨key泛化。
100%仅覆盖这个256输入的人工函数，TEST64不是密码规模的新明文证据。
手工见证把真值表写成AND/XOR，是表达检查；本表则是从透传偏置训练得到，不能混淆两者。

## 下一步：公开目标位依赖图，而非随机门堆叠

新增`representations/simon_cone.py`为**公开结构编译器**：从预定SIMON输出位回溯，
仅保留所需的明文节点、AND/XOR连接及轮密钥的符号占位节点；共享子表达式，生成拓扑有序DAG。

它不采样key、不计算真实轮密钥、不读取标签、不训练模型；没有新增含秘密值的推理接口。
符号节点只表示“这里受未知常量影响”，不是已知实际值。bit0的公开DAG规模如下，key节点数不是独立主密钥比特数或未知函数维度：

| 轮数 | 全部节点 | 明文坐标 | 符号轮key坐标 | AND | XOR |
|---|---:|---:|---:|---:|---:|
| 3 | 75 | 20 | 11 | 11 | 33 |
| 4 | 153 | 33 | 24 | 24 | 72 |
| 5 | 266 | 46 | 44 | 44 | 132 |
| 6 | 406 | 56 | 70 | 70 | 210 |

这是实现结构计数，不是可学习轮数结论；r5+密钥调度关联没有在符号图中求解。
测试中的人工解释器用小合成key与标量密码核对图，
仅是算法一致性检查，不作预测或拟合证据。

接下来接入学习时必须明确选择：

1. 公开结构特征＋学习读出：仍是带先验的位预测；
2. 在结构图上学习局部门函数：强结构先验，可能隐式辨识密钥效应，不能称纯黑盒；
3. 保留精确轮函数、直接优化符号密钥节点：是参数辨识/密钥拟合，不能隐去研究目标变化。

本轮只完成图和人工对照，不把符号key占位偷偷变成可访问的真实key输入，不自动启动下一批GPU。
本地最终验证：Ruff通过、mypy54文件通过、pytest594项通过。
全套测试包含既有tiny训练，不是整轮零更新；上表诊断的两次单独训练共600更新，另有单测更新。
来源见`provenance-structure.json`。未安装、SSH、commit或push。
