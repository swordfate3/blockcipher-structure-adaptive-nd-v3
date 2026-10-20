# 创新2进度对齐与下一步

## 到底做到哪里了

| 分支 | 已有证据 | 当前边界 |
|---|---|---|
| 旧神经明文→单bit | r3 support约77.7%—83.1%；r4多架构/Boolean约50% | 没有可靠4轮神经突破 |
| 旧公开代数 | r4、r5在各自固定key合同下持出100% | 不是神经学习，也不等于密钥恢复或跨key结论 |
| 旧r6 native | 512真实TRAIN标签、一次求解返回UNKNOWN，无模型 | 无heldout分数，不是50%、UNSAT或不可学证明 |
| 旧神经公开残差 | 完成准备，首个工程槽NVML环境检查失败 | 实际科学worker未启动、0更新；没有残差训练成绩 |
| 新后端公开残差对照 | 五项GPU实训完成；r3 81.94%，r4 direct50.18%/residual50.02%，null约50% | 旧残差假设终于得到实际测试，本设置不支持其带来突破 |
| 新后端logic16 | 2026-10-20 r3完整50epochs，TEST49.72%，exit0 | 正控未通过，按门槛停止，后三项r4未启动 |
| 结构接线人工诊断 | 相同可训练初态/300更新，aligned TEST100%，random48.44% | 仅人工布尔目标，尚非SIMON拟合或轮数推进 |
| 公开结构＋软未知常量学习 | GPU r1软/硬100%；r3软50.51%/硬49.83%，800更新 | 软梯度r3正控失败，r4未启动 |
| 公开结构＋硬参数辨识 | CPU r3/r4均TRAIN512、DEV1024、TEST16384全对；r4-null全域无一致模型 | 新后端4轮单bit正结果，仅单key/seed，不是神经成绩或完整key恢复 |
| 更高轮硬结构SAT | 新面板r5，512TRAIN、Glucose4单次CNF求解UNKNOWN，约132.18秒返回 | 无预测器/无TEST标签/无ACC，r6/null未启动，批次关闭 |
| 神经STE硬前向首批 | CPU Adam真实800更新/项，r3 TEST100%、r4 TEST74.90%、同面板null49.68% | 单面板正结果，后续未稳定复现 |
| STE四轮确认 | 新面板原init50.44%、init18 49.77%、init19 50.22%、新key61.91%，BCE均不如常量 | 原门槛未过，5轮真实/null未启动，不能称稳定4轮能力 |

历史事实仅来自只读档案：
`backups/v3-before-rebuild-20261019/docs/innovation2/`下
`PREDICTION-LEARNING-RESET-20261018.md`、`PUBLIC-RESIDUAL-NEURAL-EXECUTION-20261019.md`、
`PUBLIC-ALGEBRA-R6-FORMAL-EXECUTION-20261018.md`。它们不是新后端运行依赖或新授权。
新实测见[首批GPU结果](INNOVATION2-FIRST-GPU-RESULTS.md)。
旧新划分/训练实现/审计合同不同，不称逐字节复现历史结果。

最新推进见[结构接线诊断](INNOVATION2-STRUCTURE-DIAGNOSTIC.md)：先确认接线干预可影响人工目标学习，
并编译公开SIMON目标位依赖DAG；未新增GPU或密码拟合。
用户随后明确允许公开结构与未知参数学习，已接入[结构化SIMON候选](INNOVATION2-STRUCTURED-SIMON.md)，
并完成[首次GPU试训练](INNOVATION2-STRUCTURED-GPU-RESULTS.md)：r1正控通过、r3失败、r4未启动。
参数类似密钥不等于泄露；禁止的是实际秘密注入，而不是从TRAIN中学习未知效应。
最新[硬结构参数学习](INNOVATION2-HARD-RESULTS.md)已在新数据上取得r3/r4目标bit0未见TEST100%，
说明软梯度失败不代表公开结构中找不到有效预测器。r4-null对照无一致模型，原批关闭。
用户后续授权[5轮硬结构SAT](INNOVATION2-SAT-RESULTS.md)，新批单次返回UNKNOWN，无模型，停止r6/null。
历史5轮代数正结果与本次CNF未知分开记录。
用户纠正主线后，差分暂缓，实际[STE梯度学习](INNOVATION2-STE-RESULTS.md)r3 TEST100%、r4 74.90%、null49.68%。
但[四轮确认](INNOVATION2-STE-CONFIRM-RESULTS.md)未复现该优势：原init新面板50.44%，额外init接近50%，
新key61.91%但BCE不如常量。5轮探针按门槛未启动；神经STE74.90%只是特定面板结果，
不能称稳定4轮能力，且与枚举100%不能混报。

## 已完成候选：真正的16真值表软门

新增`model_kind=logic16`，原配置默认`mlp`，继续使用现有check/train/evaluate。
三层固定配对二输入门，每门对16个真值表的Bernoulli多线性扩展作softmax加权，最后线性单logit读出。
无key、密码实际中间状态、教师、其他密文位或历史checkpoint输入。
模型可以选full/support和direct/public-residual；下一批建议只检验direct，先不扩残差矩阵。

为什么不是重复旧Boolean：旧模型是tanh/卷积/乘法特征，不是16张真值表的门选择。
为什么现在不许宣称有效：该机制已有文献，本实现只是实验候选，不是新颖性或密码效果证据。

### 需要先排除的错误

1. **均匀16门混合**恒为1/2，输入Jacobian为0；门参数梯度不一定为0。
   本实现A透传门logit偏置+4、其余小随机扰动，避免精确均匀；这不保证深层优化成功。
2. **编号**：本地按00、01、10、11对应mask bit0..3；A=12、B=10、XOR=6。
   官方difflogic把00放高位，A=3；不能混用编号。
3. **深度**：两层二元门的实数logit多项式最高次数≤4；改为三层后上界≤8。
   这是实数软门logit次数，不是sigmoid/阈值分类次数，也不等于GF(2)代数次数。
   三层没有保证固定接线包含SIMON目标；不能因与旧形式次数数字相同就宣称有足够表示能力。
4. **相关性**：独立Bernoulli扩展不是相关隐藏线的真实联合概率；重汇合会出现平方项。
   门电路的软前向是确定性函数，不把它解释成真实密码位概率推演。
5. **覆盖与身份**：每层左线顺序覆盖输入；配置要求width≥64。固定右线不自配对（输入宽度>1时），
   所有接线保存为checkpoint整数buffer，严格加载后验证范围和非自配对；不会重新随机接线替换已学模型。
6. **评价合同**：train和eval均保持软门，不使用argmax硬化；不能称官方离散门推理复现。

来源：Petersen等2022，[Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277)；
[官方functional.py固定提交](https://raw.githubusercontent.com/Felix-Petersen/difflogic/469702c01ff0bfac9cdc6a395134252e11a56bd8/difflogic/functional.py)。
只读核查，没有安装difflogic或复制CUDA扩展。新来源记录在`provenance-logic.json`。

## 功能入口与已关闭的首批研究矩阵

CPU功能配置使用独立随机标签8/4/4行，batch4、1epoch、2更新，不生成真实密码标签：

```bash
uv run cipher-experiments check --config configs/innovation2-logic-smoke.json
uv run cipher-experiments train --config configs/innovation2-logic-smoke.json --output runs/i2-logic-null-smoke-001
```

当前准备环境源码验证可用`PYTHONPATH=src .venv/bin/python -m cipher_experiments.cli`，
无需重新安装或假定当前已安装wheel包含新实现。

以下矩阵原为提案，随后用户要求“推进试试”，已授权尝试；执行后因r3正控失败关闭，
见[logic16 GPU结果](INNOVATION2-LOGIC-GPU-RESULTS.md)。配置位于`configs/innovation2-logic-proposal/`：
r3逻辑正控、r4同期MLP-direct、r4 logic-direct、r4 logic-null。
同一新明文划分、key/init、support、width114、batch256/50epochs，DEV BCE选模后TEST一次。
每项拟用TRAIN65536/DEV8192/TEST65536、12800更新、1800秒协作上限，共4项/51200更新/7200秒。
data_seed191501、null_seed191502是新划分（不是历史orbit去重协议）；key190502、init17。
width114的逻辑候选5587可训练参数，对应MLP为r3 15619/r4 17101；不声称参数匹配。
这不是参数/吞吐完全匹配的比较；逻辑模型与MLP参数量、公开运算成本应分别报告。
不得沿用上一批已观察TEST来挑模型。先检验r3是否出现可学信号；未出现则停止后续并诊断，
建议预先固定r3门槛为TEST ACC≥70%且BCE优于两种TRAIN-only基线，未满足则停后续；
这是候选推进门槛，不是统计普适成功标准。不能借正控失败自动扩大网络或预算。
若r4模型与null/常量均相当，记录这一有界阴性；不能继续盲加epoch。

实现阶段仅完成局部合成控制；随后代码已发布为`2a969f7`，经Git同步和SSH启动一次GPU r3拟合。
r3 TEST49.72%、BCE0.693403、331秒、12800更新、exit0；后三项未启动。没有安装、重试或额外seed。
本批已关闭，不能以原批准或剩余预算自行重跑。
3层已知可表示目标的手工见证只证明该人工目标可表示；XOR小样本优化只证明局部门可训练。
它们都不代表SIMON训练成功或新方法已具有足够泛化能力。
