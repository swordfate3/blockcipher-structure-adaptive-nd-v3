# 六轮神经轮展开：梯度工程核查与固定学习率对照

## 状态与职责

用户要求不再围绕整体MLP，在已有输入结构/神经轮展开路线上“试试”。本阶段执行CPU合成梯度探针，准备正式对照，但不据此继承GPU/SSH/发布授权。
普通MLP学习率方案搁置，本地保留文件，不纳入本批发布或执行，不删除。

职责：不改模型、训练器、配置schema或报告模块。独立工程探针只在`tests/test_unrolled_gradient_probe.py`及忽略validation调用中；正式训练仍走统一CLI。
新内容只有测试、小配置、本文和工程证据。不能将探针塞入公共训练循环，或在整理中改变科学方法。

## 工程核查实际观察（非密码拟合）

固定init17、六轮bit0/width32，64条合成二进制输入及独立合成标签，CPU单batch、Adam.001一步。
模型共享8→32→1 Tanh cell、每轮/位置4个学习条件，1089参数，连续sigmoid隐状态；不是精确cipher状态、没有真实key或状态教师。

| 轮（1基） | 目标相关条件位置数 | 条件梯度L2 | 本轮cell输出梯度L2 | 条件更新L2（Adam一步后） |
|---|---:|---:|---:|---:|
| 1 | 26 | 1.42786e-6 | 5.02811e-6 | .00679151 |
| 2 | 20 | 9.44573e-6 | 3.01506e-5 | .00778229 |
| 3 | 13 | 5.94798e-5 | 1.81888e-4 | .00689944 |
| 4 | 7 | 3.73520e-4 | .00114556 | .00527537 |
| 5 | 3 | .00249649 | .00767804 | .00346381 |
| 6 | 1 | .02064594 | .06302790 | .001999998 |

- 图反向依赖集合独立推导；锥外条件梯度为0是预期，不能当断链。低半目标最后轮条件梯度应为0，另有测试。
- 各轮相关梯度有限且非零，共享cell也有梯度；第一轮条件梯度L2约为末轮1/14460。
- 各轮向量长度不同且有样本间相消，范数比只描述本点，不是逐条路径导数比或全程消失证明。
- 全参数裁剪前norm .09677198，小于1，本点没有触发缩放裁剪；step后各轮及所有参数tensor均发生变化。
- 合成BCE一步从.70166844降到.70096749，不是泛化或cipher信号；Adam归一化意味着小梯度不能直接等同于小更新。
- 抽查6个预定条件坐标（每轮第一个图相关位置的条件0）及两个cell权重，FP32autograd与不调用生产forward/cell的FP64参考中心差分一致；eps=.001/.0005。
- 这仍是同一指定架构的双实现核对，非独立密码正确性证明。仅8坐标/一个init/一个批次，不能证明每条梯度路径或全部训练轨迹正确。
- 观察支持存在明显前后轮梯度量级差异；不能归因唯一瓶颈，也不能证明调小lr会修复它。

证据：`validation/unrolled-optimization/gradient-probe.json`、源码SHA与工程测试。所有输入/标签合成，无实际密码调用、DEV/TEST或旧checkpoint。

## 正式探索矩阵（尚未运行）

| 学习率条件 | paired标签 | TRAIN/DEV/TEST | epochs | batch | 每项更新 | DEV点 |
|---|---|---|---:|---:|---:|---:|
| baseline .001 | real/null | 65536/8192/65536 | 128 | 256 | 32768 | 32 |
| medium .0003 | real/null | 同上 | 128 | 256 | 32768 | 32 |
| small .0001 | real/null | 同上 | 128 | 256 | 32768 | 32 |

固定SIMON64/128六轮MSB bit0，现有simon-unrolled/full/direct/width32，key190502仅数据侧、init17；新data194101/null194102/pool65536。
每4epoch DEV一次，更新网格1024..32768/间隔1024。六项相同新明文、初态、shuffle、模型/优化器，组内只变lr；真实/null分别相同标签。
相对历史8192步实验改变了面板与步数，只能使用同期baseline做lr对比；.001对应更长预算的同期基线，不拿旧TEST拼因果。
学习率较小可能需要更多步数；固定步数比较是预算约束下比较，不是等优化进度或全局收敛比较。

总预算：**6项、196608更新、50331648训练呈现**；三个real各139264标签加密，共417792真实调用/139264不同明文；三个null不加密。
每项1800秒协作上限、合计10800秒，不是硬截止或预测耗时。此处未授权GPU/SSH/安装/发布；若确认既有A6000执行，须先本地Git发布/远端明确checkout/现场GPU与null CLI预检，再逐项运行。
不SSH传科学脚本、不自动retry/加seed/增数据/改激活/另造模型。单项异常或缺exit按实际未知/失败报告并停后续。

## 决策与结果接收

最低DEV BCE选择checkpoint、平局最早；strict load之后才生成该项TEST标签。CLI自动保存TEST，不称强制封存。
外层先读取TRAIN/DEV/progress写决定，再展开全部TEST；若跨lr选择按真实DEV BCE最低，平局baseline→medium→small，不按TEST挑赢家。
三档×32DEV共96个真实候选，完整披露搜索自由度；这是探索，不是独立确认。
沿用历史探索信号条件：DEV ACC≥55%、较常量及paired null各≥3pp、BCE优于常量、null ACC45–55%。这是推进规则，不是显著性定理。
DEV不通过不得因TEST偶高晋级；若通过也只建议另授权新数据/init/key确认，不称稳定高轮能力。
全部real/null、selected及在线末epoch曲线一起报告；相同聚合数字不证明逐条预测一致，TRAIN在线损失不冒充末模型准确率。
正式结果取回只读保存JSON/字节SHA/身份/计数/exit，不再生标签或加载正式checkpoint前向。

若只TRAIN拟合改善、DEV不动/null类似，或三档都近基线，则关闭本有限矩阵；不能否定全部结构学习，也不自动搜索无限超参数。
合成梯度探针与正式训练分开：正式run不新增逐层日志，本批不能证明正式全程梯度有效或定位唯一因果机制。

## 配置与工程入口

- 正式候选：`configs/innovation2-unrolled-optimization/{baseline,medium,small}-{real,null}.json`。
- 随机标签CPU：`configs/innovation2-unrolled-optimization-smoke/{baseline,medium,small}.json`；8/4/4行、batch4、4epoch、每项8更新。
- 测试：`tests/test_unrolled_gradient_probe.py`、`tests/test_unrolled_optimization_configs.py`。
- 现有CLI：`cipher-experiments check/train/evaluate`；不添加新的生产入口。
