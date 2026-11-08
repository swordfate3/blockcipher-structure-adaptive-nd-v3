# 六轮MLP学习率诊断：固定三档、各真实/null（准备，尚未正式拟合）

## 当前状态：搁置，不执行

用户指出主线应回到结构学习后，本MLP矩阵搁置。配置保留为未执行提案，不能当作GPU/SSH/发布授权或待自动启动队列。
替代推进见[六轮神经轮展开优化核查](INNOVATION2-UNROLLED-OPTIMIZATION-PLAN.md)。

## 任务与模块职权

用户在后端职责迁移完成后要求继续推进实验。本阶段只新增小配置、合同测试和本文；生产模块、模型、指标、checkpoint及训练时序不再修改。
现有CLI负责执行，配置声明学习率与预算，训练器维持Adam/BCE/clip1；不向后端加调参器、调度器、registry或诊断分支。
正式GPU/SSH、代码发布和结果取回须在下列完整预算明确后再确认，旧批授权不延续。

## 可区分的问题

是否此前Adam固定lr=.001，相比更小lr，在相同六轮、数据、初态和更新预算内更难出现泛化？
本批三条件内部只改变学习率，不同时改网络、采样、batch、正则化或学习率调度。
相对历史批次，本批换新面板且各项延长至128epoch；因此只能用同期baseline对照归因，不能将相对旧TEST变化归因于lr。
本批能检验三档固定lr的有限预算效果，不能验证最优超参数、全部结构模型、全局收敛或任意晚期泛化。
选择普通MLP是为了接续已有明确的对照基线，不把其结果外推为轮展开或输入结构学习的结论。

## 冻结矩阵（6项）

| 条件 | Adam学习率 | TRAIN/DEV/TEST | epochs | batch | 更新/项 | DEV点/项 |
|---|---:|---|---:|---:|---:|---:|
| baseline real/null | .001 | 65536/8192/65536 | 128 | 256 | 32768 | 32 |
| medium real/null | .0003 | 同上 | 128 | 256 | 32768 | 32 |
| small real/null | .0001 | 同上 | 128 | 256 | 32768 | 32 |

- SIMON64/128六轮、预定MSB bit0；full/direct MLP64→128ReLU→128ReLU→1，24961参数。
- key_seed190502仅留数据侧，init17。新data194001、null194002，train_pool65536；六项共享明文、初态及shuffle，真实/null分别共享对应标签。
- 每4epoch评价DEV，所有项更新网格1024..32768/间隔1024，共32点。每项DEV最小BCE选best，平局最早。
- 六项总 **196608更新、50331648训练呈现**；每项1800秒协作上限，六项合计10800秒，不是硬截止或保证耗时。
- GPU方案仍待明确确认；若授权使用既有A6000环境，不安装、不升级、不CPU静默回退，每次启动前核现场GPU条件。
- 三个真实臂各139264次标签加密构造，总417792次调用/139264不同明文；数据集仅一份独立面板，重复训练不算更多独立数据。
- 无实际key/轮轨迹/其他输出bit/旧权重教师；从头训练，不重放历史正式checkpoint。

## 选模、跨学习率选择与TEST边界

现有CLI每项在严格加载selected之后都会生成并保存TEST，不声称访问强制封存。
外层先只读取TRAIN/DEV/progress，记录三档结果及DEV决定，再展开全部TEST数字。
若需挑一档，按真实臂DEV BCE最低、平局按baseline→medium→small；不得用TEST重选、择赢家重跑或追加seed。
三档×32DEV点共有96个真实模型候选，探索搜索自由度完整报告；不是独立确认性结论。
全部六项与null一起报告，不只报最佳；不从相似汇总数推断逐样本预测一致。

## 观察与停止

- 主观察：DEV BCE及准确率、相对TRAIN-only常量和paired null差值；结合已保存在线TRAIN BCE与DEV曲线，区分拟合和泛化。
- 延续探索门槛：真实DEV需ACC≥55%，且较常量和paired null各≥3pp，BCE优于常量；null在45–55%。这是探索推进规则，不是多重检验显著性定理。
- TEST同方向只作本批探索报告；若DEV未通过，不因TEST偶高而晋级；若通过，也需新面板/初始化/key的独立授权确认，不能自动称稳定能力。
- 若只TRAIN损失降低、DEV恶化/null类似，说明这几档下的已测更新仍未转为泛化，不证明更晚永远不会出现。
- 若三档皆近基线，关闭本有限矩阵；不声称结构学习不可行、不改成无限搜索。
- 单项失败/缺exit按实际状态报告，停止后续，不自动重试、换lr/后端、补seed、升轮或扩大数据。
- 不增加梯度日志或新指标代码；本批也不能定位饱和、裁剪或每层有效步长的唯一因果机制。

## 工程准备

配置：`configs/innovation2-learning-rate/{baseline,medium,small}-{real,null}.json`。
CPU随机标签入口：`configs/innovation2-learning-rate-smoke/{baseline,medium,small}.json`，每项8/4/4行、batch4、4epoch、8更新。
合同测试：`tests/test_learning_rate_configs.py`。仅跑随机标签工程闭环和无标签新面板预检，不生成本批正式cipher或null标签。
正式运行只使用既有`cipher-experiments check/train/evaluate`；本阶段不SSH、不GPU、不commit/push，不触碰旧runs。
