# STE四轮确认与五轮探针：执行前合同

用户在小规模4轮确认＋5轮真实/null探针建议后要求“推进试试”。
本批只用原有STE公开布尔结构＋Adam神经梯度学习，不改算法/样本量/学习率/目标位/选模规则。
无SAT、枚举、差分数据、GPU、SSH、安装、commit/push，不覆盖已关闭run。

## 固定设置

model_kind=simon-ste，CPU，目标MSB bit0，full/direct，hidden_width64占位，epochs50。
TRAIN4096 / DEV1024 / TEST8192，batch256，每项800更新/204800presentations，Adam.01、clip1。
各项max_seconds120协作预算（非硬进程截止）；一次运行，不追加seed/epoch或重试。
独立随机初始theta约±.1；硬AND/XOR前向、sigmoid叶子STE反向、固定±2logit。
只有TRAIN更新，DEV绝对BCE选择，严格加载选中checkpoint后生成TEST标签。

所有新项data_seed=192101，与上一批192001不同；init17/key190502/null191402除表中特别改动外保持一致。
新面板随机生成，不主动强制与旧批全域无交集，64位随机碰撞机会很低；manifest保存panel SHA。
新key项从随机初始化独立训练，不是把同一个模型直接迁移到新key。

| 顺序 | 配置 | 轮数 | init | key_seed | 标签 |
|---|---|---:|---:|---:|---|
| 1 | r4-panel | 4 | 17 | 190502 | cipher |
| 2 | r4-init18 | 4 | 18 | 190502 | cipher |
| 3 | r4-init19 | 4 | 19 | 190502 | cipher |
| 4 | r4-newkey | 4 | 17 | 190503 | cipher |
| 5 | r5 | 5 | 17 | 190502 | cipher |
| 6 | r5-null | 5 | 17 | 190502 | null |

配置目录`configs/innovation2-ste-confirm/`。

## 门槛与报告

- 四项4轮确认全部报告，不挑最好init当唯一结果，不因阴性自动改参数。
- 以预先指定的r4-panel（原init/key的新面板）作为进入5轮的唯一门槛：TEST≥70%且BCE优于两类TRAIN基线。
  init18/19/newkey只检验敏感性，不从它们中挑赢家替代r4-panel门槛。
- r4-panel未通过仍完成预定另外三项确认，但不启动5轮真实/null。
- 若进入5轮，无论真实臂结果如何，都执行匹配null，不因真实臂失败省掉预定对照。
- 5轮仅作为固定候选探针；高于50%但BCE不如常量不称完整通过。先完整报告准确率、BCE及null。
  若要声称“本探针正结果”，固定判据为TEST≥70%、BCE优于TRAIN固定概率、且比匹配null至少高5个百分点。
- r5+没有现成public_reference_constant（现实现只支持r1..4），不能捏造该基线成绩；
  五轮独立轮效应未强制key schedule，预测器包含真实函数但不宣称主密钥识别。
- 不使用TEST改超参数/挑位/筛选init；跨轮门槛含TEST观察所以仍属于探索性推进，非独立封存确认。
- 任一程序失败/超时即停止后续，保留失败证据，不重试。
- 批次最多六项、4800梯度更新、1228800presentations；不自动进入六轮。

此批不新增逐层梯度/符号翻转日志，以免在确认阶段改变实现。若五轮失败，只能先据现有TRAIN/DEV曲线
区分明显过拟合与无有效拟合；无法从现有记录定位唯一梯度机制，后续再单独设计诊断。
