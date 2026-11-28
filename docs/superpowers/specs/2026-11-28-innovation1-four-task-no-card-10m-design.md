# 四任务全量撤卡对照

基线为 a0288e9 的四任务全量共享 M，结果来自 innovation1-four-task-card-10m-20261127-01。用户确认同时撤掉任务特定路由输入和专属小块表。

1. 四任务共用一个从零初始化的 M 与一个 Adam。Top‑2 网络接收 13 个固定 0.5，四项均使用原通用 local、word、nibble 表。保留 SKINNY 的输入回退和 SIMON、SIMECK 的关系通道。训练参数总数为 1,087,509。
2. 原固定数据每项 900 万 TRAIN、100 万 DEV、262,144 对 TEST。沿用原种子、批次、轮换顺序、学习率和宏 DEV BCE 早停与选模规则。30 轮上限，最多 216,000 次更新和 1,201,048,576 行前向。
3. 独立分支 feat/four-task-no-card-10m，运行 ID innovation1-four-task-no-card-10m-20261128-01，独占 G:\lxy\shared-m-cipher-nd-runs\20261128-four-task-no-card-10m-01。A6000 物理 GPU0，一次尝试，16 小时协作时限。结束后按保存的 TEST ACC 和 BCE 对照基线。本轮估计两项机制合计影响，使用同一历史 TEST 和一次配对初值。
