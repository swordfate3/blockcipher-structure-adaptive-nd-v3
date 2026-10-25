# 两轮公开XOR：末轮复制项机制消融（执行前计划）

用户在新面板复查结果后再次要求“继续”。本批限一次四项CPU机制对照，不继承旧批剩余预算。
保持main及已有工作，不安装/GPU/SSH/提交/推送。

## 假设和唯一干预

前两批plaintext-xor只有小幅且不稳定的准确率提升，BCE方向一致。
假设：输出已经显式XOR公开X0后，残差学习仍接收末轮low=X0的直连，可能带来无用输入或优化干扰。
替代解释：首轮隐表示没有学到适当函数、共享cell优化困难、有限预算/初始化等；本批不能唯一隔离它们。

新增`prediction_mode=plaintext-xor-no-final-low`，范围同plaintext-xor：
仅simon-unrolled/simon/full、两轮、高半bit。保持旧plaintext-xor作为同期对照。
原cell输入是center后的[low,ROL1(high),ROL8(high),ROL2(high)]＋4条件。
新模式只将**最后一轮所有32个位置的center后low通道设为常数0**，不把未center值0误当屏蔽；
第一轮完全不变，其Feistel复制仍保留，末轮其余旋转输入/条件保持。
输出仍乘`1-2*P[:,target_bit]`，标签仍是同一个绝对密文bit、BCE不变。
不删参数/改变init/冻结权重，不引入真实状态、key或额外标签。参数仍577。

这移除的是末轮cell的直接low坐标，不移除X0对首轮隐状态的影响；不能声称残差与整个X0输入无关。
同参数量不代表有效函数类完全相同，此处干预刻意限制一个输入路径，不是纯优化等价重写。
还需注意：第一轮该通道是±1，末轮固定0会额外提供区分两轮的线索；
若改善，不能区分“去掉无用复制输入”与“共享cell更容易区分轮角色”等解释。
连续隐状态仍非真实状态。此试验不作密钥恢复/密码安全/论文复现声明。

## 预算和数据

四臂顺序：`r2-xor-real`、`r2-xor-null`、`r2-masked-real`、`r2-masked-null`。
每项CPU，SIMON64/128减到2轮、bit0、full、width32、577参数；
TRAIN4096/DEV1024/TEST8192，data193001、null193002，key190502、init17。
batch256、epochs50、Adam.001、clip1，每项800更新/204800次呈现/120秒协作上限；
合计3200更新、819200次呈现、480秒协作预算，非硬截止。
新面板与shuffle由data_seed共同控制，不与历史批作纯单因子比较。
四臂内部明文、shuffle、init完全匹配，真实/null各自标签配对。
运行前只构建无标签输入，核对与前两批面板零交集，不再生旧标签。

## 判别（新分数读取前固定）

每项用最低DEV BCE选best，平局保留早epoch；strict load后生成TEST。
当前CLI自动保存TEST并非访问强制封存；先只读DEV保存dev-decision，再一次性展开TEST。
沿用同一推进标准，候选换为masked，对照换为同期xor：

1. masked-real DEV ACC≥70%，且高于xor-real至少10个百分点；
2. masked-real DEV BCE优于TRAIN常量及既有零主密钥公开参考常量基线；
3. 两null DEV ACC都在45%–55%，masked-real高于masked-null至少10个百分点。

另外如实描述是否DEV/TEST的ACC都增加、BCE都降低，但弱方向不替代以上门槛。
null异常则标记并检查工程，不先宣称科学成功。仅TRAIN改善不称泛化。
失败不意味着所有残差结构/逐轮学习不可行；成功只支持该路径干预在当前key/init/预算下有用，
不证明真实首轮状态已恢复或末轮low是唯一瓶颈。

四项固定一次完成；技术失败停余项，不自动重试。无论结果都关闭，不自动多seed/key/epoch/轮数，
不切BiLSTM/独立cell、多bit监督、真实轨迹、旧权重、GPU。
配置在`configs/innovation2-final-low/`；全新runs输出，绝不覆盖/续训旧结果。

## 工程和证据

测试通过hooks/手工计算验证第一轮相同、末轮只屏蔽center后channel0，其余不变；
同577参数/初态/state、合法边界、符号/BCE、null禁加密、strictload和TEST标签时序。
Ruff/mypy/pytest及干净源码副本null 8/4/4、两更新check/train/evaluate。
smoke独立data193011/null193012，不算第五科学臂。
所有工程检查通过后check四配置，冻结计划/配置/源码SHA。
运行后只核保存JSON、进度、文件SHA/字节数、计数/exit与选模，不加载正式.pt、不重生标签/前向。
