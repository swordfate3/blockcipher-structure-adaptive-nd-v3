# 转向6/8轮：固定四项CPU探索

用户明确“推进高轮不要在低轮次浪费时间”。停止两轮结构/激活诊断，本批不跑任何低轮科学训练。
选取6轮和8轮，SIMON64/128完整44轮，故不称完整/接近完整轮攻击。
保持任务为未知固定key下明文→预定绝对密文bit0，不改成差分分类、SAT/key搜索或多输出状态监督。

## 候选与边界

复用现有`simon-unrolled` direct/full/width32，共享8→32→1 Tanh轮模块、sigmoid连续隐状态，
公开Feistel复制与旋转，学习[round,32,4]条件。条件不是注入真实key；隐状态未验证为密码真实状态。
仅将direct轮数能力1..6扩到1..8，旧forward/init/state语义不变；XOR模式仍仅两轮，structure仍≤6。
参数6轮1089、8轮1345（321+128r）。这是失败过浅轮的一条既有候选的有界高轮探测，
不是已有高轮有效依据的新方法；800步阴性不证明高轮不可学习或密码安全。
AN/Fix只提供机制对照与查询/计算预算口径，不给该任务高轮成功背书。

## 固定四项与预算

顺序r6-real、r6-null、r8-real、r8-null，各一次，不用6轮成绩决定是否运行8轮。
CPU，TRAIN4096/DEV1024/TEST8192，data193401/null193402/key190502/init17；
batch256、epochs50、Adam lr.001、clip1；每项800更新/204800次TRAIN呈现，
120秒协作max_seconds非硬截止；合计3200更新/819200次呈现。
不扩大数据/epoch或扫架构；lr.001沿用原unrolled，不按本批分数调参。
两个真实臂各13312次真实标量加密标签构造、合计26624次调用/13312不同明文（轮数不同），
null无真实加密；重复epoch不计新oracle查询，轮数也影响单次加密计算成本。
6/8同明文/shuffle，各自real/null同初态，跨轮数条件tensor形状不同不称同初态或同参数量。
新面板预检无标签唯一性与data192801/192901/193001/193101/193201/193301零交集。
高轮现有零key参考仅≤4轮，故本批不新增公开参考基线，不偷偷延伸实现；保留TRAIN-only常量基线。

## 判断和停止

选最低DEV BCE，平局最早；strict selected load后才请求TEST标签。CLI自动保存TEST，非访问强制封存。
外层先只读取TRAIN/DEV写dev-decision，再展开TEST。只报告整体ACC/BCE与常量基线，不做两轮特有诊断。
探索性信号条件每轮分别固定：DEV ACC≥55%，且比TRAIN-only常量与paired null均高≥3个百分点，
DEV BCE低于常量；paired null DEV45–55%。这不是此前两轮99%“学透”目标，也不倒改旧门槛。
TEST同条件单独固定报告（≥55%、两差≥3pp、BCE优于常量、null45–55%），不选择赢家或续训。
单key/init/面板无统计显著性或跨key宣称。即使达线也只称探索性信号待独立确认，不是高精度可靠预测。
技术失败停止剩余，无重试；科学失败不跳项，四项后无条件关闭、不加预算/seed/轮数。

## 工程与验收

统一check/train/evaluate，Ruff/mypy/pytest、干净源码r8 null8/4/4两更新CLI与独立只读审查后freeze。
smoke data193411/null193412、batch4/epoch1、60秒，只是工程，不算低轮科研或新科学臂。
验收只核保存文件SHA/JSON/进度/计数/选模/退出/配置源码身份；不重放正式checkpoint、不再生科学标签或forward。
使用现有环境，不安装/GPU/SSH/commit/push；main/他人分支与全部已有dirty保留。
