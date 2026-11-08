# 两轮神经候选：剩余错误定位与初始化对照

用户在上批报告后要求“推进”。本批定位两轮332参数local候选的剩余错误，
不重放上批checkpoint或重新生成旧科学标签，不把旧TEST变成设计集。不是更高轮实验或架构搜索。

## 固定研究问题与解释

- 同一新面板、相同数据顺序与预算下，init17/18是否都停在相似错误水平？
- DEV中完整模型公开输入签名已在TRAIN出现的行是否仍出错？若是，则数据未覆盖新签名不能单独解释错误。
- 错误集中于哪些公开U三bit组？局部隐藏sigmoid是否存在接近常数或不同动态范围？
  后两项仅描述，不能独立证明错误因果或隐状态学成真实密码状态。
- 新面板两init不是多key或充分统计重复；不能用两次差异证明某一种全局优化理论。
  本批对不同init同时有固定null，不按分数选赢家再声称稳定。

## 四臂与预算（新分数前固定）

按`init17-real、init17-null、init18-real、init18-null`顺序各一次。
`simon-r2-local`、SIMON64/128两轮、高半bit0、full/direct、hidden_width16；模型拓扑和332参数完全不变。
TRAIN4096/DEV1024/TEST8192；data193201/null193202/key190502，init分别17/18。
CPU、batch256、epochs50、Adam lr.01、clip1，每项800更新/204800次TRAIN呈现，
120秒协作上限而非硬限；合计3200更新/819200次呈现。
两个real各13312次真实标量加密标签构造，合计26624调用/13312不同明文，两个null无真实加密。
公开零key参考不是秘密oracle查询。初始化不影响标签与shuffle；同init real/null初态SHA必须一致。

新面板拟合前只生成无标签输入，与前四批data192801/192901/193001/193101比较零重合。
不改采样、学习率、轮数、key或更新数，不续训、不自动重试，不按某项成绩跳过其他项。
技术失败停止剩余臂；科学门槛失败仍完整报告固定四臂，然后停止。

## 诊断与TEST时序

训练选最低DEV BCE、平局最早；strict selected load后写selection.json。
随后仅对已有TRAIN/DEV作no_grad诊断，保存`diagnostics.json`，再生成TEST标签。
诊断不参与初始化/损失/选模/更新，不接收真实key、密码轨迹或额外bit标签，
不生成新科学标签，不向神经模型输入U。诊断源码只计算公开明文位函数。

报告聚合整体正确/错误/BCE，U[j+1],U[j+8],U[j+2]的8格行数/错误/残差一致性，
完整`model.indices`位签名的TRAIN覆盖、DEV seen/unseen错误与标签冲突，
三局部模块16个公共4bit角点的隐藏sigmoid范围/均值。未对比真实状态，也不拟合隐藏极性教师。
无逐条样本或TEST诊断；角点查询是确定的公开无标签工程探针，不是额外秘密oracle标签。

CLI仍自动保存TEST，不是访问强制封存。外层先读取TRAIN/DEV及诊断形成dev-decision，
只按固定规则描述错误（seen错误存在/未见、8格错误分布），之后展开最终TEST。
TEST只固定报告，不用于下一配置/加训练。本批任何结果都停止。

每个init分别按旧严格目标：TRAIN ACC≥99.9%、DEV ACC≥99%、TRAIN/DEV BCE<.05且优于两TRAIN-only基线，
paired null DEV在45–55%；最终TEST real≥99%、BCE<.05且优于基线、null45–55%。
两init都达目标才称本批两init均达标，不能挑高分init。就算均达标，也不等于跨key确认。

## 工程与验收

新增诊断模块、训练仅接入报告；旧模型参数/state schema不变。小合成/null测试、
Ruff/mypy/pytest、干净源码副本null的check/train/evaluate与只读独立审查先通过，再freeze源/配置/计划。
沿用统一CLI；正式验收仅读保存JSON/字节SHA/进度/选模/退出/配对，不重放正式.pt或再生标签/前向。
smoke配置为null8/4/4、batch4、epoch1，data193211/null193212、init17，只做2更新工程闭环。
不安装依赖/GPU/SSH/commit/push，main与全部已有dirty保留。
