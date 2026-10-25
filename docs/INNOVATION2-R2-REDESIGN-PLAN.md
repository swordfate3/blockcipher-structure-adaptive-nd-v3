# 两轮重新设计：首批CPU实验合同

用户在AN/Fix启发讨论后要求“推进下实验”。本批只测试已设计的结构化正控A和神经候选B，
不同时引入相关明文采样；不把旧批剩余预算当新许可。main与全部已有未提交工作保留。

## 模型与信息合同

A：`simon-r2-table`，4个零初始化logit。公共U=Y0 XOR F(X0)，A=U[j+1]、B=U[j+8]、
T=X0[j] XOR U[j+2]，输出(1−2T)*m[2A+B]。只由TRAIN最终bit学习四格，
不注入真实a/b/c或key。不做枚举求key、符号求解或真实状态教师。
这是强两轮结构先验下的正控，不是神经层间组合成功；学到的是固定key派生效应，不是完整key恢复。
训练内记录已有TRAIN数据的四格count/残差ones/冲突，仅用于检查覆盖与数据一致性，不增标签。

B：`simon-r2-local`，目标锥三个独立4→16→1 Tanh局部MLP，sigmoid后center，
独立3→8→1 Tanh末端，乘(1−2P_X[j])恢复绝对目标logit。332可训练参数。
无真实F值或U为教师，无四格表输出输入网络，不用旧权重，无共享参数或轮/位置条件。
只读明文与最终单bit标签，隐标量不是已验证真实状态。

两模型仅允许prediction/simon/full/direct、rounds2、高半bit、hidden_width16固定接口值。
`direct`在这里表示模型返回绝对bit logit（公开符号包含在模型内），不复用旧unrolled模式开关。
现有MLP/轮展开/创新1及旧checkpoint保持，沿用统一check/train/evaluate，不建新runner框架。

## 四臂、数据和预算

顺序table-real、table-null、local-real、local-null；四项固定一次，不看中途TEST调参。
每项CPU、标准SIMON64/128减到2轮、预定bit0，TRAIN4096/DEV1024/TEST8192；
新data193101/null193102，固定key190502/init17，batch256、epochs50、Adam lr.01、clip1，
每项800更新/204800次呈现/120秒协作上限。合计3200更新/819200次呈现/480秒，非硬限。

选择.01是在本批分数前固定的新设计优化设置，不继承旧.001；不根据结果再调。
A的凸结构不保证某个学习率下校准程度，B的表达性也不保证Adam在800步内学会。
两个模型参数量和初始化机制不同，**不要求跨模型同state SHA**；同模型真实/null必须完全同初态。
四臂相同明文与shuffle、真实/null分别标签配对；只预测同一个绝对bit，没有额外科学标签。
面板/shuffle共同随data_seed改变，运行前只生成无标签输入检查与前三批XOR相关面板零交集。
不跨key测试；单key/init/面板结果不作普遍结论。

本批用于可行性诊断，不是同函数类/同参数量架构公平性证明；B与旧577参数模型同时改变了
共享、条件、计算拓扑及本批学习率，不能将跨批改进唯一归因于“不共享”。
按现有生成器每真实臂各构造13312个最终bit标签（每次调用真实标量encrypt），
两个真实臂合计26624次加密构造调用、13312个不同明文；两null臂不调用真实加密。
这与819200次训练呈现不是一个预算概念，公开零key参考计算另列不算秘密oracle查询。

## 验收与停止（任何新科学分数前固定）

每臂最低DEV BCE选best，strict load后才生成TEST标签；现CLI自动保存TEST，不是强制封存。
四臂完结先仅提取TRAIN/DEV及TRAIN四格元数据，保存dev-decision，再展开最终TEST。
技术失败停余臂，不重试。预先不根据某臂科学分数跳过其他臂，以完整四项诊断报告收尾。

各真实模型浅轮训练/DEV门槛同时要求：

- best TRAIN ACC≥99.9%、DEV ACC≥99%，且两split BCE均<0.05；
- TRAIN与DEV BCE优于本臂TRAIN-only常量及既有零主密钥参考常量基线；
- 其配对null DEV ACC在45%–55%。

最终TEST目标单列：真实ACC≥99%、BCE<0.05且优于两基线，null ACC在45%–55%。
TEST不用于改变配置/选seed/延长训练。A另要求四格TRAIN都覆盖、真实残差格内无冲突。
若A成功而B失败：只说明强结构正控可学习，而当前B/预算未学会组合；不称神经路线成功。
若A失败：先查实现/四格覆盖/优化，不拿B的偶然分数掩盖；不自动修参重跑。
若B成功：也只建议后续独立面板/init/key确认，**本批无论结果都关闭，不自动升轮**。
旧70%门槛保持为历史合同，不因本批99%目标倒改过去结果。

## 工程预备与核验

配置目录`configs/innovation2-r2-redesign/`；smoke目录为同名前缀-smoke/，每模型独立
null8/4/4、batch4、epoch1、2更新，data193111/null193112、60秒，只做工程闭环。
小tensor/公开布尔角点控制不当科学持出成绩。不存在真实状态预训练或样本外教师。

先做Ruff/mypy/pytest、干净源码副本各模型null check/train/evaluate、独立只读审查；
四配置check和源/配置/计划SHA冻结后才开始正式拟合。输出为全新runs目录，不覆盖/续训。
结果只核保存JSON/字节SHA/进度/计数/退出/选模；不加载正式.pt、不再生科学标签、不重算科学前向。
复用现有环境，无安装/GPU/SSH/commit/push。
