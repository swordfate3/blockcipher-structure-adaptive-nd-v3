# 两轮局部通道扩展：固定四项CPU对照

用户“继续推进”后，本批考察将局部标量输出扩为4通道是否改善端到端最终bit拟合。
不升轮、不重放旧checkpoint/旧TEST，不追加秘密标签或中间状态教师。
这是事先声明的探索性容量/拓扑干预，不是已证明单标量发生信息丢失。

## 模型与假设

旧scalar=`simon-r2-local`：三个独立4→16→1 Tanh局部MLP，sigmoid/center后3→8→1 Tanh组合，332参数。
新vector=`simon-r2-vector`：三个独立4→16→4局部MLP（仅隐层Tanh），12个logit逐通道sigmoid/center后
12→8→1 Tanh组合，557参数（3×148+113）。两者接线与公开X[j]输出sign相同。
只输入各q的公开明文`[Y[q],X[q+1],X[q+8],X[q+2]]`，q=j+1,j+8,j+2。
不输入U、真实key或状态，不通过真实局部函数/其他密文bit监督，所有参数随机初始化。

假设：扩大局部输出可减轻某些优化/表示压缩困难。竞争解释：参数量、随机初态与后端输入维度同时改变，
即使提高也不能唯一归因于标量丢信息；即使失败也不能说明压缩无关或神经学习不可能。
一个实数理论上可区分16角点，原标量候选不是信息论容量不足的已证结论。
不做等参数对照或多init搜索，本批仅同新面板固定一次。

## 固定顺序和预算（新分数前）

顺序scalar-real、scalar-null、vector-real、vector-null，各一次。
SIMON64/128两轮/目标bit0，prediction/simon/full/direct/hidden_width16；
CPU，TRAIN4096/DEV1024/TEST8192，data193301/null193302/key190502/init17，
batch256、epochs50、Adam lr.01、clip1，120秒每项协作上限非硬截止。
每项800更新/204800次TRAIN呈现，合计3200更新/819200次呈现。
两真实臂各13312次真实标量加密标签构造，合计26624调用，对应13312不同明文；null无真实加密。
公开零key基线和角点报告无秘密oracle标签查询。
同模型real/null初态相同；跨模型参数形状/RNG抽样不同，不要求同初态。同面板/标签/shuffle配对。
新面板只做无标签唯一性和与data192801/192901/193001/193101/193201的零交集检查。

## 验收、诊断与停止

选择最低DEV BCE，平局最早，strict selected load后TRAIN/DEV诊断，再生成TEST标签。
诊断无梯度、不参与训练/初始化/选模，仍只有公开U三bit8格/完整模型10bit签名seen-unseen的错误与冲突统计；
vector局部16角点报告每channel min/max/mean，scalar格式保留，不拟合真实隐藏状态或极性教师。
同一10bit签名重复不等于64bit明文重复；无逐条样本/TEST诊断保存。

每模型分别要求TRAIN≥99.9%、DEV≥99%、两BCE<.05并优于TRAIN-only常量与公开零key参考常量基线，
paired null DEV45–55%。主要对照描述vector的DEV ACC差值、BCE差值、TRAIN/DEV seen错误变化；
不凭相对提高替代绝对目标。没有额外相对差值“过线”门槛。
最终TEST real≥99%、BCE<.05且优于两基线，null45–55%，只固定报告。
外层先保存TRAIN/DEV及诊断的dev-decision后展开TEST；CLI自动保存TEST不称访问强制封存。
无论胜负四项后结束；技术失败停止剩余，不重试/调参/加epoch/加seed/升轮。
不因新模型输给旧模型临时换方案。若通过，也只建议另授权新面板/init确认。

## 工程

保持统一check/train/evaluate。Ruff/mypy/pytest、两模型各干净副本null两更新CLI、只读审查后freeze源码/配置/计划。
smoke每模型null8/4/4，batch4/epoch1，data193311/null193312，60秒；不计科学臂。
验证保存JSON、文件SHA、源码/配置身份、配对、进度/计数/选模/退出，正式checkpoint不重放、不再生科学标签或前向。
旧scalar参数名/行为及旧checkpoint兼容；不动创新1，保留全部dirty与main/他人分支。
不安装/GPU/SSH/commit/push。
