# 创新2进度对齐与下一步

## 当前准备：六轮目标图有界向量单候选

用户继续推进后已实现[两项正式提案](INNOVATION2-STRUCTURE-VECTOR-PLAN.md)，`simon-vector`固定六轮bit0/full/direct/width32，10529参数。
406节点公开DAG、AND/XOR共享乘性交互、目标节点唯一读出；新增每节点L2半径2投影处理人工原型内部放大，不宣称继承人工87%或保证优化。
Ruff/mypy103/1093tests通过，干净源码CPU null CLI2更新/8呈现通过；初态和10倍参数压力探针激活受限/梯度有限，仅1次独立null工程更新。
准备时正式标签/拟合0；用户“推进实验”已确认真实/null两项各8192更新，共16384更新/4194304呈现及发布、Windows/A6000和取回，实际完成另记录。
不拿人工95%作淘汰条件，真实DEV BCE/ACC相对常量/null按预定规则描述，不将微差称显著；不自动追加确认。

## 最新机制检查：结构向量人工泛化87.11%，严格验收未通过

用户“验证下”后按[本地CPU人工合同](INNOVATION2-STRUCTURE-VECTORS-VALIDATION.md)完成三臂，[全结果](INNOVATION2-STRUCTURE-VECTORS-RESULTS.md)。
40节点/26门深的非密码人工图，向量8353参数TRAIN90.43%/DEV87.99%/TEST87.11%，null TEST49.41%，固定软门2参数TEST18.36%。
向量有部分未见输入泛化，但未达TRAIN99%/DEV95%，不证明完整门语义/六轮成功或向量普遍优越。
三fit各1024更新、共3072更新/393216呈现/exit0，无重试；没有真实密码标签/GPU/SSH/安装/发布。
深层内部向量幅度明显放大，不能据此唯一归因；机制检查结束、不自动追加或晋级六轮。
原型仅在tests，既有生产源码不改；1072tests/mypy98/Ruff通过，另有独立现有synthetic CLI回归2更新/8呈现，与机制预算分列。


## 最新完成：多通道残差CNN强拟合但未泛化

用户确认[四项方案](INNOVATION2-CONV-STRUCTURE-PLAN.md)后已发布并完成A6000训练/取回，
[完整结果](INNOVATION2-CONV-STRUCTURE-RESULTS.md)：普通环形CNN TEST49.7543%、公开旋转邻接CNN49.6429%，null约49.7%，所有DEV/TEST推进门槛失败。
64通道/6独立残差块/203329参数，只有offsets(0,1,2,3)/(0,1,2,8)不同；不是实际密码轮状态仿真。
末epoch在线TRAIN BCE real .0613/.0410，DEV BCE3.2141/3.7910，null同样拟合；说明强训练拟合与泛化分离，不称末TRAIN100%。
四项best均首次DEV点epoch4/1024更新，跨真实DEV BCE选local；没有rotation优越性证据，也非所有CNN不可能。
四exit0/32768更新/8388608呈现/48文件SHA核对/合计训练elapsed534.969s，无正式checkpoint本地重放或再生标签。
运行commit5fbf5871ac0087d9a6c9cd03b3439d47b6334435，干净发布副本1002tests/mypy91，本地完整1064tests/mypy96分别报告。
TEST由CLI自动保存/输出，外层DEV决定前读取完整JSON但不做TEST数值选择，不称sealed；全项披露无TEST-directed追加。
本批关闭，不自动加attention、数据、seed/key、学习率或步数；人工16角点成功仍仅工程，不是六轮正控。


## 最新完成：六轮轮展开三档学习率仍近基线

用户确认发布/GPU/取回后，按[轮展开优化合同](INNOVATION2-UNROLLED-OPTIMIZATION-PLAN.md)完成六项，
[完整结果](INNOVATION2-UNROLLED-OPTIMIZATION-RESULTS.md)：lr.001/.0003/.0001真实TEST50.1770%/50.2975%/50.2975%，null约50%。
按DEV BCE选baseline.001，三档DEV/TEST合取门槛全失败；TRAIN selected约50.08–50.56%，末在线BCE仍近.693，未见充分拟合。
合成探针梯度连通和数值差分通过不代表正式训练梯度全程有效，未定位唯一瓶颈；不否定全部结构学习。
六exit0/196608更新/50331648呈现、72文件核远端SHA、128progress每项、合计elapsed3118.328s，无正式checkpoint重放或标签再生。
运行commit199f3066c0ff827658d8335d7ff1113bd2c9bc13；只发布本批必要文件，干净发布副本914tests/mypy86，本地完整970tests/mypy91分别报告。
外层DEV验收曾先检查TEST聚合恒等式/有限性，后保存仅DEV选择；如实披露访问偏差，不称sealed confirm，未依TEST追加训练。
本批关闭；普通MLP学习率提案搁置，不自动扩大预算、seed、结构或数据。


## 最新完成：四倍数据／计算仍未使MLP六轮泛化

用户确认[六项方案](INNOVATION2-MLP-DATA-COMPUTE-PLAN.md)后完成发布、A6000运行与取回，
[结果](INNOVATION2-MLP-DATA-COMPUTE-RESULTS.md)：A65536 TRAIN/8192更新TEST50.1160%，
B262144 TRAIN/8192更新与C262144 TRAIN/32768更新均50.2106%，null约50%，DEV/TEST探索门槛全失败。
B比A DEV+1.07422pp但TEST仅+.09460pp；大数据继续训练仍选epoch4，未带来泛化信号。
多数据同更新缓解训练/DEV损失分化，但真实/null类似；不能说数据永远无关或模型完全没有拟合。
六轮bit0/direct MLP24961参数、同key/init、新嵌套面板193801、共同DEV8192/TEST65536，每项8次DEV选择。
六exit0/98304更新/25165824呈现，66文件核远端SHA，未本地重算科学标签/前向。
运行commit9335105b74aea724bd17096a6e8be1afacd9cac0；875tests/mypy80/Ruff、三条件CPU/CUDA null CLI通过。
预检辅助命令曾解析失败、执行前修正，无正式fit重试；本批关闭，不自动上百万数据或增加epochs。

## 最新完成：公开可逆坐标未改善六轮泛化

用户确认[本批四项方案](INNOVATION2-PUBLIC-COORDINATE-PLAN.md)后完成A6000运行与取回，
[完整结果](INNOVATION2-PUBLIC-COORDINATE-RESULTS.md)：原始输入TEST50.1526%、公开坐标50.2319%，null50.3006%/50.0458%。
DEV原始49.5239%、坐标49.4751%，坐标BCE略差；两种模式DEV/TEST探索门槛都失败。
phi(X,Y)=(Y XOR F(X),X)可逆、不是实际第一轮状态，真实状态仍相差未知K0；
同24961参数MLP、四项初始化SHA相同，无真实key/state/其他bit标签。
后期两模式真实/null在线TRAIN BCE降至约.51–.53，DEV BCE升至约.87；不能说全程无拟合或完整记忆。
新面板193701、65536/8192/65536，每项8192更新，四exit0/32768更新/8388608呈现；46文件核远端字节SHA，未重算科学标签/前向。
运行commit f9d58dde218a0586e34a26d13b6ebb998bbccf52；873tests/mypy79/Ruff、干净源码null CLI与现场CUDA null通过。
本批关闭，不自动加epochs/seed或多轮坐标，不把阴性结果当六轮不可学习证明。

## 前批完成：六轮直接MLP也未出现泛化信号

用户确认本批发布/远程预算后，按[四项合同](INNOVATION2-HIGH-ROUND-MLP-PLAN.md)完成A6000对照，
[完整结果](INNOVATION2-HIGH-ROUND-MLP-RESULTS.md)：unrolled TEST49.7238%、直接MLP49.7620%，null50.1343%/50.1495%。
同新面板65536/8192/65536，每项8192更新、四exit0；探索信号门槛DEV/TEST均失败。
MLP后期在线TRAIN BCE降到.517999，DEV BCE恶化到.870313，null类似；不能把best TRAIN近随机说成全程没拟合，
也不能据在线BCE推算未记录的最终TRAIN准确率。未观察到替代直接表示的泛化优势，不是高轮不可能证明。
发布版本3a7ba03eefd5349cd6135ac5a6d06a1bc727ada2，46文件取回核远端SHA，不重算科学标签/前向。
850tests/mypy77/Ruff、两模型干净源码null CLI及远端CUDA null预检通过。
本批32768更新/8388608呈现后关闭；不自动加预算、回低轮或升轮。

## 最新完成：远程6轮数据规模对照，无明显改善

用户确认发布与远程GPU后，按[本批合同](INNOVATION2-DATA-SCALE-PLAN.md)以A6000完成四项，
[结果](INNOVATION2-DATA-SCALE-RESULTS.md)：TRAIN4096/65536的真实TEST50.2060%/50.1068%，null50.0000%/50.1862%。
小TRAIN嵌套大TRAIN、DEV8192/TEST65536固定；每项8192更新、同32次DEV选模，全部exit0。
增16倍数据在本预算下没有让现有unrolled候选学起来，不能证明数据永远无关，也不做完整44轮安全结论。
发布/运行版本f151d0c12360fbc3050049d1456d2a2e0cfcf20e，远端无安装/升级、只fetch/checkout和CLI。
48文件已取回核远端SHA，未本地重算科学指标/标签/模型。849tests/mypy76/Ruff及本地null闭环、远端CUDA null预检通过。
四项32768更新/8388608呈现后关闭；不回低轮追分，不自动加预算或结构搜索。

## 前批方向与结果：停止低轮诊断，直接6/8轮探索

用户明确“推进高轮不要在低轮次浪费时间”，低轮结构/激活试验停止，不再以两轮达标为后续前置条件。
已按[6/8轮合同](INNOVATION2-HIGH-ROUND-PLAN.md)完成四项CPU实验，[全部结果](INNOVATION2-HIGH-ROUND-RESULTS.md)：
- 6轮TEST50.0244%、8轮50.2563%，各自ACC与TRAIN-only常量相同；两个null50.4395%。
- DEV51.0742%/49.0234%，探索信号门槛均失败；仅限共享轮模块/单key/init/800更新预算，不是高轮不可能证明。
- 4exit0、3200更新/819200呈现；839tests/mypy75/Ruff及干净源码r8 null CLI通过。
- direct构造/config仅扩上限到8，旧forward/init不变；不是架构突破，SIMON64/128完整44轮。
本批关闭，不回两轮追分、不自动扩大预算/GPU/SSH。高轮的替代路线仍需明确依据与另定预算，尚未实施。

## 前批：局部4通道扩展未改善本次学习

用户再要求“继续推进”，按[四通道合同](INNOVATION2-R2-VECTOR-PLAN.md)完成同新面板scalar/vector各real/null。
[完整结果](INNOVATION2-R2-VECTOR-RESULTS.md)：557参数vector TEST85.9863%，同期332参数scalar96.9482%，
两个null50.0244%/50.3296%。vector DEV83.3008%低于scalar95.9961%，两者均未达99%目标。
DEV seen错误scalar39/1000、vector167/1000；unseen分别2/24、4/24，多数错误仍在已见输入组合。
scalar错误区域本批转到公开U前两位11，不能把上批00组定位当固定缺陷或真实状态结论。
四项exit0、3200更新/819200次呈现；832tests/mypy73/Ruff、两模型干净源码null CLI通过。
不因扩宽失败继续扫结构/升轮，批次关闭；仅建议另行限定优化/压缩方式对照，尚未执行。

## 前批：新面板两初始化诊断，错误出在已见输入组合

用户再次“推进”后，按[诊断合同](INNOVATION2-R2-DIAGNOSTIC-PLAN.md)保持332参数模型和800步，
新面板init17/18各real/null四项已完成。[完整诊断](INNOVATION2-R2-DIAGNOSTIC-RESULTS.md)：
TEST96.6675%/92.7368%，两个null48.9624%/48.8525%，两init均未达99%与BCE<.05目标。
DEV错误42/75条全部在TRAIN已覆盖的完整10bit模型输入签名中，未见签名14条0错；
init17错误全部集中在公开U前两位00，不等于真实状态验证或唯一因果定位。
四项exit0、3200更新/819200次呈现；786tests/mypy71/Ruff和干净源码null CLI通过。
不重放旧TEST、不续训/升轮；下一步仅建议进一步局部映射/组合器诊断，尚未执行。

## 前批完成：两轮正控100%，新神经候选96.68%，仍未达99%目标

用户在AN/Fix启发讨论后要求“推进下实验”，已按[本批合同](INNOVATION2-R2-REDESIGN-PLAN.md)
完成table正控与local神经候选各真实/null，固定新面板、每项800更新、Adam.01。
[全部结果](INNOVATION2-R2-REDESIGN-RESULTS.md)：
- 四格结构化正控TRAIN/DEV/TEST100%，TEST BCE0.02966991；不是神经组合成功。
- 332参数独立局部神经候选TRAIN96.7773%、DEV97.0703%、TEST96.6797%（7920/8192），BCE0.07668525。
- 两个null TEST51.1597%/49.0601%。正控通过，神经候选未达TRAIN99.9%/DEV与TEST99%、BCE<0.05合同。
- 四项exit0，共3200更新/819200次呈现；782tests/mypy68/Ruff及两模型干净副本null CLI通过。
本批关闭；不改变采样方式、不升轮/加预算/GPU/SSH；未按TEST继续设计或训练。

## 前次决定：重设两轮目标，先诊断与设计

用户确认68.76%不足并要求推进，已完成[两轮重新设计](INNOVATION2-TWO-ROUND-REDESIGN.md)。
数学存在性构造表明原width32可以正确分类两轮目标，不能将低TRAIN直接归因于容量不足；
实际Adam能否找到仍未知。两轮单bit还可约化为公开特征上的四格未知效应表，
它只适合作独立结构化正控，不能替代神经组合学习成绩。
推荐新神经候选为目标锥上的三个不共享局部模块＋独立末端组合模块（设计332参数），
仍只用明文和最终bit监督，未实现/未训练。新浅轮目标建议TRAIN≥99.9%、DEV/TEST≥99%，
不追改旧70%合同；本轮仅公式/有限布尔检查，科学源码39份与前批冻结值一致。

## 前批结果：末轮复制项消融明显改善，仍未达推进门槛

用户在新面板复查后再次要求“继续”，已完成
[末轮复制项消融](INNOVATION2-FINAL-LOW-RESULTS.md)：同期XOR与仅屏蔽末轮center后low输入的候选，
各真实/null四项CPU，每项800更新。第一轮/监督/577参数保持，不输入真实key或轨迹。
候选TEST68.76%对同期XOR51.06%，两个null49.66%/49.60%，BCE也明显改善；
DEV66.41%虽较对照高13.09个百分点，仍低于预先70%合取门槛，因此批次关闭、不追加。
这支持该路径/轮角色问题值得研究，不证明复制项是唯一瓶颈或真实隐状态已恢复；
固定0也提供区分轮的线索，且同参数不等于同函数类。仍是单key/init/面板，未升三轮。

## 前批结果：新面板XOR效果变弱，未达推进门槛

用户随后要求“继续”，已完成独立新授权的[新面板四项复查](INNOVATION2-PLAINTEXT-XOR-PANEL-RESULTS.md)，
不改源码/key/init/预算，只更换明文面板与shuffle（data192901）及null192902。
XOR TEST51.49%、direct49.39%、两个null49.76%/49.77%；
XOR DEV51.56%略低于direct51.66%，BCE仍改善，但原推进门槛和弱信号全部方向重复条件均未通过。
四exit0/3200更新后关闭，不追加seed/epoch或升轮，不称跨key/init稳健。

### 前一面板的探索信号

用户在文献复核后明确要求“试试推荐”“试试推进”，已执行一次
[两轮公开XOR四项CPU对照](INNOVATION2-PLAINTEXT-XOR-RESULTS.md)，
[执行前计划](INNOVATION2-PLAINTEXT-XOR-PLAN.md)冻结每项800更新、同面板/同577参数初态。
绝对bit TEST：direct49.43%、公开XOR53.88%，两个null49.19%/49.65%；
XOR DEV53.42%虽BCE优于基线，但低于70%且提升不足10个百分点，预定门槛未通过。
四项均exit0，总3200更新；单key/seed仅小幅探索信号，不称两轮学会或轮数突破。
该批按原合同关闭；上述新面板批为后续用户单独授权，并非旧批续训。两批均无三轮、GPU、SSH或正式checkpoint重放。

[2026-10-22文献复核](INNOVATION2-LITERATURE-20261022.md)保留历史调研时点：
直接输出预测、整块/单bit监督、真实轨迹仿真和差分/线性攻击分别标记；
逐层最终标签学习的局部相关条件不能直接套用均匀SIMON。
原先共享/独立轮模块比较仍未实现或执行；多bit监督等仍只是候选，不自动授权新拟合。

## main当前新增：无轨迹神经轮展开

用户要求保存其他分支并回main推进；`review-fixes-20261020`原HEAD5241c62已原样保留，
额外本地引用`saved/review-fixes-before-unrolled-20261021-173603`，未合并其后续实验工作。
本main文档以下旧表不代表其他分支最新状态；原分支完整结果仍可从该引用读取。

[轮展开候选](INNOVATION2-UNROLLED.md)只用最终绝对单bit监督，不提供真实key或中间状态。
[首次CPU试验](INNOVATION2-UNROLLED-RESULTS.md)已完成：r3候选50.31%、null49.13%、
同期full MLP49.84%；每项800更新，exit0。预定DEV门槛失败，r4三项未启动，批次关闭。
没有稳定预测优势，也不能据此证明所有逐轮方法不可行。无GPU/SSH/安装/新提交发布。
随后[浅轮定位](INNOVATION2-UNROLLED-LOCAL-RESULTS.md)保持科学源码不变：人工单轮规则
两极性全角点均拟合，真实一轮TRAIN/DEV/TEST100%（null49.66%），真实两轮TEST48.94%
（null50.06%），训练也仅51.29%。一轮DEV门槛通过才执行两轮；本批止于两轮，未给真实轨迹或key。
局部模块能学，但末端单bit监督下的多轮组合仍失败，尚未隔离唯一机制；不自动追加训练。

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
| 公开结构＋未知常量学习 | 已实现simon-structure、计算见证/梯度/硬化与null CLI检查 | 未新增真实密码拟合，3/4轮学习能力待验证 |

历史事实仅来自只读档案：
`backups/v3-before-rebuild-20261019/docs/innovation2/`下
`PREDICTION-LEARNING-RESET-20261018.md`、`PUBLIC-RESIDUAL-NEURAL-EXECUTION-20261019.md`、
`PUBLIC-ALGEBRA-R6-FORMAL-EXECUTION-20261018.md`。它们不是新后端运行依赖或新授权。
新实测见[首批GPU结果](INNOVATION2-FIRST-GPU-RESULTS.md)。
旧新划分/训练实现/审计合同不同，不称逐字节复现历史结果。

最新推进见[结构接线诊断](INNOVATION2-STRUCTURE-DIAGNOSTIC.md)：先确认接线干预可影响人工目标学习，
并编译公开SIMON目标位依赖DAG；未新增GPU或密码拟合。
用户随后明确允许公开结构与未知参数学习，已接入[结构化SIMON候选](INNOVATION2-STRUCTURED-SIMON.md)。
参数类似密钥不等于泄露；禁止的是实际秘密注入，而不是从TRAIN中学习未知效应。

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
