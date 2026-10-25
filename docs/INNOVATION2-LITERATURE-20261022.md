# 创新2文献复核：输出预测、逐轮学习与监督信息

日期：2026-10-22。调研完成时状态：**阶段性调研完成，当时模型修改和训练仍暂停**。
本轮调研只检索、阅读、核对数学条件和整理文档；未执行新拟合、checkpoint前向、GPU、SSH、安装、提交或推送。
后续状态：用户另行要求试推荐方案，已完成[公开XOR四项CPU对照](INNOVATION2-PLAINTEXT-XOR-RESULTS.md)，
其TEST53.88%但DEV推进门槛未通过，批次关闭。以下第6节保留调研当时的候选表述，不代表后续未执行。

## 1. 结论先行

1. **直接输出预测是已有研究路线，但本轮未核实到与当前合同完全一致的标准SIMON64/128深轮成功结果。**
   这不是“不存在”的证明，也不是系统综述。最相关的Simon论文是Watanabe等的ePrint **2024/1310**，
   其目的包括发现结构弱点，再用扩散/线性分析解释；不能把它描述成已经解决我们的单bit任务。
2. 很多可读实验用**最终整块输出**监督，而本项目当前只给最终一个绝对密文位。
   整块监督不是自动泄露真实密钥，但提供了额外标签，可能显著改变组合学习；不得静默加入。
3. **LSTM沿输入序列计算，不等于按密码轮数学习真实状态。** 深层网络也不等于逐密码轮监督。
   可读输出预测论文不支持“只要把密码展开成多轮神经模块就一定容易训练”。
4. 确有只用最终标签的逐层学习理论[7]，但它要求树形接线和局部相关等条件。
   均匀明文的SIMON存在真内部门与最终标签不相关的情形，不能直接套用其保证。
5. “弱化PRESENT成功11轮”[3]的具体含义是**16-bit小分组、替换弱S-box、整块精确命中率高于极低随机基线**，
   不是标准64-bit PRESENT的11轮单bit高准确率，更不是全轮标准密码攻破。
6. 暂不继续原先共享/独立轮模块六臂比较。更优先的是核对**监督合同、公开可消除的XOR掩码和局部可学习信号**，
   再选单一干预；第6节只有候选建议，不是训练授权。

## 2. 比较基准与证据级别

当前基准：标准SIMON64/128的约定减轮版本；固定未知key；均匀唯一明文划分；
输入仅明文；只用最终预先指定绝对密文bit；不提供真实key/轮密钥/中间轨迹/其他密文位；
允许公开结构、公开明文函数及从TRAIN学习未知效应；DEV选模后TEST一次。

减轮标准密码不是toy；缩小分组、替换S-box/旋转常数、缩小密钥空间则需分别标记。

- **正文已读**：取得原论文的相关方法、公式和结果段；仍未独立复现其代码或成绩。
- **部分原文**：摘要、出版元数据、公开附录/表格；正文缺失的协议字段保留未知。
- **摘要级**：只能采信摘要明确描述的任务，不补造实现细节。
- **本项目推导**：从公开SIMON方程推导，独立于论文作者的实验声称，不是新拟合结果。

## 3. 核心证据矩阵

| 来源 | 对象/任务 | 监督、密钥及输出 | 本轮证据与预算 | 与当前合同的关系 |
|---|---|---|---|---|
| [1] Xiao–Hao–Yao，2019 | 减轮DES的明文→密文；另有Hitag2任务 | 固定映射E(P,k)，最终64个并行二分类输出；不是单bit监督。Hitag2另为内部输入状态→输出bit，不混作未知状态下攻击 | 正文§III–IV已读；DES示例TRAIN/TEST各2^16，最大350epochs、batch1000；浅宽1000节点优于其深窄/级联候选 | 可参考浅宽基线、逐bit与复制半部基线；不能据此断言SIMON或所有深网络更差 |
| [2] Kimura等，ePrint2021/401、出版2022 | small PRESENT/AES/TWINE及32/64-bit扩展；CP/PR | 原始完整方法未取得；不能用后续论文协议填充本篇空白 | 部分原文：摘要、Springer附录A–F与表格。附录F的64-bit数据范围2^10–2^19、TEST2^16；表14/16的PRESENT四轮CP/PR均无成功值 | 是直接输出预测核心来源，但AES-like/TWINE-like是作者构造，不是标准AES/TWINE；概览表与详细表轮数需结合缺失方法解释 |
| [3] Kimura等，2023 | 两种弱S-box的small PRESENT-[4]，分组16bit；CP/PR | §3.1明确同一秘密key；§3.3仅P→最终C或反向，输出16bit，无真实中间状态监督；跨key独立拟合，不是一个模型零样本跨key | 正文已读。每key TRAIN2^15、TEST剩余2^15；100epochs、batch250、LSTM；调参用20keys、100候选，最终另100keys | 最清楚的可读协议；说明最终标签可用于直接学习，但标签多、分组小、部件弱、调参成本大，不能直接挪用轮数 |
| [4] Watanabe–Ito–Ohigashi，ePrint2024/1310 | Simon及Simon-like结构的输出预测；扩散性质和线性密钥恢复解释 | 公开表1：输入/输出16或32节点，非一个输出节点；每key/跨key细节、真实状态监督等正文信息仍未完整核实 | 部分原文：摘要、Springer公开内容与表1。LSTM300节点、1层、MSE/Adam、lr.01、batch250、100epochs、TRAIN/TEST各2^15（表中带脚注，未补猜其适用范围） | 最相关的Simon文献；不是标准SIMON64/128预定单bit的已验证基线，也不能把后续线性攻击混作NN预测成绩 |
| [5] Jeong–Ahmadzadeh–Moon，2024 | DES/AES/SPECK32/64、SDES/SAES；EE/PR/KR/文本分类分别报告 | 位数组EE为P→最终整块C，PR相反；BiLSTM按输入半块组织序列，不是密码轮展开；EE固定key细节本轮未从正文定位到充分说明，不以2026篇补全 | 正文§3.3、§4.1–4.3已读；位数组300epochs、batch128、AdamW.001；标准大分组实验TRAIN至2^22、TEST2^15 | 表3报告SPECK32/64三轮EE：FC .587、BiLSTM .883，**是整块平均bit准确率**；这是作者报告而非本项目复现，不是SIMON单bit成绩 |
| [6] Jeong–Park–Moon，2026 | 同一研究线的集中/联邦EE、PR；DES/SPECK等1–4轮 | §3.2.1明确全体样本同一固定秘密key；最终整块监督，未描述真实逐轮状态监督；序列RNN不等于逐密码轮训练 | 正文已读；TRAIN2^20、TEST2^15；联邦T10×本地E10、batch4096、lr.0001，3层256 BiLSTM；客户端顺序模拟 | 表5/7中集中DES/SPECK二轮起约.5；未展示高轮突破。16bit toy用2^20样本且未见去重保证，不能认定全部持出明文未见 |
| [7] Malach–Shalev-Shwartz，2019/2020v2 | 树形Boolean电路的逐层梯度学习理论，不是密码攻击实测 | Algorithm1只给最终(x,y)，对每层平均池化输出用最终y；无真实内部门标签 | 正文§3–5已读；hinge+破对称正则、固定随机第二层权重、局部相关LCA及分布条件；不是当前Adam+BCE算法 | 提供“最终标签也可能逐层学习”的条件性根据，**不提供均匀SIMON上的可学保证** |
| [8] Bellini等，SecITC2021/出版2022 | 黑盒Boolean函数/密码仿真的限制、输入依赖与资源估计 | 强调随机Boolean函数和结构未知黑盒；明确不讨论利用公开内部结构的技术 | 已读26页accepted manuscript；§3/§4及结论；不是对所有结构化神经密码分析的无条件下界 | 可用于反驳“通用逼近所以必然容易学”，不能证明本项目公开结构方法无效或任意单bit优势不可能 |
| [9] Singh，ePrint2025/1069 | PRESENT逐轮APG/MLP链式仿真 | 摘要明确“excluding the key”，聚焦SPN；只在摘要层确认逐轮输入输出关系，训练标签如何取得仍未知 | 仅摘要；PDF403。预算、代码、完整key处理均未核实 | **排除作为未知key标准PRESENT全轮攻击证据**；不得把历史笔记中的修改细节冒充本轮已读正文 |
| [10] Kim等，2023 Revisited | S-DES/S-AES/缩减key空间S-SPECK，输入(P,C)→key | 真实key作为训练标签；输出密钥bit，不是P→C_j | 正文§2–4已读；残差/GLU可作架构线索，但结果与受限key空间绑定 | 不纳入直接输出预测轮数比较；“参数减少93.16%”属于S-DES密钥预测，不能转述成SIMON输出预测提升 |

### 3.1 必须保留的指标区别

- [1]的cipher match rate与[5][6]的BAP_avg都是**跨输出位平均**，不能推断预先指定bit0有同样准确率。
- [3]§3.3 Step4明确使用“percentage of exact match”，即**整块完全命中**。
  16bit、半数域TRAIN时，作者成功阈值为2^-15。
  表7弱S-box1的11轮CP为2^-14.36，表8弱S-box2的8轮CP为2^-14.35。
  这些数既不是85%或95%的bit准确率，也不是可任意选择明文的高成功仿真。
- [2]附录表14/16：64bit PRESENT三轮CP在2^19 TRAIN下列2^-0.74，四轮各数据量列0；
  不把综述/概览表的“4轮OP”直接变成“正文已复核标准PRESENT四轮单bit成功”。
- PR（密文→明文）、KR（输入含P/C、目标key）、差分/线性区分器和原始P→C_j分开。
- [1]明确一轮Feistel整块的复制半部基线为75%，而非50%。本项目固定计算半部bit0是必要控制，不能靠复制位抬高成绩。

### 3.2 可信度与协议注意事项

1. [1]Algorithm1反复用同一M2比较网络、样本量并停止；按本项目标准这是探索/选模用途，不能当未经使用的最终确认集。
2. [3]调参keys和最终keys分开是有价值的设计，但其报告仍不是单个网络不经新样本拟合就跨key预测。
   论文中的“whitebox analysis”主要指事后分析经典攻击性质，不自动表示模型训练读到了真实key/状态。
3. [5]表3的三轮SPECK结果比当前单bit试验使用更多监督，且2^22×300约12.58亿样本呈现；
   不能简单解释为“换BiLSTM即可复现”，也不能把该预算自动转为新的训练批准。
4. [6]表9正文把DES两轮FC/BiLSTM数值与表内一轮数值混写，且.8273被用来说明优于.4999；引用以表和上下文分开保留，不据此选模型。
   其“p不显著即安全”表述不成立；全块相关bit也不能未经检查当作相互独立Bernoulli重复。
5. [6]16bit域仅65536个输入却使用1048576训练样本；正文未建立唯一明文隔离。
   这是泛化证据不足的风险，不是已经实证其代码发生泄漏。
6. 没有下载/运行这些论文的作者科学代码，本轮未确认核心结果具有可直接执行的公开复现包。
   论文未提供链接与断言“绝无代码”不同。

## 4. “只用最终标签逐层学”究竟成立在什么条件下？

[7]Algorithm1的输入确为最终样本S⊆X×Y。它从底层开始，每次固定已学层，
对新层输出做平均池化，再相对同一个最终标签y优化；没有要求提供真实中间门答案。

但是它的保证依赖：

- 已知的满二叉树接线，叶子对应输入位置；不是有复制、重汇合、共享参数的任意密码DAG。
- 每个有影响门g与最终标签Y有足够局部相关，原文LCA要求|E[gY]|超过标签偏置一个正裕量。
- 定理还包含输入/分布及模式概率条件、特定初始化/正则/优化过程。
- §5.1的parity正结果使用各坐标概率与1/2有间隔的乘积分布，**不能遗漏其远离均匀分布的条件**。
- §4.1的困难结论针对相应函数类与带噪梯度/SQ框架，不是每一个具体SIMON实例不可学习的证明。

### 4.1 SIMON两轮的公开数学检查：影响不等于相关

以下为本项目推导，不是新实验或论文结果。令X0、Y0为独立均匀32bit初态，K0、K1为任意固定轮密钥，
F(x)=(ROL1(x) AND ROL8(x)) XOR ROL2(x)。按本项目高半字约定：

```text
X1 = Y0 XOR F(X0) XOR K0,  Y1 = X0
X2 = X0 XOR F(X1) XOR K1,  Y2 = X1
```

对任意x,z，给定X0=x、X1=z唯一确定Y0，因此P(X0=x,X1=z)=2^-64。
所以(X0,X1)仍是独立均匀的32bit向量对。对计算半部任一预定j：

```text
P(X2_j=1 | X1=z) = 1/2
```

即**整个X1单独都不能预测X2_j**，虽然其中坐标通过F参与了计算。
此外F_j在均匀X1下平衡：独立公平的旋转2输入bit对旋转1/8的AND做XOR。
于是X2_j也独立于整个X0单独，但(X0,X1)联合可完全确定它。

含义不是“信息不足无法预测”：明文含X0,Y0，固定key效应仍可从TRAIN学习，神经表示也不必等于真实状态。
它只说明：不要把“首轮每个真状态坐标会影响终点”误当作“每个坐标有可用于贪心学习的相关信号”。
仅看X1的辅助预测头在此分布下没有信号；能联合看(X0,X1)的非线性头不受此结论排除。
复制输出Y2=X1不适用这个结论；非均匀/条件化输入也不能照搬。
这不是当前共享模块失败的唯一原因诊断，不证明端到端反传无效。

## 5. 对此前推进思路的修正

| 思路 | 本轮复核后的判断 |
|---|---|
| 逐轮输入真实状态，再预测下一轮 | 能变成有监督仿真，但改变合同；SIMON真实相邻状态直接暴露轮密钥，不能声称完全不含秘密信息 |
| 只给最终bit、内部按轮组织 | 合同允许；当前一轮成功两轮失败。文献不保证该归纳偏置容易优化，也不证明不可行 |
| 每个隐藏位置加同一最终bit辅助头 | 不优先；先检查其输入与目标是否有信号，不能机械套[7] |
| LSTM/BiLSTM直接P→最终C_j | 合同允许；有整块输出预测的邻近文献支持，单bit收益与SIMON适用性尚无验证；不把序列位置称密码轮数 |
| 最终整块C监督、仅报告预定bit0 | 可作为另一个合理合同，不直接给实际key/轨迹，但额外给了63个标签；必须显式变更并报告样本数与标签bit数 |
| 神经结果最终用SAT/代数搜到函数 | 不能替代神经学习效果；作为经典对照单列 |
| 因失败增加数据、epochs或更大模型 | 本轮没有支持自动扩大预算；先分辨监督信息、任务难度和优化干预 |

## 6. 下一步排序：候选而非执行

### 优先候选A：不增加标签，先做公开XOR跳连/残差的两轮因果对照

由上式，令g(P)=X0_j（输入明文中对应高半位），则同一个终点标签可确定：

```text
D_j = C_j XOR g(P) = F(X1)_j XOR K1_j
C_j = D_j XOR g(P)
```

g完全公开，不调用真实key、真实状态或其他密文位。可通过已知符号翻转logit等价实现绝对标签BCE，
不需要额外加密查询。此思路避免让模型在输出端同时学一个公开的XOR掩码，
也可能暴露被该公平掩码遮掉的局部相关。

**这是有机制根据但未实测的新假设**：不能由它推出端到端优化必然成功，也不能直接套[7]定理。
现有`simon-unrolled`已经输入复制路径，但“把公开XOR显式提出输出端”和“指望MLP自己学会XOR”不是同一参数化。
此前零主密钥公开参考残差的阴性结果仍保留；此处g只是一条明文位，不能宣称旧结果已验证本干预。

若用户决定继续，先单独冻结一个新两轮协议：direct与公开跳连残差、各自null；相同基础模型、
新明文面板、相同初态与更新预算；唯一主要干预为输出参数化。
在DEV预定门槛前不查看TEST挑方案；失败即停，不自动扩到更深轮。
本轮未新建这批配置、未改源码、未训练。

### 候选B：允许新增最终多bit标签时，先对齐文献监督再比较架构

如果研究目标允许“最终整块标签，但无实际key/轨迹”，可另立合同比较单bit与多输出辅助监督，
统一报告预定bit0、所有bit、两半部和整块命中；不给复制半部平均分冒充优势。
同明文数量不等于同标签信息量，应同时记N和N×监督bit数；多输出head的参数差异也应披露。
LSTM/BiLSTM可作为文献基线，而不是先赌更深的按轮仿真。
这是合同变更选项，**不在当前最终单bit批准内**。

### 暂缓

共享/独立模块消融、额外seed/epoch、大规模GPU拟合、真实状态教师、跨key预训练等都先不启动。
先解决信息合同和最小因果对照，比在未对齐协议上继续堆候选更有价值。

## 7. 来源、书目信息及可追溯路径

1. Ya Xiao, Qingying Hao, Danfeng (Daphne) Yao. **Neural Cryptanalysis: Metrics, Methodology, and Applications in CPS Ciphers**，2019。
   [arXiv1911.04020v3](https://arxiv.org/html/1911.04020v3)，§III、§IV-A/C、Algorithm1。
2. Hayato Kimura, Keita Emura, Takanori Isobe, Ryoma Ito, Kazuto Ogawa, Toshihiro Ohigashi.
   **Output Prediction Attacks on Block Ciphers Using Deep Learning**。
   [ePrint2021/401](https://eprint.iacr.org/2021/401)；出版于2022，
   [DOI10.1007/978-3-031-16815-4_15](https://doi.org/10.1007/978-3-031-16815-4_15)，LNCS13285，248–276。
   ePrint原始HTML、BibTeX与出版信息均含首作者Hayato Kimura；部分可读抽取漏显示该名字，
   因此作者表按原始页面与出版信息核对，不将抽取缺失当作原网页缺失。
   [表14](https://link.springer.com/chapter/10.1007/978-3-031-16815-4_15/tables/14)、
   [表16](https://link.springer.com/chapter/10.1007/978-3-031-16815-4_15/tables/16)。
3. 同上六位作者。**A Deeper Look into Deep Learning-based Output Prediction Attacks Using Weak SPN Block Ciphers**。
   Journal of Information Processing31，550–561，2023。
   [DOI10.2197/ipsjjip.31.550](https://doi.org/10.2197/ipsjjip.31.550)，
   [原PDF](https://www.jstage.jst.go.jp/article/ipsjjip/31/0/31_550/_pdf)，§3.1–3.3、§4.1、表6–8。
4. Hayato Watanabe, Ryoma Ito, Toshihiro Ohigashi.
   **On the Effects of Neural Network-based Output Prediction Attacks on the Design of Symmetric-key Ciphers**。
   [ePrint2024/1310](https://eprint.iacr.org/2024/1310)；
   [会议DOI10.1007/978-3-031-76934-4_13](https://doi.org/10.1007/978-3-031-76934-4_13)，在线2024/印刷2025；
   [表1](https://link.springer.com/chapter/10.1007/978-3-031-76934-4_13/tables/1)。
   后续期刊条目DOI10.1016/j.jisa.2025.104016（2025）不等于本轮已读其完整实验正文。
   ePrint导出的doi字段为会议网站URL，不是有效DOI，不原样录入参考文献库。
5. Ongee Jeong, Ezat Ahmadzadeh, Inkyu Moon.
   **Comprehensive Neural Cryptanalysis on Block Ciphers Using Different Encryption Methods**。
   Mathematics12，1936，2024。[DOI10.3390/math12131936](https://doi.org/10.3390/math12131936)，§3.3、§4.1–4.3、表3。
6. Ongee Jeong, Seonghwan Park, Inkyu Moon.
   **Scalable Neural Cryptanalysis of Block Ciphers in Federated Attack Environments**。
   Mathematics14，373，2026。[DOI10.3390/math14020373](https://doi.org/10.3390/math14020373)，§3.2、§4.1、表5/7/9。
7. Eran Malach, Shai Shalev-Shwartz. **Learning Boolean Circuits with Neural Networks**。
   2019，读取2020v2。[原文](https://arxiv.org/html/1910.11923v2)，Algorithm1、Assumption1、§4.1、§5.1。
8. Emanuele Bellini, Anna Hambitzer, Matteo Protopapa, Matteo Rossi.
   **Limitations of the Use of Neural Networks in Black Box Cryptanalysis**。
   SecITC2021，出版2022，LNCS13195，100–124。
   [DOI10.1007/978-3-031-17510-7_8](https://doi.org/10.1007/978-3-031-17510-7_8)，
   [机构仓储](https://iris.polito.it/handle/11583/2975409)。读取accepted manuscript，§3–4及结论。
9. Gopal Singh. **PRESENT Full Round Emulation : Structural Flaws and Predictable Outputs**。
   2025预印本。[ePrint2025/1069摘要](https://eprint.iacr.org/2025/1069)。仅摘要级。
10. Hyunji Kim, Sejin Lim, Yeajun Kang, Wonwoong Kim, Dukyoung Kim, Seyoung Yoon, Hwajeong Seo.
    **Deep-Learning-Based Cryptanalysis of Lightweight Block Ciphers Revisited**。
    Entropy25，986，2023。[DOI10.3390/e25070986](https://doi.org/10.3390/e25070986)，§2.3、§3、§4。

### 检索与本地留存范围

检索关键词：`neural network block cipher output prediction`、`round neural cipher emulation intermediate states`、
`learning boolean circuits neural networks correlation`，并沿档案线索/原论文引用追踪上述来源。
OpenAlex/Crossref在线三查询脚本200秒超时；随后offline仅恢复第一查询两家各5条，去重8条，
其余两查询cache_miss，report明确`partial: true`。Exa429、Brave无key；ePrint核心PDF与若干MDPI HTML403。
没有更改provider/凭据、启用托管回退或安装软件。MDPI静态PDF和J-STAGE原PDF取得成功，
没有将HTML拒绝访问误报为全文已读。

本地研究产物（Git忽略）：`validation/literature-innovation2-20261022/`：

- `search.log`：在线超时的部分日志，不是完整检索结果。
- `offline-search.log`、`search-results.jsonl`、`search-results.jsonl.report.json`：部分缓存检索。
- `kimura2023.pdf`、`jeong2024.pdf`、`federated2026.pdf`、`kim2023-revisited.pdf`：原PDF。
- `sources/`：可读网页原始HTML、已有read转换文本及从HTML去标签的辅助文本；转换文本不替代PDF公式核对。
- `source-manifest.json`：下载来源、字节数、SHA；仅对明确重新抓取的网页记录抓取时间，未倒填旧下载时间。

旧档案阅读笔记只作检索线索，不能升级为本轮正文证据。上述论文没有在本项目独立复现。
本次只改研究文档，不重新跑已关闭的训练或715项测试；此前715passed/mypy62files仍是此前源码验证记录。
本轮核对39份科学源码SHA与浅轮批次冻结身份一致、本地Markdown链接有效、来源快照SHA一致、
`git diff --check`通过。只读复核确认了两轮推导与公开残差信息边界，并修正一处把网页抽取漏项误认作原网页漏作者的描述；
未独立复现论文结果，部分论文正文缺失的限制仍然保留。
