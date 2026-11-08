# 创新2进度对齐与下一步

## 最新：非零背景三轮xor2复验58.836%，零背景高分未保持

按[两臂合同](INNOVATION2-NONZERO-BACKGROUND-PLAN.md)完成高半固定0xA5A5A5A5/低半随机的重训，见[结果](INNOVATION2-NONZERO-BACKGROUND-RESULTS.md)。xor2 TEST58.836365%/BCE.653554310，TRAIN常数50.234985%/.693150074；null49.551392%/.693451450。真实所选epoch32，TRAIN67.7521%/DEV58.2397%，null所选epoch1但也完整训练32轮。相比零背景99.6384%，存在明显背景敏感性，但新背景仍有部分预测能力。不是零背景模型跨背景迁移，也不保证任意背景/key。

任意固定B的F(B)仍为常量，不能说非零恢复首轮随机非线性；背景同时改变后续目标及MLP固定1输入对偏置/梯度的影响，未区分优化/函数复杂度。若继续，优先设计同非零目标、输入固定通道归零的公开表示对照，不能靠本轮断言唯一机制。

发布4596264d409fd0b98676496aeff8c8f1c75d5bf7仅9任务文件，新严格uint32 plaintext_fixed_value默认0省略保旧身份；旧空间枚举历史名称保留、metadata写实际背景。主模型diff/旧新配置核仅name+固定值；161tests/9排除、干净103文件Ruff/mypy、新91tests及两CPU/null CLI4updates通过，独立实现和结果静态审查未见核心矛盾。2exit0/16384更新/4194304呈现/64DEV/retry0，elapsed126.718s；推算139264cipher calls+139264nullbit，无独立逐调用计数。22文件467103字节完整取回核SHA/身份/选模，无模型/标签重算。stage UTC为启动预留而非结束时间；TEST自动输出非封存，单key/init探索。CLOSED不再加fit。

## 最新：三轮固定半明文空间，部分目标明显可学

按[两空间五标签合同](INNOVATION2-PLAINTEXT-SUBSPACE-PLAN.md)完成10fits，见[全部结果](INNOVATION2-PLAINTEXT-SUBSPACE-RESULTS.md)。A高32固定0/低随机，B低32固定0/高随机，同32位去重随机流、固定未知key、相同MLP24961参数。按bit/xor2/xor3/xor-and/null：A TEST ACC **100%/99.6384%/50.0443%/87.0758%/49.5117%**，B **99.8169%/50.1053%/50.2548%/66.4337%/50.1465%**。各TEST65536行，TRAIN常数准确率49.527–50.632%，故高分非单纯多数类。B-xor3微小BCE变化不算有效发现；100%有限TEST不等于2^32全域证明。

A/B bit及A-xor2、两侧xor-and支持不同程度的固定子空间同分布泛化；xor3与B-xor2未学出显著能力。A首轮延后随机非线性是解释候选，不能作为已证实唯一机制；xor-and高分可能只用子表达式/相关性，非精确逻辑恢复。不推广到完整64随机明文/新背景/跨key；仍单key/init、CLI暴露TEST的探索。null同流不同输入，非独立跨key确认。

发布255ed9757b05ae620a087c7d64a1a1150a0c71a2，仅25本任务文件；数据层新plaintext_space枚举，默认序列化/输入不变，不改cipher/model/labels/checkpoint/evaluate算法。独立审查P2移动HEAD测试已固定fixture，236tests、干净102文件Ruff/mypy与十null CLI20更新通过。10正式exit0、81920更新/20971520呈现/320DEV，retry0、elapsed合计601.125s。推算1114112cipher calls/278528nullbits，未装运行时逐调用计数。110文件2329299字节完整取回核SHA/身份/选模，未本地模型/标签重算。独立结果静态审查无数字/选模矛盾。CLOSED不自动加背景/轮数/seed；下一步应有界复验成功目标，而非混称已破高轮。

## 最新降轮对照：三／四轮仍未见直接MLP泛化改善

用户要求降轮后，完成[固定10项](INNOVATION2-CIPHER-EXPRESSION-LOW-ROUND-PLAN.md)，见[全部降轮结果](INNOVATION2-CIPHER-EXPRESSION-LOW-ROUND-RESULTS.md)。按bit/xor2/xor3/xor-and顺序：r3 TEST ACC50.3891%/50.0763%/50.0061%/50.3433%，r4为49.7894%/49.8993%/50.0534%/50.0702%；两个null49.5163%与r6同轨迹，是重复对照、不算独立确认。所有TEST BCE差于各自TRAIN常数。部分指标比r6数值上升，但未见明确泛化改善；本次bit MLP也未学会可靠泛化，不能单归因于多bit标签难。

关键例外：r3-bit已有公开结构参考＋TRAIN残差常数在TEST达到62.6099%/BCE.660999501，未作为full/direct输入；证明不能把MLP失败写成目标无结构信号。r4该参考50.0916%。非bit不套单bit参考。

同key/init/data与r6，只有rounds/name变化；发布d6c3c5d3b89afe5dd494a948f6b313b43c1f6d39仅22新文件，无生产代码变更。10fits/81920更新/20971520呈现/320DEV/10exit0，正式retry0，elapsed合计630.687s。推算1114112cipher调用（139264不同明文）＋278528nullbit，未装独立调用计数器。取回110文件2315277字节核SHA/身份/选模，未本机密码或模型重算。工程157tests、干净101文件mypy/Ruff、10null CLI20updates通过；独立配置/结果静态审查未见可证实矛盾。CLOSED，无自动增轮/增seed/新表达式；单key/init、已见TEST的探索而非确认。

## 最新正式训练：六轮密文布尔表达式，未见泛化优势

用户要求明文输入、预测多个密文bit的布尔表达式值并授权试跑/发布后，按[五臂合同](INNOVATION2-CIPHER-EXPRESSION-PLAN.md)完成训练并取回，见[全部结果](INNOVATION2-CIPHER-EXPRESSION-RESULTS.md)。C0、C0 XOR C8、C0 XOR C8 XOR C16、C0 XOR(C8 AND C16)、null的TEST ACC分别50.4547%/49.8520%/49.7559%/49.9939%/49.5163%，BCE .69329980/.69330457/.69342497/.69360306/.69372270；全部BCE劣于各自TRAIN常数基线，不支持组合表达式更易学。

同MLP24961参数、相同x及初始化SHA、固定key190502/init17、新data195601；每臂8192更新/2097152呈现/32 DEV，合计40960更新/10485760呈现/160 DEV，5exit0、正式重试0。所选epoch1/1/1/2/1；第32轮在线TRAIN BCE约.519–.526、DEV约.881–.900，null也同样过拟合，在线值不是last.pt全TRAIN评估。

发布commit13e939bc8f305874a6f787012ded60a8fef3457f仅批准15文件，Windows exact Git同步、46源码身份核验；指定GPU1，现有环境无安装。五项保存elapsed合计295.313s；取回55文件1146688字节核SHA/身份/计数/选模，无本地模型或密码重算。由代码和split规模推算557056 cipher调用/139264不同明文（不是独立逐调用计数），null139264bit。preCLI旧模板占位符失败一次，SSHfit0后修正，不算第六fit；跨机时钟有偏差，不能用裸UTC独立排序冻结先后。独立静态审查指标一致、语义边界保留。TEST自动输出已暴露，单key/init探索，CLOSED不追加。

## 最新推进：显式COPY全局division人工原型验收

代理恢复后按[方法桥接](INNOVATION2-INTEGRAL-LITERATURE-BRIDGE.md)继续，取得[人工原型实测结果](INNOVATION2-INTEGRAL-BITDP-TOY-RESULTS.md)。不再扫描已排除高字k2：新toy原型按使用端口显式COPY、传播全局切面前沿，能在未化简`g=x XOR y; f=g AND g`中保留共享约束，两活跃判零、单活跃保留可达；不是原节点投影support换名字。

主模型独立实测8测试/26子测试、430cube/614输入域子集/27382切面矩通过，Ruff/strict mypy两文件通过；独立快照审查未发现可证实P0–P3。图严格n<=4、参数q<=2、无SIMON/生产encrypt/solver/科学标签；未取得Todo全文，故只称自包含division抽象toy prototype，不称标准DP复现、十一轮再现或新密码结果。论文摘要之外具体输入/输出/复杂度仍未知，检索受403/429限制。

另补旧496项扫描的事后独立有限审查：未发现推翻反例的科学语义缺陷；P3失败项残留字节漏计在496全成功路径未触发，冻结原件不改，未来复用先修。审查仅抽首末原始记录，无Git基线/执行，不冒称496项独立复验。当前方法/人工工程阶段已收尾，尚未扩64bit高轮实验。

## 继续推进：公平轮转仍饱和，转固定cube精确反例判定

用户明确“是推进”后，按[分散轮转合同](INNOVATION2-INTEGRAL-DIVERSITY-PLAN.md)真实完成新八轮公开搜索，见[结果](INNOVATION2-INTEGRAL-DIVERSITY-RESULTS.md)。8起点均访问，各2400000pair；80query=65完整UNKNOWN+15RESOURCE，全部完整支持仍全空间，0proof，COMPLETE/exit0，total19200872pair/CPU16.826s，真encrypt0。覆盖饿死问题已解决，不能把未找到证书扩成八轮无积分；本轮关闭无九十轮。
随后按[精确miter合同](INNOVATION2-INTEGRAL-MITER-PLAN.md)**正式完成两项**，见[精确结果](INNOVATION2-INTEGRAL-MITER-RESULTS.md)：八轮{0,2}和{0..6}均SAT_VERIFIED，非UNKNOWN。前者4份输出[1,0,1,1]、XOR1；后者128份69个1、XOR1。vars/clauses分别2310/7934、67901/253950，CPU.033／.781s；单次load/solve/delete均成功，各exit0、无重试。main另独立解析所有CNF子句及完整模型、packed32公开符号递推核全部copytargets、全部artifactSHA通过。独立review只读聚合核对未见不一致，不冒称其执行SHA或全CNF检查。
这排除**两个指定cube在独立轮常量模型下对所有背景／常量恒零**，不是合法主密钥反例、其它cube不存在、真实密码预测／密钥恢复或零和证明。trueencrypt0，无GPU/SSH/安装/发布；现成PySAT常规适配，未自研solver。最终78fake/公开小图测试＋Ruff/strictmypy独验，正式前修复常量域／缺模型补值等错误；另披露代理越界2次旧encoder-only观察，solver0但不隐瞒extraencoder。
两项正式求解额度耗尽并CLOSED，未追加第三cube／更高轮。下一项应比较标准bit-DP显式COPY约束换cube与合法key调度精确判定，不重复已被该模型反例否定的恒零目标。

### 八轮高字全部两位置：496/496合法key反例（最新完成）

用户随后要求继续尝试，按[扫描合同](INNOVATION2-INTEGRAL-PAIR-SCAN-PLAN.md)冻结高字0..31全部496对、固定目标X8[0]，结果[详见记录](INNOVATION2-INTEGRAL-PAIR-SCAN-RESULTS.md)。496/496 SAT_VERIFIED，UNKNOWN/ERROR/重试均0；每项都存在合法主密钥和背景使四点XOR1，262项一个1、234项三个1。逐子句独立核4,697,120条CNF，第二套packed32数学递推核1,984点；生产encrypt调用0。周期wall79.583s，原件178,773,114字节，账本CLOSED。

这穷尽排除了**八轮固定目标的高字内k2零和cube族**，不是仅旧{0,2}失败；不排除跨字、更高维、别的目标或原论文积分构造，不意味着每把key失败。执行时代理鉴权失败，限定主模型实现/验收，当时没有独立代理审查；后补有限快照审查见上节。32无生产调用测试、Ruff/strict mypy三文件通过。上一周期“约80调用”不精确且超原合同，已在新结果披露，旧原件保留，不继承为本轮授权。

### 合法主密钥约束下的续测（已完成）

随后按[合法密钥合同](INNOVATION2-INTEGRAL-KEY-MITER-PLAN.md)把同一个反例公式加上公开SIMON64/128密钥调度，再判同样两个八轮cube，见[结果](INNOVATION2-INTEGRAL-KEY-MITER-RESULTS.md)。
两项仍SAT_VERIFIED，且witness是**合法主密钥**：{0,2}四点输出0,0,1,0（XOR1，2690vars/9470clauses/CPU.049s）；{0..6}128点中59个1（XOR1，68281/255486/.933s）。主模型用生产`encrypt`逐点重算4／128个输出，与符号target完全一致；轮密钥与生产`_round_keys`逐字相等；全部子句与完整模型独立核验通过。
所以这两组小cube在八轮对全部key**不**恒零，是真实失效而不是独立常量假设过强；但不说明八轮无积分（别的cube未测），也不是密钥恢复或UNSAT零和证明。
正式前曾把密钥调度`z3`位误当整字翻转（会使全部派生轮密钥错误），由“CNF固定key与生产密码逐点一致”的对照测出并在任何正式求解前修正。
下一项有信息量的动作是换cube扫描（如k=2全部496对），仍需单独冻结与授权。

## 最新连续周期：六轮8点／七轮4点实测通过，八轮公开搜索碰墙停止

用户“继续推进，等推不动再停”后按[有界至十轮合同](INNOVATION2-INTEGRAL-CYCLE-PLAN.md)连续执行，见[完整结果](INNOVATION2-INTEGRAL-CYCLE-RESULTS.md)。
六轮bit0/{0,4,10}每cube8点，真实16/16零和／null10/16；七轮bit0/{0,2}每cube4点，真实16/16零和／null6/16，分别128和64次encrypt。两项是新cube首次真实验证，不冒用旧7维结果。
随后八轮公开搜索196query=192UNKNOWN+4RESOURCE_LIMIT，0proof，20000000pair精确触顶，exit3/BUDGET_STOP，CPU14.714s/墙钟14.283s；第一起点未完成，后两起点NOT_STARTED，无stage choice，无八轮真实或九／十轮启动。
192完整支持全部等于2^k全mask空间，分数全1导致排序退到字典序，沿连续小位加深；当前评分饱和／证明成本墙，不是八轮不可达或真实不平衡，不宜原样加预算。
全周期真实attempted/completed192/192、null192、0fit/GPU/SSH/安装/发布/科学重试；49344上限中未用49152不延续许可，CLOSED。
生产源码／模型／CLI未改；validate44fake＋search58fake/公开测试及各Ruff/严格mypy独立通过，正式前证书绑定／nullpartial／缓存与最终预算P2已修。未跑完整后端训练回归，不虚构。
保存rawbits/points身份/byteSHA/源绑定／pair计数只读验收，无密码重算；原件runs/integral-cycle-20261029-{172532-validate-r6,172602-validate-r7,173002-search-r8}/。
积分是选择明文结构分析，不是单点预测成功、新颖性、神经学习或密钥恢复；下一周期先比较更强证明／有信息评分或多起点公平分配，不自动追加。

## 最新公开分析：六轮8点、七轮4点积分候选，尚未真实验证

用户“那继续推进”后完成[有界公开搜索](INNOVATION2-INTEGRAL-FRONTIER-PLAN.md)，见[结果](INNOVATION2-INTEGRAL-FRONTIER-RESULTS.md)。
六轮bit0已验证7位来源全部127非空子集中，{0,4,10}有零和充分证书，支持Q={0..6}缺全集7；该固定来源／判定器可证最小维度3，每cube8点，不是全局最优。
七轮输出高字bit0三个固定轨迹都得到{0,2}，Q={0,1,2}缺全集3，每cube4点，最终单删均UNKNOWN，只称判定器下局部极小，不称实际无单点积分或全局最小。
主模型另用全字公开递推重算两个固定证书一致；真实加密0/真实key0/科学标签0/fit0/GPU0/SSH0，不能继承旧六轮128点16/16为这两个新证书实测通过。
201unique=127六轮+74七轮，7cache，0截断/失败，892625pair尝试，exit0/CPU0.860907s/墙钟0.842102s；5文件524243字节/完整支持与SHA核查，公开源接口未扩。
26项fake/公开小对照/Ruff/两文件严格mypy独立通过；ignored脚本未发布，生产源/模型/训练/CLI未改。一个非blocking跨轨迹限额标志问题因0截断未触发，记录而不重跑。
run `runs/integral-frontier-20261029-165914/`，本轮CLOSED；建议另行授权两个固定cube共192次真实验证，不自动追加，未声称文献创新或key恢复。

## 最新完成：六轮积分结构交叉验证，真实16/16零和

用户在[方法核查](INNOVATION2-INTEGRAL-METHOD-REVIEW.md)后“推进”，按[固定合同](INNOVATION2-INTEGRAL-SIX-CHECK-PLAN.md)完成一次本地CPU真实验证，见[结果](INNOVATION2-INTEGRAL-SIX-CHECK-RESULTS.md)。
六轮SIMON64/128、输出高字MSB bit0、活跃明文位(0,2,4,10,14,15,18)，16个不相交128点cube：真实16/16 XOR0，随机函数bit对照8/16 XOR0，与公开支持集充分证书一致。
真实attempted/completed2048/2048、2048不同明文、null2048bit、exit0、CPU0.022122s/墙钟0.026454s、6文件28271字节；无重试/fit/梯度更新/GPU/SSH/发布。
预定全16组零和事件在理想随机函数下概率2^-16；这是16个统计而非2048个独立积分统计，不是结果错误概率。
此为选择明文积分关系，不等于原随机明文单bit预测成功、神经创新、全key实验枚举、全轮攻击或密钥恢复，也未与文献最佳复杂度比较。
一次性ignored诊断脚本调用现有密码API，生产源码/CLI/训练/checkpoint未改；31项fake测试/Ruff/两文件严格mypy独立通过，不称本次完整项目回归。
保存bit/points SHA/文件SHA/源码与冻结合同身份只读核对通过，无再生真实标签；独立保存记录审查未见矛盾但未另算字节SHA，区分其覆盖范围。
run `runs/integral-six-check-20261029-161843/`，本轮CLOSED/预算耗尽；下一步应另定更高轮/少查询证书或有额外用途的学习辅助，不自动追加实验。

## 最新完成：五轮结构向量仍未明显拟合

用户“5轮试试／是”后按[五轮合同](INNOVATION2-STRUCTURE-VECTOR-R5-PLAN.md)完成两项，见[完整结果](INNOVATION2-STRUCTURE-VECTOR-R5-RESULTS.md)。
真实TEST50.0046%／BCE0.69314756，null49.9588%／0.69315322；两项均选epoch4，TRAIN/DEV/TEST准确率均与各自TRAIN多数常量完全相同。
仅聚合相同不能证明逐样本恒定，但无输入相关预测规律证据；真实末在线TRAIN BCE0.693258、null0.693266，没有明显拟合，不是单纯六轮深度门槛证据。
同门公式、半径2、width32、lr0.001、同明文面板与key/init；五轮266节点／20门深／9697参数，非六轮同容量实验。
运行commit01ca85a34e532791d9d7b27ece05be523c4d04d6，干净发布index1052tests/mypy99/Ruff，六轮固定旧版本兼容探针实际执行通过。
两项exit0、共16384更新／4194304呈现／16DEV点，24文件250542字节完整取回核SHA，保存elapsed合计778.313秒；现场CUDA null另2更新/8呈现，无科学重算/重试。
TEST自动生成/输出、完整JSON读取已披露，先冻DEV再展示TEST，非封存确认。本批关闭，不自动四轮或参数扫描。

## 最新完成：六轮目标图有界向量，未见可靠改善

[两项合同](INNOVATION2-STRUCTURE-VECTOR-PLAN.md)与[完整结果](INNOVATION2-STRUCTURE-VECTOR-RESULTS.md)：真实TEST50.1694%／BCE0.69321714，null49.8474%／0.69316327。
`simon-vector`固定六轮bit0/full/direct/width32，10529参数，406节点公开DAG、AND/XOR共享乘性交互、半径2投影、目标节点唯一读出。
真实DEV50.5371%对null50.1953%，满足原定描述性DEV改善规则，但真实TEST BCE劣于常量和null；不称统计证据或有效突破。
真实selected TRAIN/DEV/TEST正确数恰等各split bit1数量，与始终预测1相容；聚合记录不能证明逐样本恒定，但不能作为学到输入条件规律的证据。
末在线TRAIN BCE real0.693284/null0.693258，没有明显TRAIN拟合，不沿用上一CNN强拟合但泛化失败解释。
代码ae2c7ac160bda758d6ee716816da303ebfaa9870，干净发布index1037tests/mypy98/Ruff，现场CUDA null2更新/8呈现。
两项exit0、共16384更新／4194304呈现／16DEV点、24文件取回核SHA，保存训练elapsed合计971.797秒；无科学重算/拟合重试。
启动前Windows长命令与路径格式故障已留原证据，确认未进入正式CLI后修正；科学源码仅Git同步。
TEST由CLI自动生成/输出、外层完整JSON已读，先冻结DEV再展示TEST，不称sealed。批次关闭，不追加参数扫描或自动确认。

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

## 最新完成：固定64mask的TRAIN-only核查未检出稳定相关

用户明确“确认”后按[一次CPU合同](INNOVATION2-PARITY-CORRELATION-PLAN.md)执行，[完整结果](INNOVATION2-PARITY-CORRELATION-RESULTS.md)。
65536 TRAIN两半、固定64mask、real/null联合256估计，alpha.05/eps约.02374，真实/null稳定检测均0。
真实A最大absrho .01654，选中index37到B rho+.00574而A为−.01654，方向反转；A锁方向B49.7131%，不是TEST成绩。
929tests/mypy84/Ruff与旧CLI null通过；正式65536次真实加密、65536null标签、0模型更新、exit0/elapsed1.928s。
8保存文件验身份/聚合整数/字节SHA，未再生科学标签；没有DEV/TEST、GPU/SSH/安装或commit/push。
辅助脚本两项审查发现于正式执行前修复并用合成失败fixture验证，无正式重试。
本批关闭，不自动加字典/数据或拟合head；未检测不等rho=0，不能排除更弱相关或多特征组合。

## 最新机制核查：公开Walsh候选不足以直接支持GPU拟合

用户“继续”后完成[无标签推导与编译](INNOVATION2-LINEAR-FEASIBILITY.md)，[核查结果](INNOVATION2-LINEAR-FEASIBILITY-RESULTS.md)：
六轮bit0/beam64/局部变量限额12得到64路径、64不同奇偶mask（5..11位），最大单路径系数1/1024。
局部谱精确、全局路径截断，未知key只作符号占位；路径大小不是实际key相关或界，没有测任何TRAIN/DEV/TEST相关性。
892tests/mypy82/Ruff与干净源码旧CLI null4更新闭环通过。生产model/config/training/evaluation未改。
审查发现测试覆盖不足，已补共享变量/跨界及默认六轮存活路径检查；没有确认生产数学错误或科学fit重试。
本轮无科学拟合/SSH/GPU/安装/commit/push；不足以直接启动新大规模训练，若继续需另定小型TRAIN-only相关性核查范围。
不把频谱结构基线替代最终神经组合学习目标，不自动降低历史门槛或回低轮追分。

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
