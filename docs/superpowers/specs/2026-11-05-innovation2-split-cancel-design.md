# 创新2积分分支 A→B：结构扩展认证与有限符号消项设计

日期：2026-11-05；状态：**DESIGN / 待用户批准本文，未授权实现或正式运行**。

用户已选择A→B并要求制定一次有明确预算/停止条件的新周期设计。本文件的参数是主控制器提出的方案，不是已获授权的事实。旧神经周期与核积分周期继续CLOSED，额度不延用。批准本文后才进入writing-plans；设计批准本身是否包含执行，按末尾授权包确认。

## 1. 目标、非目标与方法选择

精确目标：在标准SIMON64/128、项目MSB位序、同cube同key条件下，(A)用新实现核验七轮证书及其输入/输出扩展，(B)使用q=1精确后缀和健全但保守的前缀oracle，求degree≤2输出组合的**符号认证空间**，判断其中是否超出本周期明确列出的直接结构继承基线。

这是选择明文积分研究，不替换原创新2“全随机明文→预定绝对密文bit”的任务，不训练模型/恢复key。以下8/9/10轮均按实际轮数报告，另列7轮基底及扩展来源。标准延伸不称新记录；不以工程测试数、核维数或多几个轮数字当创新证据。

| 方案 | 得失 | 选择 |
|---|---|---|
| A→B | 先认证旧证书与标准扩展，再做有限符号消项，容易定位证据变化 | 用户已选；本设计限定一个固定族 |
| 直接B | 更快进入论文矩阵，但位序/常量量词/扩展正控尚未工程化 | 不先做 |
| C：2080维经验核 | 可挖候选，但需新采数据，不能替代全key认证 | 本周期不做 |

科学假设：精确后缀中的**相同未定单项式**可以相消，得到逐输出观测量单独认证之外的关系；但即便得到这些关系，也可能全由已知Feistel结构导出。主要终点因此不是总轮数，而是相对明确基线W的认证空间增量。若保守oracle饱和，结果为“本oracle/固定族未得到增量”，不能推断论文方法或积分路线无效。

来源：本地《On Extending Integral Distinguishers》§4.1–4.3/Algorithm1/Thm2，PDF SHA `9a35913441e274bec37d7f15c2724438087279c8f5b0792d6bdd8033eb9193c7`；推导审计 `result/20261105-innovation2-split-cancel-audit/report.md`。官方仓库访问被安全守卫阻断，版本与SIMON64实现未核；本周期是**论文原理指导的自包含保守变体**，不是作者代码复现、不是论文18(19)轮/2^63数据复现。

## 2. 数学语义与关键正确性

状态为(X高字,Y低字)，字内MSB0，输出坐标0..31为X，32..63为Y。所有字下标模32。

```
X_(t+1)[j] = Y_t[j] XOR X_t[j+1]X_t[j+8] XOR X_t[j+2] XOR k_t[j]
Y_(t+1)[j] = X_t[j]
h_(r,j) = X_r[j] XOR Y_r[j+1]Y_r[j+8] XOR Y_r[j+2]
         = X_(r-2)[j] XOR k_(r-1)[j]
```

原高字cube H={0,2}已存七轮X7[0]支持充分证书Q={0,1,2}、全集3缺失。独立轮key全称涵盖合法schedule；背景/轮key的值从不进入符号模块。低字cube L={32,34}第一轮后成为高字H模式，key只改背景，故八轮X8[0]零和；H上h9、L上h10由偶数四点消key常量得到。不得将L输入的X8结论混写为H输入的Y8复制。

### 2.1 精确后缀

输出观测顺序固定为64个单位bit g_i=c_i，其后2016个c_i*c_j（0≤i<j<64，按i,j字典序），共2080；不包含恒1，不含重复平方项。组合向量coordinate i对应1<<(2079-i)。

每个后缀单项式身份为(state_mask64, suffix_key_mask32)，均用MSB打包；空mask代表乘法单位1。多项式是这些单项式的GF(2)集合，XOR用对称差，乘法按两mask分别OR、重复项按偶数次消去，Boolean平方幂等。只在这个**精确后缀域**相消，绝不对前缀过近似支持作GF(2)消项。

q固定1，最多每个高字bit4项（含key），任一二次观测展开≤16项。纯key项可以因为|S|=4消去；混合state×key项不能丢。不同key_mask绝不合并为同一未知系数，也不把符号key赋成0/随机值。

### 2.2 前缀oracle：健全、不完备、不调用SAT

公开SIMON递推对k=2活跃变量传播支持mask（仅0..3）。活跃叶为单位mask，固定背景与各独立轮key叶为{0}；XOR取并，AND取两两OR；实际ANF支持始终包含于过近似支持。

对边界单项式state_mask，取其全部边界bit的支持OR乘积；若全集mask3缺失，返回CERTIFIED_ZERO，否则UNRESOLVED。state_mask=0为常函数，其四点和恒0，明确返回CERTIFIED_ZERO（不能仅靠“全集是否出现”巧合处理常数）。

oracle身份包含p/cube/位序/递推版本及输入单项式。ZERO须保存可重放支持证据；UNRESOLVED不是反例，不臆测系数奇偶。只复用新代码内缓存，不能import validation/backups/runs。缓存耗时/计数披露，节余不用来扩族。

这比论文的monomial-prediction oracle更保守；允许错过更多关系，但不允许错误认证。若精度不够，不在周期内换solver/放宽规则。

### 2.3 矩阵与认证

每个未定身份(u,v)为一列，每个observable为一行；M[i,(u,v)]是精确后缀的出现奇偶。通过对M转置求右核得到输出组合K=ker_left(M)，等价于“每个未定列的2080bit incidence向量”为约束行。

**逐个组合的最终检查不能只信消元器**：对K每个规范基从原始suffix ANF重构完整精确后缀XOR，检查每一个未相消项的state_mask都具有ZERO证据（不要求整个ANF为空）；同时独立验证lambda*M=0。另检查规范RREF、基独立性、完整约束rank与dimK=2080-rank(M)，不能仅以残差为零声称完整核。所有未定项必须精确相消，不能靠数据全零替代。认证空间的完整性仅相对于本固定oracle与observable族，不指全部真实零和函数。

## 3. 有限科学族与阶段门禁

### A：固定认证与6个实测case

新代码先认证原H/r7/X0，核对旧Q与目标身份；对下表固定case冻结源代码/公式后实测。每case64组、每组4点；A3/A4是同一面板的两个观测视图，共用采样域case_id=`A34`、key/背景/同次密文，其余A1/A2/A5/A6四个面板各自独立。因此6个case仅5份面板，不按结果增加。

| case | 轮数/cube | 观测与处理 |
|---|---|---|
| A1 | 7/H | X7[0]，必过的旧证书正控 |
| A2 | 8/L | X8[0]，必过的输入扩展 |
| A3 | 8/H | X8[0]，已知该cube不具有普遍零和，仅记录新panel反例数；64组未遇反例不当正证 |
| A4 | 8/H | 同次密文计算h_(8,j)，j固定{1,2,3,8,26,27,31}，都是旧经验核线索，不是必过正控 |
| A5 | 9/H | h_(9,0)，必过的输出扩展 |
| A6 | 10/L | h_(10,0)，必过的组合扩展 |

A4对应Y7[j]=X6[j]，对这7个固定单bit前缀先做符号认证，不能因经验通过升格。另固定核X7[25]（H）这一经验线索，总固定目标oracle请求9项（原X7[0]+7个X6[j]+X7[25]）。原Q重算失败、必过case非零、证据重放失败，立即停止整个周期，不进入B。A4中UNRESOLVED线索出现经验反例或无反例都不触发更换候选，记录后照原矩阵推进B；**任何已获ZERO认证的A4关系出现非零，立即停止全周期**。UNKNOWN与反例不得覆盖已认证状态，优先适用认证链失败门禁。

### B：只做6个符号cell，q=1

```
(p,cube) = (7,H),(7,L),(8,H),(8,L),(9,H),(9,L)
总轮数r=p+1 = 8,9,10。
```

全部2080观测；固定p与cube顺序，不筛cube，不调q，不加degree3，不扫新seed/更多轮数。所有cell的符号K、基线W和候选名单在**任何B真实调用前**落盘。

基线W在同一2080坐标空间中生成，且每个生成元必须获本oracle认证并检查属于K：

1. 逐observable单独后缀认证为ZERO的单位向量；
2. 前缀Xp[j]已认证为ZERO ⇒ 最终Y_(p+1)[j]；
3. 前缀Yp[j]已认证为ZERO ⇒ 最终h_(p+1,j)（四点消常量key）。

每cell固定查询全部64个前缀单位bit，以免按结果择基线。“已认证”只接受当前cell对应同一p/cube的符号证据，不接受旧经验存活。每个W生成元均直接对原始incidence验证，不只查询消元结果。W是**明确列出的结构继承基线，不是全部平凡/已知关系的全集**；不能将dim(K)-dim(W)>0直接叫创新。W⊆K违例视为认证链不一致（含oracle未捕获继承）并停止，不能静默删生成元或截负数。

主要终点：每cell的dimK、dimW、d_extra=dimK-dimW及全6cell表。另记录相对“单observable已认证空间”的纯矩阵相消增量，避免W掩盖方法作用。K和W按完整规范RREF保存；按K规范基顺序以W为初始span，依次加入独立基，得到确定商空间代表；最多取前4个/每cell（最多24个）用于新panel核对，不依经验好坏替换。

若该cell d_extra=0，仍允许核对最多4个K规范基作为实现正控，但不能叫新增候选；K=0则不调用该cell的密码，记NOT_APPLICABLE而非“实测0”。所有非空cell各256个新key/背景组、每组4点，同一批密文同时计算选定≤4个组合。每cell前32组记check、后224组记预冻结confirmation；两者都不参与候选选择。任一认证组合出现非零即停止全周期，保留反例计数但不泄露key/trace，不改协议重试。

## 4. 预算与时间（本文件的提案，需批准）

| 项目 | 最大量 |
|---|---:|
| A真实cipher block attempts | 5份独立面板×64×4 = 1280（A3/A4共享） |
| B真实cipher block attempts | 6×256×4 = 6144 |
| 正式科学真实调用合计 | **7424** |
| 单独工程KAT/标量vs批量接口检查 | **512**块上限 |
| 全周期真实cipher attempts合计 | **7936** |
| B随机函数null | 6×128组×4点 = **3072个64bit words** |
| 模型fit/更新/梯度/DEV选模/evaluate | **0** |
| 正式运行次数/失败自动重试 | **1 / 0** |

同一明文在不同轮数加密分开计attempt；A与B面板不共享key，除明确共域的A3/A4外，case间不共享key；B的两阶段也不得重复key。不同observables同批复用不重复计加密。预算按提交前reserve、错误不退款；attempt/completed与attempt_unique/completed_unique(精确key/plaintext对、忽略round)分开。两种“unique”不能当统计独立性证明。真实数据层私有ledger使用域+key_index+plaintext，不输出key或可读私有ledger。

B null为各cell独立随机函数：在4个不同点各取独立均匀64bit输出，对已冻结非零组合计算组XOR。不可取“独立随机特征bit”替代真实多项式观测。每cell128组；任一被检查组合128组全零则null报警、全周期停止；不称其误报概率必为2^-128，因为非线性函数可能偏置。null使用独立域，不求经验核、不用于选择/修正符号空间。

资源硬数量上限：每cell2080 observables、每observable≤16 suffix terms、出现项总数≤33280、唯一未定项≤33280；最多6cell；所有prefix monomial请求（首次cache miss含baseline/A目标）≤**220000**；k2支持每节点≤4mask。每cell组合basis重构的observable处理数≤2080²=4326400、单项式XOR合入次数≤69222400；超过视为RESOURCE_LIMIT，不靠跳过部分基降低验证范围。越界为RESOURCE_LIMIT/UNKNOWN，不截断后伪造ZERO。逐cell处理，落盘后释放大型matrix，单cell内不丢unknown列。保存压缩/编码方式交实现者决定，但必须完整可重放，不保存真实key。

时间：拟批准后，以实际实施启动为T0，**6小时协作窗口**，包含设计后的计划、实现、测试、审查、一次科学CLI与交付；启动时写绝对UTC截止。超时停止新增操作，保留未完状态，不自动续期。正式CLI内CPU和wall各≤1800秒；每符号cell各≤180秒、A符号≤120秒、每实测cell≤60秒，累积不清零。未实测成本为估计，预算是上限而非预计花满。

检查在有界循环/批次/消元/持久化边界完成，**不是硬进程树限时/内存保证**，可能I/O或单调用越界，报告实际时钟。SIGINT/取消优先停止，不运行剩余矩阵。若难在窗口内完成，收尾为BLOCKED/TIME_LIMIT，不减少验证或自动换模型接管。

## 5. 模块归属与兼容

旧`integral_linear.py`限定width≤64，`data/integral.py`固定高字k≤6/round≤8/旧协议root与deadline；两者不是新周期的可调参数容器。保留其旧行为与源码，不改旧实验截止、不偷偷扩宽接口。

拟新增的具体职责（最终文件拆分由计划确定，不建registry）：

- `representations/simon_suffix.py`：只负责q1精确Boolean ANF、observable顺序、h与输入模式公开代数；无真实key/配置/I/O/训练。
- `integral_prefix.py`：只负责公开前缀图、支持过近似oracle、逐项证据与复核；不采样/调用cipher/选候选。
- `integral_symbolic.py`：组incidence、纯packed GF2(最大2080组合坐标/33280约束)、K/W与固定quotient；不读key/cipher/文件。无需为复用造通用GF2框架；小消元核心有独立穷举测试。
- `data/integral_observables.py`：只持有新私有key/背景、合法生产cipher适配、每点观测组XOR/null、保守计数；输出聚合，不求核/选候选/写报告。不读其他cipher标签或真实trace。
- `split_cancel_config.py`：严格身份(name/experiment/protocol)，建议experiment=`integral-symbolic`、protocol=`simon64-split-cancel-v1`；参数是已批准协议常量，无任意override开关。
- `split_cancel_run.py`：只组装固定A→全部B符号→冻结选择→null→B核对，期限/状态/预算；不内嵌密码算法、ANF/GF2或大段报告。
- `split_cancel_reports.py`：接收公开结果/证据写身份与artifact；无反向依赖runner、无key/采样/GF2决策。
- `cli.py`：check/train中显式新分派；check0cipher且不导入执行器。积分train保持fit0、不产checkpoint、不支持evaluate。不制造第二CLI/科学worker，不执行SSH/Git同步。

不变：原prediction/innovation1默认序列化SHA、训练/选模/TEST/checkpoint/模型与cipher算法；旧积分config/结果与停止条件；backup/papers/historical runs只读。main原位工作，保持全部dirty。无commit/push/worktree/reset/clean/stash。

## 6. 数据身份、产物与证据等级

新root建议198101，派生编码固定为ASCII JSON `[protocol,198101,role,case_id,phase,purpose]`，separators=(',',':')，SHA256大端转int，`random.Random`本机Python3.12版本冻结。purpose区分key/background/null；role A/B/engineering；phase A64整panel、Bcheck/confirm、null分域；case_id含cube/round（A3/A4均用同一`A34`身份，其余互异）。key为合法128bit主key，用完整key全局私有集合拒绝碰撞；背景清零active后保证同case/phase结构不重复；drawcap=16×target，超限即停止不改seed。每cube四点按active升序二进制顺序枚举。原生产密钥调度不变。

正式运行前冻结：协议常量、实际SHA逐文件、完整6case/6cell、observable顺序、数据派生规则与deadline、旧query142来源SHA。输出目录必须exclusive先占用，之后才可生产cipher调用；失败目录不覆盖、不resume。

保存：manifest/config、A固定符号证据和组parity、B每cell suffix精确展开/完整unknown身份及incidence/所有oracle返回及ZERO重放支持/K/W/quotient与检查结果；selection在B真实调用前保存；null及check/confirm每组≤4组合parity、原始完整输出bit XOR供审计（不是点级密文）；counts/progress/stages/exit。生产point ciphertext只在data内短暂计算观测，不写key、轮key、实际中间状态或点级trace。

独立保存证据复核不得重新加密。真实调用仅核对所列定理/候选与实现是否一致，不把有限panel通过当成证明；全key保证来自sound符号认证。重复同panel等价统计不作独立证据。任一counterexample推翻的对象是其具体普遍零和声明/实现，不说整个密码无结构。

结果另存`result/YYYYMMDD-innovation2-split-cancel-cycle[-n]/{report.md,index.html,source.json}`，原始run独立`runs/<fresh-id>/`；报告全6cell，包括UNKNOWN/NOT_RUN，不挑最好一项。浏览器未启动不说可访问。

## 7. 验证、分工与权限门禁

先由implement-deep设计核心算法实现，常规数据/CLI测试可交implement（不重叠文件）；主控制器讨论/拆包/验收，不默认接管编码。code-review独立审查，代理不得递归派人、无自己的科学运行许可。

TDD必须覆盖：Boolean重复项相消与幂等、keymixed不丢、不同monomial身份不混；人工小图oracle ZERO与穷举完全域对照，故意UNKNOWN；q1独立公开位函数与精确表达对照；MSB packing；常项四点消失；宽GF2小维全枚举/转置/完整核与商；旧Q重放；W⊆K；prefix oracle坏证明/缺列损坏拒绝；预算提交前预扣、partial不补0、exclusive输出前置；CLI新旧默认身份/惰性导入与无evaluate。均使用合成小图或fake，无自动真实cipher test。

工程真实调用只由显式单次KAT包执行并计入512：核标准已知向量与新适配器1..10轮有限对照，具体清单在实现计划冻结，不能任意跑全pytest获取额外cipher。干净源码副本只新check＋fake-oracle执行小闭环，无正式生产train，旧神经训练0。

主控制器执行scoped pytest/Ruff、`PYTHONPATH=src .venv/bin/python -m mypy`（外部验证文件加MYPYPATH=src）、干净源码检查，核diff/源码身份。正式CLI仅在TDD/独立审查/控制器验收后一次启动；保存证据以独立小复核器核矩阵残差/所有组合展开/行SHA/计数/退出，不重加密。实际命令待入口实现后在计划/记录给出，当前不提供可误执行的未实现命令。

### 提请一次性批准的执行包

若用户确认本文完整执行包，允许：按本文本地实施/合成测试/审查→工程cipher≤512→正式一次A→B CPU周期cipher≤7424、null≤3072words→无加密保存复核→三文件报告，在6小时窗口内连续推进，不逐case再次询问。

始终不允许：GPU、SSH、远端、安装/升级、solver、训练/梯度/evaluate、Git提交发布、访问生产Hub/真实Pi会话测试、新seed/cube/q/degree/候选族或第二正式尝试。失败重试0、预算节余不扩大研究。用户批准前，仅允许设计与只读审查；本文不自动授权自身。
