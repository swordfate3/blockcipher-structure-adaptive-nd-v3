# 创新2多周期研究与实现设计

日期：2026-11-02。状态：**LOCAL_IMPLEMENTATION_APPROVED**。用户阅过设计／白话说明要求“推进”，并在实施计划三项审查缺口修订后回复“是”确认本地实施；正式科学运行／GPU／SSH／发布仍按第11节逐项确认，不扩展推断。

## 1. 已确认事项与本设计权限

用户已选择[路线比较稿](../../INNOVATION2-MULTICYCLE-ROADMAP-PROPOSAL.md)中的C：神经预测为主、积分独立支线；接受核心实现／审查按当前`implement-deep/code-review→auto/strong`，常规实现按`implement→auto/glm-flash`，执行按`experiment→auto/minimax-flash`。不改代理配置，不再询问总体路线或这次路由替代。

本设计将路线中尚未固定的接口、候选身份、统计和停止规则落实。**用户上一条“是，推进”是总体路线／分工确认，不追溯批准当时尚不存在的设计细节。** 按用户新确认的Superpowers规则，本文件独立可实施性检查后交用户批准；批准后下一步先用writing-plans，不跳过到编码。

当前进入本地实施准备：先writing-plans并交付实施计划审阅，再按已选代理方式编码和有限无密码工程。正式标签、GPU／SSH／发布／积分正式solver的完整权限在第11节单独选择。安装／主动旧checkpoint evaluate默认禁止。

## 2. 研究问题、证据与路线取舍

原始主任务为SIMON64/128，固定未知key、完整随机64位明文→绝对密文MSB bit0。六轮为起点，有条件到八／十轮，非完整44轮攻击。N1三轮xor2只定位机制，不能代替主成果。积分是选择明文集合关系，不与单点神经预测混报。

路线A仅神经：最贴近原任务，但六轮可能失败；路线B仅积分：可推进符号结构研究，但任务和贡献改变；用户选择C，保留两条证据路径，不在一条失败后无界扩另一条。历史依据与负面证据沿用比较稿第2节，不重跑或从旧许可恢复执行。

方法依据是本项目公开SIMON递推、受限背景结果和人工COPY抽象；不是已证明的新颖方法或未读论文的复制。Todo论文仍只读到摘要，不能猜十一轮构造。公开特征／残差也不能因三轮参考有效就保证六轮有效。

**好结果的定义**：预先冻结的任务上得到有实际效果、胜基线并独立确认的预测；或得到非平凡且证据链核清的积分充分关系。完整阴性／UNKNOWN是诚实交付，不冒称阳性。没有候选无限重试直到成功。

## 3. 固定研究协议与候选身份

### 3.1 各fit共同协议

TRAIN65536／DEV8192／TEST65536、pool65536，batch256，32epochs，width128，Adam lr.001、clip1。每epoch DEV最小BCE选模，严格小于更新best，平局最早。每fit8192更新、2097152呈现、32次在线DEV选模评估，另1次selected末端DEV（共33次DEV评估），最多139264个真实标签或同数nullbit。训练及标签实现只用原入口`check/train/evaluate`，不会发明DEV-only参数。

绝对位判定继续`logit>0`；零判0。常数多数平局判0，概率继续TRAIN Jeffreys平滑。现有checkpoint schema、raw配置SHA严格绑定、selected load后TEST标签时序不变。TEST仍自动保存／输出，非封存；只DEV决定晋级，看到TEST不修改矩阵。

### 3.2 面板／key／init

下表seed为设计决定，批准后固定；当前未生成科学标签。key seed只是标签侧派生身份，不是秘密保护承诺。模型不读取key seed，更不读取key值。

| 阶段 | data seed起点 | null seed | init | key seed | 标签／空间 |
|---|---:|---:|---:|---:|---|
| N1 | 196101 | 196102 | 17 | 190502 | r3，C0 XOR C8，X0固定0xA5A5A5A5 |
| N2 | 196201 | 196202 | 17 | 190502 | r6，C0，full |
| N3-r8 | 196301 | 196302 | 17 | 190502 | r8，C0，full |
| N3-r10 | 196401 | 196402 | 17 | 190502 | r10，C0，full |
| N4-K1 | 196501 | 196502 | 29 | 190502 | 冻结最高合格轮数，C0，full |
| N4-K2 | 196601 | 196602 | 41 | 190502 | 同上 |
| N4-K3 | 196701 | 196702 | 53 | 196703 | 同上，第二把key从零重训 |

同阶段所有臂共享明文和shuffle；相同real目标共享标签值，同一null流只作配对负控，不算多份独立面板。

**无标签去重规则**：生成候选面板只用现有make_inputs；同阶段split严格去重。N4各面板在所有训练／DEV／TEST合集中不得与本周期N1-N3或其它N4面板重合；不得只查TEST。若有重合，data seed在其起点加10*i，i=0..9，取第一个无重叠面板，null/init/key不变。只看无标签输入，不查看输出；十个候选都不合格就停，不扩大范围。N1/N2/N3也按同规则避免本周期面板重合（同一阶段配对除外）。跨全部历史runs不作未核查的零重合声明；已知同seed历史面板另列而不读取历史标签。N4是对本周期独立的新面板，不声称全部历史从未出现某明文。

不在训练程序中实现seed搜寻；主模型在正式前按固定规则预检并冻结完整配置，单fit只消费已冻结seed。

## 4. 神经设计与科学分支

### 4.1 N1：固定背景机制，6 fits

真实输入始终P=[B,L]，B=0xA5A5A5A5；目标y=C0 XOR C8。三种模型各real/null：

- R：旧MLP直接读P。
- Z：同形状同参数初态，前向将高半置0，不改标签侧P。
- M：前向同Z；初始化时首层b←b+W_H·bits(B)。只用公开B和初始参数。与R在该子空间上初始函数数学相同，浮点工程容差见第10节；后续Adam轨迹不保证相同。

R/Z与M初始化记录分开：R/Z参数SHA相同但初始函数不同；R/M参数SHA不同但函数匹配。初始化变换必须仅发生一次，并且严格checkpoint重建后覆盖state不能再补偿一次。不能给旧R加归零行为。

每臂所有更新照共同协议完成，末端保存selected的预测，不借统计追加科学forward。N1效果：相对R的DEV ACC至少+3pp、BCE至少−.01作为实际重要筛选；只作探索，不作已确认机制。

**N1 null报警固定规则**：任一null臂的selected DEV相对其自身TRAIN常数，ACC提高至少.03且BCE降低至少.005，即报警停查；不使用单bit公开参考或q6，不把Z/M仅相对R的null训练差异本身当异常。该阈值是工程审查触发器，不是零假设显著性界。

**分支**：Z/M都改善支持参数化参与；只有Z改善保留初始函数解释；都无改善记阴性。无论机制方向，只要身份／控制有效均进入N2。不追加背景／低轮。初始匹配失败、标签配对失败、上述null报警触发则停查工程，正式retry0。

### 4.2 N2：六轮五方法，10 fits

公开Z=phi(P)=(Y XOR F(X),X)。固定Psi=[Z, Z_H[i+1] AND Z_H[i+8]，i=0..31]，索引模32，96维binary。Z部分可逆保留原64位信息；增加32项不包含真实状态，不代表联合分布被正确恢复。

q_r(P)按公开轮递推r轮、每轮常量全置0，返回bit0。该无key模型**r>=5不叫零主密钥SIMON**；现有public_reference(1..4)不扩上限、不改解释。残差读出ell=(1−2q_r)g，仍以原y计算BCE。

| ID | 模型 | 公开输入 | 读出 | 参数 |
|---|---|---|---|---:|
| D | 旧MLP | P64 | direct | 24961 |
| F | 独立研究MLP | Psi96 | direct | 29057 |
| R | 独立研究MLP | P64 | q6残差 | 24961 |
| FR | 独立研究MLP | Psi96 | q6残差 | 29057 |
| S | 独立线性logit | Psi96 | q6残差 | 97 |

各real/null一次。原宽度128两层ReLU不改。S损失固定`BCE+1e-4*sum(abs(weight))`，bias不正则；Adam／clip不变，DEV选模仍只看原BCE；日志分开train_bce和train_objective。该单值正则是本次设计决定，不是调参最优结论。

同形状F/FR、D/R从相同seed生成相同权重／bias，变换和reference不消耗RNG。D/R需兼容旧MLP初始化顺序；96维和64维不称同完整初态，不作纯容量匹配因果声明。S独立初始化按同init seed，不声称与MLP可比初态。

每臂均保存TRAIN常数和`q6+TRAIN残差常数`基线；D不改变前向，额外公开参考仅报告。新的q参考字段独立命名，不覆盖旧public_reference_constant。

**能力门槛**：DEV ACC≥TRAIN多数ACC+.02；BCE≤TRAIN常数BCE−.005且≤q参考常数BCE−.005。各自都要满足，不后验挑最弱基线。

**null报警**：若对应null在DEV同样满足上述能力门槛，或其ACC高于自身TRAIN多数至少.03且BCE优于自身常数至少.005，视为异常控制，暂停新科学启动核数据、RNG、split及指标；不是仅因real-null准确率差小就自动报警。

通过者先按DEV BCE最低选；差绝对值≤1e-8视为平局，参数较少优先，再按D,S,R,F,FR固定顺序。如果想主张神经结构机制收益，还要探索性胜同期D和S；没有胜S则只主张预测能力／特征成果。

全部不通过：神经周期停止，N3/N4不启动；完成其余已冻结臂用于完整比较，但发现工程／控制异常则立即停。不得临时加网络、数据、lr或表达式。

### 4.3 N3：八／十轮阶梯，最多8 fits

只冻结N2所选候选，其特征／读出／正则／width／训练预算不再变；q参考轮数随任务轮数确定改变，不是维持旧q6。

每轮候选real/null＋D real/null，若候选就是D则去重只2 fits，节余额度不补其它run。先r8；非D候选必须能力门槛通过，且对D的DEV BCE改善≥.005、ACC不下降，才启动r10；D只需能力门槛。

为保持N4主张一致，记录两层资格：

1. 能力资格：胜TRAIN常数及q_r参考，满足2pp／.005。
2. 方法优势资格：非D除能力外，对D再胜.005 BCE、ACC不下降。

N2的所选候选也按同规则记录方法优势。最终N4优先选择最高具有方法优势的轮数；若从无此资格，则选最高能力轮数，明确只确认预测能力。八轮只有能力、六轮有方法优势时，确认六轮，八轮仍完整报告。不能让高轮弱主张吞掉低轮强主张的确认预算。

r8能力资格失败不启动r10，回到合格r6确认；r8仅方法优势资格失败也不启动r10，但若此前任何轮都无方法优势则保留r8能力资格进入N4；若r6有方法优势则按优先规则确认r6。r10能力／方法优势未通过时，同样从全部已测轮数按上述双层优先规则选N4，不笼统以“失败”丢弃能力资格。执行UNKNOWN停止新启动，由主模型核是否是局部资源问题或共用代码错误；不凭未知结果跳更高轮。共用代码错误降级所有受影响结果，无自动正式重试。

### 4.4 N4：确认最多12 fits

按第3节K1/K2/K3，冻结同一候选、轮数、主张和统计，依次完成三块，每块候选real/null＋D real/null，D候选去重，不self-comparison。阶段全部配置在首块前冻结；K1结果不改变K2/K3。

- 能力确认：每块TEST仍胜TRAIN常数和q_r常数，BCE各至少.005、ACC相对TRAIN多数至少+.02；对应null不报警。
- 方法优势确认：非D若以此主张进入，还需对同期D的BCE改善≥.005、ACC不下降以及第7节区间支持；D不与自己比较。
- K1/K2均过，支持所测key的重复有效；K3过，仅增加第二把已测key的复验，不声称任意key稳健或原模型跨key迁移。
- 未通过则降级探索，部分通过报告适用范围，不加第4块／新seed。

本预算没有S在N4的同期重训：因此无论N2怎样，都不能声称独立确认了神经优于S，包括原key和第二把key。若论文核心要该主张，必须另改设计和预算，本周期不偷加。

## 5. 神经模块边界与接口

### 5.1 新配置，默认省略保护历史

新增字段建议锁定为：

- `public_input`: identity（默认）／background-zero／simon-interactions。
- `public_readout`: direct（默认）／zero-round-constant-residual。
- `background_init`: standard（默认）／match-direct。
- `l1_penalty`: 0.0（默认）；S固定1e-4。
- `record_predictions`: false（默认）。

新增`model_kind=public-mlp/public-linear`承载研究模型；旧`mlp`默认完全不变。D使用旧mlp，但本周期可启用record_predictions。新增字段默认从to_dict省略，保留历史raw mapping／SHA。非默认字段均进入config身份。

约束：public模型仅prediction/simon/full-features，旧prediction_mode保持direct（新字段明确区分，不复用老public-residual）。background-zero只允许非full固定半、background_init match-direct只允许此输入；N1只high32-zero／xor2，暂不扩任意表达式组合。simon-interactions和零轮常量残差只允许full明文、bit0、rounds∈{6,8,10}；public-linear固定interactions＋残差＋l1=1e-4，width字段仍128但不代表线性层宽度。旧model_kind不得接受非默认研究字段；record_predictions是所有prediction模型的可选报告开关，不改变训练／标签。l1_penalty只能用于public-linear；不加全模型通用惩罚注册表。

N1 R使用旧mlp，Z/M为public-mlp，身份配对由参数内容核，不以类名相同替代。background参数由factory接收公开plaintext_space／fixed_value，仅用作mask和初始化；不向模型传key_seed、数据、标签或整个config对象。

### 5.2 归属

- `representations/prediction_public.py`（新）：纯公开tensor变换、q_r与必要独立描述。只依赖torch／标准库，不依赖cipher、data、训练、config文件、旧run。不从现有prediction.py导入导致模型↔表示循环；若需抽取旧phi必须保持旧接口包装及oracle兼容，优先新模块无重复科学规则、旧phi包装调用同实现。
- `models/public_prediction.py`（新）：public predictor、宽度96 MLP／线性头、单次背景初态匹配、raw logit恢复。不给旧BitMLP简单放宽输入上限；研究类独立支持96维，保留旧64维限定。
- `models/prediction.py`：唯一显式factory接线，旧路径参数／state键／RNG不变；新路径明确返回类型。
- `config.py`：声明和严格合法组合，不做变换或审批。
- `checkpoints.py`：只扩factory参数透传／类型；schema、raw SHA、strict state不放松。初始化补偿必须构造确定且载入state后不重复执行。研究特征和reference身份进config／metadata，若持久buffer则全部规范核对。
- `prediction_reports.py`：只读metadata与初始化描述；不写文件、不forward模型、不改RNG／state。
- `training.py`：只组装factory参数、接入已定义L1标量、透传报告收集器；不写特征公式、bootstrap或候选决策。候选专属L1求值放模型／独立小函数，训练显式调用，不新增registry。
- `evaluation.py`：可选收集既有最终评估的logit和label；默认指标公式和返回字段不变。最后selected TRAIN/DEV/TEST每批只forward一次。
- `prediction_evidence.py`（新）：给定捕获预测和公开输入身份，打包／校验／纯统计；不造标签、不load模型、不选模。详细要求第7节。
- data/cipher/CLI：采样、标签及入口均不变；主动evaluate兼容新增模型重建但不在默认接收验收时运行。

共享文件只有一个实现所有者。新表示／模型核心完成接口冻结后，再让常规实现代理写config矩阵和测试，不能两代理同时改factory/config。

## 6. 积分支线设计与阻塞处理

### I0：独立小型公开分析工具

旧toy及生产simon_cone不扩上限。新研究工具先放职责明确的`validation/innovation2-multicycle/integral/`，不import旧科学run／历史solver实现。需要公开算法复用时只读当前生产公共图1..6作兼容基准，不import生产cipher。

边界文件：public_graph.py（1..11轮共享DAG）、path_cnf.py（COPY线性端口和指数CNF）、verify_path.py（独立关系验收）、solver_call.py（一次求解与计数）、proof_check.py（已有外部checker适配，不自称实现完整proof checker）、records.py（严格规格／产物）。每个文件具体职责而非通用workflow。科学输入为r／bit／active；输出为公式／路径或proof身份和明确状态。

门关系：COPY a=b+c；XOR c=a+b（整数和，11无）；AND c=a OR b；ID相等；DROP入指数0；未知固定参数初指数0。数据bit不参与求解，不编码真实key。最终k_out≤target：单输出目标1时输出指数自由，不强制1。

健全性证明与路径／前沿闭包等价必须先写清并审查。人工测试包括全部门行、共享g²双／单活跃、x XOR x抽象假阳性、重复输出、未使用活跃输入、多重集、损坏门／模型／映射。1..6×64共384规格对照完整接线和坐标，7..11独立公开递推核结构；不通过支持集合相同放过旋转错误。

**工具链现实**：本轮仅在PATH查找drat-trim／gratgen，均未找到。这不是证明整个机器没有checker，也未执行solver。实现前只读检查现有可执行路径与版本；不安装。如果没有可用独立checker：I0可完成人工编码／路径验收，但**I1不启动**，报告BLOCKED_CHECKER；另请用户选择安装可信checker或允许未认证探索，不能自行降格跑完再叫成功。

I0工具还需现有PySAT可用性核对；无依赖时阻塞不下载。不得因旧文档说可用就当本轮实测。

### I1：固定8..11轮矩阵

默认16规格：r∈{8,9,10,11}，target0，k∈{16,32,48,63}，pi_(2j)=5j mod32，pi_(2j+1)=32+(5j mod32)，S_k为前k项的集合。实现时仅展开确定公式成完整规格，不自选新位置。

预筛：非空非64、所有活跃位在目标cone内；可靠次数基线、旧投影支持基线在同运算预算内先判；缺失依赖／次数已证／同轮已证子集继承标明并跳过正式solver，不能包装成新发现。基线最多16 query，每query≤100000 pair、全周期≤1000000 pair；超限UNKNOWN，不替换为实际不存在。纯次数上界传播O(节点)单列。

顺序r递增、k递增。最高轮无证据不扩轮数。每solver只一次完整公式load/solve/delete，默认Glucose4现有版本，单配置；任一身份／编码／验收错误立即停。单资源UNKNOWN收口可继续同冻结矩阵，同轮连续两项资源UNKNOWN关闭后续新科学启动。

正控8项：人工可穷举零／可达各一；r6/{0,4,10}和r7/{0,2}；r8/{0,1}、{0,2}、{0,31}、{15,16}。后四已有反例不能与真实数据混淆，只预期抽象可达。r6/r7若无覆盖复现失败，先核编码／抽象强弱，不临时改门表过关。

原图到线性Program另有独立结构验收：绑定原图完整身份，逐项核plaintext/parameter坐标、原门操作数、重复用途、COPY树、DROP和目标输出；不能只核Program内部关系。CNF冻结每线一个变量、无辅助变量，完整模型范围显式传入且与线数一致。SAT模型由独立检查器先核原图翻译，再核完整赋值、每门／端口、初始／最终关系；不是合法key反例。UNSAT同样必须通过原图到Program的独立结构验收；再核独立checker绑定原始CNF，以及图／量词／编码可信链，才叫有条件零和充分证据。checker通过不自动证明密码翻译正确。

### I2／I3：收口

SAT_PATH_VERIFIED／UNSAT_UNCERTIFIED／UNSAT_PROOF_CHECKED／ZERO_SUFFICIENT_CHECKED／UNKNOWN_RESOURCE／ERROR分开，不用bool balanced混淆。高维不启动2^k真实加密枚举，不转精确miter。论文只摘要就不称十一轮复现；有阳性也不自动加神经辅助搜索。

## 7. 预测证据、统计与确认比较族

### 7.1 保存而不重算模型

本周期record_predictions=true。在selected末端原有三个split评估中捕获每批现成logits、targets，按输入原顺序落盘；不捕获所有epoch预测，不引入额外科学forward。

每split保存严格NPZ（allow_pickle=false读取），仅float32 logits、uint8 labels，形状[N]；manifest记录split rows、输入x_sha、y_sha、配置／checkpoint SHA、数组dtype／shape／字节SHA及文件SHA。禁止保存key、轮状态和完整ciphertext。原数据顺序及x_sha是配对前提，错配即拒绝，不靠只比标签分布。

q_r基线预测由同一公开输入和TRAIN残差多数／概率计算，在最终已有评估中生成并保存同样有身份的reference bits／残差多数位／概率参数，无新真实标签；所有real/null使用自身TRAIN估计，不使用DEV/TEST重新拟合。ACC分类直接用reference_bits XOR reference_majority，残差多数平局判0，所以平局时分类仍等于reference_bits，不由概率阈值重建；TRAIN常数分类直接用保存majority bit。概率仅参与BCE。

原ACC／BCE聚合计算保持；独立float64 `logaddexp(0,z)-y*z`供配对统计，并记录它与原fp32累加BCE的数值差，超过1e-5停查，不悄悄覆盖历史指标。候选和D的模型预测按logit>0口径一致；常数与q基线分类按上一段保存的多数位规则，不把概率=.5的q预测全部改成0；不对主BCE裁剪。

接收默认只核保存原件／身份／计数，不进行模型加载或标签再生。新统计是此设计明确计划的**保存预测后处理**，只能在单独报告步骤执行并记账，不能说它也是“仅文件SHA验收”。

### 7.2 统计冻结

探索N1-N3使用DEV门槛和完整效果表，任何区间都仅描述；不将DEV选checkpoint后的区间叫未选择推断。

N4每块预声明最多3个比较（TRAIN常数、q_r常数、同期D；D获选删除第三项但不收回显著性额度），每个比较2个指标（ACC、BCE），三块共18格family。alpha总.05，每格两侧区间置信度1−.05/18；配对IID bootstrap10000次，行有放回、同时重采模型和比较对象。固定使用percentile区间，分位数为alpha_cell/2与1−alpha_cell/2，numpy.quantile(method="linear")；不使用basic／BCa，不做后验切换。seed按196801＋100*block＋10*comparator＋metric确定，block／comparator／metric从0起；使用显式numpy PCG64，版本进环境身份。批量分块最多64重采样，避免大矩阵OOM。

ACC效应=候选correct−基线correct，BCE效应=基线loss−候选loss，均正为改善；方法优势BCE区间下界>0且点估计≥.005，ACC点估计不下降并报告其区间；能力主张对常数和q参考的BCE区间下界>0且点估计≥.005，ACC对TRAIN多数至少+.02。ACC区间不另偷换成必要的显著性门槛；完整18格均报告。

bootstrap是给定训练模型和面板的近似样本推断，不涵盖训练seed／key分布，尾部分位有Monte Carlo误差，不称严格密码证明。10k重采对高置信尾较粗，原样披露，不看结果再加重采次数。

后处理最多3块×6比较指标、每项10000重采，CPU/wall合计30分钟、进程内存建议2GiB，串行。超限报告STATISTICS_INCOMPLETE，不省略失败格仍称全确认。N1配对差只算点效应、必要摘要，不额外bootstrap挤占预算。

## 8. 工程验证与错误分支

本地有限无密码工程：生产encrypt调用0，科学标签0；fake输出和人工Boolean真值允许。优化更新累计≤64、训练呈现≤512，代理与主验收都计，重复pytest不免费。工程fit每次启动前记账，超额停。纯前向／人工oracle测试CPU+wall各30分钟上限，无solver测试另计。

分配：核心代理最多12更新／96呈现；外围假集成最多12／96；主验收最多12／96；干净源码最多14个2更新／8呈现null CLI＝28／112；总64／400，余112呈现不能转额外更新或正式训练。若超过14种入口需要smoke，减少重复test或在正式前改设计，不能暗加。

验收必需：

- N1真实标签侧P不变／初态匹配；CPU FP64独立线性oracle差≤1e-12，FP32模型logit差≤1e-5；若对给定无标签测试面板失败即停，不以不同求和顺序静默放宽。
- Psi独立标量旋转／AND oracle、q_r全零轮常量递推，r6/8/10；不调用生产cipher，不把q称零主密钥。
- 同形状初始state／参数SHA、旧默认字典／SHA／state键及RNG冻结fixture，研究字段严格bool/int／范围／组合拒绝。
- strict save/load、新旧checkpoint、state篡改拒绝、背景初始化不重复补偿。
- record_predictions=false时旧返回／产物行为保持；true捕获不增加forward计数，不提前请求TEST标签；pure report不改RNG／模式／梯度。
- 假进程缺exit、超时、非零、残留产物计量、strict JSON拒重复key／NaN／溢出，未知消耗不释放后再跑。
- Ruff、mypy、限定pytest、干净源码null CLI实际验证；不跑未计数的全仓真实cipher测试，不冒称完整全仓回归。

错误先systematic-debugging，独立review发现的问题核对后修；正式retry0。缺exit／连接中断是执行UNKNOWN，不记50%／失败科学结论。TRAIN拟合与DEV恶化、TRAIN不拟合分开解释，不追加epoch或模型补救。

## 9. 累计预算

| 神经阶段 | fits | real/null | updates | presentations | DEV |
|---|---:|---:|---:|---:|---:|
| N1 | 6 | 3/3 | 49152 | 12582912 | 192 |
| N2 | 10 | 5/5 | 81920 | 20971520 | 320 |
| N3 | 8 | 4/4 | 65536 | 16777216 | 256 |
| N4 | 12 | 6/6 | 98304 | 25165824 | 384 |
| 最大 | 36 | 18/18 | 294912 | 75497472 | 1152 |

上表DEV列均为在线选模机会：最大1152次；另每fit1次selected末端DEV，最多36次，**全部DEV评估上限1188次**，不新增选模机会。最终selected TRAIN／DEV／TEST共108次完整split前向（每次按batch分块），与在线训练／DEV分账，禁止把1152叫成全部DEV调用数。真实encrypt最大2506752，null同数；每fit139264不等于每fit新增独立明文数。重复方法／轮数共用输入有身份和计数。D分支去重只减消耗、不转预算。原公开基线及q计算最多每个fit每split一次公开reference panel加模型前向内q计算，reference次数／处理行数另记，不算cipher调用。

每fit1800秒协作，训练累计≤64800秒（18小时）不是预计实际用时。工程实施验证总wall≤4小时（不含人工等待／设计讨论）、远端同步／预检／取回协作≤2小时，达到任一上限停并报剩余。科学运行不声称硬截止／断线存活。未消耗额度在周期关闭即失效。

积分：toy solver64×2秒、正控8×15秒、正式16×120秒，共88次；checker最多20×90秒，分配4次人工／正控＋16次正式，每类耗尽不借别类，未认证单列；未检查的零正控不得当成已认证。toy前沿／枚举oracle用于人工正控认可，不冒称外部proof检查。公式≤250k vars／1m clauses，单proof512MiB、总产物4GiB、solver/checker进程树建议2GiB、周期含生成检查90分钟。现有工具如果不能强制资源限制则记录协作观测，不写成硬沙箱。失败残留也计；计量失败UNKNOWN并停。正式加密0／GPU0。

## 10. 实施次序、委派边界与交付

本设计批准后先writing-plans逐文件拆任务；以下是职责，不是现在已派编码。

1. 核心接口：implement-deep负责公开表示／模型、config/factory/checkpoint必要接口；人工oracle/TDD，旧行为保护。
2. 接口冻结后：implement负责独立configs／矩阵测试／报告外围，不能同时编辑核心代理拥有的共享文件。
3. 统计核心与采集接线：implement-deep；常规测试／序列化外围implement，独立文件。训练只透传，不塞入bootstrap。
4. 积分I0：不同核心代理负责图／CNF和独立路径验收，不共享实现门表；现有checker缺失时不把正式求解排入可启动列表。
5. code-review只读审查、主模型核实际diff／测试／计数／依赖，代理不再派代理，不把代理报告当科学执行证据。
6. 有完整正式权限才由experiment执行冻结配置；不改源码、不自行加候选；主模型按分支决定和验收。

源码留main，保留当前dirty、papers、旧refs和runs；不stash/reset/clean/stage-all，不自行commit。Git发布若获批只精确列明新任务路径，并从真实index副本验证；Windows只exact fetch/checkout，不远端commit/push、不SCP科学源码。结果新目录runs/received，默认不evaluate。

每阶段交付：矩阵／源码config身份／预算及实际消耗／exit／selected TRAIN/DEV与online分开／全部TEST及基线／证据状态／不确定性／下个合同内分支和剩余额度。最终报告主任务／子空间／积分分表，适用轮数／key／协议和负面证据完整。

## 11. 权限确认与阶段状态

| 项目 | 当前状态 | 若批准本设计 |
|---|---|---|
| 总路线C、当前strong替代与常规代理路由 | 用户已确认 | 不再询问 |
| 本地代码／TDD／有限无密码工程 | 实施计划修订后用户“是”确认，Task1已委派 | 按第8节限定执行，逐任务独立审查 |
| 神经正式36-fit条件矩阵、真实标签 | 未授权 | 需用户明确批准正式矩阵 |
| 指定远端GPU、SSH同步／CLI／取回 | 未授权 | 需一并明确批准，沿已有主机路径和GPU门槛，不安装 |
| 本地精确路径commit/push | 未授权 | 需要明确批准；拟最多3次实现发布（N1、N2、确认统计），无科学产物入Git |
| 积分I0人工solver工程88总额中的64次 | 未授权 | 可随设计明确批准，无生产密码 |
| 积分正控／16正式solver | 未授权且checker门槛未过 | 必须批准且已有checker可用后才运行 |
| checker安装／其它依赖升级 | 禁止 | 本设计不申请自动安装 |
| 旧checkpoint主动evaluate／取回后模型重放 | 禁止 | 本设计不申请 |

独立builder初审发现四个重要歧义：N1 null阈值、bootstrap构造／分位数、r8仅能力资格的确认去向、在线与末端DEV计数。主模型对照当前training调用链后已分别明确：N1独立报警、percentile+linear、双层资格优先、1152在线＋36末端＝1188总DEV。此为设计检查，不是科学验证。

独立builder复核回复“nothing important left”。用户看过本设计及白话说明后要求“推进”，现进入writing-plans；下一门槛是审阅逐文件实施计划，而非重选总路线或代理方式。正式矩阵／GPU／SSH／精确发布／取回仍需明确确认；权限不得由代理或设计文件自我授予。批准范围内连续推进，不每个小步骤重复问。
